#!/usr/bin/env python3
"""Prepare one immutable derived subset for fair local TTS model comparison."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from tiv_tts.audio import load_audio, resample_audio, save_wav
from tiv_tts.text import CharacterTokenizer, normalize_text


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping.")
    return config


def load_signal_warnings(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8", newline="") as handle:
        return {
            str(Path(row["audio_path"]).resolve()): row["warnings"]
            for row in csv.DictReader(handle)
        }


def select_rows(config: dict[str, Any]) -> list[dict[str, str]]:
    selection = config["selection"]
    signal_warnings = load_signal_warnings(Path(config["signal_quality_csv"]))
    audit = json.loads(Path(config["audit_json"]).read_text(encoding="utf-8"))
    minimum_character_count = int(selection["minimum_corpus_character_count"])
    allowed_characters = {
        item["character"]
        for item in audit["transcripts"]["character_inventory"]
        if int(item["count"]) >= minimum_character_count
    }
    with Path(config["audit_csv"]).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    eligible: list[dict[str, str]] = []
    for row in rows:
        duration = float(row["duration_seconds"])
        normalized = normalize_text(row["text"])
        if row["source_group"] != selection["source_group"]:
            continue
        if row["validation_status"] != "valid" or row["mapping_status"] != "mapped":
            continue
        if not (
            float(selection["minimum_duration_seconds"])
            <= duration
            <= float(selection["maximum_duration_seconds"])
        ):
            continue
        if normalized.warnings:
            continue
        if any(character not in allowed_characters for character in normalized.cleaned):
            continue
        if selection["require_no_signal_warnings"] and signal_warnings.get(
            str(Path(row["audio_path"]).resolve())
        ):
            continue
        row["cleaned_text"] = normalized.cleaned
        eligible.append(row)
    rng = random.Random(int(config["seed"]))
    rng.shuffle(eligible)
    selected = eligible[: int(selection["sample_count"])]
    if len(selected) != int(selection["sample_count"]):
        raise RuntimeError(
            f"Only {len(selected)} eligible rows; "
            f"{selection['sample_count']} requested."
        )
    return selected


def assign_split(index: int, config: dict[str, Any]) -> str:
    selection = config["selection"]
    train_end = int(selection["train_count"])
    validation_end = train_end + int(selection["validation_count"])
    if index < train_end:
        return "train"
    if index < validation_end:
        return "validation"
    return "test"


def write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")


def prepare(config_path: Path) -> dict[str, Any]:
    config = load_yaml(config_path)
    selected = select_rows(config)
    output_dir = Path(config["output_dir"])
    wav_dir = output_dir / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)
    target_rate = int(config["audio"]["sample_rate"])
    records: list[dict[str, Any]] = []
    for index, row in enumerate(selected):
        source_path = Path(row["audio_path"])
        waveform, source_rate = load_audio(source_path)
        waveform = resample_audio(waveform, source_rate, target_rate)
        wav_path = wav_dir / f"{row['key']}.wav"
        save_wav(wav_path, waveform, target_rate)
        records.append(
            {
                "sample_id": row["sample_id"],
                "key": row["key"],
                "source_group": row["source_group"],
                "source_audio_path": str(source_path),
                "audio_path": str(wav_path),
                "original_text": row["text"],
                "text": row["cleaned_text"],
                "duration_seconds": waveform.shape[-1] / target_rate,
                "sample_rate": target_rate,
                "split": assign_split(index, config),
                "transformations": [f"mono_resample_{source_rate}_to_{target_rate}"],
            }
        )

    expected = {
        "train": int(config["selection"]["train_count"]),
        "validation": int(config["selection"]["validation_count"]),
        "test": int(config["selection"]["test_count"]),
    }
    actual = Counter(record["split"] for record in records)
    if dict(actual) != expected:
        raise RuntimeError(f"Split mismatch: expected {expected}, got {dict(actual)}")

    tokenizer = CharacterTokenizer.from_texts(
        [record["text"] for record in records]
    )
    write_lines(
        output_dir / "manifest.jsonl",
        [json.dumps(record, ensure_ascii=False) for record in records],
    )
    (output_dir / "vocab.json").write_text(
        json.dumps(tokenizer.vocabulary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    # Coqui's LJSpeech formatter expects key|raw text|normalized text and wavs/.
    for split in ("train", "validation", "test"):
        subset = [record for record in records if record["split"] == split]
        write_lines(
            output_dir / "coqui" / f"metadata_{split}.csv",
            [
                f"{record['key']}|{record['original_text']}|{record['text']}"
                for record in subset
            ],
        )
        # Matcha filelists accept wav path|text. Absolute paths avoid cwd ambiguity.
        write_lines(
            output_dir / "matcha" / f"{split}.txt",
            [
                f"{Path(record['audio_path']).resolve()}|{record['text']}"
                for record in subset
            ],
        )

    summary = {
        "config": str(config_path),
        "source_group": records[0]["source_group"],
        "samples": len(records),
        "splits": dict(actual),
        "duration_seconds": sum(record["duration_seconds"] for record in records),
        "duration_minutes": sum(
            record["duration_seconds"] for record in records
        )
        / 60,
        "sample_rate": target_rate,
        "vocabulary_size": len(tokenizer.vocabulary),
        "vocabulary": tokenizer.vocabulary,
        "minimum_corpus_character_count": int(
            config["selection"]["minimum_corpus_character_count"]
        ),
        "raw_dataset_modified": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/tiv_bakeoff.yaml"),
    )
    args = parser.parse_args()
    print(json.dumps(prepare(args.config), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
