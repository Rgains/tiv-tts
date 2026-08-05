#!/usr/bin/env python3
"""Build the derived smoke-test dataset without training anything.

This runs only the selection and preparation half of the smoke test: it picks
deterministic, audit-clean samples, decodes them to derived 22.05 kHz mono WAVs,
extracts mel features, and writes a manifest and vocabulary under ``work_dir``.
No model is constructed and no checkpoint is written, so it can be run before a
training smoke test has been approved.

Source files under the dataset root are opened read-only.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from tiv_tts.smoke import load_config, prepare_records, select_samples, set_deterministic_seed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/tiv_smoke_gpu.yaml"),
    )
    args = parser.parse_args()

    config = load_config(args.config)
    set_deterministic_seed(int(config["seed"]))
    selected = select_samples(config)
    records, tokenizer = prepare_records(config, selected)

    work_dir = Path(config["work_dir"])
    splits = collections.Counter(record["split"] for record in records)
    durations = [record["duration_seconds"] for record in records]
    summary = {
        "status": "prepared",
        "purpose": "Derived smoke-test dataset only; no training was performed.",
        "config": str(args.config),
        "work_dir": str(work_dir),
        "manifest_path": str(work_dir / "manifest.jsonl"),
        "vocabulary_path": str(work_dir / "vocab.json"),
        "sample_count": len(records),
        "splits": dict(sorted(splits.items())),
        "source_group": records[0]["source_group"],
        "vocabulary_size": len(tokenizer.vocabulary),
        "sample_rate": int(config["audio"]["sample_rate"]),
        "total_duration_seconds": round(sum(durations), 3),
        "duration_seconds_min": round(min(durations), 3),
        "duration_seconds_max": round(max(durations), 3),
    }
    (work_dir / "prepare_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
