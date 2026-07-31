#!/usr/bin/env python3
"""Audit the Tiv TTS source dataset without modifying it.

The script reads mapping TSV files and audio metadata, computes file hashes,
checks transcript Unicode, and writes derived reports outside the dataset.
It intentionally does not preprocess, rename, delete, or rewrite source files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import unicodedata
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


REQUIRED_MAPPING_COLUMNS = ("audio_filename", "key", "sentence", "attempts")
AUDIO_SUFFIXES = {".mp3", ".wav", ".flac", ".m4a", ".ogg", ".opus"}
BATCH_PATTERN = re.compile(
    r"^(?P<prefix>.+)_(?P<batch_id>\d+)_"
    r"(?P<clips>\d+)clips_(?P<seconds>\d+)s_(?P<timestamp>\d{8}-\d{4})$"
)
AFINFO_FORMAT_PATTERN = re.compile(
    r"Data format:\s+(?P<channels>\d+)\s+ch,\s+"
    r"(?P<sample_rate>[\d.]+)\s+Hz,\s+(?P<codec>\S+)"
)
AFINFO_DURATION_PATTERN = re.compile(
    r"estimated duration:\s+(?P<duration>[\d.]+)\s+sec"
)
AFINFO_BITRATE_PATTERN = re.compile(r"bit rate:\s+(?P<bitrate>\d+)\s+bits per second")


@dataclass(slots=True)
class AudioProbe:
    """Metadata and integrity information for one audio file."""

    path: str
    sha256: str = ""
    readable: bool = False
    duration_seconds: float | None = None
    sample_rate: int | None = None
    channels: int | None = None
    codec: str = ""
    container: str = ""
    bit_depth: int | None = None
    bit_rate: int | None = None
    probe_backend: str = ""
    probe_error: str = ""


@dataclass(slots=True)
class SampleRecord:
    """One mapped or unmatched dataset sample."""

    sample_id: str
    source_group: str
    audio_path: str
    audio_filename: str
    key: str
    text: str
    attempts: str
    mapping_path: str
    mapping_row: int | None
    mapping_status: str
    duration_seconds: float | None = None
    sample_rate: int | None = None
    channels: int | None = None
    codec: str = ""
    container: str = ""
    bit_depth: int | None = None
    bit_rate: int | None = None
    sha256: str = ""
    validation_status: str = "valid"
    recommended_exclusion: bool = False
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Create a read-only audit of a Tiv TTS dataset."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Root containing source-group directories and mapping.tsv files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/dataset_audit"),
        help="Derived JSON and CSV output directory (must be outside the dataset).",
    )
    parser.add_argument(
        "--markdown-report",
        type=Path,
        default=Path("docs/dataset_audit.md"),
        help="Path for the human-readable Markdown report.",
    )
    parser.add_argument(
        "--rare-character-threshold",
        type=int,
        default=5,
        help="Report characters occurring fewer than this many times.",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=0.5,
        help="Recommend exclusion of clips shorter than this many seconds.",
    )
    parser.add_argument(
        "--long-duration",
        type=float,
        default=20.0,
        help="Warn on clips longer than this many seconds.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(8, os.cpu_count() or 1)),
        help="Parallel workers for audio probing and hashing.",
    )
    return parser.parse_args()


def is_relative_to(path: Path, parent: Path) -> bool:
    """Return whether ``path`` is inside ``parent``."""

    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def validate_paths(dataset: Path, output_dir: Path, markdown_report: Path) -> None:
    """Protect the source tree by rejecting output paths inside it."""

    if not dataset.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset}")
    dataset_resolved = dataset.resolve()
    for label, candidate in (
        ("output directory", output_dir.resolve()),
        ("Markdown report", markdown_report.resolve()),
    ):
        if is_relative_to(candidate, dataset_resolved):
            raise ValueError(f"{label} must be outside the raw dataset: {candidate}")


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest for ``path``."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def probe_with_afinfo(path: Path) -> AudioProbe:
    """Probe an audio file with macOS ``afinfo``."""

    result = AudioProbe(path=str(path), probe_backend="afinfo")
    try:
        completed = subprocess.run(
            ["afinfo", str(path)],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        result.probe_error = str(exc)
        return result

    output = f"{completed.stdout}\n{completed.stderr}"
    format_match = AFINFO_FORMAT_PATTERN.search(output)
    duration_match = AFINFO_DURATION_PATTERN.search(output)
    bitrate_match = AFINFO_BITRATE_PATTERN.search(output)
    type_match = re.search(r"File type ID:\s+(\S+)", output)

    if format_match:
        result.channels = int(format_match.group("channels"))
        result.sample_rate = round(float(format_match.group("sample_rate")))
        result.codec = format_match.group("codec").lstrip(".")
    if duration_match:
        result.duration_seconds = float(duration_match.group("duration"))
    if bitrate_match:
        result.bit_rate = int(bitrate_match.group("bitrate"))
    if type_match:
        result.container = type_match.group(1)

    result.readable = (
        completed.returncode == 0
        and result.duration_seconds is not None
        and result.duration_seconds > 0
        and result.sample_rate is not None
        and result.channels is not None
    )
    if not result.readable:
        concise = " ".join(line.strip() for line in output.splitlines() if line.strip())
        result.probe_error = concise[-500:] or f"afinfo exited {completed.returncode}"
    return result


def probe_with_ffprobe(path: Path) -> AudioProbe:
    """Probe an audio file with FFmpeg's ``ffprobe``."""

    result = AudioProbe(path=str(path), probe_backend="ffprobe")
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        (
            "stream=codec_name,sample_rate,channels,bits_per_sample,bit_rate:"
            "format=format_name,duration,bit_rate"
        ),
        "-of",
        "json",
        str(path),
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        result.probe_error = str(exc)
        return result

    if completed.returncode != 0:
        result.probe_error = completed.stderr.strip()[-500:]
        return result
    try:
        payload = json.loads(completed.stdout)
        stream = payload.get("streams", [{}])[0]
        format_data = payload.get("format", {})
        result.codec = str(stream.get("codec_name") or "")
        result.container = str(format_data.get("format_name") or "")
        result.sample_rate = int(stream["sample_rate"]) if stream.get("sample_rate") else None
        result.channels = int(stream["channels"]) if stream.get("channels") else None
        bits = stream.get("bits_per_sample")
        result.bit_depth = int(bits) if bits not in (None, "", "0", 0) else None
        duration = format_data.get("duration")
        result.duration_seconds = float(duration) if duration is not None else None
        bitrate = stream.get("bit_rate") or format_data.get("bit_rate")
        result.bit_rate = int(bitrate) if bitrate else None
        result.readable = (
            result.duration_seconds is not None
            and result.duration_seconds > 0
            and result.sample_rate is not None
            and result.channels is not None
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        result.probe_error = f"Could not parse ffprobe output: {exc}"
    return result


def probe_and_hash(path_text: str, backend: str) -> AudioProbe:
    """Worker entry point for hashing and metadata probing."""

    path = Path(path_text)
    probe = probe_with_ffprobe(path) if backend == "ffprobe" else probe_with_afinfo(path)
    try:
        probe.sha256 = sha256_file(path)
    except OSError as exc:
        probe.probe_error = f"{probe.probe_error}; hash failed: {exc}".strip("; ")
        probe.readable = False
    return probe


def select_probe_backend() -> str:
    """Select an installed, non-mutating audio metadata probe."""

    if shutil.which("ffprobe"):
        return "ffprobe"
    if shutil.which("afinfo"):
        return "afinfo"
    raise RuntimeError("Neither ffprobe nor afinfo is installed; cannot validate audio.")


def probe_audio_files(
    paths: list[Path], backend: str, workers: int
) -> dict[str, AudioProbe]:
    """Probe and hash audio files in parallel."""

    probes: dict[str, AudioProbe] = {}
    total = len(paths)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(probe_and_hash, str(path.resolve()), backend): path
            for path in paths
        }
        for completed_count, future in enumerate(as_completed(futures), start=1):
            path = futures[future]
            try:
                probe = future.result()
            except Exception as exc:  # pragma: no cover - defensive worker boundary
                probe = AudioProbe(
                    path=str(path.resolve()),
                    probe_backend=backend,
                    probe_error=f"Worker failed: {exc}",
                )
            probes[str(path.resolve())] = probe
            if completed_count % 100 == 0 or completed_count == total:
                print(
                    f"Probed {completed_count}/{total} audio files",
                    file=sys.stderr,
                    flush=True,
                )
    return probes


def parse_source_group_name(name: str) -> dict[str, Any]:
    """Parse declared clip count and duration from an export directory name."""

    match = BATCH_PATTERN.match(name)
    if not match:
        return {
            "source_group": name,
            "batch_id": None,
            "declared_clips": None,
            "declared_seconds": None,
            "export_timestamp": None,
        }
    groups = match.groupdict()
    return {
        "source_group": name,
        "batch_id": int(groups["batch_id"]),
        "declared_clips": int(groups["clips"]),
        "declared_seconds": int(groups["seconds"]),
        "export_timestamp": groups["timestamp"],
    }


def read_mapping(mapping_path: Path) -> tuple[list[dict[str, str]], list[str]]:
    """Read and validate a source mapping TSV."""

    errors: list[str] = []
    try:
        with mapping_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            columns = tuple(reader.fieldnames or ())
            missing_columns = [
                column for column in REQUIRED_MAPPING_COLUMNS if column not in columns
            ]
            if missing_columns:
                errors.append(
                    f"missing required columns: {', '.join(missing_columns)}"
                )
            rows = [dict(row) for row in reader]
    except (OSError, UnicodeError, csv.Error) as exc:
        return [], [f"could not read UTF-8 TSV: {exc}"]
    return rows, errors


def unicode_character_rows(counter: Counter[str], rare_threshold: int) -> list[dict[str, Any]]:
    """Build a stable, descriptive Unicode character inventory."""

    rows: list[dict[str, Any]] = []
    for char, count in sorted(counter.items(), key=lambda item: ord(item[0])):
        rows.append(
            {
                "character": char,
                "display": repr(char),
                "codepoint": f"U+{ord(char):04X}",
                "unicode_name": unicodedata.name(char, "UNNAMED"),
                "category": unicodedata.category(char),
                "count": count,
                "rare": count < rare_threshold,
            }
        )
    return rows


def confusable_groups(characters: Iterable[str]) -> list[dict[str, Any]]:
    """Report corpus characters that collapse under conservative compatibility keys."""

    cross_script_skeleton = {
        "Α": "A",
        "Κ": "K",
        "ο": "o",
        "τ": "t",
        "υ": "u",
        "К": "K",
        "а": "a",
        "е": "e",
        "к": "k",
        "о": "o",
        "р": "p",
        "т": "t",
    }
    by_key: dict[str, set[str]] = defaultdict(set)
    for char in characters:
        if char.isspace():
            key = "SPACE"
        elif char in "'’‘ʼ`´":
            key = "APOSTROPHE"
        elif char in "-‐‑‒–—―":
            key = "DASH"
        elif char in '"“”„‟':
            key = "DOUBLE_QUOTE"
        elif char in cross_script_skeleton:
            key = f"HOMOGLYPH:{cross_script_skeleton[char]}"
        else:
            key = unicodedata.normalize("NFKC", char)
            if key in cross_script_skeleton.values():
                key = f"HOMOGLYPH:{key}"
        by_key[key].add(char)

    groups: list[dict[str, Any]] = []
    for key, variants in sorted(by_key.items()):
        if len(variants) < 2:
            continue
        ordered = sorted(variants, key=ord)
        groups.append(
            {
                "normalization_key": key,
                "characters": ordered,
                "codepoints": [f"U+{ord(char):04X}" for char in ordered],
            }
        )
    return groups


def percentile(values: list[float], fraction: float) -> float | None:
    """Return a simple interpolated percentile."""

    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def add_issue(record: SampleRecord, severity: str, message: str) -> None:
    """Attach a normalized issue to a sample record."""

    target = record.errors if severity == "error" else record.warnings
    if message not in target:
        target.append(message)


def finalize_status(record: SampleRecord) -> None:
    """Derive validation status and exclusion recommendation."""

    if record.errors:
        record.validation_status = "invalid"
        record.recommended_exclusion = True
    elif record.warnings:
        record.validation_status = "warning"
    else:
        record.validation_status = "valid"


def relative_or_absolute(path: Path, base: Path) -> str:
    """Prefer paths relative to the current working directory."""

    try:
        return str(path.resolve().relative_to(base.resolve()))
    except ValueError:
        return str(path.resolve())


def build_records(
    dataset: Path,
    probes: dict[str, AudioProbe],
    min_duration: float,
    long_duration: float,
) -> tuple[list[SampleRecord], list[dict[str, Any]], list[str], list[str]]:
    """Join mappings to audio and classify sample-level issues."""

    records: list[SampleRecord] = []
    source_groups: list[dict[str, Any]] = []
    dataset_errors: list[str] = []
    dataset_warnings: list[str] = []
    cwd = Path.cwd()

    directories = sorted(path for path in dataset.iterdir() if path.is_dir())
    if not directories:
        dataset_errors.append("Dataset contains no source-group directories.")

    for source_dir in directories:
        group_info = parse_source_group_name(source_dir.name)
        mapping_path = source_dir / "mapping.tsv"
        audio_paths = sorted(
            path
            for path in source_dir.iterdir()
            if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES
        )
        audio_by_name = {path.name: path for path in audio_paths}

        if not mapping_path.is_file():
            rows: list[dict[str, str]] = []
            mapping_errors = ["mapping.tsv is missing"]
        else:
            rows, mapping_errors = read_mapping(mapping_path)
        if mapping_errors:
            dataset_errors.extend(
                f"{source_dir.name}/mapping.tsv: {error}" for error in mapping_errors
            )

        referenced_names: Counter[str] = Counter()
        group_durations: list[float] = []
        for row_number, row in enumerate(rows, start=2):
            audio_filename = (row.get("audio_filename") or "").strip()
            key = (row.get("key") or "").strip()
            text = row.get("sentence") or ""
            attempts = (row.get("attempts") or "").strip()
            referenced_names[audio_filename] += 1
            audio_path = audio_by_name.get(audio_filename)
            intended_path = audio_path or (source_dir / audio_filename)
            record = SampleRecord(
                sample_id=f"{source_dir.name}:{key or Path(audio_filename).stem}",
                source_group=source_dir.name,
                audio_path=relative_or_absolute(intended_path, cwd),
                audio_filename=audio_filename,
                key=key,
                text=text,
                attempts=attempts,
                mapping_path=relative_or_absolute(mapping_path, cwd),
                mapping_row=row_number,
                mapping_status="mapped" if audio_path else "transcript_without_audio",
            )

            if not audio_filename:
                add_issue(record, "error", "empty audio filename")
            if audio_filename and Path(audio_filename).name != audio_filename:
                add_issue(record, "error", "audio filename contains a directory component")
            if not key:
                add_issue(record, "warning", "empty key")
            elif audio_filename and key != Path(audio_filename).stem:
                add_issue(record, "warning", "key does not match audio filename stem")
            if not text.strip():
                add_issue(record, "error", "empty transcript")
            if text != unicodedata.normalize("NFC", text):
                add_issue(record, "warning", "transcript is not Unicode NFC")
            if attempts and not attempts.isdigit():
                add_issue(record, "warning", "attempts is not an integer")
            elif attempts.isdigit() and int(attempts) > 10:
                add_issue(record, "warning", "attempts is unusually high (greater than 10)")
            if "\\" in text:
                add_issue(
                    record,
                    "warning",
                    "literal backslash escape requires transcript review",
                )
            non_latin_scripts = sorted(
                {
                    unicodedata.name(char, "UNNAMED").split()[0]
                    for char in text
                    if unicodedata.category(char).startswith("L")
                    and "LATIN" not in unicodedata.name(char, "")
                }
            )
            if non_latin_scripts:
                add_issue(
                    record,
                    "warning",
                    "non-Latin letters require transcript review: "
                    + ", ".join(non_latin_scripts),
                )
            if any(unicodedata.category(char) == "So" for char in text):
                add_issue(record, "warning", "emoji or other symbol requires transcript review")
            if referenced_names[audio_filename] > 1:
                add_issue(record, "error", "audio filename is referenced more than once")

            if audio_path is None:
                add_issue(record, "error", "transcript references missing audio")
            else:
                probe = probes.get(str(audio_path.resolve()))
                if probe is None or not probe.readable:
                    detail = probe.probe_error if probe else "audio was not probed"
                    add_issue(record, "error", f"corrupt or unreadable audio: {detail}")
                else:
                    record.duration_seconds = probe.duration_seconds
                    record.sample_rate = probe.sample_rate
                    record.channels = probe.channels
                    record.codec = probe.codec
                    record.container = probe.container
                    record.bit_depth = probe.bit_depth
                    record.bit_rate = probe.bit_rate
                    record.sha256 = probe.sha256
                    group_durations.append(probe.duration_seconds or 0.0)
                    if (probe.duration_seconds or 0.0) < min_duration:
                        add_issue(
                            record,
                            "error",
                            f"duration below {min_duration:g} seconds",
                        )
                    if (probe.duration_seconds or 0.0) > long_duration:
                        add_issue(
                            record,
                            "warning",
                            f"duration above {long_duration:g} seconds",
                        )
                    if probe.channels != 1:
                        add_issue(record, "warning", f"audio has {probe.channels} channels")

            finalize_status(record)
            records.append(record)

        for audio_filename, audio_path in sorted(audio_by_name.items()):
            if audio_filename in referenced_names:
                continue
            probe = probes.get(str(audio_path.resolve()))
            record = SampleRecord(
                sample_id=f"{source_dir.name}:{audio_path.stem}",
                source_group=source_dir.name,
                audio_path=relative_or_absolute(audio_path, cwd),
                audio_filename=audio_filename,
                key=audio_path.stem,
                text="",
                attempts="",
                mapping_path=relative_or_absolute(mapping_path, cwd),
                mapping_row=None,
                mapping_status="audio_without_transcript",
            )
            if probe:
                record.duration_seconds = probe.duration_seconds
                record.sample_rate = probe.sample_rate
                record.channels = probe.channels
                record.codec = probe.codec
                record.container = probe.container
                record.bit_depth = probe.bit_depth
                record.bit_rate = probe.bit_rate
                record.sha256 = probe.sha256
            add_issue(record, "error", "audio file has no transcript mapping")
            finalize_status(record)
            records.append(record)

        actual_duration = sum(group_durations)
        group_info.update(
            {
                "mapping_rows": len(rows),
                "audio_files": len(audio_paths),
                "readable_audio_files": len(group_durations),
                "actual_duration_seconds": actual_duration,
                "duration_difference_seconds": (
                    actual_duration - group_info["declared_seconds"]
                    if group_info["declared_seconds"] is not None
                    else None
                ),
                "actual_to_declared_duration_ratio": (
                    actual_duration / group_info["declared_seconds"]
                    if group_info["declared_seconds"]
                    else None
                ),
                "mapping_errors": mapping_errors,
            }
        )
        source_groups.append(group_info)
        if (
            group_info["declared_clips"] is not None
            and group_info["declared_clips"] != len(audio_paths)
        ):
            dataset_warnings.append(
                f"{source_dir.name}: directory declares "
                f"{group_info['declared_clips']} clips but contains {len(audio_paths)}"
            )
        if group_info["declared_seconds"] is not None and abs(
            actual_duration - group_info["declared_seconds"]
        ) > max(5.0, group_info["declared_seconds"] * 0.02):
            dataset_warnings.append(
                f"{source_dir.name}: directory name declares "
                f"{group_info['declared_seconds']} seconds, while selected MP3 files "
                f"total {actual_duration:.3f} seconds; the declared value may include "
                "discarded attempts or session time"
            )

    return records, source_groups, dataset_errors, dataset_warnings


def flag_duplicates(records: list[SampleRecord]) -> dict[str, Any]:
    """Flag exact duplicate audio and transcript groups."""

    audio_groups: dict[str, list[SampleRecord]] = defaultdict(list)
    transcript_groups: dict[str, list[SampleRecord]] = defaultdict(list)
    for record in records:
        if record.sha256:
            audio_groups[record.sha256].append(record)
        if record.text.strip():
            transcript_groups[record.text.strip()].append(record)

    duplicate_audio = []
    for digest, group in sorted(audio_groups.items()):
        if len(group) < 2:
            continue
        paths = [record.audio_path for record in group]
        duplicate_audio.append({"sha256": digest, "count": len(group), "paths": paths})
        for record in group:
            add_issue(record, "warning", f"exact duplicate audio in group of {len(group)}")
            finalize_status(record)

    duplicate_transcripts = []
    for text, group in sorted(transcript_groups.items()):
        if len(group) < 2:
            continue
        duplicate_transcripts.append(
            {
                "text": text,
                "count": len(group),
                "sample_ids": [record.sample_id for record in group],
            }
        )
        for record in group:
            add_issue(
                record,
                "warning",
                f"exact duplicate transcript in group of {len(group)}",
            )
            finalize_status(record)

    return {
        "duplicate_audio_groups": duplicate_audio,
        "duplicate_transcript_groups": duplicate_transcripts,
    }


def transcript_analysis(
    records: list[SampleRecord], rare_threshold: int
) -> dict[str, Any]:
    """Analyze corpus text without changing it."""

    texts = [record.text for record in records if record.mapping_row is not None]
    character_counts = Counter("".join(texts))
    non_nfc = [
        record.sample_id
        for record in records
        if record.text and record.text != unicodedata.normalize("NFC", record.text)
    ]
    empty = [
        record.sample_id
        for record in records
        if record.mapping_row is not None and not record.text.strip()
    ]
    rows_with_digits = [
        record.sample_id for record in records if any(char.isdigit() for char in record.text)
    ]
    rows_with_urls_or_email = [
        record.sample_id
        for record in records
        if re.search(r"(?:https?://|www\.|\S+@\S+\.)", record.text, re.IGNORECASE)
    ]
    rows_with_all_caps_tokens = [
        record.sample_id
        for record in records
        if re.search(r"\b[A-ZÀ-ÖØ-Þ]{2,}\b", record.text)
    ]
    rows_with_literal_backslashes = [
        record.sample_id for record in records if "\\" in record.text
    ]
    rows_with_non_latin_letters = [
        record.sample_id
        for record in records
        if any(
            unicodedata.category(char).startswith("L")
            and "LATIN" not in unicodedata.name(char, "")
            for char in record.text
        )
    ]
    rows_with_emoji_or_symbols = [
        record.sample_id
        for record in records
        if any(unicodedata.category(char) == "So" for char in record.text)
    ]
    attempts_distribution = Counter(
        record.attempts
        for record in records
        if record.mapping_row is not None and record.attempts
    )
    punctuation_counts = {
        char: count
        for char, count in character_counts.items()
        if unicodedata.category(char).startswith(("P", "S"))
    }
    return {
        "encoding": "UTF-8",
        "normalization_target": "NFC",
        "unique_character_count_including_whitespace": len(character_counts),
        "character_inventory": unicode_character_rows(
            character_counts, rare_threshold
        ),
        "rare_character_threshold": rare_threshold,
        "non_nfc_sample_count": len(non_nfc),
        "non_nfc_sample_ids": non_nfc,
        "empty_transcript_count": len(empty),
        "empty_transcript_sample_ids": empty,
        "rows_with_digits_count": len(rows_with_digits),
        "rows_with_digits_sample_ids": rows_with_digits,
        "rows_with_urls_or_email_count": len(rows_with_urls_or_email),
        "rows_with_urls_or_email_sample_ids": rows_with_urls_or_email,
        "rows_with_all_caps_tokens_count": len(rows_with_all_caps_tokens),
        "rows_with_all_caps_tokens_sample_ids": rows_with_all_caps_tokens,
        "rows_with_literal_backslashes_count": len(rows_with_literal_backslashes),
        "rows_with_literal_backslashes_sample_ids": rows_with_literal_backslashes,
        "rows_with_non_latin_letters_count": len(rows_with_non_latin_letters),
        "rows_with_non_latin_letters_sample_ids": rows_with_non_latin_letters,
        "rows_with_emoji_or_symbols_count": len(rows_with_emoji_or_symbols),
        "rows_with_emoji_or_symbols_sample_ids": rows_with_emoji_or_symbols,
        "attempts_distribution": dict(sorted(attempts_distribution.items())),
        "punctuation_and_symbol_counts": dict(
            sorted(punctuation_counts.items(), key=lambda item: ord(item[0]))
        ),
        "visually_similar_character_groups": confusable_groups(character_counts),
        "mixed_language_assessment": (
            "Not automatically classified: Tiv and English share Latin-script "
            "vocabulary patterns; native-speaker or language-ID review is required."
        ),
    }


def count_values(
    records: Iterable[SampleRecord], attribute: str, *, include_none: bool = False
) -> dict[str, int]:
    """Count normalized values of one record attribute."""

    counts: Counter[str] = Counter()
    for record in records:
        value = getattr(record, attribute)
        if value is None and not include_none:
            continue
        counts[str(value)] += 1
    return dict(sorted(counts.items()))


def summarize(
    dataset: Path,
    records: list[SampleRecord],
    source_groups: list[dict[str, Any]],
    duplicates: dict[str, Any],
    transcript: dict[str, Any],
    dataset_errors: list[str],
    dataset_warnings: list[str],
    backend: str,
) -> dict[str, Any]:
    """Build the complete machine-readable audit payload."""

    mapped = [record for record in records if record.mapping_status == "mapped"]
    readable = [record for record in mapped if record.duration_seconds is not None]
    durations = [
        record.duration_seconds
        for record in readable
        if record.duration_seconds is not None
    ]
    statuses = Counter(record.validation_status for record in records)
    exclusions = [record for record in records if record.recommended_exclusion]
    warnings = [record for record in records if record.validation_status == "warning"]

    return {
        "audit": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "dataset_path": str(dataset.resolve()),
            "read_only_policy": (
                "Source files were opened only for reading; all outputs are outside "
                "the dataset tree."
            ),
            "probe_backend": backend,
            "signal_quality_assessment": {
                "status": "not_assessed",
                "fields": [
                    "long_silences",
                    "excessive_background_noise",
                    "clipping",
                    "very_low_volume",
                ],
                "reason": (
                    "No supported signal-analysis decoder is installed. Metadata "
                    "probing cannot establish waveform-level quality."
                ),
            },
        },
        "dataset": {
            "source_group_count": len(source_groups),
            "source_groups": source_groups,
            "declared_export_duration_seconds": sum(
                group["declared_seconds"] or 0 for group in source_groups
            ),
            "declared_export_duration_note": (
                "Directory-name duration totals exceed selected MP3 duration and may "
                "include discarded attempts or full session time; they are not used "
                "as training duration."
            ),
            "speaker_count": None,
            "speaker_metadata_status": (
                "Unknown: mapping.tsv and directory names contain no explicit "
                "speaker, gender, age, dialect, or region fields."
            ),
            "session_metadata_status": (
                "Fourteen export directories are available as source groups, but "
                "their relationship to speakers and recording sessions is unverified."
            ),
            "license_and_consent_status": (
                "Unverified: no licence, consent, release, or provenance file was "
                "found in the dataset root."
            ),
            "mapping_format": "UTF-8 TSV",
            "mapping_columns": list(REQUIRED_MAPPING_COLUMNS),
            "mapped_rows": len(mapped),
            "total_sample_records_including_unmatched": len(records),
            "readable_mapped_audio_files": len(readable),
            "duration_seconds": sum(durations),
            "duration_hours": sum(durations) / 3600 if durations else 0.0,
            "duration_statistics_seconds": {
                "minimum": min(durations) if durations else None,
                "maximum": max(durations) if durations else None,
                "mean": statistics.fmean(durations) if durations else None,
                "median": statistics.median(durations) if durations else None,
                "p05": percentile(durations, 0.05),
                "p95": percentile(durations, 0.95),
            },
            "sample_rates": count_values(readable, "sample_rate"),
            "channels": count_values(readable, "channels"),
            "codecs": count_values(readable, "codec"),
            "containers": count_values(readable, "container"),
            "bit_depths": count_values(readable, "bit_depth"),
            "bit_depth_note": (
                "Compressed MP3 metadata does not expose a PCM bit depth; null is "
                "expected until decoding or conversion."
            ),
            "validation_status_counts": dict(sorted(statuses.items())),
            "recommended_exclusion_count": len(exclusions),
            "warning_sample_count": len(warnings),
            "dataset_errors": dataset_errors,
            "dataset_warnings": dataset_warnings,
        },
        "duplicates": duplicates,
        "transcripts": transcript,
        "recommended_exclusions": [
            {
                "sample_id": record.sample_id,
                "audio_path": record.audio_path,
                "reasons": record.errors,
            }
            for record in exclusions
        ],
        "warning_samples": [
            {
                "sample_id": record.sample_id,
                "audio_path": record.audio_path,
                "warnings": record.warnings,
            }
            for record in warnings
        ],
    }


def format_number(value: float | int | None, digits: int = 3) -> str:
    """Format optional numeric report values."""

    if value is None:
        return "not available"
    if isinstance(value, int):
        return f"{value:,}"
    return f"{value:,.{digits}f}"


def character_table(inventory: list[dict[str, Any]]) -> str:
    """Render the full character inventory compactly."""

    parts = []
    for item in inventory:
        display = item["display"].replace("|", "\\|")
        parts.append(
            f"{display} ({item['codepoint']}, {item['unicode_name']}, {item['count']})"
        )
    return "; ".join(parts)


def render_markdown(report: dict[str, Any]) -> str:
    """Render the requested human-readable audit format."""

    dataset = report["dataset"]
    transcript = report["transcripts"]
    duration = dataset["duration_statistics_seconds"]
    statuses = dataset["validation_status_counts"]
    duplicates = report["duplicates"]
    signal = report["audit"]["signal_quality_assessment"]

    source_rows = []
    for group in dataset["source_groups"]:
        source_rows.append(
            "| {source_group} | {audio_files} | {mapping_rows} | {readable} | "
            "{declared} | {duration:.3f} | {ratio:.1%} |".format(
                source_group=group["source_group"],
                audio_files=group["audio_files"],
                mapping_rows=group["mapping_rows"],
                readable=group["readable_audio_files"],
                declared=group["declared_seconds"] or "n/a",
                duration=group["actual_duration_seconds"],
                ratio=group["actual_to_declared_duration_ratio"] or 0.0,
            )
        )

    exclusion_lines = [
        f"- `{item['sample_id']}`: {', '.join(item['reasons'])}"
        for item in report["recommended_exclusions"][:100]
    ]
    if len(report["recommended_exclusions"]) > 100:
        exclusion_lines.append(
            f"- …and {len(report['recommended_exclusions']) - 100} more; see samples.csv."
        )
    if not exclusion_lines:
        exclusion_lines = ["- None from metadata and mapping checks."]
    dataset_warning_lines = [
        f"- {warning}" for warning in dataset["dataset_warnings"]
    ] or ["- None."]

    rare = [
        item
        for item in transcript["character_inventory"]
        if item["rare"] and not item["character"].isspace()
    ]
    rare_text = "; ".join(
        f"{item['display']} ({item['codepoint']}, {item['count']})" for item in rare
    )
    if not rare_text:
        rare_text = "None."

    return f"""# Tiv TTS dataset audit

Generated: {report["audit"]["generated_at_utc"]}

## CURRENT STATE

- Repository: No existing source code, Git repository, README, configuration, or dependency manifest was present at inspection time. The audit script is the first project code added.
- Dataset: `{report["audit"]["dataset_path"]}` contains {dataset["source_group_count"]} source-group exports.
- Framework: No TTS framework or pretrained model has been selected or installed.
- Environment: Python {sys.version.split()[0]}; audio metadata probed with `{report["audit"]["probe_backend"]}`. PyTorch, torchaudio, FFmpeg, and common Python audio packages were not present at inspection time.
- Risks: Speaker/session identity is unverified; licence and consent are unverified; waveform-level quality remains unassessed; the directory is not currently a Git repository.

## DATASET AUDIT

- Files: {dataset["mapped_rows"]:,} mapped clips; {dataset["total_sample_records_including_unmatched"]:,} total sample records including unmatched files.
- Duration: {format_number(dataset["duration_hours"], 4)} hours ({format_number(dataset["duration_seconds"], 3)} seconds).
- Export-name duration: {format_number(dataset["declared_export_duration_seconds"] / 3600, 4)} hours; this is not used as selected-audio duration because every source-group name overstates the sum of its final MP3 files.
- Speakers: Unknown. No explicit speaker, gender, age, dialect, or region metadata is present.
- Source groups: {dataset["source_group_count"]} export directories; do not treat these as speakers until provenance is confirmed.
- Audio formats: codecs `{json.dumps(dataset["codecs"], ensure_ascii=False)}`; containers `{json.dumps(dataset["containers"], ensure_ascii=False)}`.
- Sample rates: `{json.dumps(dataset["sample_rates"])}`.
- Channels: `{json.dumps(dataset["channels"])}`.
- Bit depth: Not available for compressed MP3 without decoding.
- Transcript format: UTF-8 TSV with columns `{", ".join(dataset["mapping_columns"])}`.
- Invalid samples: {statuses.get("invalid", 0):,}.
- Warning samples: {statuses.get("warning", 0):,}.
- Valid samples: {statuses.get("valid", 0):,}.
- Recommended exclusions: {dataset["recommended_exclusion_count"]:,}; only objective mapping/audio-integrity failures and clips below the configured minimum are automatically recommended.
- Duration statistics: minimum {format_number(duration["minimum"])} s; maximum {format_number(duration["maximum"])} s; mean {format_number(duration["mean"])} s; median {format_number(duration["median"])} s; p05 {format_number(duration["p05"])} s; p95 {format_number(duration["p95"])} s.
- Audio without transcripts: {sum(1 for item in report["recommended_exclusions"] if "audio file has no transcript mapping" in item["reasons"]):,}.
- Transcripts without audio: {sum(1 for item in report["recommended_exclusions"] if "transcript references missing audio" in item["reasons"]):,}.
- Duplicate audio: {len(duplicates["duplicate_audio_groups"]):,} exact SHA-256 groups.
- Duplicate transcripts: {len(duplicates["duplicate_transcript_groups"]):,} exact-text groups; these are warnings, not automatic exclusions.
- Empty transcripts: {transcript["empty_transcript_count"]:,}.
- Unicode NFC inconsistencies: {transcript["non_nfc_sample_count"]:,}.
- Character inventory: {transcript["unique_character_count_including_whitespace"]:,} unique characters including whitespace. Full inventory: {character_table(transcript["character_inventory"])}
- Rare characters (fewer than {transcript["rare_character_threshold"]} occurrences): {rare_text}
- Visually similar character groups: `{json.dumps(transcript["visually_similar_character_groups"], ensure_ascii=False)}`.
- Rows containing digits: {transcript["rows_with_digits_count"]:,}.
- Rows containing all-capital tokens: {transcript["rows_with_all_caps_tokens_count"]:,}.
- Rows containing literal backslash escapes: {transcript["rows_with_literal_backslashes_count"]:,}; these require conservative quote-normalisation review.
- Rows containing non-Latin letters: {transcript["rows_with_non_latin_letters_count"]:,}; Greek, Cyrillic, and Hebrew fragments require manual review.
- Rows containing emoji or other standalone symbols: {transcript["rows_with_emoji_or_symbols_count"]:,}.
- Recording-attempt distribution: `{json.dumps(transcript["attempts_distribution"])}`; values above 10 are flagged for review.
- Mixed English/Tiv: {transcript["mixed_language_assessment"]}
- Licensing/consent status: {dataset["license_and_consent_status"]}

### Source-group totals

| Source group | Audio | TSV rows | Readable | Named s | Actual s | Ratio |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(source_rows)}

### Dataset-level warnings

{chr(10).join(dataset_warning_lines)}

### Recommended exclusions

{chr(10).join(exclusion_lines)}

## AUDIT LIMITATIONS

- Signal-level checks are `{signal["status"]}`: {", ".join(signal["fields"])}.
- Reason: {signal["reason"]}
- Background-language classification and transcript correctness require native-Tiv review.
- The 14 source groups cannot safely be used as speaker IDs or session IDs until their provenance is confirmed.

## RECOMMENDED NEXT STEP

- Action: Confirm speaker/session metadata, contributor consent, dataset licence, and whether the 14 export directories belong to one or multiple speakers; then add a pinned local audio-analysis dependency and run the waveform-quality extension of this audit.
- Reason: Those facts determine leakage-safe splits, model architecture, release eligibility, and which samples need manual review.
- Expected output: Verified provenance metadata plus a complete audit covering silence, noise, clipping, loudness, and a reviewed exclusion list.
- Command to reproduce this report: `python3 scripts/audit_dataset.py --dataset Tiv-TTS-Dataset`

Machine-readable details are in `outputs/dataset_audit/audit.json` and the per-sample classifications are in `outputs/dataset_audit/samples.csv`.
"""


def write_csv(path: Path, records: list[SampleRecord]) -> None:
    """Write a per-sample audit CSV."""

    fieldnames = [
        "sample_id",
        "source_group",
        "audio_path",
        "audio_filename",
        "key",
        "text",
        "attempts",
        "mapping_path",
        "mapping_row",
        "mapping_status",
        "duration_seconds",
        "sample_rate",
        "channels",
        "codec",
        "container",
        "bit_depth",
        "bit_rate",
        "sha256",
        "validation_status",
        "recommended_exclusion",
        "errors",
        "warnings",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = asdict(record)
            row["errors"] = " | ".join(record.errors)
            row["warnings"] = " | ".join(record.warnings)
            writer.writerow(row)


def main() -> int:
    """Run the audit."""

    args = parse_args()
    dataset = args.dataset.resolve()
    output_dir = args.output_dir.resolve()
    markdown_report = args.markdown_report.resolve()
    validate_paths(dataset, output_dir, markdown_report)

    mapping_paths = sorted(dataset.rglob("mapping.tsv"))
    audio_paths = sorted(
        path
        for path in dataset.rglob("*")
        if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES
    )
    print(
        f"Found {len(mapping_paths)} mapping files and {len(audio_paths)} audio files",
        file=sys.stderr,
    )
    backend = select_probe_backend()
    probes = probe_audio_files(audio_paths, backend, args.workers)
    records, source_groups, dataset_errors, dataset_warnings = build_records(
        dataset,
        probes,
        min_duration=args.min_duration,
        long_duration=args.long_duration,
    )
    duplicates = flag_duplicates(records)
    transcript = transcript_analysis(records, args.rare_character_threshold)
    report = summarize(
        dataset,
        records,
        source_groups,
        duplicates,
        transcript,
        dataset_errors,
        dataset_warnings,
        backend,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    markdown_report.parent.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "audit.json"
    csv_path = output_dir / "samples.csv"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_csv(csv_path, records)
    markdown_report.write_text(render_markdown(report), encoding="utf-8")

    print(f"Wrote {json_path}", file=sys.stderr)
    print(f"Wrote {csv_path}", file=sys.stderr)
    print(f"Wrote {markdown_report}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
