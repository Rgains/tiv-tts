#!/usr/bin/env python3
"""Run waveform-level quality checks without modifying source audio."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from tiv_tts.quality import SignalQuality, analyze_signal


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--samples-csv",
        type=Path,
        default=Path("outputs/dataset_audit/samples.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/dataset_audit"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("docs/dataset_signal_audit.md"),
    )
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def analyze(path: str) -> SignalQuality:
    return analyze_signal(Path(path).resolve())


def write_csv(path: Path, results: list[SignalQuality]) -> None:
    rows = [result.to_dict() for result in results]
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            row["warnings"] = " | ".join(row["warnings"])
            writer.writerow(row)


def build_summary(results: list[SignalQuality]) -> dict[str, object]:
    readable = [result for result in results if result.readable]
    warning_counts = Counter(
        warning for result in readable for warning in result.warnings
    )

    def stats(attribute: str) -> dict[str, float] | None:
        values = [
            float(value)
            for result in readable
            if (value := getattr(result, attribute)) is not None
        ]
        if not values:
            return None
        return {
            "minimum": min(values),
            "maximum": max(values),
            "mean": statistics.fmean(values),
            "median": statistics.median(values),
        }

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": (
            "Decoded source audio read-only with libsndfile. Warnings are "
            "screening heuristics and require listening review."
        ),
        "files": len(results),
        "readable": len(readable),
        "unreadable": len(results) - len(readable),
        "files_with_warnings": sum(bool(result.warnings) for result in readable),
        "warning_counts": dict(sorted(warning_counts.items())),
        "peak_dbfs": stats("peak_dbfs"),
        "rms_dbfs": stats("rms_dbfs"),
        "silence_frame_fraction": stats("silence_frame_fraction"),
        "leading_silence_seconds": stats("leading_silence_seconds"),
        "trailing_silence_seconds": stats("trailing_silence_seconds"),
        "snr_proxy_db": stats("snr_proxy_db"),
    }


def render_report(summary: dict[str, object]) -> str:
    return f"""# Tiv TTS signal-quality audit

Generated: {summary["generated_at_utc"]}

- Files analyzed: {summary["files"]:,}
- Readable: {summary["readable"]:,}
- Unreadable: {summary["unreadable"]:,}
- Files with heuristic warnings: {summary["files_with_warnings"]:,}
- Warning counts: `{json.dumps(summary["warning_counts"], ensure_ascii=False)}`
- Peak dBFS statistics: `{json.dumps(summary["peak_dbfs"])}`
- RMS dBFS statistics: `{json.dumps(summary["rms_dbfs"])}`
- Silence-frame statistics: `{json.dumps(summary["silence_frame_fraction"])}`
- Leading-silence statistics: `{json.dumps(summary["leading_silence_seconds"])}`
- Trailing-silence statistics: `{json.dumps(summary["trailing_silence_seconds"])}`
- SNR-proxy statistics: `{json.dumps(summary["snr_proxy_db"])}`

These are conservative screening heuristics, not automatic deletion rules. Every
flagged sample remains in the raw dataset and should be reviewed by listening
before exclusion.
"""


def main() -> int:
    args = parse_args()
    with args.samples_csv.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    paths = sorted(
        {
            row["audio_path"]
            for row in rows
            if row["mapping_status"] == "mapped" and row["audio_path"]
        }
    )
    results: list[SignalQuality] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(analyze, path): path for path in paths}
        for completed, future in enumerate(as_completed(futures), start=1):
            try:
                results.append(future.result())
            except Exception as exc:  # pragma: no cover - worker boundary
                results.append(
                    SignalQuality(
                        audio_path=str(Path(futures[future]).resolve()),
                        readable=False,
                        error=str(exc),
                    )
                )
            if completed % 100 == 0 or completed == len(paths):
                print(
                    f"Signal analysis {completed}/{len(paths)}",
                    file=sys.stderr,
                    flush=True,
                )
    results.sort(key=lambda item: item.audio_path)
    summary = build_summary(results)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "signal_quality.csv", results)
    (args.output_dir / "signal_quality.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    args.report.write_text(render_report(summary), encoding="utf-8")
    print(json.dumps(summary, indent=2), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

