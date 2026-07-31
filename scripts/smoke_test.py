#!/usr/bin/env python3
"""Run the configured Tiv TTS end-to-end diagnostic smoke test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tiv_tts.smoke import run_smoke_test


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/tiv_smoke.yaml"),
    )
    args = parser.parse_args()
    metrics = run_smoke_test(args.config)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

