#!/usr/bin/env python3
"""Generate a WAV from a diagnostic Tiv smoke-test checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tiv_tts.inference import infer_checkpoint


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    metadata = infer_checkpoint(
        checkpoint_path=args.checkpoint,
        text=args.text,
        output_path=args.output,
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

