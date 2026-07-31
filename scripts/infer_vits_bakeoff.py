#!/usr/bin/env python3
"""Synthesize a sentence from a completed Tiv VITS bake-off checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from TTS.config import load_config
from TTS.tts.models.vits import Vits
from TTS.tts.utils.text import cleaners as coqui_cleaners


def tiv_character_cleaner(text: str) -> str:
    return " ".join(text.split())


setattr(coqui_cleaners, "tiv_character_cleaner", tiv_character_cleaner)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument(
        "--target-peak",
        type=float,
        default=0.9,
        help="Normalize generated audio to this peak amplitude.",
    )
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    config_path = checkpoint.parent / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing VITS configuration: {config_path}")

    torch.manual_seed(args.seed)
    config = load_config(str(config_path))
    model = Vits.init_from_config(config)
    model.load_checkpoint(config, checkpoint, eval=True, strict=True)

    token_ids = model.tokenizer.text_to_ids(args.text, language="tiv")
    if model.tokenizer.not_found_characters:
        missing = ", ".join(repr(item) for item in model.tokenizer.not_found_characters)
        raise ValueError(f"Text contains characters outside the trained vocabulary: {missing}")
    tokens = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0)
    with torch.inference_mode():
        waveform = model.inference(tokens)["model_outputs"].squeeze().cpu().numpy()

    waveform = waveform.astype(np.float32)
    peak = float(np.max(np.abs(waveform))) if waveform.size else 0.0
    if peak > 0:
        waveform *= args.target_peak / peak
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.output, waveform, config.audio.sample_rate)
    info = sf.info(args.output)
    print(
        f"Wrote {args.output.resolve()} "
        f"({info.duration:.3f}s, {info.samplerate} Hz, {info.channels} channel)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
