"""Standalone inference for the diagnostic TinyTTS smoke checkpoint."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import soundfile as sf
import torch

from tiv_tts.audio import mel_to_waveform, save_wav
from tiv_tts.model import TinyTTS
from tiv_tts.text import CharacterTokenizer, normalize_text


def infer_checkpoint(
    *,
    checkpoint_path: Path,
    text: str,
    output_path: Path,
) -> dict[str, Any]:
    """Load a smoke checkpoint and synthesize one diagnostic WAV."""

    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    config = checkpoint["config"]
    model_config = config["model"]
    audio_config = config["audio"]
    tokenizer = CharacterTokenizer(checkpoint["vocabulary"])
    normalized = normalize_text(text)
    if not normalized.cleaned:
        raise ValueError("Inference text is empty after normalization.")
    if normalized.warnings:
        raise ValueError(
            "Inference text requires manual review: "
            + ", ".join(normalized.warnings)
        )
    token_ids = tokenizer.encode(normalized.cleaned)
    unknown_id = tokenizer.char_to_id[tokenizer.UNK]
    if unknown_id in token_ids:
        unknown_characters = sorted(
            {
                character
                for character, token_id in zip(
                    normalized.cleaned, token_ids, strict=True
                )
                if token_id == unknown_id
            }
        )
        raise ValueError(
            "Checkpoint vocabulary does not contain: "
            + ", ".join(repr(char) for char in unknown_characters)
        )

    model = TinyTTS(
        vocabulary_size=len(tokenizer.vocabulary),
        n_mels=int(audio_config["n_mels"]),
        embedding_dim=int(model_config["embedding_dim"]),
        hidden_dim=int(model_config["hidden_dim"]),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    tokens = torch.tensor(token_ids).unsqueeze(0)
    output_frames = max(
        10,
        round(len(token_ids) * float(checkpoint["frames_per_character"])),
    )
    predicted_normalized = model.infer(tokens, output_frames)
    predicted_mel = (
        predicted_normalized * float(checkpoint["mel_std"])
        + float(checkpoint["mel_mean"])
    )
    waveform = mel_to_waveform(
        predicted_mel,
        sample_rate=int(audio_config["sample_rate"]),
        n_fft=int(audio_config["n_fft"]),
        hop_length=int(audio_config["hop_length"]),
        n_mels=int(audio_config["n_mels"]),
        f_min=float(audio_config["f_min"]),
        f_max=float(audio_config["f_max"]),
        griffin_lim_iterations=int(audio_config["griffin_lim_iterations"]),
    )
    save_wav(output_path, waveform, int(audio_config["sample_rate"]))
    info = sf.info(output_path)
    metadata = {
        "model_version": "tiny-tts-smoke-0.1.0",
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": int(checkpoint["step"]),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "text": normalized.cleaned,
        "normalization_changes": list(normalized.changes),
        "audio_path": str(output_path),
        "sample_rate": int(info.samplerate),
        "duration_seconds": float(info.duration),
    }
    metadata_path = output_path.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return metadata

