#!/usr/bin/env python3
"""Run a bounded, character-level Matcha-TTS trial on the derived Tiv subset."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
import types
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

# Matcha imports an English eSpeak cleaner eagerly. This character-level Tiv run
# uses no cleaners, so provide an intentionally empty module instead of requiring
# an unused system eSpeak installation.
sys.modules.setdefault(
    "matcha.text.cleaners", types.ModuleType("matcha.text.cleaners")
)
import matcha.text
from matcha.data.text_mel_datamodule import TextMelBatchCollate, TextMelDataset
from matcha.models.matcha_tts import MatchaTTS
from matcha.utils.utils import intersperse


SAMPLE_RATE = 22_050
N_FFT = 1_024
N_MELS = 80
HOP_LENGTH = 256
WIN_LENGTH = 1_024
F_MIN = 0
F_MAX = 8_000


def load_config(path: Path) -> dict[str, Any]:
    import yaml

    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping.")
    return config


def load_symbols(vocab_path: Path) -> list[str]:
    vocabulary = json.loads(vocab_path.read_text(encoding="utf-8"))
    symbols = ["_"] + [
        symbol for symbol in vocabulary if symbol not in {"<pad>", "<unk>"}
    ]
    if len(symbols) != len(set(symbols)):
        raise ValueError("Matcha vocabulary contains duplicate symbols.")
    return symbols


def install_symbols(symbols: list[str]) -> None:
    """Install a run-local vocabulary without changing the installed package."""
    matcha.text._symbol_to_id = {  # type: ignore[attr-defined]
        symbol: index for index, symbol in enumerate(symbols)
    }
    matcha.text._id_to_symbol = {  # type: ignore[attr-defined]
        index: symbol for index, symbol in enumerate(symbols)
    }


def make_dataset(
    filelist: Path,
    statistics: dict[str, float] | None,
    seed: int,
) -> TextMelDataset:
    return TextMelDataset(
        filelist_path=str(filelist),
        n_spks=1,
        cleaners=[],
        add_blank=True,
        n_fft=N_FFT,
        n_mels=N_MELS,
        sample_rate=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        f_min=F_MIN,
        f_max=F_MAX,
        data_parameters=statistics,
        seed=seed,
        load_durations=False,
    )


def calculate_statistics(dataset: TextMelDataset) -> dict[str, float]:
    count = 0
    total = 0.0
    total_squared = 0.0
    for index in range(len(dataset)):
        mel = dataset[index]["y"].double()
        count += mel.numel()
        total += mel.sum().item()
        total_squared += torch.square(mel).sum().item()
    mean = total / count
    variance = max(total_squared / count - mean * mean, 1e-12)
    return {"mel_mean": mean, "mel_std": math.sqrt(variance)}


def model_configuration() -> tuple[Any, Any, Any]:
    encoder = OmegaConf.create(
        {
            "encoder_type": "RoPE Encoder",
            "encoder_params": {
                "n_feats": N_MELS,
                "n_channels": 192,
                "filter_channels": 768,
                "filter_channels_dp": 256,
                "n_heads": 2,
                "n_layers": 6,
                "kernel_size": 3,
                "p_dropout": 0.1,
                "spk_emb_dim": 64,
                "n_spks": 1,
                "prenet": True,
            },
            "duration_predictor_params": {
                "filter_channels_dp": 256,
                "kernel_size": 3,
                "p_dropout": 0.1,
            },
        }
    )
    decoder = OmegaConf.create(
        {
            "channels": [256, 256],
            "dropout": 0.05,
            "attention_head_dim": 64,
            "n_blocks": 1,
            "num_mid_blocks": 2,
            "num_heads": 2,
            "act_fn": "snakebeta",
        }
    )
    cfm = OmegaConf.create(
        {"name": "CFM", "solver": "euler", "sigma_min": 1e-4}
    )
    return encoder, decoder, cfm


def build_model(
    symbol_count: int, statistics: dict[str, float]
) -> MatchaTTS:
    encoder, decoder, cfm = model_configuration()
    return MatchaTTS(
        n_vocab=symbol_count,
        n_spks=1,
        spk_emb_dim=64,
        n_feats=N_MELS,
        encoder=encoder,
        decoder=decoder,
        cfm=cfm,
        data_statistics=statistics,
        out_size=32,
        optimizer=None,
        scheduler=None,
        prior_loss=True,
        use_precomputed_durations=False,
    )


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def total_loss(model: MatchaTTS, batch: dict[str, Any]) -> tuple[torch.Tensor, dict[str, float]]:
    losses = model.get_losses(batch)
    loss = sum(losses.values())
    values = {name: float(value.detach().cpu()) for name, value in losses.items()}
    values["total_loss"] = float(loss.detach().cpu())
    return loss, values


def text_ids(text: str) -> torch.Tensor:
    ids, cleaned = matcha.text.text_to_sequence(text, [])
    if cleaned != text:
        raise RuntimeError("Matcha unexpectedly changed the Tiv inference text.")
    return torch.tensor(intersperse(ids, 0), dtype=torch.long).unsqueeze(0)


def write_griffin_lim(mel: torch.Tensor, path: Path) -> None:
    magnitude = np.exp(mel.detach().cpu().numpy())
    waveform = librosa.feature.inverse.mel_to_audio(
        magnitude,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        fmin=F_MIN,
        fmax=F_MAX,
        power=1.0,
        n_iter=16,
    )
    peak = float(np.max(np.abs(waveform))) if waveform.size else 0.0
    if peak > 0.98:
        waveform = waveform * (0.98 / peak)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, waveform.astype(np.float32), SAMPLE_RATE)


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config)
    seed = int(config["comparison"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(args.cpu_threads)

    data_dir = Path(config["output_dir"]).resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    symbols = load_symbols(data_dir / "vocab.json")
    install_symbols(symbols)

    unnormalized_train = make_dataset(
        data_dir / "matcha" / "train.txt", None, seed
    )
    statistics = calculate_statistics(unnormalized_train)
    train_dataset = make_dataset(
        data_dir / "matcha" / "train.txt", statistics, seed
    )
    validation_dataset = make_dataset(
        data_dir / "matcha" / "validation.txt", statistics, seed
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        collate_fn=TextMelBatchCollate(1),
        generator=torch.Generator().manual_seed(seed),
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=TextMelBatchCollate(1),
    )

    device = torch.device("cpu")
    model = build_model(len(symbols), statistics).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    requested_steps = args.steps or int(config["comparison"]["local_max_steps"])

    history: list[dict[str, float | int]] = []
    iterator = iter(train_loader)
    started = time.perf_counter()
    model.train()
    for step in range(1, requested_steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        loss, values = total_loss(model, batch)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite Matcha loss at step {step}: {loss}")
        loss.backward()
        gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        )
        if not math.isfinite(gradient_norm):
            raise FloatingPointError(
                f"Non-finite Matcha gradient norm at step {step}."
            )
        optimizer.step()
        values.update({"step": step, "gradient_norm": gradient_norm})
        history.append(values)
        if step == 1 or step % 10 == 0 or step == requested_steps:
            print(
                f"matcha step={step}/{requested_steps} "
                f"loss={values['total_loss']:.6f} "
                f"grad={gradient_norm:.4f}",
                flush=True,
            )
    elapsed = time.perf_counter() - started

    validation_losses: list[float] = []
    model.eval()
    with torch.inference_mode():
        for batch in validation_loader:
            _, values = total_loss(model, move_batch(batch, device))
            validation_losses.append(values["total_loss"])

    checkpoint_path = output_dir / "checkpoint_final.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "steps": requested_steps,
            "symbols": symbols,
            "data_statistics": statistics,
            "seed": seed,
            "architecture": "official Matcha-TTS default",
        },
        checkpoint_path,
    )
    reloaded = build_model(len(symbols), statistics)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    reloaded.load_state_dict(state["model"], strict=True)
    reloaded.eval()

    test_text = (data_dir / "matcha" / "test.txt").read_text(
        encoding="utf-8"
    ).splitlines()[0].split("|", 1)[1]
    x = text_ids(test_text)
    x_lengths = torch.tensor([x.shape[-1]], dtype=torch.long)
    with torch.inference_mode():
        synthesis = reloaded.synthesise(
            x=x,
            x_lengths=x_lengths,
            n_timesteps=10,
            temperature=0.667,
            length_scale=1.0,
        )
    mel = synthesis["mel"][0, :, : int(synthesis["mel_lengths"][0])]
    audio_path = output_dir / "sample_griffin_lim.wav"
    write_griffin_lim(mel, audio_path)
    audio_info = sf.info(audio_path)

    result = {
        "model": "Matcha-TTS",
        "implementation": "matcha-tts 0.0.7.2",
        "architecture": "official default Matcha-TTS configuration",
        "device": str(device),
        "steps": requested_steps,
        "parameter_count": parameter_count,
        "elapsed_seconds": elapsed,
        "seconds_per_step": elapsed / requested_steps,
        "initial_train_loss": history[0]["total_loss"],
        "final_train_loss": history[-1]["total_loss"],
        "validation_loss_mean": float(np.mean(validation_losses)),
        "all_losses_finite": True,
        "checkpoint_reload": True,
        "checkpoint": str(checkpoint_path),
        "sample_audio": str(audio_path),
        "sample_text": test_text,
        "sample_duration_seconds": audio_info.duration,
        "sample_rate": audio_info.samplerate,
        "vocoder": "Griffin-Lim diagnostic inversion; not a trained neural vocoder",
        "data_statistics": statistics,
        "symbols": symbols,
        "raw_dataset_modified": False,
        "history": history,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: value for key, value in result.items() if key != "history"}, ensure_ascii=False, indent=2))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path("configs/tiv_bakeoff.yaml")
    )
    parser.add_argument("--steps", type=int)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/bakeoff/matcha")
    )
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
