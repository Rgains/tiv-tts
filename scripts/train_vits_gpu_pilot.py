#!/usr/bin/env python3
"""Fixed-budget GPU pilot for official VITS on a larger derived Tiv subset.

This extends the local CPU bake-off (`train_vits_bakeoff.py`, 50 one-sample
steps, for architecture selection only) into a real multi-epoch, real-batch
training run on GPU, per the "Required gate before full AWS training" step in
docs/model_bakeoff_results.md: a fixed-budget cloud pilot for VITS retaining
held-out test utterances. It is still a bounded pilot, not full production
training. The bake-off script and its config are left untouched.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
import yaml
from trainer import Trainer, TrainerArgs

from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.configs.shared_configs import CharactersConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import Vits, VitsAudioConfig
from TTS.tts.utils.text import cleaners as coqui_cleaners


def tiv_character_cleaner(text: str) -> str:
    """Preserve Tiv graphemes while removing file-format whitespace."""
    return " ".join(text.split())


setattr(coqui_cleaners, "tiv_character_cleaner", tiv_character_cleaner)


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping.")
    return config


def character_config(vocab_path: Path) -> CharactersConfig:
    vocabulary = json.loads(vocab_path.read_text(encoding="utf-8"))
    punctuation_set = {" ", ",", "-", ".", ":", ";"}
    punctuations = "".join(
        symbol for symbol in vocabulary if symbol in punctuation_set
    )
    characters = "".join(
        symbol
        for symbol in vocabulary
        if symbol not in {"<pad>", "<unk>"} and symbol not in punctuation_set
    )
    return CharactersConfig(
        pad="<PAD>",
        eos=None,
        bos=None,
        blank="<BLNK>",
        characters=characters,
        punctuations=punctuations,
        is_unique=False,
        is_sorted=False,
    )


def build_configuration(
    data_dir: Path,
    output_root: Path,
    pilot: dict[str, Any],
    seed: int,
) -> VitsConfig:
    dataset = BaseDatasetConfig(
        formatter="ljspeech",
        dataset_name="tiv_vits_gpu_pilot",
        path=str(data_dir),
        meta_file_train="coqui/metadata_train.csv",
        meta_file_val="coqui/metadata_validation.csv",
        language="tiv",
    )
    return VitsConfig(
        output_path=str(output_root),
        run_name="tiv_vits_gpu_pilot",
        run_description="Fixed-budget GPU pilot, multi-epoch, real batching.",
        dashboard_logger="tensorboard",
        audio=VitsAudioConfig(
            sample_rate=22_050,
            fft_size=1_024,
            win_length=1_024,
            hop_length=256,
            num_mels=80,
            mel_fmin=0,
            mel_fmax=8_000,
        ),
        characters=character_config(data_dir / "vocab.json"),
        datasets=[dataset],
        use_phonemes=False,
        text_cleaner="tiv_character_cleaner",
        add_blank=True,
        enable_eos_bos_chars=False,
        batch_size=int(pilot["batch_size"]),
        eval_batch_size=int(pilot["eval_batch_size"]),
        num_loader_workers=2,
        num_eval_loader_workers=2,
        epochs=int(pilot["epochs"]),
        run_eval=True,
        test_delay_epochs=999,
        print_step=int(pilot["print_step"]),
        plot_step=100_000,
        save_step=int(pilot["save_step"]),
        save_n_checkpoints=int(pilot["save_n_checkpoints"]),
        save_all_best=False,
        save_checkpoints=True,
        model_param_stats=False,
        mixed_precision=bool(pilot["mixed_precision"]),
        training_seed=seed,
        test_sentences=[],
        cudnn_benchmark=False,
        cudnn_deterministic=True,
    )


def safe_json_values(values: dict[str, Any] | None) -> dict[str, float]:
    if not values:
        return {}
    result = {}
    for key, value in values.items():
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().item()
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            result[key] = float(value)
    return result


def save_waveform(path: Path, waveform: torch.Tensor) -> sf.SoundFile:
    array = waveform.detach().cpu().squeeze().numpy().astype(np.float32)
    peak = float(np.max(np.abs(array))) if array.size else 0.0
    if peak > 0.98:
        array *= 0.98 / peak
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, array, 22_050)
    return sf.info(path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    bakeoff = load_config(args.config)
    pilot = dict(bakeoff["pilot"])
    if args.epochs is not None:
        pilot["epochs"] = args.epochs
    if args.batch_size is not None:
        pilot["batch_size"] = args.batch_size
        pilot.setdefault("eval_batch_size", args.batch_size)

    seed = int(bakeoff["comparison"]["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)

    data_dir = Path(bakeoff["output_dir"]).resolve()
    output_root = args.output_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    log_dir = args.log_dir.resolve()
    log_dir.mkdir(parents=True, exist_ok=True)

    config = build_configuration(
        data_dir=data_dir, output_root=output_root, pilot=pilot, seed=seed
    )
    train_samples, validation_samples = load_tts_samples(
        config.datasets, eval_split=True
    )
    model = Vits.init_from_config(
        config, samples=train_samples + validation_samples
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    averages: list[dict[str, Any]] = []

    def record_step(trainer: Trainer) -> None:
        values = safe_json_values(
            trainer.keep_avg_train.avg_values
            if trainer.keep_avg_train is not None
            else None
        )
        values["step"] = trainer.total_steps_done + 1
        averages.append(values)

    trainer = Trainer(
        TrainerArgs(),
        model.config,
        str(output_root),
        model=model,
        train_samples=train_samples,
        eval_samples=validation_samples,
        callbacks={"on_train_step_end": record_step},
        parse_command_line_args=False,
    )
    device = "cuda" if trainer.use_cuda else "cpu"
    started = time.perf_counter()
    trainer.fit()
    elapsed = time.perf_counter() - started
    trainer.save_checkpoint()
    checkpoint_path = max(
        Path(trainer.output_path).glob("checkpoint_*.pth"),
        key=lambda path: path.stat().st_mtime,
    )

    reloaded = Vits.init_from_config(
        model.config, samples=train_samples + validation_samples
    )
    reloaded.load_checkpoint(
        model.config, checkpoint_path, eval=True, strict=True
    )
    test_text = (
        (data_dir / "matcha" / "test.txt")
        .read_text(encoding="utf-8")
        .splitlines()[0]
        .split("|", 1)[1]
    )
    ids = reloaded.tokenizer.text_to_ids(test_text, language="tiv")
    if reloaded.tokenizer.not_found_characters:
        raise RuntimeError(
            "VITS discarded Tiv inference characters: "
            f"{reloaded.tokenizer.not_found_characters}"
        )
    tokens = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
    if device == "cuda":
        reloaded = reloaded.to("cuda")
        tokens = tokens.to("cuda")
    with torch.inference_mode():
        generated = reloaded.inference(tokens)["model_outputs"]
    audio_path = Path(trainer.output_path) / "sample.wav"
    audio_info = save_waveform(audio_path, generated)

    train_averages = safe_json_values(
        trainer.keep_avg_train.avg_values
        if trainer.keep_avg_train is not None
        else None
    )
    validation_averages = safe_json_values(
        trainer.keep_avg_eval.avg_values
        if trainer.keep_avg_eval is not None
        else None
    )
    loss_keys = [
        key
        for key in train_averages
        if "loss" in key.lower() and "avg" in key.lower()
    ]
    all_finite = all(
        math.isfinite(value)
        for values in averages
        for value in values.values()
        if isinstance(value, float)
    )
    result = {
        "model": "VITS",
        "implementation": "coqui-tts 0.27.5",
        "architecture": "official default VITS configuration",
        "purpose": (
            "Fixed-budget GPU pilot per docs/model_bakeoff_results.md gate 3; "
            "not a full production training run."
        ),
        "device": device,
        "torch_version": torch.__version__,
        "batch_size": pilot["batch_size"],
        "epochs_configured": pilot["epochs"],
        "steps": trainer.total_steps_done,
        "parameter_count": parameter_count,
        "elapsed_seconds": elapsed,
        "seconds_per_step": elapsed / max(trainer.total_steps_done, 1),
        "train_averages": train_averages,
        "validation_averages": validation_averages,
        "reported_loss_keys": loss_keys,
        "all_losses_finite": all_finite,
        "checkpoint_reload": True,
        "checkpoint": str(checkpoint_path),
        "sample_audio": str(audio_path),
        "sample_text": test_text,
        "sample_duration_seconds": audio_info.duration,
        "sample_rate": audio_info.samplerate,
        "vocoder": "VITS integrated waveform decoder",
        "symbols": reloaded.tokenizer.characters.vocab,
        "raw_dataset_modified": False,
        "history": averages,
    }
    metrics_path = Path(trainer.output_path) / "metrics.json"
    metrics_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    shutil.copy(metrics_path, log_dir / "vits-gpu-pilot-metrics.json")
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "history"},
            ensure_ascii=False,
            indent=2,
        )
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path("configs/tiv_vits_gpu_pilot.yaml")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/ubuntu/tiv-tts/checkpoints/vits-gpu-pilot"),
    )
    parser.add_argument(
        "--log-dir", type=Path, default=Path("/home/ubuntu/tiv-tts/logs")
    )
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    args = parser.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
