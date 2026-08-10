#!/usr/bin/env python3
"""Full VITS training run across the whole prepared Tiv corpus subset.

Long-running: intended for a tmux/screen session, not a single foreground
command. Checkpoints save periodically (see configs/tiv_vits_full.yaml
`training.save_step`) so the run can be stopped and resumed at any time with
--continue-path, and so a checkpoint is never more than a few minutes old if
the process is killed. This is real production-directed training, not a
bounded pilot -- see docs/licensing.md for the recorded permission this
depends on.
"""

from __future__ import annotations

import argparse
import json
import math
import os
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
    training: dict[str, Any],
    seed: int,
) -> VitsConfig:
    dataset = BaseDatasetConfig(
        formatter="ljspeech",
        dataset_name="tiv_vits_full",
        path=str(data_dir),
        meta_file_train="coqui/metadata_train.csv",
        meta_file_val="coqui/metadata_validation.csv",
        language="tiv",
    )
    return VitsConfig(
        output_path=str(output_root),
        run_name="tiv_vits_full",
        run_description="Full training run across the whole prepared subset.",
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
        batch_size=int(training["batch_size"]),
        eval_batch_size=int(training["eval_batch_size"]),
        num_loader_workers=4,
        num_eval_loader_workers=2,
        epochs=int(training["epochs"]),
        run_eval=True,
        test_delay_epochs=999,
        print_step=int(training["print_step"]),
        plot_step=100_000,
        save_step=int(training["save_step"]),
        save_n_checkpoints=int(training["save_n_checkpoints"]),
        save_best_after=0,
        save_all_best=False,
        save_checkpoints=True,
        model_param_stats=False,
        mixed_precision=bool(training["mixed_precision"]),
        training_seed=seed,
        test_sentences=[],
        cudnn_benchmark=True,
        cudnn_deterministic=False,
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
    full = load_config(args.config)
    training = dict(full["training"])
    if args.epochs is not None:
        training["epochs"] = args.epochs
    if args.batch_size is not None:
        training["batch_size"] = args.batch_size
        training.setdefault("eval_batch_size", args.batch_size)

    seed = int(full["comparison"]["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)

    data_dir = Path(full["output_dir"]).resolve()
    output_root = args.output_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    log_dir = args.log_dir.resolve()
    log_dir.mkdir(parents=True, exist_ok=True)

    config = build_configuration(
        data_dir=data_dir, output_root=output_root, training=training, seed=seed
    )
    train_samples, validation_samples = load_tts_samples(
        config.datasets, eval_split=True
    )
    model = Vits.init_from_config(
        config, samples=train_samples + validation_samples
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    averages: list[dict[str, Any]] = []
    history_path = log_dir / "vits-full-training-log.jsonl"

    def record_step(trainer: Trainer) -> None:
        values = safe_json_values(
            trainer.keep_avg_train.avg_values
            if trainer.keep_avg_train is not None
            else None
        )
        values["step"] = trainer.total_steps_done + 1
        averages.append(values)
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(values, ensure_ascii=False) + "\n")

    trainer_args = TrainerArgs(
        continue_path=args.continue_path or "",
        restore_path=args.restore_path or "",
    )
    trainer = Trainer(
        trainer_args,
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
        "purpose": "Full training run intended to produce a usable Tiv voice.",
        "device": device,
        "torch_version": torch.__version__,
        "batch_size": training["batch_size"],
        "epochs_configured": training["epochs"],
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
    }
    metrics_path = Path(trainer.output_path) / "metrics.json"
    metrics_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    shutil.copy(metrics_path, log_dir / "vits-full-metrics.json")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path("configs/tiv_vits_full.yaml")
    )
    # TIV_WORK_ROOT lets the training host set its own layout; the default
    # matches the EC2 box these runs were executed on.
    work_root = Path(os.environ.get("TIV_WORK_ROOT", "/home/ubuntu/tiv-tts"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=work_root / "checkpoints/vits-full",
    )
    parser.add_argument("--log-dir", type=Path, default=work_root / "logs")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument(
        "--continue-path",
        type=str,
        default=None,
        help="Resume from the last checkpoint in this run folder.",
    )
    parser.add_argument(
        "--restore-path",
        type=str,
        default=None,
        help="Start a new run but load weights from this checkpoint file.",
    )
    args = parser.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
