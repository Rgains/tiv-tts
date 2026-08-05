"""End-to-end local Tiv TTS smoke-test implementation."""

from __future__ import annotations

import csv
import json
import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
import yaml

from tiv_tts.audio import (
    load_audio,
    log_mel_spectrogram,
    mel_to_waveform,
    mel_transform,
    peak_dbfs,
    resample_audio,
    rms_dbfs,
    save_wav,
)
from tiv_tts.model import TinyTTS
from tiv_tts.quality import analyze_signal
from tiv_tts.text import CharacterTokenizer, normalize_text


def load_config(path: Path) -> dict[str, Any]:
    """Load a UTF-8 YAML smoke-test configuration."""

    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Smoke configuration must be a YAML mapping.")
    return config


def set_deterministic_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def choose_device() -> torch.device:
    """Choose a local device, favoring CUDA/MPS when actually available."""

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _source_group_number(name: str) -> int:
    pieces = name.split("_")
    for piece in reversed(pieces):
        if piece.isdigit():
            return int(piece)
    return sys.maxsize


def read_signal_warnings(path: Path) -> dict[str, str]:
    """Load signal warnings keyed by resolved source path."""

    if not path.is_file():
        return {}
    with path.open(encoding="utf-8", newline="") as handle:
        return {
            str(Path(row["audio_path"]).resolve()): row["warnings"]
            for row in csv.DictReader(handle)
        }


def select_samples(config: dict[str, Any]) -> list[dict[str, str]]:
    """Select deterministic, clean, short samples from one source group."""

    audit_path = Path(config["audit_csv"])
    data_config = config["data"]
    with audit_path.open(encoding="utf-8", newline="") as handle:
        audited = list(csv.DictReader(handle))
    signal_warnings = read_signal_warnings(
        audit_path.parent / "signal_quality.csv"
    )
    candidates = []
    for row in audited:
        duration = float(row["duration_seconds"])
        normalized = normalize_text(row["text"])
        absolute_audio = str(Path(row["audio_path"]).resolve())
        if row["validation_status"] != "valid" or row["mapping_status"] != "mapped":
            continue
        if not (
            float(data_config["minimum_duration_seconds"])
            <= duration
            <= float(data_config["maximum_duration_seconds"])
        ):
            continue
        if normalized.warnings:
            continue
        if signal_warnings.get(absolute_audio):
            continue
        row["cleaned_text"] = normalized.cleaned
        candidates.append(row)

    requested_group = data_config.get("source_group")
    if requested_group:
        source_group = str(requested_group)
    else:
        by_group: dict[str, list[dict[str, str]]] = {}
        for row in candidates:
            by_group.setdefault(row["source_group"], []).append(row)
        if not by_group:
            raise RuntimeError("No eligible samples remain after audit filters.")
        source_group = sorted(
            by_group,
            key=lambda group: (-len(by_group[group]), _source_group_number(group)),
        )[0]
    candidates = [row for row in candidates if row["source_group"] == source_group]
    sample_count = int(data_config["sample_count"])
    if len(candidates) < sample_count:
        raise RuntimeError(
            f"Source group {source_group} has only {len(candidates)} eligible samples; "
            f"{sample_count} requested."
        )
    rng = random.Random(int(config["seed"]))
    rng.shuffle(candidates)
    selected = candidates[:sample_count]
    selected.sort(key=lambda row: row["sample_id"])
    return selected


def assign_splits(records: list[dict[str, Any]]) -> None:
    """Assign a tiny deterministic train/validation/test split."""

    if len(records) < 5:
        raise ValueError("Smoke test needs at least five samples.")
    for index, record in enumerate(records):
        if index == len(records) - 1:
            record["split"] = "test"
        elif index == len(records) - 2:
            record["split"] = "validation"
        else:
            record["split"] = "train"


def prepare_records(
    config: dict[str, Any],
    selected: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], CharacterTokenizer]:
    """Create derived WAVs, features, manifest records, and vocabulary."""

    work_dir = Path(config["work_dir"])
    audio_dir = work_dir / "audio"
    feature_dir = work_dir / "features"
    audio_dir.mkdir(parents=True, exist_ok=True)
    feature_dir.mkdir(parents=True, exist_ok=True)
    audio_config = config["audio"]
    target_rate = int(audio_config["sample_rate"])
    transform = mel_transform(
        sample_rate=target_rate,
        n_fft=int(audio_config["n_fft"]),
        hop_length=int(audio_config["hop_length"]),
        n_mels=int(audio_config["n_mels"]),
        f_min=float(audio_config["f_min"]),
        f_max=float(audio_config["f_max"]),
    )
    tokenizer = CharacterTokenizer.from_texts(
        [row["cleaned_text"] for row in selected]
    )
    records: list[dict[str, Any]] = []
    for row in selected:
        source_path = Path(row["audio_path"])
        waveform, source_rate = load_audio(source_path)
        derived = resample_audio(waveform, source_rate, target_rate)
        wav_path = audio_dir / f"{row['key']}.wav"
        feature_path = feature_dir / f"{row['key']}.pt"
        save_wav(wav_path, derived, target_rate)
        mel = log_mel_spectrogram(derived, transform)
        torch.save({"mel": mel}, feature_path)
        tokens = tokenizer.encode(row["cleaned_text"])
        records.append(
            {
                "sample_id": row["sample_id"],
                "source_group": row["source_group"],
                "source_audio_path": str(source_path),
                "derived_audio_path": str(wav_path),
                "feature_path": str(feature_path),
                "original_text": row["text"],
                "text": row["cleaned_text"],
                "token_ids": tokens,
                "source_sample_rate": source_rate,
                "sample_rate": target_rate,
                "duration_seconds": derived.shape[-1] / target_rate,
                "mel_frames": mel.shape[-1],
                "text_length": len(tokens),
                "transformations": [f"resample_{source_rate}_to_{target_rate}"],
            }
        )
    assign_splits(records)
    manifest_path = work_dir / "manifest.jsonl"
    manifest_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    (work_dir / "vocab.json").write_text(
        json.dumps(tokenizer.vocabulary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return records, tokenizer


def load_mel(record: dict[str, Any]) -> torch.Tensor:
    """Load one trusted, locally generated feature tensor."""

    payload = torch.load(record["feature_path"], map_location="cpu", weights_only=True)
    return payload["mel"]


def mel_statistics(records: list[dict[str, Any]]) -> tuple[float, float]:
    """Compute scalar normalization statistics from the training subset."""

    values = torch.cat(
        [load_mel(record).reshape(-1) for record in records if record["split"] == "train"]
    )
    return float(values.mean()), float(values.std().clamp_min(1e-5))


def make_batch(
    records: list[dict[str, Any]],
    tokenizer: CharacterTokenizer,
    mel_mean: float,
    mel_std: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate records into padded token and normalized-mel tensors."""

    token_lists = [torch.tensor(tokenizer.encode(record["text"])) for record in records]
    mels = [(load_mel(record) - mel_mean) / mel_std for record in records]
    text_lengths = torch.tensor([tokens.numel() for tokens in token_lists])
    mel_lengths = torch.tensor([mel.shape[-1] for mel in mels])
    tokens = torch.zeros(
        len(records), int(text_lengths.max()), dtype=torch.long
    )
    targets = torch.zeros(
        len(records),
        mels[0].shape[0],
        int(mel_lengths.max()),
        dtype=torch.float32,
    )
    for index, (token_ids, mel) in enumerate(zip(token_lists, mels, strict=True)):
        tokens[index, : token_ids.numel()] = token_ids
        targets[index, :, : mel.shape[-1]] = mel
    return (
        tokens.to(device),
        text_lengths.to(device),
        targets.to(device),
        mel_lengths.to(device),
    )


def masked_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    """Compute L1 loss without padded frames."""

    frame_ids = torch.arange(prediction.shape[-1], device=prediction.device)
    mask = frame_ids.unsqueeze(0) < lengths.unsqueeze(1)
    mask = mask.unsqueeze(1).expand_as(prediction)
    return torch.abs(prediction - target).masked_select(mask).mean()


def evaluate(
    model: TinyTTS,
    record: dict[str, Any],
    tokenizer: CharacterTokenizer,
    mel_mean: float,
    mel_std: float,
    device: torch.device,
) -> float:
    """Evaluate one small validation item."""

    model.eval()
    tokens, text_lengths, targets, mel_lengths = make_batch(
        [record], tokenizer, mel_mean, mel_std, device
    )
    with torch.no_grad():
        prediction = model(tokens, text_lengths, mel_lengths)
        loss = masked_l1(prediction, targets, mel_lengths)
    return float(loss)


def save_checkpoint(
    path: Path,
    *,
    model: TinyTTS,
    optimizer: torch.optim.Optimizer,
    step: int,
    config: dict[str, Any],
    tokenizer: CharacterTokenizer,
    mel_mean: float,
    mel_std: float,
    frames_per_character: float,
) -> None:
    """Save all state needed for deterministic resume and inference."""

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "step": step,
            "config": config,
            "vocabulary": tokenizer.vocabulary,
            "mel_mean": mel_mean,
            "mel_std": mel_std,
            "frames_per_character": frames_per_character,
        },
        path,
    )


def train_steps(
    *,
    model: TinyTTS,
    optimizer: torch.optim.Optimizer,
    train_records: list[dict[str, Any]],
    tokenizer: CharacterTokenizer,
    mel_mean: float,
    mel_std: float,
    device: torch.device,
    start_step: int,
    steps: int,
    batch_size: int,
    seed: int,
    log_every: int,
    log_path: Path,
) -> tuple[list[float], float]:
    """Run a deterministic number of finite-loss optimization steps."""

    rng = random.Random(seed + start_step)
    losses: list[float] = []
    last_gradient_norm = 0.0
    model.train()
    with log_path.open("a", encoding="utf-8") as log:
        for offset in range(1, steps + 1):
            step = start_step + offset
            batch_records = [
                train_records[rng.randrange(len(train_records))]
                for _ in range(batch_size)
            ]
            tokens, text_lengths, targets, mel_lengths = make_batch(
                batch_records, tokenizer, mel_mean, mel_std, device
            )
            optimizer.zero_grad(set_to_none=True)
            prediction = model(tokens, text_lengths, mel_lengths)
            loss = masked_l1(prediction, targets, mel_lengths)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at step {step}: {loss}")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            if not torch.isfinite(gradient_norm):
                raise RuntimeError(f"Non-finite gradient at step {step}")
            optimizer.step()
            loss_value = float(loss.detach().cpu())
            last_gradient_norm = float(gradient_norm.detach().cpu())
            losses.append(loss_value)
            row = {
                "step": step,
                "loss": loss_value,
                "gradient_norm": last_gradient_norm,
            }
            log.write(json.dumps(row) + "\n")
            if step % log_every == 0 or offset == 1 or offset == steps:
                print(
                    f"step={step} loss={loss_value:.6f} "
                    f"gradient_norm={last_gradient_norm:.6f}",
                    flush=True,
                )
    return losses, last_gradient_norm


def run_smoke_test(config_path: Path) -> dict[str, Any]:
    """Prepare data, train, resume, infer, and verify a local smoke run."""

    config = load_config(config_path)
    seed = int(config["seed"])
    set_deterministic_seed(seed)
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(config.get("log_dir") or output_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "training_log.jsonl"
    log_path.write_text("", encoding="utf-8")

    selected = select_samples(config)
    records, tokenizer = prepare_records(config, selected)
    train_records = [record for record in records if record["split"] == "train"]
    validation_record = next(
        record for record in records if record["split"] == "validation"
    )
    test_record = next(record for record in records if record["split"] == "test")
    mel_mean, mel_std = mel_statistics(records)
    frames_per_character = float(
        np.median(
            [
                record["mel_frames"] / record["text_length"]
                for record in train_records
            ]
        )
    )
    device = choose_device()
    model_config = config["model"]
    audio_config = config["audio"]
    training_config = config["training"]
    model = TinyTTS(
        vocabulary_size=len(tokenizer.vocabulary),
        n_mels=int(audio_config["n_mels"]),
        embedding_dim=int(model_config["embedding_dim"]),
        hidden_dim=int(model_config["hidden_dim"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(training_config["learning_rate"])
    )

    initial_losses, initial_gradient = train_steps(
        model=model,
        optimizer=optimizer,
        train_records=train_records,
        tokenizer=tokenizer,
        mel_mean=mel_mean,
        mel_std=mel_std,
        device=device,
        start_step=0,
        steps=int(training_config["initial_steps"]),
        batch_size=int(training_config["batch_size"]),
        seed=seed,
        log_every=int(training_config["log_every"]),
        log_path=log_path,
    )
    initial_step = int(training_config["initial_steps"])
    checkpoint_path = output_dir / f"checkpoint_step_{initial_step}.pt"
    save_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        step=initial_step,
        config=config,
        tokenizer=tokenizer,
        mel_mean=mel_mean,
        mel_std=mel_std,
        frames_per_character=frames_per_character,
    )

    # Prove load and resume using fresh model and optimizer instances.
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    resumed_model = TinyTTS(
        vocabulary_size=len(checkpoint["vocabulary"]),
        n_mels=int(audio_config["n_mels"]),
        embedding_dim=int(model_config["embedding_dim"]),
        hidden_dim=int(model_config["hidden_dim"]),
    ).to(device)
    resumed_optimizer = torch.optim.AdamW(
        resumed_model.parameters(), lr=float(training_config["learning_rate"])
    )
    resumed_model.load_state_dict(checkpoint["model_state"])
    resumed_optimizer.load_state_dict(checkpoint["optimizer_state"])
    resume_steps = int(training_config["resume_steps"])
    resumed_losses, resumed_gradient = train_steps(
        model=resumed_model,
        optimizer=resumed_optimizer,
        train_records=train_records,
        tokenizer=tokenizer,
        mel_mean=mel_mean,
        mel_std=mel_std,
        device=device,
        start_step=int(checkpoint["step"]),
        steps=resume_steps,
        batch_size=int(training_config["batch_size"]),
        seed=seed,
        log_every=int(training_config["log_every"]),
        log_path=log_path,
    )
    final_step = initial_step + resume_steps
    final_checkpoint = output_dir / "checkpoint_final.pt"
    save_checkpoint(
        final_checkpoint,
        model=resumed_model,
        optimizer=resumed_optimizer,
        step=final_step,
        config=config,
        tokenizer=tokenizer,
        mel_mean=mel_mean,
        mel_std=mel_std,
        frames_per_character=frames_per_character,
    )
    validation_loss = evaluate(
        resumed_model,
        validation_record,
        tokenizer,
        mel_mean,
        mel_std,
        device,
    )

    token_ids = tokenizer.encode(test_record["text"])
    inference_tokens = torch.tensor(token_ids, device=device).unsqueeze(0)
    output_frames = max(10, round(len(token_ids) * frames_per_character))
    resumed_model.eval()
    predicted_normalized = resumed_model.infer(inference_tokens, output_frames)
    predicted_mel = predicted_normalized * mel_std + mel_mean
    generated = mel_to_waveform(
        predicted_mel,
        sample_rate=int(audio_config["sample_rate"]),
        n_fft=int(audio_config["n_fft"]),
        hop_length=int(audio_config["hop_length"]),
        n_mels=int(audio_config["n_mels"]),
        f_min=float(audio_config["f_min"]),
        f_max=float(audio_config["f_max"]),
        griffin_lim_iterations=int(audio_config["griffin_lim_iterations"]),
    )
    generated_path = output_dir / "generated_smoke.wav"
    save_wav(generated_path, generated, int(audio_config["sample_rate"]))

    reference_mel = load_mel(test_record)
    reference_reconstruction = mel_to_waveform(
        reference_mel,
        sample_rate=int(audio_config["sample_rate"]),
        n_fft=int(audio_config["n_fft"]),
        hop_length=int(audio_config["hop_length"]),
        n_mels=int(audio_config["n_mels"]),
        f_min=float(audio_config["f_min"]),
        f_max=float(audio_config["f_max"]),
        griffin_lim_iterations=int(audio_config["griffin_lim_iterations"]),
    )
    reference_path = output_dir / "reference_reconstruction.wav"
    save_wav(reference_path, reference_reconstruction, int(audio_config["sample_rate"]))

    decoded, generated_rate = sf.read(
        generated_path, dtype="float32", always_2d=True
    )
    generated_tensor = torch.from_numpy(decoded[:, 0]).unsqueeze(0)
    if decoded.size == 0 or not np.isfinite(decoded).all():
        raise RuntimeError("Generated WAV is empty or non-finite.")

    metrics = {
        "status": "passed",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "Diagnostic pipeline test only; generated speech quality is not a "
            "model-quality result."
        ),
        "device": str(device),
        "torch_version": torch.__version__,
        "sample_count": len(records),
        "train_samples": len(train_records),
        "validation_samples": 1,
        "test_samples": 1,
        "source_group": records[0]["source_group"],
        "vocabulary_size": len(tokenizer.vocabulary),
        "initial_step": initial_step,
        "resumed_to_step": final_step,
        "checkpoint_loaded": True,
        "resume_verified": final_step > int(checkpoint["step"]),
        "initial_loss": initial_losses[0],
        "last_initial_phase_loss": initial_losses[-1],
        "last_resumed_loss": resumed_losses[-1],
        "validation_loss": validation_loss,
        "last_gradient_norm": resumed_gradient or initial_gradient,
        "all_losses_finite": all(
            math.isfinite(value) for value in [*initial_losses, *resumed_losses]
        ),
        "mel_mean": mel_mean,
        "mel_std": mel_std,
        "frames_per_character": frames_per_character,
        "test_text": test_record["text"],
        "test_original_text": test_record["original_text"],
        "generated_audio_path": str(generated_path),
        "generated_sample_rate": generated_rate,
        "generated_duration_seconds": len(decoded) / generated_rate,
        "generated_peak_dbfs": peak_dbfs(generated_tensor),
        "generated_rms_dbfs": rms_dbfs(generated_tensor),
        "reference_reconstruction_path": str(reference_path),
        "initial_checkpoint": str(checkpoint_path),
        "final_checkpoint": str(final_checkpoint),
        "training_log_path": str(log_path),
        "manifest_path": str(Path(config["work_dir"]) / "manifest.jsonl"),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "inference_metadata.json").write_text(
        json.dumps(
            {
                "model_version": "tiny-tts-smoke-0.1.0",
                "checkpoint_step": final_step,
                "generated_at_utc": metrics["generated_at_utc"],
                "text": test_record["text"],
                "source_sample_id": test_record["sample_id"],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return metrics

