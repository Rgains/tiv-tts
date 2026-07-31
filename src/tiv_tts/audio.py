"""Audio loading, derivation, mel features, and diagnostic inversion."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio


def load_audio(path: Path) -> tuple[torch.Tensor, int]:
    """Decode an audio file as a mono float tensor without changing the source."""

    data, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    mono = np.mean(data, axis=1, dtype=np.float32)
    waveform = torch.from_numpy(mono).unsqueeze(0)
    if waveform.numel() == 0 or not torch.isfinite(waveform).all():
        raise ValueError(f"Unreadable or non-finite audio: {path}")
    return waveform, int(sample_rate)


def resample_audio(
    waveform: torch.Tensor, source_rate: int, target_rate: int
) -> torch.Tensor:
    """Resample mono audio with torchaudio's bandlimited sinc implementation."""

    if source_rate == target_rate:
        return waveform
    return torchaudio.functional.resample(waveform, source_rate, target_rate)


def save_wav(path: Path, waveform: torch.Tensor, sample_rate: int) -> None:
    """Write a derived mono PCM WAV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    samples = waveform.detach().cpu().squeeze(0).numpy()
    sf.write(path, samples, sample_rate, subtype="PCM_16")


def peak_dbfs(waveform: torch.Tensor) -> float:
    """Return peak amplitude in dBFS."""

    peak = float(waveform.abs().max())
    return 20.0 * math.log10(max(peak, 1e-12))


def rms_dbfs(waveform: torch.Tensor) -> float:
    """Return RMS level in dBFS."""

    rms = float(torch.sqrt(torch.mean(waveform.square())))
    return 20.0 * math.log10(max(rms, 1e-12))


def mel_transform(
    *,
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    n_mels: int,
    f_min: float,
    f_max: float,
) -> torchaudio.transforms.MelSpectrogram:
    """Construct the shared mel-spectrogram transform."""

    return torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        win_length=n_fft,
        hop_length=hop_length,
        f_min=f_min,
        f_max=f_max,
        n_mels=n_mels,
        power=2.0,
        center=True,
        norm="slaney",
        mel_scale="slaney",
    )


def log_mel_spectrogram(
    waveform: torch.Tensor,
    transform: torchaudio.transforms.MelSpectrogram,
) -> torch.Tensor:
    """Return log-power mel features shaped ``[n_mels, frames]``."""

    mel = transform(waveform).squeeze(0)
    return torch.log(torch.clamp(mel, min=1e-5))


def mel_to_waveform(
    log_mel: torch.Tensor,
    *,
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    n_mels: int,
    f_min: float,
    f_max: float,
    griffin_lim_iterations: int,
) -> torch.Tensor:
    """Invert log-mel features for a diagnostic sample using Griffin-Lim."""

    safe_log_mel = torch.clamp(log_mel.detach().cpu(), min=-11.5, max=6.0)
    mel_power = torch.exp(safe_log_mel)
    inverse_mel = torchaudio.transforms.InverseMelScale(
        n_stft=n_fft // 2 + 1,
        n_mels=n_mels,
        sample_rate=sample_rate,
        f_min=f_min,
        f_max=f_max,
        norm="slaney",
        mel_scale="slaney",
    )
    linear_power = inverse_mel(mel_power)
    griffin_lim = torchaudio.transforms.GriffinLim(
        n_fft=n_fft,
        win_length=n_fft,
        hop_length=hop_length,
        power=2.0,
        n_iter=griffin_lim_iterations,
        momentum=0.99,
        rand_init=False,
    )
    waveform = griffin_lim(torch.clamp(linear_power, min=1e-10)).unsqueeze(0)
    peak = waveform.abs().max()
    if peak > 0:
        waveform = waveform * (0.95 / peak)
    return waveform

