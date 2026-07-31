"""Signal-level audio quality metrics used for read-only dataset review."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import soundfile as sf


@dataclass(slots=True)
class SignalQuality:
    """Waveform metrics and conservative heuristic warnings."""

    audio_path: str
    readable: bool
    sample_rate: int | None = None
    channels: int | None = None
    duration_seconds: float | None = None
    peak_dbfs: float | None = None
    rms_dbfs: float | None = None
    dc_offset: float | None = None
    clipped_sample_fraction: float | None = None
    silence_frame_fraction: float | None = None
    leading_silence_seconds: float | None = None
    trailing_silence_seconds: float | None = None
    noise_floor_dbfs_p10: float | None = None
    active_level_dbfs_p90: float | None = None
    snr_proxy_db: float | None = None
    warnings: tuple[str, ...] = ()
    error: str = ""

    def to_dict(self) -> dict[str, object]:
        """Return a JSON/CSV-friendly dictionary."""

        row = asdict(self)
        row["warnings"] = list(self.warnings)
        return row


def _db(value: np.ndarray | float) -> np.ndarray | float:
    return 20.0 * np.log10(np.maximum(value, 1e-12))


def analyze_signal(path: Path, frame_ms: float = 20.0) -> SignalQuality:
    """Decode and analyze one file without writing beside it."""

    try:
        data, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    except Exception as exc:
        return SignalQuality(audio_path=str(path), readable=False, error=str(exc))

    if data.size == 0 or not np.isfinite(data).all():
        return SignalQuality(
            audio_path=str(path),
            readable=False,
            error="empty or non-finite decoded waveform",
        )
    mono = data.mean(axis=1, dtype=np.float32)
    absolute = np.abs(mono)
    peak = float(absolute.max())
    rms = float(np.sqrt(np.mean(np.square(mono, dtype=np.float64))))
    frame_size = max(1, round(sample_rate * frame_ms / 1000.0))
    padded_size = math.ceil(len(mono) / frame_size) * frame_size
    padded = np.pad(mono, (0, padded_size - len(mono)))
    frames = padded.reshape(-1, frame_size)
    frame_rms = np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=1))
    frame_db = _db(frame_rms)
    silent = frame_db < -50.0

    leading_frames = 0
    for value in silent:
        if not value:
            break
        leading_frames += 1
    trailing_frames = 0
    for value in silent[::-1]:
        if not value:
            break
        trailing_frames += 1

    noise_floor = float(np.percentile(frame_db, 10))
    active_level = float(np.percentile(frame_db, 90))
    snr_proxy = active_level - noise_floor
    clip_fraction = float(np.mean(absolute >= 0.999))
    warnings: list[str] = []
    if peak >= 1.0 or clip_fraction > 0.0001:
        warnings.append("possible_clipping")
    if float(_db(rms)) < -35.0:
        warnings.append("very_low_volume")
    if leading_frames * frame_ms / 1000.0 > 1.0:
        warnings.append("long_leading_silence")
    if trailing_frames * frame_ms / 1000.0 > 1.0:
        warnings.append("long_trailing_silence")
    if float(np.mean(silent)) > 0.35:
        warnings.append("high_silence_fraction")
    if noise_floor > -30.0 and snr_proxy < 15.0:
        warnings.append("possible_high_background_noise")

    return SignalQuality(
        audio_path=str(path),
        readable=True,
        sample_rate=int(sample_rate),
        channels=int(data.shape[1]),
        duration_seconds=float(len(mono) / sample_rate),
        peak_dbfs=float(_db(peak)),
        rms_dbfs=float(_db(rms)),
        dc_offset=float(np.mean(mono)),
        clipped_sample_fraction=clip_fraction,
        silence_frame_fraction=float(np.mean(silent)),
        leading_silence_seconds=leading_frames * frame_ms / 1000.0,
        trailing_silence_seconds=trailing_frames * frame_ms / 1000.0,
        noise_floor_dbfs_p10=noise_floor,
        active_level_dbfs_p90=active_level,
        snr_proxy_db=snr_proxy,
        warnings=tuple(warnings),
    )
