import numpy as np
import soundfile as sf

from tiv_tts.quality import analyze_signal


def test_signal_quality_detects_long_leading_silence(tmp_path) -> None:
    sample_rate = 8000
    silence = np.zeros(sample_rate * 2, dtype=np.float32)
    tone_time = np.arange(sample_rate, dtype=np.float32) / sample_rate
    tone = 0.2 * np.sin(2 * np.pi * 220 * tone_time)
    path = tmp_path / "sample.wav"
    sf.write(path, np.concatenate([silence, tone]), sample_rate)
    result = analyze_signal(path)
    assert result.readable
    assert "long_leading_silence" in result.warnings

