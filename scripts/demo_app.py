#!/usr/bin/env python3
"""Streamlit demo: type Tiv text, hear it synthesized by the trained VITS model.

Run with: streamlit run scripts/demo_app.py
Loads a fixed checkpoint at startup (default: the full-training run's
best_model.pth). Intended for a listening-review demo, not production serving.
"""

from __future__ import annotations

import io
import os
from pathlib import Path

import numpy as np
import soundfile as sf
import streamlit as st
import torch

from TTS.config import load_config
from TTS.tts.models.vits import Vits
from TTS.tts.utils.text import cleaners as coqui_cleaners

REPO_ROOT = Path(__file__).resolve().parents[1]
EC2_RUN = Path(
    "/home/ubuntu/tiv-tts/checkpoints/vits-full/"
    "tiv_vits_full-July-31-2026_04+37PM-4834afa"
)
HF_REPO_ID = os.environ.get("TIV_HF_REPO", "ejnuma/tiv-tts")
CHECKPOINT_NAME = "best_model.pth"

EXAMPLE_SENTENCES = [
    "Kpa ior kpishi hemba soon er a yila wan iti er Korwua",
    "Nahan anterev mba tumbun kor sha ku la yô, Anzun Gbaka fatyô u suan kor ga.",
    "U kaa imo la nahan shima nyian mo!",
    "Tagude lu tindi amin hen tar na la.",
    "Tiv mba er injakwagh ne u vendan imbwase i kaan or.",
]


def tiv_character_cleaner(text: str) -> str:
    return " ".join(text.split())


setattr(coqui_cleaners, "tiv_character_cleaner", tiv_character_cleaner)


LFS_POINTER_MAGIC = b"version https://git-lfs.github.com/spec/v1"


def checkpoint_problem(checkpoint: Path) -> str | None:
    """Return a readable reason the checkpoint cannot be loaded, or None if it can."""
    if not checkpoint.exists():
        return (
            f"No checkpoint at {checkpoint}. Point TIV_RUN_DIR at a directory "
            f"holding config.json and {CHECKPOINT_NAME}."
        )
    with checkpoint.open("rb") as handle:
        head = handle.read(len(LFS_POINTER_MAGIC))
    if head == LFS_POINTER_MAGIC:
        return (
            f"{checkpoint} is an unresolved Git LFS pointer of "
            f"{checkpoint.stat().st_size} bytes, not the model itself. The host "
            "cloned this repository without Git LFS support, or the repository is "
            "over its LFS bandwidth quota."
        )
    return None


def hub_token() -> str | None:
    """Read the Hub token from the environment, falling back to Streamlit secrets."""
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    try:
        return st.secrets["HF_TOKEN"]
    except Exception:
        # No secrets file locally, or the key is absent on the host.
        return None


@st.cache_resource(show_spinner="Fetching the model from the Hub...")
def fetch_from_hub() -> Path:
    """Download config and checkpoint from the Hub, returning their shared directory."""
    from huggingface_hub import hf_hub_download

    token = hub_token()
    config = hf_hub_download(HF_REPO_ID, "config.json", token=token)
    hf_hub_download(HF_REPO_ID, CHECKPOINT_NAME, token=token)
    return Path(config).parent


def resolve_run_dir() -> Path:
    """TIV_RUN_DIR, then the local model/ copy, then the EC2 run, then the Hub."""
    override = os.environ.get("TIV_RUN_DIR")
    if override:
        return Path(override)
    for candidate in (REPO_ROOT / "model", EC2_RUN):
        if (candidate / "config.json").exists():
            return candidate
    return fetch_from_hub()


@st.cache_resource
def load_model(checkpoint_str: str) -> Vits:
    checkpoint = Path(checkpoint_str)
    torch.set_num_threads(int(os.environ.get("TIV_THREADS", "4")))
    config_path = checkpoint.parent / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing VITS configuration: {config_path}")
    config = load_config(str(config_path))
    model = Vits.init_from_config(config)
    model.load_checkpoint(config, checkpoint, eval=True, strict=True)
    if torch.cuda.is_available():
        model = model.to("cuda")
    model.eval()
    return model


def synthesize(model: Vits, text: str) -> tuple[int, np.ndarray]:
    device = next(model.parameters()).device
    token_ids = model.tokenizer.text_to_ids(text, language="tiv")
    if model.tokenizer.not_found_characters:
        missing = ", ".join(repr(c) for c in model.tokenizer.not_found_characters)
        raise ValueError(
            f"Text contains characters outside the trained vocabulary: {missing}"
        )
    tokens = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)
    with torch.inference_mode():
        waveform = model.inference(tokens)["model_outputs"].squeeze().cpu().numpy()
    waveform = waveform.astype(np.float32)
    peak = float(np.max(np.abs(waveform))) if waveform.size else 0.0
    if peak > 0:
        waveform *= 0.9 / peak
    return model.config.audio.sample_rate, waveform


def main() -> None:
    st.set_page_config(page_title="Tiv TTS demo", page_icon="🗣️")
    st.title("Tiv TTS -- listening review demo")
    st.caption(
        "Research demo of a VITS model trained on the Tiv-TTS-Dataset "
        "(Mozilla Data Collective), built for a Tiv-language early-warning "
        "advisory pilot. Not a production system. Speech quality has not "
        "yet been validated by Tiv speakers -- your listening feedback is "
        "exactly what this demo is for."
    )

    checkpoint = resolve_run_dir() / CHECKPOINT_NAME
    problem = checkpoint_problem(checkpoint)
    if problem:
        st.error(problem)
        st.stop()

    model = load_model(str(checkpoint))

    if "text_input" not in st.session_state:
        st.session_state.text_input = EXAMPLE_SENTENCES[0]

    st.subheader("Try an example")
    cols = st.columns(len(EXAMPLE_SENTENCES))
    for col, sentence in zip(cols, EXAMPLE_SENTENCES):
        short = sentence if len(sentence) <= 20 else sentence[:17] + "..."
        if col.button(short, help=sentence, use_container_width=True):
            st.session_state.text_input = sentence

    text = st.text_area("Tiv text", key="text_input", height=100)

    if st.button("Synthesize", type="primary"):
        if not text.strip():
            st.warning("Enter some Tiv text first.")
        else:
            try:
                with st.spinner("Synthesizing..."):
                    sample_rate, waveform = synthesize(model, text.strip())
                buffer = io.BytesIO()
                sf.write(buffer, waveform, sample_rate, format="WAV")
                st.audio(buffer.getvalue(), format="audio/wav")
                st.caption(f"{waveform.shape[-1] / sample_rate:.2f}s at {sample_rate} Hz")
            except ValueError as exc:
                st.error(str(exc))


if __name__ == "__main__":
    main()
