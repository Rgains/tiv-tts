# Tiv TTS

This repository provides a reproducible, read-only dataset audit, an end-to-end
diagnostic smoke test, and matched local trials of official VITS and Matcha-TTS
implementations. It does not yet contain a trained production Tiv voice.

## Current result

The diagnostic smoke test passed on 10 derived samples:

- 8 training, 1 validation, and 1 test sample
- UTF-8/NFC Tiv character tokenization
- MP3 decoding and derived 22.05 kHz mono WAV creation
- mel-spectrogram extraction and batching
- finite forward loss and backpropagation
- checkpoint save at step 30
- checkpoint load into fresh model and optimizer instances
- resume through step 35
- standalone inference from the final checkpoint
- 22.05 kHz mono PCM sample WAV generation

The TinyTTS model is intentionally small. Its generated audio proves pipeline
functionality; it is not a model-quality result and should not be presented as
a finished Tiv voice.

The paper-backed bake-off also passed at 50 steps for both VITS and Matcha-TTS.
VITS is the recommended first AWS pilot because its waveform decoder is
integrated. See `docs/model_bakeoff_results.md` for the controlled comparison
and its limitations.

## Setup

```bash
UV_CACHE_DIR="$PWD/.cache/uv" uv sync --all-groups
```

## Reproduce the audits

```bash
python3 scripts/audit_dataset.py --dataset Tiv-TTS-Dataset
UV_CACHE_DIR="$PWD/.cache/uv" uv run python scripts/audit_signal_quality.py
```

Neither command writes into `Tiv-TTS-Dataset/`.

## Run the sample smoke test

```bash
UV_CACHE_DIR="$PWD/.cache/uv" \
  uv run python scripts/smoke_test.py --config configs/tiv_smoke.yaml
```

The run creates derived data under `data/interim/tiv_smoke/` and test artifacts
under `outputs/smoke_test/`.

## Run standalone checkpoint inference

```bash
UV_CACHE_DIR="$PWD/.cache/uv" \
  uv run python scripts/infer.py \
  --checkpoint outputs/smoke_test/checkpoint_final.pt \
  --text "Ve tôv sha zayol la ve hime iyongo shon." \
  --output outputs/smoke_test/sample_from_checkpoint.wav
```

## Reproduce the paper-backed local bake-off

Prepare the shared derived subset:

```bash
UV_CACHE_DIR="$PWD/.cache/uv" \
  uv run --group research python scripts/prepare_bakeoff_data.py
```

Run the candidates and comparison:

```bash
MPLCONFIGDIR="$PWD/.cache/matplotlib" \
XDG_CACHE_HOME="$PWD/.cache" \
  uv run --group research python scripts/train_matcha_bakeoff.py

MPLCONFIGDIR="$PWD/.cache/matplotlib" \
XDG_CACHE_HOME="$PWD/.cache" \
  uv run --group research python scripts/train_vits_bakeoff.py

uv run --group research python scripts/summarize_bakeoff.py
```

All model inputs are derived under `data/interim/`; the commands do not write
into `Tiv-TTS-Dataset/`.

## Tests

```bash
UV_CACHE_DIR="$PWD/.cache/uv" uv run pytest
```

## Important constraints

- The corpus is the Mozilla Data Collective `Tiv-TTS-Dataset`, released under the
  Nwulite Obodo Open Data Licence 1.0 (NOODL-1.0).
- **Its Forbidden Usage terms name generative AI and voice cloning, and limit
  the data to research and scientific use.** Written permission from the legal
  owner is required before any training run intended to produce a usable voice.
  See `docs/licensing.md`.
- The dataset is a single speaker (Ihyarev dialect), so splits carry no
  speaker-leakage risk and single-speaker VITS is the right architecture.
- No pretrained model weights were used in the diagnostic smoke run.
- No raw dataset files should be renamed, rewritten, normalized, or deleted.
- The dataset must not be re-hosted or redistributed, including to cloud
  storage that is publicly readable.
- Full training must wait for licence permission, framework/base-checkpoint
  review, and an AWS smoke test.
