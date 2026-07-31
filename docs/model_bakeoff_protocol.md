# Tiv TTS pre-AWS model bake-off

## Goal

Choose one architecture for the AWS L40S cloud smoke test using reproducible
evidence from the same small Tiv subset. This is a pipeline and optimization
comparison, not a speech-quality ranking after full convergence.

## Candidates

### VITS

- Paper: *Conditional Variational Autoencoder with Adversarial Learning for
  End-to-End Text-to-Speech* (ICML 2021).
- Why test it: mature end-to-end baseline, stochastic duration modeling,
  integrated waveform decoder, strong low-resource precedent.
- Official code licence: MIT.
- Local integration: current Coqui-TTS VITS implementation, MPL-2.0 framework.

### Matcha-TTS

- Paper: *Matcha-TTS: A Fast TTS Architecture with Conditional Flow Matching*
  (ICASSP 2024).
- Why test it: compact memory footprint, fast non-autoregressive inference,
  monotonic alignment, and official custom-dataset training.
- Official code licence: MIT.
- Vocoder: must be evaluated and licensed separately.

## Reviewed but not admitted to the local trial

### StyleTTS 2

The paper reports excellent quality, but the official training stack is
two-stage, depends on pretrained speech-language components, and its repository
still documents incomplete or untested training paths. It is not a fair,
low-cost local gate.

### F5-TTS

The code is MIT, but official pretrained weights are CC-BY-NC because of their
training data. Training or meaningful adaptation from scratch is also too
heavy for this local pre-AWS comparison. It remains a research-only alternative
unless a compatible base checkpoint is found.

## Shared dataset

- 60 short, clean samples from one source group
- 50 train, 5 validation, 5 test
- identical fixed split and random seed
- mono 22,050 Hz PCM WAV derived from untouched source MP3
- UTF-8 NFC transcripts with conservative normalization
- no transcript-audit or signal-audit warnings
- character input for both models to avoid an English phonemizer changing Tiv
  text

Speaker identity is still unverified; the selected export directory is treated
only as a source group.

## Local trial budget

- no pretrained weights
- maximum 50 optimization steps per candidate
- batch size adjusted only for memory
- checkpoint save and reload required
- inference artifact required
- no conclusion about naturalness from unconverged output

## Comparison fields

- setup success
- data-loader success
- character coverage and unknown-token count
- forward/backward success
- finite loss
- wall-clock seconds per step
- peak process memory when available
- checkpoint size
- checkpoint reload/resume
- inference success
- inference wall time and real-time factor
- generated WAV validity
- code licence
- pretrained-weight licence
- AWS L40S compatibility and expected training complexity

## Selection rule

An architecture is eligible for AWS only if its official implementation can
load the Tiv subset, preserve the character inventory, train with finite loss,
save/reload/resume, and synthesize a valid WAV. Among eligible models, prefer
the simpler and more licence-compatible option unless listening evaluation
after a cloud smoke test shows a clear quality advantage.

