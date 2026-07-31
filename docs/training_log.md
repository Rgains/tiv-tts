# Training log

## Local diagnostic smoke test

- Date: 2026-07-30
- Configuration: `configs/tiv_smoke.yaml`
- Device: CPU
- PyTorch: 2.13.0
- Source group: `tts_Tiv_dataset_20_175clips_1866s_20260418-1226`
- Samples: 10 total (8 train, 1 validation, 1 test)
- Vocabulary: 36 entries including `<pad>` and `<unk>`
- Initial training: 30 steps
- Resume training: 5 steps
- Initial loss: 0.825327
- Step-30 loss: 0.567341
- Step-35 loss: 0.616860
- Validation loss: 0.676076
- Checkpoint reload: passed
- Resume verification: passed
- Standalone inference: passed
- Generated audio: 22,050 Hz, mono PCM WAV, 4.493 seconds

The model is a deliberately small diagnostic text-to-mel network with
Griffin-Lim inversion. The audio is expected to sound poor. This test proves
pipeline mechanics, not intelligibility or naturalness.

Machine-readable metrics and per-step logs are stored under
`outputs/smoke_test/`.

## Paper-backed local bake-off

- Date: 2026-07-30
- Configuration: `configs/tiv_bakeoff.yaml`
- Device: CPU
- PyTorch: 2.8.0
- Candidates: official VITS and Matcha-TTS implementations
- Samples: 60 total (50 train, 5 validation, 5 test)
- Optimization budget: 50 steps per model
- VITS: finite losses, checkpoint reload passed, integrated WAV inference passed
- Matcha-TTS: loss 5.1481 to 3.7867, checkpoint reload passed, diagnostic WAV
  inference passed
- Recommendation: VITS for the first end-to-end AWS pilot; Matcha-TTS remains
  the lower-compute alternative after adding a licensed neural vocoder

Detailed results and limitations are in `docs/model_bakeoff_results.md`.
