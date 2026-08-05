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

## Full GPU training run (VITS)

- Dates: 2026-07-31 16:37 UTC to 2026-08-01 07:54 UTC (15h17m)
- Configuration: `configs/tiv_vits_full.yaml`, `scripts/train_vits_full.py`
- Device: CUDA (AWS g6.2xlarge, NVIDIA L4), PyTorch 2.8.0+cu128
- Data: all 14 export groups (single-speaker corpus), 1,176 clips, 1,096
  train / 40 validation / 40 test, 2.52 hours of audio, 102-symbol vocabulary
- Batch size 16, 1,500 epochs, 103,500 steps, ~0.53 s/step sustained
- Final train mel loss 22.87; validation mel loss 23.84
- All losses finite for the entire run; checkpoint reload verified after
  training; inference from the reloaded checkpoint on a held-out test
  sentence passed
- Checkpoints: `~/tiv-tts/checkpoints/vits-full/tiv_vits_full-July-31-2026_04+37PM-4834afa/`
  (`checkpoint_103500.pth`, `best_model.pth` by validation loss)
- Full metrics: `~/tiv-tts/logs/vits-full-metrics.json`

This is a real training run under the permission recorded in
`docs/licensing.md`, not a diagnostic pipeline check. Finite, converging
losses show the model learned; they do not establish intelligibility or
naturalness. Per the bake-off's "Required gate" section, a listening review
by Tiv speakers is still needed before treating this checkpoint as usable
for the NEMA early-warning system.
