# Tiv TTS local model bake-off results

## Outcome

Both official model implementations passed the 50-step local gate on the same
derived Tiv sample. VITS is the recommended first AWS pilot because it is
end-to-end and emits waveform audio from one model. Matcha-TTS is the
cost-focused alternative, but it still needs a separately selected, licensed,
and validated neural vocoder.

This is not a naturalness or intelligibility verdict. Fifty optimization steps
from random initialization can validate data flow, optimization, checkpoints,
and inference, but cannot produce a trained voice.

## Controlled setup

- Date: 2026-07-30
- Device: CPU
- PyTorch: 2.8.0
- Source group: `tts_Tiv_dataset_20_175clips_1866s_20260418-1226`
- Samples: 60 total; 50 train, 5 validation, 5 test
- Derived audio: mono 22,050 Hz PCM WAV
- Training budget: 50 one-sample optimization steps per model
- Input: direct Tiv characters, with no English phonemization or transliteration
- Pretrained weights: none
- Tiv-specific characters retained: `ô`, `õ`, `ö`, `ō`
- Raw manifest hash before/after:
  `4487c442fc1481bb77173a67a90ebf57a5c8b32667a587c73e99747c33030784`

## Results

| Field | VITS | Matcha-TTS |
|---|---:|---:|
| Official architecture | Default VITS | Default Matcha-TTS |
| Training parameters | 83,044,588 including discriminator | 18,180,385 |
| CPU time per step | 0.837 s including validation overhead | 0.053 s |
| Initial loss | generator 213.09; discriminator 6.06 | total 5.15 |
| Final step loss | generator 57.53; discriminator 2.79 | total 3.79 |
| Validation loss | generator 58.35; discriminator 2.59 | total 4.42 |
| Finite forward/backward | Pass | Pass |
| Checkpoint save/reload | Pass | Pass |
| Valid 22.05 kHz WAV | Pass | Pass |
| Waveform path | Integrated VITS decoder | Diagnostic Griffin-Lim only |
| Checkpoint size | 997.7 MB with optimizer states | 218.5 MB with optimizer state |

Loss magnitudes are architecture-specific and are not comparable across the
two models. Matcha's measured step time also excludes the one-time mel
statistics pass and diagnostic Griffin-Lim inversion.

## Selection

### First AWS pilot: VITS

VITS passed all gates and produced audio with its own waveform decoder after a
fresh checkpoint reload. Its integrated acoustic model and decoder make the
first cloud experiment simpler to reproduce and license. The tradeoff is a
larger training stack and checkpoint.

### Alternative: Matcha-TTS

Matcha-TTS was much lighter and faster in this CPU test and optimized stably.
It is attractive when training cost is the priority. It should not be treated
as a complete production voice until a neural vocoder is selected and tested
under the same licence and Tiv listening protocol.

## What the audio means

The samples are expected to sound noisy, quiet, or unintelligible. They came
from random initialization with only 50 updates. They prove that each model can
consume Tiv text and audio, train with finite gradients, reload, and synthesize
a valid waveform. They do not show which model will sound better after
convergence.

## Required gate before full AWS training

1. Confirm that the chosen source group represents one speaker and one
   recording domain.
2. Confirm dataset consent, provenance, and a licence permitting cloud training
   and model release.
3. Run a fixed-budget cloud pilot for VITS, retaining held-out test utterances.
4. Evaluate intelligibility and naturalness with Tiv speakers before committing
   to a long run.
5. If budget permits, run Matcha-TTS for the same audio-hours and optimizer
   budget with a licensed neural vocoder, then conduct a blinded listening
   comparison.

Machine-readable results are written to
`outputs/bakeoff/comparison.json`.
