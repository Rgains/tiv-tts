# Building a Text-to-Speech System for Tiv: A Low-Resource, Single-Speaker Case Study

**Status:** research baseline, not production. No native Tiv speaker has evaluated the output yet — see Limitations.

## Abstract

We describe the construction of a character-level VITS text-to-speech
system for Tiv, a low-resource Nigerian language, from a single-speaker
corpus of prompted read speech. The work was motivated by a proposed
Tiv-language early-warning advisory system built with Nigeria's National
Emergency Management Agency (NEMA), intended to reach communities where
English-language advisories are not accessible. We document the full
pipeline: dataset audit, licence clearance for generative-AI use, a
controlled architecture comparison between VITS and Matcha-TTS, staged
GPU validation (smoke test, pilot, full run), and the resulting model.
Training converged over 103,500 steps (15h17m on a single NVIDIA L4) with
finite losses throughout. We report training dynamics and give an honest
account of what remains before this is a usable system: native-speaker
intelligibility review, and — separately — the absence of any usable
English-to-Tiv translation resource, which we found no existing corpus or
model adequately covers.

## 1. Introduction

Nigeria is linguistically diverse, and emergency advisories issued only in
English do not reach every community. This project's motivating goal is a
text-to-speech voice for Tiv (spoken primarily in Benue State, Nigeria, by
an estimated several million speakers) that could read emergency advisory
text aloud as part of a NEMA-affiliated early-warning system, with the
resulting model released open source.

TTS for a language like Tiv has no existing solution to build on: Tiv is
absent from major multilingual TTS and MT systems we checked (Section 6),
and no pretrained checkpoint exists for it. The project therefore had to
establish, from scratch and in this order: (1) whether a usable licensed
corpus existed at all, (2) whether training on it was legally permitted,
(3) which architecture was viable at this data scale, and only then (4)
whether training actually produced a stable, convergent model.

## 2. Dataset

### 2.1 Source and provenance

| Field | Value |
|---|---|
| Source | Mozilla Data Collective, dataset `cmo4nmfam00nxny07rssox2tj` |
| Name | Tiv-TTS-Dataset |
| Licence | Nwulite Obodo Open Data Licence 1.0 (NOODL-1.0) |
| Steward | Institute of African Digital Humanities |
| Legal owner | Daniel Nyitse |
| Speaker | One speaker, Ihyarev dialect |
| Content | Prompted read speech: expository prose, folk narrative, proverbs |
| Published duration | 5h 50m 43s across 2,443 clips |

### 2.2 Audit

A read-only audit (`scripts/audit_dataset.py`) established, without
modifying the source corpus:

- **2,443 mapped clips**, 0 unmatched files, 0 duplicate audio (exact
  SHA-256), 0 duplicate transcripts, 0 empty transcripts, 0
  audio-without-transcript or transcript-without-audio mismatches.
- **5.87 hours** of selected audio (21,116.9 s); note that the 14
  source-group directory names collectively overstate this by ~1.6 hours,
  consistent with each group's export including discarded recording
  attempts not present in the final MP3 set.
- Uniform format: MP3, 48 kHz, mono, throughout.
- Duration distribution: min 2.14 s, median 8.30 s, mean 8.64 s, p95
  14.14 s, max 31.78 s.
- Transcript format: UTF-8 TSV (`audio_filename, key, sentence,
  attempts`); 2,330 valid rows, 113 rows carrying a review warning
  (literal backslash escapes, non-Latin-script fragments, an emoji, or an
  unusually high retry count), 0 rows automatically excluded.
- **130 unique characters** in the raw transcripts, including Tiv-specific
  graphemes (`ô õ ö ō`) alongside a long tail of single-occurrence
  characters from Greek, Cyrillic, and Hebrew scripts and one emoji —
  plausibly transcription artifacts rather than Tiv orthography, flagged
  for review rather than silently dropped.
- 14 export directories exist; their relationship to distinct speakers or
  recording sessions was **not** established from metadata and they were
  never treated as speaker labels.

A follow-up signal-quality pass (waveform decode, not just metadata) found
a mean SNR-proxy of 41.8 dB across the corpus and no files that failed to
decode.

### 2.3 Licensing

NOODL-1.0's Forbidden Usage clause names generative AI and voice cloning
explicitly, conditioned on the legal owner's permission rather than an
absolute bar. Written permission was sought from the legal owner and
steward, stating the specific intended use (VITS training for the NEMA
early-warning system), the open-source release plan, and a staged
pilot-before-full-training approach. Permission was reported granted on
2026-07-31 and is recorded, with scope and date, in `docs/licensing.md` —
the diagnostic and pipeline-validation work described below (Sections 4–5)
predates and does not depend on that permission, as it used derived
features and unintelligible-by-design diagnostic output rather than
training a usable voice.

## 3. Architecture selection

Two candidates were evaluated under a controlled, identical local
protocol before any cloud spend, per `docs/model_bakeoff_protocol.md`:
**VITS** (Conditional VAE with adversarial learning, ICML 2021; MIT
licence; mature low-resource precedent) and **Matcha-TTS** (conditional
flow matching, ICASSP 2024; MIT licence; lighter, but needs a separately
licensed vocoder). F5-TTS was considered and excluded at this stage:
its code is MIT, but its official pretrained checkpoint is CC-BY-NC-4.0,
and training it from scratch on this data scale was judged too heavy to
justify given VITS's precedent in the low-resource TTS literature.

**Protocol.** 60 samples from one source group (50 train / 5 validation /
5 test), identical fixed split and seed, mono 22.05 kHz PCM WAV derived
from untouched source MP3, direct Tiv characters with no phonemizer, no
pretrained weights, 50 one-sample optimization steps per model, CPU.

**Results** (2026-07-30):

| Field | VITS | Matcha-TTS |
|---|---:|---:|
| Parameters | 83,044,588 (incl. discriminator) | 18,180,385 |
| CPU time/step | 0.837 s | 0.053 s |
| Initial loss | gen 213.09, disc 6.06 | 5.15 |
| Final step loss | gen 57.53, disc 2.79 | 3.79 |
| Validation loss | gen 58.35, disc 2.59 | 4.42 |
| Finite forward/backward | Pass | Pass |
| Checkpoint save/reload | Pass | Pass |
| Valid 22.05 kHz WAV output | Pass | Pass |
| Waveform path | Integrated decoder | Griffin-Lim only (diagnostic) |
| Checkpoint size | 997.7 MB | 218.5 MB |

Fifty steps from random initialization validates data flow, optimization
stability, and inference mechanics — it is not a naturalness or
intelligibility signal, and loss magnitudes are architecture-specific and
not comparable across models. **VITS was selected** as the first cloud
pilot: an integrated waveform decoder means one model and one licence
surface to manage, versus Matcha-TTS's lighter footprint plus a
separately-sourced vocoder dependency.

## 4. Staged validation before full training

Rather than committing directly to a long, costly training run, validation
proceeded in four increasing-cost stages, each gating the next:

**4.1 Diagnostic smoke test.** A deliberately tiny non-VITS network (data
load → train → checkpoint → resume → infer) on 10 samples, GPU, 35 steps
(30 + 5 resumed). All losses finite, checkpoint reload verified, produced
a valid WAV. Confirms pipeline mechanics only.

**4.2 GPU-scale VITS pilot.** Real VITS, real batching (batch size 8),
113 samples from one source group, ~1,080 steps. Batch sizes 4/8/16/32
were benchmarked for throughput and memory; 8 was used for this pilot.
Satisfies "run a fixed-budget cloud pilot for VITS" from the bake-off's
required gate before full training.

**4.3 Full training run.** Scaled to all 14 export groups — 1,176 clips
(1,096 train / 40 validation / 40 test), 2.52 hours of audio, 102-symbol
vocabulary (batch-16 calibration showed near-identical throughput to
batch 8, so batch 16 was kept for a better gradient estimate at no real
cost). Config: `configs/tiv_vits_full.yaml`; training script:
`scripts/train_vits_full.py`.

## 5. Full training results

| | |
|---|---|
| Hardware | NVIDIA L4, AWS `g6.2xlarge`, on-demand |
| Software | PyTorch 2.8.0+cu128, Coqui-TTS 0.27.5 |
| Steps | 103,500 (1,500 epochs × 69 steps/epoch) |
| Batch size | 16 |
| Wall-clock | 15h17m (2026-07-31 16:37 UTC → 2026-08-01 07:54 UTC), 0.532 s/step sustained |
| Parameters | 83,053,612 |
| All losses finite | Yes, for the entire run |
| Checkpoint reload | Verified (strict, fresh model + optimizer instances) |

The sustained per-step time (0.532 s) was substantially faster than an
initial single-epoch timing estimate (1.53 s/step) taken before `cudnn`
had settled into its fastest algorithm selection for this workload — the
lesson being that short calibration runs can meaningfully overestimate
total training time for this kind of adversarial training loop.

A synthesis was generated from the final checkpoint on a held-out test
sentence ("*Kpa ior kpishi hemba soon er a yila wan iti er Korwua*",
22.05 kHz, 7.55 s) and eight further held-out test sentences were
synthesized for listening review (`outputs/vits_full_samples/`), none of
which were seen during training.

**What convergence does and does not establish.** Finite, stable losses
across 103,500 steps show the model learned a consistent mapping from Tiv
text to mel-spectrogram and waveform without diverging or collapsing.
It does not establish intelligibility, naturalness, or correct
pronunciation — those require a Tiv speaker's ear, not a loss curve.

## 6. The translation gap

The motivating use case — reading English-authored emergency advisories
aloud in Tiv — requires translation, which is a categorically different
problem from speech synthesis and was not solved by this work. A
deliberate search found no usable existing resource:

- **NLLB-200** (FLORES-200 language list): Tiv absent.
- **Google Cloud Translation** (documented language list): Tiv absent.
- **Google Translate's 2024 110-language expansion**: Tiv not among the
  named additions.
- **OPUS-translatewiki**: nominally has a `tiv` corpus, but on inspection
  it is 85 lines of MediaWiki software-interface strings ("Cancel",
  "Project:...") across only 5 of 1,407+ documents in that release — not
  natural sentences, and not cleanly sentence-aligned with the English
  side in the distributed format.
- **Masakhane / MAFAND-MT**: no confirmed Tiv pair among established
  benchmarks (which cover Swahili, Yoruba, Hausa, and others).

**Recommendation.** For a life-safety application, we recommend against
building or depending on general-purpose machine translation at all. A
bounded set of human-translated alert templates (e.g. "flood warning for
{area}", "evacuate now", "shelter in place"), translated once by a
qualified Tiv translator with placeholders for variable content, is both
more tractable than sourcing or building an MT corpus and inherently
safer than trusting machine-translated safety-critical text.

## 7. Limitations

- **No native-speaker evaluation.** This is the single most important
  open item. Every result in this report is a pipeline-correctness or
  optimization-convergence result, not a quality result.
- **Speaker/session provenance unverified.** The 14 export directories
  were combined on the stated assumption of a single speaker; this was
  not independently confirmed from audio or metadata.
- **Character inventory noise.** Rare non-Tiv-script characters (Greek,
  Cyrillic, Hebrew, one emoji) were present in fewer than 5 transcripts
  each and were not filtered out of the training vocabulary; they are
  very unlikely to have influenced training materially given their
  rarity, but a cleaner transcript pass would remove this ambiguity.
- **Translation is unsolved** (Section 6) and is likely the harder of the
  two problems this project set out to eventually address.
- **Verification status of the reported licence permission.** The
  generative-AI permission is recorded as reported by the project team in
  `docs/licensing.md`; the written confirmation itself was not
  independently reviewed as part of this audit process.

## 8. Reproducibility

All configs, scripts, and generated reports referenced above are checked
into this repository:

- Dataset audit: `docs/dataset_audit.md`, `docs/dataset_signal_audit.md`
- Licensing record: `docs/licensing.md`
- Architecture comparison: `docs/model_bakeoff_protocol.md`,
  `docs/model_bakeoff_results.md`
- Training log (all stages): `docs/training_log.md`
- Full training config/script: `configs/tiv_vits_full.yaml`,
  `scripts/train_vits_full.py`
- Final checkpoint metrics: `checkpoints/vits-full/tiv_vits_full-July-31-2026_04+37PM-4834afa/metrics.json`
- Listening samples: `outputs/vits_full_samples/`

## 9. Next steps

1. Native Tiv speaker review of `outputs/vits_full_samples/` and the
   Streamlit demo (`scripts/demo_app.py`) — blocking everything below.
2. Resolve the translation approach (Section 6) with whoever owns the
   NEMA alert content — template-based is recommended.
3. If quality warrants it, consider target-speaker fine-tuning or
   additional data before any production framing of this system.
4. Independent verification of the generative-AI licence permission's
   written evidence (Section 7).
