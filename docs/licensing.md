# Licensing status

## Dataset

Provenance was identified by the project owner on 2026-07-30.

| Field | Value |
|---|---|
| Source | Mozilla Data Collective, dataset `cmo4nmfam00nxny07rssox2tj` |
| Name | Tiv-TTS-Dataset |
| Licence | Nwulite Obodo Open Data Licence 1.0 (NOODL-1.0) |
| Steward | Institute of African Digital Humanities |
| Point of contact | Emmanuel Ngue Um |
| Legal owner | Daniel Nyitse |
| Speaker | One speaker, Ihyarev dialect |
| Content | Prompted read speech: expository prose, folk narrative, proverbs |
| Published duration | 5 h 50 min 43 s across 2,443 clips |

Source: <https://mozilladatacollective.com/datasets/cmo4nmfam00nxny07rssox2tj>

### Blocking restriction for this project

The dataset page states, under Forbidden Usage, that the user agrees not to use
the data for:

- Generative AI
- Voice cloning or speaker imitation
- Reproduction, duplication, modification, or redistribution
- Commercial use without explicit permission

It also states the dataset is "for research and scientific use only" and must
not be re-hosted or redistributed.

Training a TTS model on this corpus is generative AI, and because the corpus is
a single speaker, the resulting voice is that speaker's. Both named
restrictions apply directly to this project's stated goal. NOODL frames the
generative-AI restriction as conditional on permission from the dataset's legal
owner rather than as an absolute bar, so the resolution is written permission,
not a workaround.

### Required before any cloud training run

1. Obtain written permission from the legal owner (Daniel Nyitse) and the
   steward (Institute of African Digital Humanities), via the Mozilla Data
   Collective request-access route or direct contact.
2. State explicitly what is being requested: training a Tiv TTS model on the
   corpus, the intended use of the resulting model, whether the model weights
   will be released, and to whom.
3. Confirm the NOODL tier that applies to this project. NOODL grants different
   terms by the user's geographic and economic position, and African users
   receive the most permissive tier — but the generative-AI restriction is
   documented as requiring owner permission regardless of tier.
4. Record the permission, its scope, and its date in this file before the first
   paid AWS run.

Work completed so far — a read-only audit, derived local features, and
short diagnostic training runs producing unintelligible output — is defensible
as research and scientific use. Nothing has been re-hosted or redistributed,
and the raw corpus is unmodified. This paragraph is an engineering record, not
a legal opinion.

## Diagnostic smoke model

The local TinyTTS smoke model uses original project code and no pretrained
model weights. It exists only to validate data loading, tokenization, gradient
flow, checkpointing, resume, and inference.

This does not remove the need to review the training dataset licence or the
licence of any future pretrained base model.

## Installed dependency inventory

The reproducible environment is recorded in `uv.lock`. Installed package
metadata reports:

| Package | Version | Installed metadata |
|---|---:|---|
| torch | 2.13.0 | Composite open-source licence expression in wheel metadata |
| torchaudio | 2.11.0 | Review upstream licence and bundled components before distribution |
| numpy | 2.4.6 | Composite permissive licence expression in wheel metadata |
| soundfile | 0.14.0 | BSD 3-Clause |
| PyYAML | 6.0.3 | MIT |

These entries are an engineering inventory, not legal advice.

## Pretrained models

No pretrained checkpoint was downloaded or used. Meta MMS was considered
because it covers many low-resource languages, but Tiv is not in its official
TTS list and the MMS code/model release is CC-BY-NC 4.0. F5-TTS, XTTS, NeMo,
Coqui VITS, ESPnet, Piper, and other candidates require separate code,
checkpoint, training-data, and deployment-licence review before selection.

