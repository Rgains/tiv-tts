# Draft permission request — NOT YET SENT

This is a draft only. Fill in the bracketed placeholders, independently
verify the recipient addresses against the Mozilla Data Collective dataset
page (do not trust them solely from this draft), and have a project
principal review and send it. Recording the permission, its scope, and its
date in `docs/licensing.md` is still required once granted, per that file's
"Required before any cloud training run" section.

---

**To:** Daniel Nyitse <[OWNER EMAIL — verify on the Mozilla Data Collective dataset page]>
**Cc:** Emmanuel Ngue Um <[CONTACT EMAIL — verify on the Mozilla Data Collective dataset page]>; Institute of African Digital Humanities
**Subject:** Permission request — Tiv-TTS-Dataset for a humanitarian early-warning voice system (generative AI use, NOODL-1.0)

Dear Mr. Nyitse,

My name is [YOUR NAME], [YOUR TITLE/ROLE] at [YOUR ORGANIZATION]. I'm writing
to request written permission to use the Tiv-TTS-Dataset
(`cmo4nmfam00nxny07rssox2tj`, Mozilla Data Collective) for a use that falls
under the dataset's Forbidden Usage restriction on generative AI and voice
synthesis, and so requires your explicit permission under NOODL-1.0.

**What we are building.** In collaboration with [Nigeria's] National
Emergency Management Agency (NEMA), we are building a text-to-speech system
for Tiv as part of an early-warning and disaster-advisory service. The goal
is to deliver emergency alerts and safety guidance in Tiv for communities
where English-language advisories are not accessible or effective. This is a
public-safety application, not a commercial product.

**What we are requesting permission for.** Training a TTS model (we are
currently evaluating VITS and Matcha-TTS) directly on the audio and
transcripts in the Tiv-TTS-Dataset, to produce a synthetic Tiv voice capable
of reading emergency advisories aloud.

**Release plan.** The resulting model and code will be released open source.
[Confirm licence — e.g., a permissive open-source licence / a copyleft
licence / not yet decided.] [Confirm whether "open source" means public
release to anyone, or release restricted to a specific set of
organizations/partners.]

**Attribution and giving back.** We would like to credit the Institute of
African Digital Humanities and you as the dataset's legal owner in any
release. We're also glad to contribute back to the dataset itself — for
example [additional recordings / corrected transcripts / documentation of
issues we found during a data audit] — if that would be useful, rather than
only drawing from it.

**Staged approach.** We would like to run a small-scale technical pilot
first — training on a limited GPU budget to validate the pipeline and
produce early listening samples — before committing to a full production
training run, and would welcome your input on whether that staged approach
is acceptable under this permission or whether you'd prefer to review
before each stage.

**Tier confirmation.** NOODL-1.0 applies different terms depending on the
requester's geographic and economic position, with African users receiving
the most permissive tier. [YOUR ORGANIZATION] is based in [LOCATION] —
please let us know which tier applies to this request.

Please let us know if you need any further information, and thank you for
making this corpus available for Tiv-language work.

Sincerely,
[YOUR NAME]
[YOUR ORGANIZATION / ROLE]
[CONTACT EMAIL / PHONE]

---

## Notes for internal tracking (not part of the email)

- Steward contact (Institute of African Digital Humanities) has no email on
  file yet from the page fetch — cc via Emmanuel Ngue Um or the dataset's
  "Request Access" button until a direct steward address is confirmed.
- Once permission is received, record it in `docs/licensing.md` under
  "Required before any cloud training run" with date and scope, per that
  file's own checklist.
