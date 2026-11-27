# Stage 4: emotional Urdu speech (data curation plan)

This is a hand-off document. Give it to a coding agent together with access to this repository
(`hassan-31x/orpheus-ft-aslp-urdutts`) and read access to the Hugging Face repos listed below.
The agent's job is to **build and publish the stage-4 training dataset**. Training is a later step.

Written 2026-11-27. Stages 1–3 are treated as done: the model (Orpheus-3B + LoRA r32, fine-tuned on ~41 h of
Urdu from ASLP, UAT and UrduTTS) speaks correct Urdu. Stage 4 teaches it to speak **with a requested emotion**.

---

## 1. Goal and constraints

- The model must render the same Urdu text as neutral, happy, sad, angry, fearful, etc., on request, **without
  losing pronunciation, intelligibility or the voices it already has** (no catastrophic forgetting).
- Main new data is **UrduSER**, which is small (≈3.5–4 h). Every clip matters, so curation has to be careful and auditable.
- Everything stays **private and for this course project**. Credit all datasets in the report.

## 2. Source datasets

### 2.1 UrduSER (new, the emotion source)
- Mendeley Data, DOI `10.17632/jcpfjnk5c2` (use the latest version, currently v4; record the version you used).
  Paper: Akhtar et al., *UrduSER: A comprehensive dataset for speech emotion recognition in Urdu language*,
  Data in Brief 60 (2025) 111627.
- Facts from the paper. **Verify every one after downloading.**
  - 3,500 WAV clips: **10 professional actors (5 male, 5 female) × 7 emotions × 50 clips**.
  - Emotions: Angry, Fear (fear and anxiety merged), Boredom, Disgust, Happy, Neutral, Sad. 500 clips each.
  - Clips are 2–6 s, giving an estimated 3.5–4 h total. Audio is 44.1 kHz, 32-bit float.
  - Layout: `<actor>/<emotion>/<file>.wav`. Filenames are `actor_gender_emotion_index` (for example `2_1_3_04.wav`).
  - Urdu dialogue scripts are in an accompanying Excel file. The texts are almost all unique, so the same
    sentence is generally *not* repeated across emotions.
  - Clips were cut from Pakistani drama serials, telefilms and reality shows on YouTube. The authors removed music
    with Audacity + OpenVINO source separation and suppressed noise, then trimmed silence at −60 dB.
  - Licence: CC BY 4.0. The underlying audio comes from TV dramas, so the data stays private and is credited, and
    we never release voices that imitate identifiable actors.
- Known weaknesses to plan for:
  - The audio is acted, not spontaneous.
  - The music separation may have left artefacts or residual music.
  - The scripts were transcribed by humans, so some may not match the audio.
  - Disgust and Boredom are hard to hear even for people.
  - The clips come from the same kind of source as ASLP's drama category, so **the two datasets may overlap**.

### 2.2 Existing data (for replay and extra emotion)
All of these are private, owned by `hassan-31x`, and need read access.

| Repo | What it is |
|---|---|
| `hassan-31x/urdu-orpheus-stage3-data` (revision `ace55d2970e84fad801932e5ae8bf601ad6a1315`) | The v4 training set used in stage 3: 41.4 h train (ASLP 13.8 h, UAT 17.6 h, UrduTTS 10.0 h) and 870 validation clips. It contains `manifests/train.jsonl`, `labels/v4.jsonl` and `report.json`. |
| `hassan-31x/urdu-orpheus-stage2-data` | The scored source pools (`pool/aslp`, `pool/urdutts`, `pool/uat`). Each row has DNSMOS/VAD metrics and SHA-256 hashes. |
| `hassan-31x/orpheus-urdu-stage2-checkpoints` | Model checkpoints, dataset pointers (`datasets/*.json`), hard-word mining (`mining/v1/`) and evaluation prompts. |

ASLP already carries emotion labels. Hours by emotion in the **v4 train split** (measured 2026-11-27):

| ASLP emotion | Hours | Speakers |
|---|---|---|
| neutral | 3.93 | 1062 |
| serious | 3.43 | 938 |
| happy | 1.98 | 540 |
| angry | 1.72 | 546 |
| fearful | 0.94 | 392 |
| sad | 0.84 | 415 |
| excited | 0.71 | 160 |
| surprised | 0.23 | 140 |

These are many speakers with short, noisy, conversational clips. That complements UrduSER, which has few speakers
with many clips each and every emotion per speaker. Treat ASLP emotion labels as **noisier** than UrduSER's, and
check them before relying on them (see section 5.6).

## 3. What the literature says, and what we adopt

| Finding | Source | What we do |
|---|---|---|
| Fine-tuning an LLM-based TTS on limited new-style data damages its earlier abilities (word accuracy, speakers). Updating only some parameters reduces this. | Efficient Emotion and Speaker Adaptation in LLM-Based TTS via Characteristic-Specific Partial Fine-Tuning, arXiv 2501.14273 | Keep LoRA, use a low learning rate, and **replay** stage-3 data in every batch. Monitor forgetting on the old eval prompts. |
| Orpheus is trained to follow a text prompt and supports inline tags in the text (`<laugh>`, `<sigh>`, …). The multilingual research models were fine-tuned the same way. Unsloth recommends starting from a fine-tuned checkpoint and keeping tags as plain text. | Orpheus TTS README and multilingual collection (canopylabs); Unsloth TTS fine-tuning guide | Express emotion **as plain words in the prompt**, as our stage-1–3 voice tag already does. Do not add new special tokens. |
| Newly invented tags trained on a small set can be ignored by the model, while tags it already knows keep working. | VoxCPM issue #292 (a different model, so a warning only) | Reuse the emotion slot the model has already seen (ASLP tags carry emotion). Keep the tag vocabulary small and consistent. |
| Emotional LLM-TTS models are trained with an emotion description plus text, and judged by an SER classifier together with WER and listening tests. | EmoVoice (arXiv 2504.12867); EMORL-TTS (arXiv 2510.05758) | Evaluate with an SER judge, CER and listening (section 7). |
| Preference tuning with **neutral speech as the negative** for the same text sharpens emotion. | Emo-DPO (arXiv 2409.10157) | Optional future work. Section 5.9 prepares the data so it stays possible. |

## 4. Design decisions (please keep these)

1. **Prompt format is unchanged:** `"<speaker> | <gender> | <style> | <emotion>"`, followed by the text, exactly as
   in stages 2–3 (`labels/*.jsonl` field `voice`; see `stage2_labels.py`). UrduSER examples:
   - `urduser_f03 | female | acted | angry`
   - `urduser_m01 | male | acted | neutral`
2. **Canonical emotion vocabulary** (lower-case English words): `neutral, happy, sad, angry, fearful, disgusted, bored, surprised`.
   - UrduSER mapping: Fear→fearful, Boredom→bored, Disgust→disgusted, and the rest map to themselves.
   - ASLP mapping: excited→happy and serious→neutral in the canonical field.
   - **Always keep the raw label** in its own column, so the mapping can change later without a rebuild.
3. **Keep UrduSER neutral clips.** Each actor speaks all seven emotions, so the only thing that changes inside an
   actor is the emotion. That is what teaches the model that the emotion word controls delivery, not identity.
4. **Speaker identity:** each UrduSER actor gets a stable tag (`urduser_m01` … `urduser_f05`). These are the
   candidate pool for the project's later named voices (e.g. "Ali", "Ahmed", a female voice). Those names are a
   later mapping. Do not rename now.
5. **Utterance-level emotion first.** UrduSER labels whole clips, so phase A gives one emotion per clip.
   "Emotion tags in between" (switching mid-sentence) is phase B, an optional augmentation (section 5.8).
6. **Text normalisation identical to stages 2–3:** reuse the functions in `stage3_build.py`, `build_stage2.py` and
   `stage2_labels.py`. Stage-3 text has no punctuation, so strip it here too. Record the original script text in a
   separate column, because `?` and `!` may be useful later.

## 5. Curation pipeline (what the agent builds)

Model it on `stage3_build.py`. Write a script `stage4_build.py` with resumable phases and unit tests in
`tests/test_stage4_build.py` that use synthetic data. Generate the Kaggle notebook the same way
`make_stage3_notebooks.py` does, with a smoke run on a few clips first, then the full build. The heavy steps
(alignment, SER, DNSMOS) need a GPU, so run them on Kaggle.

### 5.0 Repo map: where to start

All paths are relative to the repository root. Read these before writing code; **copy their patterns, import their helpers, do not rewrite them.**

| Need | File | Use |
|---|---|---|
| The model to copy | `stage3_build.py` | Phased, resumable dataset build (`--phase uat/align/select/all`, `--limit`, `--no-hub`). Reuse: `PoolSource`, `uat_worker` (convert + DNSMOS/VAD per clip, with its first-use race retry), `ctc_scores` / `phase_align` (MMS-1b-all CTC text-audio scoring), `uat_gate`, `select_v4`, `verify_final`, `make_labels`, `phase_select` (select, verify, upload, pointer). |
| Hub, shards, upload | `build_stage2.py` | `Hub`, `PartWriter`, `read_parts`, `write_part`, `to_flac`, `scorer`, `norm_text`, `sha256_bytes`, `hours`, `summarize`. |
| Voice tags and emotion labels | `stage2_labels.py` | `voice(row)` builds `"speaker \| gender \| style \| emotion"`; `emotion(text)` and `gender(g)` are the existing mappings. UrduSER tags must use the same format. |
| Audio quality | `quality.py` | `QualityScorer` (DNSMOS + VAD), `acoustic_metrics`, `to_16k`, `text_problem`, `transcript_errors`, `normalize_asr`. |
| How the trainer reads a dataset | `stage2_data.py` (`prepare_dataset`), `stage2_plan.py` (`epoch_config`, `load_data_pointer`, `LEARNING_RATES`) | Your dataset has to load through `prepare_dataset`; the pointer file is `datasets/<name>.json` on the checkpoint repo. |
| Loss weights for hard words | `weighted_loss.py`, `mining/v1/loss_weights.json` on the checkpoint repo | Source names other than `aslp`/`urdutts` get weight 1.0, so UrduSER (`urduser`) needs no change. Remember to add `"urduser"` to the per-source validation loop in `finetune_mixed.py` (`for source in ("aslp", "urdutts", "uat"):`). |
| Notebook generator | `make_stage3_notebooks.py`, `kaggle_submit_stage3.py` | Copy these to `make_stage4_notebooks.py` / `kaggle_submit_stage4.py`. They produce a smoke-then-full Kaggle notebook and publish it with `--no-run`. |
| Tests | `tests/test_stage3_build.py` | Pattern for synthetic-pool tests (unique texts, end-to-end `phase_select`, CTC and worker tests). Run: `python -m unittest discover -s tests`. |
| Past decisions and numbers | `STAGE3_PLAN.md`, `README.md`, `research/RESEARCH.md` | Why the data is shaped as it is. |
| Eval code to reuse | `eval_pronunciation.py`, `kaggle_eval_pron.ipynb`, `listen_samples.py`, `kaggle_listen.ipynb` | CER/WER with Whisper large-v3 and paired bootstrap; listening-sample generation. |

**Files to create** (new, so they do not collide with stage 3):

- `stage4_build.py`: the build, in phases such as `ingest`, `convert`, `score`, `judge`, `select`, `all`.
- `stage4_labels.py`: the UrduSER filename and folder parser plus the emotion mapping tables from section 4.
- `tests/test_stage4_build.py`: synthetic-data tests, one test per gate in section 5 and per acceptance check in section 8.
- `make_stage4_notebooks.py`, `kaggle_stage4_data.ipynb` (generated), `kaggle_submit_stage4.py`.
- `configs/stage4_mix.json`: the block shares from section 6 as numbers, so training can retune them.
- `STAGE4_PLAN.md`: a short record of the decisions you make while building, in the style of `STAGE3_PLAN.md`.

**Conventions to keep.**
- Python 3.13, the `requirements-kaggle.txt` stack. Large work goes to `/kaggle/working`; never commit audio, parquet or model files (see `.gitignore`).
- Never write a Hugging Face or Kaggle token into any file, notebook, log or commit. Read `HF_TOKEN` from the environment.
- Pin every input to a revision or checksum, as `stage3_build.py` does (`UAT_REV`, `POOL_REV`, `MINING_REV`).
- Each phase must be resumable and idempotent. The full build is one Kaggle run that the owner starts from the web UI.

**5.1 Ingest.**
- Download UrduSER and store its checksum and version.
- Parse the Excel scripts and join them to files.
- Assert exact counts: 10 actors × 7 emotions × 50 clips.
- Parse each filename and check it against the folder name. **The folder is the source of truth.** Log every mismatch.
- Record duration, sample rate, channels, peak and clipping fraction for each clip.

**5.2 Audio conversion.**
- Convert 44.1 kHz float to **24 kHz mono 16-bit** (SNAC's rate) with a high-quality resampler.
- Trim leading and trailing silence, keeping about 150 ms of padding.
- **Do not loudness-normalise each clip.** Loudness is part of emotion: an angry clip should stay louder than a sad
  one. Apply one gain per actor instead, so each actor's overall level matches, and limit peaks to −1 dBFS.

**5.3 Text–audio agreement.**
- Reuse the MMS-1b-all CTC scoring from `stage3_build.py` (`ctc_scores`, per-character NLL and greedy-decode CER),
  and also compute Whisper large-v3 CER.
- Drop clips with CER > 0.5, unalignable characters, or empty or Latin-only text.
- Send the worst ~10% by alignment cost to **human review** rather than auto-dropping them, because emotional speech
  aligns worse.

**5.4 Audio quality.**
- Run `quality.QualityScorer` (DNSMOS + VAD).
- Use **emotion-aware thresholds**: angry and fearful speech scores lower on DNSMOS SIG for good reasons. Set
  thresholds per emotion, for example at the 5th percentile of that emotion, rather than one global cut.
- Detect leftover music with an audio tagger (PANNs or YAMNet "Music" probability). Drop clips with clear music.
- Flag band-limited clips (energy above about 7 kHz near zero), which point to low-quality YouTube sources.

**5.5 Dedup and leakage.**
- Match by audio hash, near-duplicate audio (chromaprint or embedding similarity) and normalised text against
  the ASLP, UAT and UrduTTS pools and the 150 evaluation prompts.
- Overlapping clips must not appear in any test or validation split. Report how many there are.

**5.6 Emotion label check.**
- Score every UrduSER and ASLP-emotional clip with an SER "judge":
  1. Train a linear probe on frozen `emotion2vec_plus_large` (or WavLM) features using the **UrduSER train split only**.
  2. Report its accuracy on the held-out split. That number is the ceiling for later evaluation.
- Store each clip's full probability vector and a `label_confidence`.
- For **UrduSER**, do not auto-drop on disagreement. Drop a clip only when the judge is confident in a *different*
  emotion and a listener agrees.
- For **ASLP**, keep emotional clips only when the judge agrees above a threshold. Its labels are noisier, and there
  are hours to choose from.

**5.7 Splits.**

| Split | What goes in it | Clips |
|---|---|---|
| **test-emotion** (never trained) | Per actor × emotion, 5 clips whose text is unseen | 350 |
| **validation** | Per actor × emotion, 3 clips, plus 300 replay clips taken from the existing v4 validation split | 210 + 300 |
| **train** | Everything else | about 2,900 UrduSER clips |

- Stratify every split by actor × emotion. Fix the random seed and write the seed and procedure into `report.json`.
- **Cross-speaker test:** 60 neutral-voice prompts (ASLP and UAT speakers) paired with each emotion word. These
  have no ground-truth audio. They measure whether emotion *transfers* to voices that never had it.

**5.8 Optional phase B: mid-sentence emotion switching.**
- Build up to about 400 synthetic examples by concatenating two clips from the **same actor** with different
  emotions, joined by 250–400 ms of silence. Put the emotion inline in the text, for example
  `<happy> text one <sad> text two`, and set the prompt's emotion field to `mixed`.
- Keep these in their own `augmentation` column and their own subset, so training can turn them on or off.
- Do not include these in test-emotion.

**5.9 Optional: pairs for Emo-DPO later.** For each test-emotion text, record a neutral counterpart from the same
actor when one exists. Write a pairing table only; this needs no extra audio work.

## 6. Training mixture (what the dataset should contain)

Target **about 12–14 h per epoch**. At stage-3 speed (~41 h in ~8.5 h on a T4), that is roughly 2.5–3 GPU hours
per epoch, so 2–3 epochs fit in one or two Kaggle sessions.

| Block | Hours | Share | Content |
|---|---|---|---|
| **A. UrduSER train** | ~3.3 | ~25% | All 7 emotions including neutral, all 10 actors. Upsample ×2 at most; more risks memorising 50 lines per actor-emotion. |
| **B. ASLP emotional** (judge-verified) | ~3.0 | ~22% | happy/excited, angry, fearful, sad, surprised. Cap each speaker at about 3 min. Many speakers, so emotion generalises beyond the 10 actors. |
| **C. Replay, conversational** | ~2.5 | ~18% | ASLP neutral and serious clips from the v4 train split, with the same speakers as v4. Keeps the conversational default voice. |
| **D. Replay, read speech** | ~2.5 | ~18% | UAT clips from v4: wide vocabulary, the uat_narrator voice. |
| **E. Replay, hard words** | ~2.0 | ~15% | UrduTTS clips carrying the most hard-word signal (`mining/v1`), keeping pronunciation gains. |

So about 47% of each epoch is emotional and about 53% is replay, close to half and half. Expose the shares as config
values (`report.json` plus a JSON config), because the training run may need to shift them: increase block A if
emotion is weak, increase replay if CER regresses. Take replay from **stage-3 v4 clip IDs** rather than fresh
sampling, so the model sees familiar data and forgetting can be measured cleanly.

Balance within the emotional blocks: no canonical emotion should be below about 8% of emotional hours.
`disgusted` and `bored` exist only in UrduSER, so they stay at UrduSER's natural share. **Do not** upsample them
further. Report per-emotion hours, clips and speakers.

## 7. Evaluation the dataset must support

These are the training team's metrics. The dataset must ship the prompts and the references they need.

1. **Emotion accuracy:** the SER judge from section 5.6, run on generated audio for test-emotion and for the
   cross-speaker prompts. Compare against the judge's accuracy on real held-out audio, which is the ceiling.
2. **Intelligibility:** Whisper large-v3 CER/WER on test-emotion and on the existing 150 UrduTTS + 150 UAT prompts.
   The old prompts must **not regress** beyond the paired-bootstrap 95% CI. This is the forgetting check.
3. **Speaker similarity:** ECAPA/WavLM-SV cosine between generated and real audio of the same actor.
4. **Listening:** a blind test with about 8 sentences × 4 emotions, using the `listening-compare` page.

## 8. Deliverables and acceptance checks

**Publish** a private HF dataset, `hassan-31x/urdu-orpheus-stage4-emotion-data`, laid out like stage 3:

- `train/` and `validation/` parquet shards with 24 kHz audio, in the same columns as v4.
- `test_emotion/`.
- `manifests/*.jsonl` with every metric: DNSMOS, VAD, CTC NLL, CER, music probability, SER probabilities,
  raw and canonical labels.
- `labels/v5.jsonl`, with the same schema as `labels/v4.jsonl`: `id`, `split`, `keep`, `voice`.
- `eval/` holding the cross-speaker prompts CSV, the test-emotion prompts CSV and the DPO pair table.
- `report.json`, plus a `README.md` dataset card that credits UrduSER, ASLP, UAT and UrduTTS.

Also publish the pointer `datasets/stage4-v5.json` in the checkpoint repo, in the same format as
`datasets/stage3-v4.json`.

**Acceptance (the agent must check and print each one):**
- [ ] UrduSER counts match 10 × 7 × 50, or every deviation is listed with its reason.
- [ ] No clip appears in more than one split. No test-emotion or validation clip overlaps ASLP, UAT, UrduTTS or the
      150 evaluation prompts, checked by hash, near-duplicate audio and text.
- [ ] Every training clip is 24 kHz mono and 1.0–20 s long, has text after normalisation, and has CER ≤ 0.5.
- [ ] `report.json` contains:
  - drops at every gate, by reason;
  - hours, clips and speakers per emotion and per block;
  - the actor × emotion matrix;
  - the SER judge's held-out accuracy and confusion matrix.
- [ ] A human audit listened to at least 2 clips per actor × emotion (140 clips) and logged the results in `audit.csv`.
- [ ] The trainer's own loader (`stage2_data.prepare_dataset`, as in `consumer_check()` in `make_stage3_notebooks.py`)
      loads the published dataset.
- [ ] Unit tests pass, and the smoke run passes on Kaggle before the full build.

## 9. Things to watch

- **Label noise beats data size.** With about 3.5 h, a few hundred mislabelled clips can blur an emotion. The audit
  and the SER judge come before any filtering by thresholds.
- **Speaker–emotion confounding in ASLP:** some ASLP speakers appear in only one emotion. Cap each speaker per
  emotion so the model does not learn "this voice = angry".
- **Loudness and pitch are emotion cues.** Never normalise them away per clip (section 5.2).
- **Prosody vs residual music:** source separation can leave a faint music bed that the model will learn to
  generate. The music-tag gate matters.
- **Unseen text:** test-emotion texts must not appear anywhere in training, including inside replay.
- **Ethics:** the actors are real people. Use their voices only for course evaluation, and do not publish a model
  that imitates them.

## Sources

- UrduSER on Mendeley Data: https://data.mendeley.com/datasets/jcpfjnk5c2
- UrduSER paper (Data in Brief, 2025): https://pmc.ncbi.nlm.nih.gov/articles/PMC12151234/
- Orpheus TTS README (emotive tags, fine-tuning): https://github.com/Mu-L/Orpheus-TTS (mirror of canopyai/Orpheus-TTS)
- Orpheus multilingual research release: https://huggingface.co/collections/canopylabs/orpheus-multilingual-research-release
- Unsloth TTS fine-tuning guide: https://unsloth.ai/docs/basics/text-to-speech-tts-fine-tuning.md
- Characteristic-specific partial fine-tuning (forgetting in LLM-TTS): https://arxiv.org/abs/2501.14273
- EmoVoice: https://arxiv.org/abs/2504.12867
- Emo-DPO: https://arxiv.org/abs/2409.10157
- EMORL-TTS: https://arxiv.org/abs/2510.05758
- VoxCPM issue on new tags being ignored: https://github.com/OpenBMB/VoxCPM/issues/292
- Older Urdu emotion set (4 emotions, talk shows; not used, listed for completeness):
  https://huggingface.co/datasets/Pak-Speech-Processing/urdu-emotions
