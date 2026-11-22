# Stage 3 plan: more natural, wider-coverage Urdu (epochs 6–8)

Written 2026-11-22 after epoch 5, updated the same day with the owner's decisions.
Goals, in priority order: (1) natural, conversational delivery, (2) wide vocabulary, (3) keep improving pronunciation.

## Decisions made

| Question | Decision |
|---|---|
| Default voice | **Conversational** (an ASLP non-acted speaker with neutral emotion). Named speakers (Ali, Ahmed, a female voice…) come later, each with its own tag. |
| Re-mine hard words? | **No.** The epoch-5 weights (`mining/v1`) stay pinned, so words the model already says correctly keep being trained and improve further. |
| New data | UAT (`mahwizzzz/UAT`), screened for text/audio agreement, mixed with ASLP and UrduTTS. |
| GPU account | The friend's Kaggle account for the data build and training, each as a single run with no manual steps beyond attaching the secret. |
| Epoch 5 vs 4 | Epoch 5 is the base (heard improvement on some hard words; automatic metrics tied). |

## Why not just more epochs on the same data

Eval loss plateaued from step 2800 of epoch 5 (4.1246), and CER/WER were a statistical tie with epoch 4.
New data and a higher learning rate for the first epoch on it are what can move the model. The old plan's epoch 6 at 1e-5
would barely change anything, so epoch 6 now restarts at 4e-5, epoch 7 at 2e-5, epoch 8 at 1e-5.

## The data (v4, about 40 h)

| Source | Stage 2 (v2/v3) | v4 | Notes |
|---|---|---|---|
| ASLP conversational | 13.4 h (v3 capped acted speech at 7 h → 9.6 h) | about 14–15 h | Same gates, same DNSMOS tier and same speaker-disjoint holdout as before; per-speaker cap 30 → 45 min; **no acted-speech cap**, since conversational hours are the scarce resource. Emotion stays in the voice tag. |
| UAT read speech | 0 | about 11–14 h after screening | 18.3 h raw, 20,444 clips, mean 3.2 s. Very clean audio (first clip: DNSMOS 3.6 / 4.1 / 3.35). About 17% of clips are shorter than 1.5 s and are dropped. |
| UrduTTS audiobook | 30.7 h | about 11–12 h | Prioritises clips carrying the most hard-word signal (75%), then the cleanest others. |

- **Why 40 h:** an epoch of the 44 h mixture took about 8.5 h plus about 1 h of first-time token encoding. 40 h keeps an epoch plus the smoke test inside one 12 h session, so the friend never has to resume.
- **Conversational share:** about 36% of hours (was 24% in stage 2). That is the most the usable ASLP data allows; the rest is read speech. The conversational default voice and the voice tags carry the natural style.
- **Validation:** the same ASLP holdout speakers and the same 250 UrduTTS clips as before (identical selection procedure), plus 250 held-out UAT clips. Per-source validation losses are logged separately (`aslp_`, `urdutts_`, `uat_`).
- **Held-out eval prompts:** 150 UAT sentences (`eval/uat_heldout_prompts.csv`) and the existing 150 UrduTTS prompts (`mining/v1/eval_prompts.csv`).

### UAT risks and how they are handled
- **Transcript quality:** spellings like سادت for سعادت mean some text does not match the audio. Every UAT clip is scored with MMS-1b-all (Urdu) CTC: per-character alignment cost of the transcript plus the greedy-decode CER. The worst 15% by alignment cost and anything with CER above 0.5 are dropped; clips with Latin letters or unalignable characters are dropped; duplicates are removed.
- **Licence:** the dataset card has none. The built dataset is **private, course project only**, not for redistribution; credit the source in the report; ask the author (`mahwizzzz`) before releasing any model trained on it.
- **Punctuation:** UAT text has none, like UrduTTS, so it is consistent with the bulk of the data.

## Training

| Epoch | Init | LR | Notes |
|---|---|---|---|
| 6 | epoch 5 | 4e-5 | v4 data, `mining/v1` weights with `oversample=1` (v4 has almost no hard-word-free UrduTTS clips left, so "repeat hard, drop easy" would delete all easy clips) |
| 7 | epoch 6 | 2e-5 | same data, new shuffle |
| 8 | epoch 7 | 1e-5 | only if epoch 7 beat epoch 6 on the gate |

The dataset revision is read at run time from `datasets/stage3-v4.json` in the checkpoint repo, so no code change sits between
the data build and training (and the smoke receipt, which is tied to the code commit, stays valid).

### Gate after every epoch (about 1 GPU h)
Same seeds; the conversational default voice and the narrator voice.
1. CER, WER, hard-word accuracy on the 150 UrduTTS prompts (comparable with epochs 4/5) and the 150 UAT held-out prompts, with a paired bootstrap; a difference counts only outside the 95% CI.
2. Naturalness proxies on CPU: F0 spread, speaking rate and pause ratio against real conversational (ASLP) and read (UrduTTS) clips.
3. Blinded A/B listening on about 12 sentences.

Keep an epoch only if it does not regress on 1 and wins on 2 or 3.

## Run book (what the friend does)

Everything is pushed to the repo at one commit; both notebooks are pinned to it. Each notebook is one run:

1. **`orpheus-urdu-stage3-data`** (about 2.5–3 GPU h): attach the `HF_TOKEN` secret, enable GPU + Internet, Save & Run All.
   Smoke test on 48 clips first, then the full build, verification, upload, pointer, and a consumer check with the trainer's own loader. Success: the last cell prints `STAGE-3 DATA READY`.
2. **`orpheus-urdu-stage3-train`** (about 10 GPU h): same steps. Smoke run, then the full epoch 6; if Kaggle stops the session, run it again and it resumes. Success: status `complete`.

The owner first runs both smoke paths once on their own account (about 10 min and about 30 min) so the friend's run does not meet a first-time bug:
`kaggle_submit_stage3.py data --smoke-only` and, after the data exists, `kaggle_submit_stage3.py train --run-mode smoke`
(the smoke receipt lands in the shared Hub repo, so the friend's training run skips its own smoke test).

`HF_TOKEN` must be a Hugging Face token with write access to the owner's dataset and checkpoint repos; the friend pastes it into their own Kaggle secret.

## Budget (estimates from past runs)
| Step | GPU h |
|---|---|
| Owner's smoke runs | about 0.7 |
| Data build (friend) | about 2.5–3 |
| Epoch 6 (friend) | about 10 |
| Gate eval | about 1 |
| Epoch 7 (+ gate) | about 11 |
| Optional epoch 8 (+ gate) | about 11 |

## Risks
- **Untested on a GPU until the smoke runs:** the MMS alignment step and the Hub upload run only on Kaggle. They are covered by unit tests with synthetic data and by the smoke runs, not by a local end-to-end run.
- **Hub storage:** the private storage cap. The v4 dataset adds about 5 GB; the UAT pool stays on Kaggle's disk and is not uploaded.
- **Voice drift** toward the UAT narrator: it has its own speaker tag; always prompt with a fixed tag.
- **Forgetting** of epoch-5 gains: the hard-word-heavy UrduTTS clips are kept and the gate re-checks the old 150 prompts.
- **Session length:** about 10 h expected against 11.75 h; the notebook budgets the remaining time for the full run and saves before the limit.
