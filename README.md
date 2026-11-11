# Conversational Urdu adaptation: ASLP + UrduTTS

Separate stage-2 code and artifacts for the course project. The existing ASLP and UrduTTS repositories remain separate.

- Code: [hassan-31x/orpheus-ft-aslp-urdutts](https://github.com/hassan-31x/orpheus-ft-aslp-urdutts)
- Private model artifacts: [hassan-31x/orpheus-ft-aslp-urdutts](https://huggingface.co/hassan-31x/orpheus-ft-aslp-urdutts)
- Starting weights: the **completed** run `urdutts-r32-lr1e4-seed3407-epoch1` in [the original checkpoint repository](https://huggingface.co/hassan-31x/orpheus-urdutts-checkpoints).

At inspection on 2026-11-08, its latest verified snapshot was **checkpoint-3895**, `paused_for_resume`, epoch **0.7746426351771286**, with a scheduled total of **5029 steps**. This repository deliberately refuses that unfinished snapshot as its stage-2 parent. Finish the original epoch first. The new data mix is a hypothesis to evaluate; a perfect speaking model is not an assured outcome.

## Finish the preceding UrduTTS epoch

Import **[resume_urdutts_epoch1.ipynb](resume_urdutts_epoch1.ipynb)** into Kaggle. It is configured with:

```python
REPO_URL = "https://github.com/hassan-31x/orhpeus-ft-urdutts.git"
REPO_REF = "7ed00e155b4d668122dd5bfd41a0020a97e5b007"
RUN_ID_OVERRIDE = "urdutts-r32-lr1e4-seed3407-epoch1"
TEST = False
INIT_FROM_RUN = None
```

All six recorded training source hashes match that commit. The notebook restores the original optimizer, scheduler, random state and data position; it does not create a new epoch or reset the optimizer. It reinstalls the recorded project versions, including the data-conversion packages needed for identical token-cache fingerprints. It requires the original `torch==2.11.0+cu128` runtime and stops if the Kaggle image is incompatible. Enable Internet, a T4 GPU and the existing `HF_TOKEN` secret for that notebook. Original full checkpoints, not model weights alone, are required for exact continuation.

Before declaring it complete, confirm the original run's verified `latest.json` points to a snapshot with `status.json: complete`, final optimizer step at least its scheduled maximum, and a durable remote backup. Loose `monitoring/status.json` may lag behind the atomic snapshot; inspect snapshot status. Current small write/read probes succeeded on both Hub repositories, but that does not establish that a multi-GB checkpoint upload will succeed.

## Stage-2 data selection

Defaults in **[configs/mixed.json](configs/mixed.json)** target **60% ASLP / 40% UrduTTS by measured training audio seconds**. Total size follows the amount of qualifying conversational ASLP; there is no duplication to manufacture hours.

ASLP is pinned to `ASLP-lab/UrduSpeech@16dd380cfd9049a3db7f06a98e878086916bf833`. Only `corpus/US-Std` is eligible. The independent benchmark, code-switched subset, accented-English subset, prose, poetry, film, comedy, vlogs and reviews are excluded.

| Within the ASLP portion | Policy |
|---|---|
| Interview + podcast | At least 65% of measured ASLP hours; selected first |
| Roadside interview | At most 20% |
| Drama | At most 12% |
| News | At most 3% |

These are maxima, not quotas that force weak audio into training. Missing secondary categories reduce the total; core speech remains the majority. The default cap is three ASLP hours and 180 seconds per category-scoped speaker group. A full run requires at least six minutes of surviving ASLP; a small surviving dataset should prompt review of the reports and listening samples before interpreting model improvements.

The actual release has no transcript/paralinguistic JSONLs for **short interview** and **short news**, so those unpaired files are not used. Several roadside rows share the same `audio_id`, and some full `Audio_Clip` keys have conflicting records. Joins use the full clip filename; conflicting keys are excluded. This release inspection yielded 113 eligible interview and 549 eligible podcast metadata records before audio screening. Metadata eligibility is not final quality acceptance.

Selection layers:

1. Require Urdu-script text, confidence >=0.98, explicit Standard Pakistani Urdu accent, clear pronunciation metadata, 2–18 second clips and at most 210 text characters. Reject singing, recitation, monologue/narration and explicit speaker overlap or extreme delivery descriptions. Preserve transcript punctuation, repetitions and spoken fillers.
2. Decode the actual audio; verify measured duration against metadata. Reject clipping, invalid waveforms, extreme loudness and insufficient speech. Use Silero VAD to score speech occupancy without cutting training audio.
3. Require non-personalized **DNSMOS P.835** SIG >=3.4, BAK >=3.5 and OVRL >=3.1. These are screening proxies, not measured Urdu human MOS. The reference model and its SHA-256 are pinned. Short clips are repeated only inside the scorer, as in the reference implementation.
4. Cross-check the transcript with pinned multilingual Whisper large-v3, using Urdu transcription without previous-text conditioning. Require normalized CER <=0.15 and WER <=0.35. Keep the source transcript; do not replace it with the ASR output. The normalizer is for comparison only. Good speech can be rejected because Urdu ASR is imperfect.
5. Rank passing clips, enforce category/speaker budgets and create deterministic category+speaker holdouts. Remove exact waveform and normalized-text duplicates, giving validation priority. Check composition again after deduplication, SNAC filtering and GPU-memory exclusions.

UrduTTS is pinned to the same dataset commit as epoch 1. Only released `train` supplies replay; released `eval` supplies monitoring. `test` is untouched. The full workflow acoustically scans **all training shards**, retaining only a bounded highest-scoring shortlist on disk, then ASR-checks that shortlist. It selects the highest-ranked survivors up to the replay-hours budget. It does not claim to ASR-check every released training clip or to establish a globally optimal subset. The top shortlist is three times the required replay duration by default. The eval scan is bounded to avoid spending hours on monitoring preparation.

Each Arrow shard downloads into a temporary Hub cache before parsing, with up to five resumable transfer attempts. The same cache is kept between attempts; partial downloads never yield training rows. The shard cache is deleted after iteration or failure, so raw shards do not accumulate across the full release.

No denoising, time stretching, speech truncation, fabricated emotion tags or new speaker tokens are introduced. Audio is converted to 24 kHz mono PCM16. UrduTTS has no speaker labels; no global speaker-disjointness or gender balance is claimed. Its transcripts were themselves generated with Whisper, so agreement with another Whisper model is partly correlated evidence. ASLP paralinguistic descriptions do not constitute independently verified overlap detection.

## Stage-2 training and Kaggle

**Wait until the original epoch is finished.** Then import **[kaggle_run.ipynb](kaggle_run.ipynb)** with Internet, a T4 GPU and `HF_TOKEN` access to both private repos.

GPU setup preserves Kaggle's Torch/CUDA packages and explicitly reinstalls xFormers 0.0.35 from the matching official PyTorch CUDA index (cu126, cu128 or cu130). The PyPI wheel can lack CUDA operators. Supported Torch builds are 2.10–2.12; the causal-attention forward/backward check must pass on the actual GPU before training starts. See the [upstream installation instructions](https://github.com/facebookresearch/xformers#installing-xformers).

1. Keep `TEST=True`. The smoke run selects up to 20 training and 10 validation examples across both sources, uses a separate run ID and goes through the real quality, SNAC, training and remote-backup path. Screening and large-v3 downloads make this more expensive than a 20-row test of already-prepared data.
2. The notebook resolves the completed source to a fixed Hub commit/checkpoint. It verifies the snapshot pointer and checkpoint seal, imports adapter weights only and records their SHA-256. It refuses paused/smoke parents, incompatible base revisions/ranks, or corrupted weights.
3. A passing smoke run requires finite validation loss, final adapter, sealed full checkpoint and current remote backup. It publishes a receipt tied to code, source weights and training settings. Listen to generated speech separately; the receipt does not certify naturalness.
4. Set `TEST=False`. A matching receipt is required. The full run starts afresh from the completed **UrduTTS parent**, never from the smoke adapter. Pin `REPO_REF` to the printed commit for resumes.

Defaults retain the existing r32/alpha64 LoRA and Orpheus/SNAC token format. The learning rate is **2e-5**, with a fresh AdamW optimizer, 3% warmup and cosine decay for **one pass over the newly selected mixture**. This is not an optimizer continuation of the earlier UrduTTS schedule.

Training visits three approximately equal-duration sections targeting **50%, 60%, then 70% ASLP**. Every selected clip appears once; each source is shuffled deterministically inside that schedule. Trainer uses a sequential sampler over the recorded order so it cannot silently undo the curriculum. Replay is present throughout. This schedule and the ratio are engineering choices to ablate, not published optimal Urdu settings. Changing transformer layers or freezing selected projections is not necessary for this experiment.

ASR screening runs in a separate process and releases its GPU allocation before SNAC/Orpheus loading. Screening resumes locally by verified quality records and per-shard cursors. Final selected WAVs can be regenerated from pinned sources without selecting a different dataset. Full GPU token caches, checkpoint recovery, upload retries and verified post-encoding WAV reclamation are inherited from the working workflow. Source-adapter staging uses hardlinks on the same filesystem, and parent weights are excluded from destination checkpoint snapshots to avoid redundant storage.

Artifacts remain private under `runs/<run_id>/`. Important evidence includes `dataset_source.json` (selected rows and composition), `curriculum.json`, `init_source.json`, `token_statistics.json`, `effective_training_manifests.json`, metrics, samples, full optimizer checkpoints and `backup_status.json`. Local preparation additionally retains per-candidate quality records with rejection reasons. A saved snapshot restores the exact selected rows and manifest checksums; missing WAVs regenerate without reranking. An interruption during screening before a preparation report is uploaded can require repeated screening in a fresh runtime. SNAC caches restore remotely when identities match.

Interrupted stage 2: rerun the same commit/config/run ID to restore its own full checkpoint. A later completed mixed-data epoch can seed a new run through `INIT_FROM_RUN`; use a new run ID. Code/settings changes require a new experiment and smoke receipt.

For the November 9 interrupted Arrow download, `kaggle_retry.ipynb` pins training code to `816474a` and mounts the failed full notebook's outputs. Its embedded `restore_screening_cache.py` can copy compatible ASLP WAVs/conversion records and quality caches into the new preparation directory. It permits only the recorded transport-only source change; data revisions, quality code/thresholds, packages, split seed and subset settings must match. It records both recipes and imported-file counts, leaves an already selected dataset untouched, and never imports optimizer state. The full retry still requires a passing smoke receipt for `816474a`.

## Curated stage-2 epochs (`kaggle_stage2.ipynb`)

Epochs 2, 3, 4, ... train on the curated dataset `hassan-31x/urdu-orpheus-stage2-data`; `stage2_plan.py` defines each epoch's run ID, seed, learning rate and parent run. Launch with `python kaggle_submit_stage2.py smoke --epoch N`, then `full --epoch N` (default budget 11.75 h: Kaggle's 12 h limit minus a save margin; an unfinished epoch resumes on any account with the same command).

- **Epoch 2** used the v2 data (44.1 h, 30% ASLP) with plain-text prompts.
- **From epoch 3**, the pinned `labels/v3.jsonl` caps acted ASLP dialogue (drama, comedy, film) at 7 h, preferring single-turn clips, which leaves 40.3 h of training data (24% ASLP). It also prefixes every prompt with a voice tag `speaker | gender | style | emotion: text`:
  - **speaker:** `urdutts_narrator` for UrduTTS, or `aslp_<category>_<id>` (1,700 ASLP speakers).
  - **gender:** `male` / `female` / `unknown`, from the ASLP annotation.
  - **style:** `audiobook` or `conversational`.
  - **emotion:** one of `neutral happy excited sad angry fearful surprised serious`, mapped from ASLP's free-text emotion notes by keyword.

  For clean narration, prompt with `urdutts_narrator | unknown | audiobook | neutral`. To vary voice and emotion, swap in another speaker or emotion tag.
- `eval_pronunciation.py` generates the benchmark prompts from several adapters with identical seeds and scores Whisper large-v3 CER/WER per adapter.

## Evaluation and validation

Compare the completed UrduTTS parent and mixed adapter on identical unseen prompts/seeds. Use **[benchmark_prompts.csv](benchmark_prompts.csv)** for questions, casual responses, numbers and pronunciation contrasts. **[compare_adapters.py](compare_adapters.py)** generates paired samples sequentially. Source-specific validation losses are saved alongside aggregate loss. Blind native-listener ratings should assess intelligibility and conversational naturalness separately. ASR CER/WER are intelligibility proxies and must be calibrated on real Urdu references; loss and pitch variation alone do not establish naturalness.

See **[research/RESEARCH.md](research/RESEARCH.md)** for the paper review, evidence, limitations and proposed ablations. Local probes in `research/audio_screening_probe.json` cover 19 ASLP files plus 20 UrduTTS examples; they test conversion/VAD/DNSMOS with real data, not GPU training or ASR acceptance. The initial stricter DNSMOS settings rejected all ASLP probe files; current thresholds are provisional and still require listening and full selection diagnostics.

CPU regression tests:

```bash
python -m pip install -r requirements-cpu.txt
python -m unittest discover -s tests -q
```

GPU training, CUDA ASR execution, dependency installation on Kaggle and multi-GB uploads require the remote smoke run. They are not established by mocked CPU training-control tests.

## Attribution

[ASLP UrduSpeech](https://huggingface.co/datasets/ASLP-lab/UrduSpeech) is released under CC BY 4.0; cite Haq et al., arXiv:2605.17846. [UrduTTS](https://huggingface.co/datasets/kaab4321/UrduTTS) is CC BY-NC 4.0. Preserve both attributions and the noncommercial restriction for this course project. DNSMOS follows the Microsoft DNS Challenge license and attribution in `research/THIRD_PARTY.md`. The Bangla paper PDF remains in the original local reference folder and is not republished here.

## Optional Kaggle API submission

Install the Kaggle CLI/SDK separately (`pip install kaggle six`) and authenticate through its supported credential mechanism. `kaggle_submit.py draft` publishes the mixed notebook privately without running it. `kaggle_submit.py resume --existing-kernel OWNER/SLUG` submits a new version of the existing original notebook. In the tested API submission, the new version could not read the existing secret: open Edit → Add-ons → Secrets, enable HF_TOKEN, then Save Version → Save & Run All in Kaggle. New notebooks require enabling HF_TOKEN in the Secrets UI; the API does not grant secret access. `smoke` and `full` are for after the parent epoch completes. API submission is not evidence that a training process reached the first optimizer step.

## Preserve screening across session limits

The original full notebook only backed up the completed selection. Partial ASLP
quality records and UrduTTS shard cursors stayed on the Kaggle filesystem.
For the already-running primary session, leave it running and retain its saved
outputs. After it ends, `kaggle_screening_resume.ipynb` attaches those outputs,
imports only an exactly matching screening recipe, and publishes a recovery
snapshot before continuing. It refuses to start from scratch if the first import
is unavailable. Never run it concurrently with the source job under the same run ID.

The embedded `screening_recovery.py` supervisor leaves the smoke-tested training
code pinned to commit `816474a`. It saves atomic Hugging Face snapshots every ten
minutes and when the child exits, including quality records, completed-shard
cursors and shortlist audio. Interrupted snapshots cannot replace the last good
pointer; live shortlist checksums must match. A mid-shard interruption repeats
that shard, not completed shards. Previously rejected ASLP audio may be downloaded
again to identify its cached score, but the matching Whisper/DNSMOS result is reused.
Hard termination can lose work since the last successful backup. The currently
running older notebook cannot gain this supervisor in place: its saved Kaggle
outputs are required for the first migration. Output availability is checked on
resume, not assumed.

For later epochs, retain this supervisor and attach the preceding completed run's
outputs, use a new run ID and set `INIT_FROM_RUN` to the completed mixed run.
Keep the preparation recipe unchanged to reuse its selected dataset. The full-run
smoke gate still applies to training-configuration changes.

Before spending GPU quota on the first recovery, run
`kaggle_screening_rescue_cpu.ipynb` after the source session's outputs become
available. Its accelerator is disabled. It uploads the matching cache and restores
it again with archive and shortlist checksums, writing `recovery_verified.json`.
If it fails, do not launch another GPU session. It cannot access live files while
the older committed notebook is still running.
