# Evidence and decisions for conversational Urdu adaptation

Reviewed 2026-11-08. Primary papers, publisher dataset documentation, the supplied local IEEE PDF and reference implementations were inspected. Dataset/model facts are distinguished from experiment choices below. None of these papers establishes an optimal ASLP/UrduTTS ratio or guarantees a specific Urdu outcome.

## Relevant research

**Sutradhar et al., Human-like Bangla Text-to-Speech Synthesis using the Orpheus TTS Model Fine-tuning**, ICECTE 2026, [DOI](https://doi.org/10.1109/ICECTE69292.2026.11429426). The supplied local paper uses 1,891 clips, about 2 h 56 min, six male speakers, 24 kHz resampling and parameter-efficient Orpheus adaptation through Unsloth. Its 15 epochs and 7,095 steps concern a different data regime. The published CER/WER examples cover only two generated utterances, and the reported SNAC-based similarity is not a validated speaker metric. Retain the codec/adapter workflow; do not copy an unreported learning rate, repeat 15 epochs automatically or interpret a tiny training loss as naturalness evidence. The paper does not test mixed narration/conversation training. The copyrighted PDF is not included in this repository.

**Basher et al., BnTTS: Few-Shot Speaker Adaptation in Low-Resource Setting**, NAACL Findings 2025, [paper](https://aclanthology.org/2025.findings-naacl.279/), [full text](https://arxiv.org/html/2502.05729v1). BnTTS continues from XTTS, using a large Bengali corpus and separate high-quality speaker adaptation data. Its text evaluation separates phonetic diversity, named entities and short everyday utterances. This motivates testing Urdu pronunciation and conversational phrases separately rather than relying on one aggregate loss. Its architecture uses reference-audio speaker conditioning, so its partial/complete prompting stages cannot be copied directly into text-only Orpheus adaptation. Its 2e-5 training rate is a useful scale reference, not evidence that the same rate is optimal for our adapter and dataset.

**Yan et al., AdaSpeech 3: Adaptive Text to Speech for Spontaneous Style**, Interspeech 2021, [authors' publication](https://www.microsoft.com/en-us/research/publication/adaptive-text-to-speech-for-spontaneous-style/). This explicitly adapts a reading-style model to spontaneous speech and addresses filled pauses and variable rhythm. It supports the task framing: pronunciation and spontaneous delivery require different evidence. Preserve real pauses, repetitions and fillers rather than stripping them into formal written Urdu. The paper's dedicated pause predictor and rhythm experts belong to another architecture and are not transplanted into Orpheus. Category mixing alone cannot guarantee those capabilities.

**Ma et al., WenetSpeech4TTS**, Interspeech 2024, [paper](https://arxiv.org/html/2406.05763v2). The authors refine in-the-wild ASR audio for TTS using transcript, audio-quality and speaker-homogeneity checks, and report benefits from higher-quality subsets. This motivates staged screening and ranking instead of choosing a genre as a quality label. Their quality tiers use DNSMOS **P.808**; this implementation uses non-personalized **P.835 SIG/BAK/OVRL**. Those scores and thresholds are not interchangeable. Their denoising, source-timestamp boundary extension and speaker clustering are not reproduced here; the released Urdu clips do not supply enough trusted original-source timing to make boundary reconstruction safe.

**Kwon et al., Parameter-Efficient Fine-Tuning for Low-Resource Text-to-Speech via Cross-Lingual Continual Learning**, Interspeech 2025, [paper](https://www.isca-archive.org/interspeech_2025/kwon25_interspeech.pdf). The study investigates adapters for F5-TTS and the tradeoff between linguistic accuracy and speaker similarity. It supports evaluating those dimensions independently and retaining parameter-efficient adaptation under limited compute. Its conditioning/prompt adapters and DropPath experiments target a diffusion-based architecture; they are not justification for arbitrary layer freezing, replacing the current Orpheus token protocol or increasing LoRA rank during a warm start.

**UtterTune**, 2025, [paper](https://arxiv.org/html/2508.09767v1). This uses LoRA and language-specific pronunciation/prosody notation for Japanese CosyVoice2 adaptation. It illustrates that pronunciation control can require explicit linguistic supervision. It does not establish that adding IPA from eSpeak to a model already adapted to raw Urdu text will improve this stage. Retain the existing native-script prompt format and evaluate difficult Urdu contrasts instead of changing the input protocol mid-experiment.

## Primary data evidence

[ASLP-lab/UrduSpeech](https://huggingface.co/datasets/ASLP-lab/UrduSpeech), [paper](https://arxiv.org/html/2605.17846v1), commit `16dd380cfd9049a3db7f06a98e878086916bf833`: the release distinguishes Standard Pakistani Urdu from code-switched speech and Pakistani-accented English, and provides confidence and free-text paralinguistic metadata. These labels are not direct independent acoustic measurements. Inspection found missing short-interview/news annotation files, singing inside an interview category and conflicting roadside filenames. The implementation excludes missing/conflicting pairs and applies audio checks.

[UrduTTS](https://huggingface.co/datasets/kaab4321/UrduTTS), commit `a012374328f140d2647e9138e6ceae85d4ee9f2c`: stored as a saved DatasetDict with embedded WAV bytes. Preserve released train/eval/test roles. It has no speaker labels, four reported male speakers and Whisper-generated native-script transcriptions. It supplies useful read-speech replay, with no claim that all its clips are equally clean or correctly aligned. No training on the evaluation/test partitions is allowed.

Actual metadata eligibility before audio scoring:

| Category | Eligible records | Screening candidate cap |
|---|---:|---:|
| Interview | 113 | 2,000 |
| Podcast | 549 | 2,000 |
| Drama | 7,491 | 2,000 |
| News | 67 | 500 |
| Roadside interview | 532 | 1,200 |

The drama pool is much larger than conversational core speech. Without explicit caps, quality ranking alone could reproduce another acted/narrative bias. Missing files and later waveform/ASR failures further reduce usable hours. Metadata eligibility is not a trained-data count.

## Chosen experiment

The **60/40 audio-hour mixture**, **50/60/70% ASLP curriculum**, **2e-5 learning rate**, category budgets and minimum-hours floor are explicit engineering hypotheses. Replay is intended to reduce pronunciation forgetting while a conversational majority shifts delivery. Neither papers nor metadata prove the intended effect; the training run and listening comparison must test it.

Use the existing adapter dimensions and base/codec commits. Import completed UrduTTS weights, then reset optimizer/schedule because the new selected dataset and run have a different objective distribution. Resume interruptions within stage 2 using the stage-2 optimizer checkpoint. A deterministic, no-replacement sequence provides exact epoch coverage and reproducible skip-on-resume behavior. Do not freeze arbitrary transformer blocks or fabricate emotion controls.

A **19-file ASLP / 20-row UrduTTS acoustic probe** verified real decoding, resampling, VAD and DNSMOS inference. Initially proposed P.835 thresholds 3.5 SIG / 3.8 BAK / 3.4 OVRL rejected every ASLP probe. The current provisional thresholds are 3.4 / 3.5 / 3.1, followed by transcript checks and quality ranking. This small probe is not a calibration of Urdu MOS and does not establish full-corpus pass rates. Exact score records and current-gate indicators are in `audio_screening_probe.json`. Stronger thresholds remain a useful ablation if enough conversational material survives.

## Evaluate before claiming success

- Compare the completed parent and mixed model with identical seeds, decoding parameters and unseen prompt text. Include short conversational responses, questions, exclamations, numbers, named entities and aspirated/retroflex contrasts.
- Save ASLP and UrduTTS validation loss separately. Lower mixed loss may conceal regression on read speech. Validation in the same media genre still does not establish conversational quality on new topics.
- Measure generated-speech CER/WER using an Urdu ASR system, and run it on human reference recordings to expose recognizer error. Training-time Whisper agreement and evaluation with the same ASR family are not independent proof.
- Use blind native-listener ratings for pronunciation and conversational naturalness separately. Do not use DNSMOS, training loss, speaker cosine similarity, F0 variance or one good audio sample as a substitute for those ratings.
- Retain the untouched external ASLP benchmark and UrduTTS test partition for future controlled evaluation. ASLP category-scoped speaker holdouts and UrduTTS utterance splits do not prove global speaker/source disjointness.

Budget permitting, compare the same selected core with 50/50, 60/40 and 70/30 replay, fixed mixing versus curriculum, and a 1e-5 versus 2e-5 rate. Change one factor at a time under separate run IDs. If insufficient clean core speech survives, obtain corrected conversational data rather than silently relaxing gates or allowing drama/news to dominate.
