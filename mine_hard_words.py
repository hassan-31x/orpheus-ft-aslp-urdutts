#!/usr/bin/env python3
"""Find the words a checkpoint mispronounces and turn them into training loss weights (Kaggle GPU).

  python mine_hard_words.py --name v1 \
      --adapter epoch4=hassan-31x/orpheus-urdu-stage2-checkpoints:stage2-r32-lr3e5-seed3409-epoch4:checkpoint-3185

Stages (each resumable; state is mirrored to <repo>/mining/<name>/ so another session continues it):
  1 select    UrduTTS training sentences to probe (2-12 s), from the same pinned data as epochs 3-4
  2 generate  the checkpoint speaks every probe sentence with --seeds different seeds (batched)
  3 asr       Whisper large-v3 on the synthetic audio, the REAL probe recordings and every real ASLP clip
  4 score     a (word, seed) trial counts only when Whisper heard the word right in the real recording;
              per-word model error rate -> multiplier; ASLP clip CER -> transcript reliability
  5 align     MMS-1b-all CTC forced alignment of every UrduTTS training clip that contains a hard word
              -> time spans of the hard words
  6 publish   loss_weights.json (used by finetune_mixed.py via loss_weighting) + eval_prompts.csv
Models run one at a time and are removed from the Hub cache afterwards (Kaggle /kaggle/working is 20 GB).
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import shutil
import sys
import time
from pathlib import Path

from orpheus_utils import CODEC_ID, SPECIAL, atomic_json, decode_frames, prompt_ids, read_json
from pron_words import ctc_align, normalize, score_words, word_spans, words

# The Xet token endpoint counts against the Hub's 1000 API requests / 5 min quota; with thousands of
# small clips every download needs one, so fetch through the plain resolve route instead.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

LOG = logging.getLogger("mine")
WHISPER = "openai/whisper-large-v3"
MMS = "facebook/mms-1b-all"
MMS_LANG = "urd-script_arabic"
AUDIO_TOKENS_PER_SECOND = 7 * 24000 / 2048


class State:
    """Local folder mirrored to the Hub so a later session (any account with the token) resumes."""

    def __init__(self, root, repo, name, token):
        from huggingface_hub import HfApi
        self.root, self.repo, self.prefix, self.token = Path(root), repo, f"mining/{name}", token
        self.api = HfApi(token=token)
        self.root.mkdir(parents=True, exist_ok=True)
        self.last_push = time.monotonic()

    def pull(self):
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import HfHubHTTPError
        names = [f for f in self.api.list_repo_files(self.repo) if f.startswith(self.prefix + "/")]
        if not names:
            return
        with_cache = self.root.parent / ".pull-cache"
        for attempt in range(1, 31):  # resumes: files already downloaded are skipped on each retry
            try:
                snapshot_download(self.repo, allow_patterns=[self.prefix + "/*", self.prefix + "/**"],
                                  local_dir=with_cache, token=self.token, max_workers=8)
                break
            except HfHubHTTPError as exc:
                if exc.response is None or exc.response.status_code != 429 or attempt == 30:
                    raise
                LOG.warning("Hub rate limit while resuming (attempt %d); waiting 5 minutes", attempt)
                time.sleep(300)
        source = with_cache / self.prefix
        for path in source.rglob("*"):
            if path.is_file():
                target = self.root / path.relative_to(source)
                if not target.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(path), target)
        shutil.rmtree(with_cache, ignore_errors=True)
        LOG.info("Resumed state from %s/%s", self.repo, self.prefix)

    def push(self, message, force=True):
        if not force and time.monotonic() - self.last_push < 1800:
            return
        self.api.upload_folder(repo_id=self.repo, folder_path=str(self.root), path_in_repo=self.prefix,
                               ignore_patterns=["*.tmp", "adapter/**", "data/**"], commit_message=message)
        self.last_push = time.monotonic()
        LOG.info("State pushed: %s", message)

    def done(self, stage):
        return (self.root / f"{stage}.done").is_file()

    def mark(self, stage, **info):
        atomic_json(self.root / f"{stage}.done", dict(finished=time.time(), **info))
        self.push(f"mining {self.prefix}: {stage} done")


def free_model(repo_id):
    """Delete a model from the Hub cache; disk is the binding constraint on Kaggle."""
    from huggingface_hub import scan_cache_dir
    try:
        cache = scan_cache_dir()
        hashes = [rev.commit_hash for repo in cache.repos if repo.repo_id == repo_id for rev in repo.revisions]
        if hashes:
            cache.delete_revisions(*hashes).execute()
            LOG.info("Removed %s from the Hub cache", repo_id)
    except Exception as exc:  # cache cleanup is best effort
        LOG.warning("Cache cleanup for %s failed (%s)", repo_id, type(exc).__name__)


def load_rows(repo_root, data_work):
    """Materialise the epoch 3-4 curated data (v3 labels) and return train/validation rows."""
    import soundfile as sf
    sys.path.insert(0, str(repo_root))
    import stage2_plan
    from stage2_data import prepare_dataset
    cfg = stage2_plan.epoch_config(4, stage2_plan.load_base(repo_root))
    root = prepare_dataset(cfg, data_work)
    rows = {}
    for split in ("train", "validation"):
        with (root / f"{split}.csv").open(encoding="utf-8") as f:
            rows[split] = [dict(r, id=Path(r["audio"]).stem, path=str(root / r["audio"])) for r in csv.DictReader(f)]
    report = read_json(root / "dataset_source.json")
    durations = {r["id"]: r["duration"] for split in ("train", "validation") for r in report["rows"][split]}
    for split in rows:
        for r in rows[split]:
            r["duration"] = durations.get(r["id"]) or sf.info(r["path"]).duration
    return rows


def select(rows, count, seed):
    pool = [r for r in rows["train"] if r["source"] == "urdutts" and 2.0 <= r["duration"] <= 12.0]
    random.Random(seed).shuffle(pool)
    return [dict(id=r["id"], text=r["text"], voice=r.get("voice", ""), duration=r["duration"]) for r in pool[:count]]


def generate(state, probes, adapter_spec, seeds, batch_size, deadline, cfg):
    """Batched sampling with the same decoding settings as every earlier evaluation."""
    from unsloth import FastLanguageModel  # must precede torch/transformers (Unsloth patches them)
    import torch
    import soundfile as sf
    from peft import PeftModel
    from snac import SNAC
    from eval_pronunciation import fetch_adapter
    pending = [(s, p) for s in range(seeds) for p in probes
               if not (state.root / f"synth/seed{s}/{p['id']}.flac").is_file()]
    if not pending:
        return True
    _, adapter = fetch_adapter("adapter=" + adapter_spec, state.root)
    tagged = bool(read_json(adapter / "run_identity.json")["config"].get("speaker_column"))
    # Load exactly as synthesize.py and training do: Unsloth resizes the embeddings to 156,940 rows, which
    # a plain Transformers load (156,939) cannot take the adapter's embedding/head weights into.
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["model_id"], revision=cfg["model_revision"], use_exact_model_name=True,
        max_seq_length=cfg["max_length"], dtype=None, load_in_4bit=cfg["precision"] == "4bit")
    model = PeftModel.from_pretrained(model, str(adapter), is_trainable=False)
    FastLanguageModel.for_inference(model)
    model.eval()
    codec = SNAC.from_pretrained(CODEC_ID, revision=cfg["codec_revision"]).eval().cuda()
    # Similar lengths share a batch, so short sentences do not wait for long ones.
    pending.sort(key=lambda item: (item[0], item[1]["duration"]))
    finished = True
    for b in range(0, len(pending), batch_size):
        if time.monotonic() > deadline:
            finished = False
            break
        batch = pending[b:b + batch_size]
        prefixes = [prompt_ids(tokenizer, p["text"], p["voice"] if tagged else None) for _, p in batch]
        width = max(map(len, prefixes))
        ids = torch.tensor([[SPECIAL["pad"]] * (width - len(x)) + x for x in prefixes], device="cuda")
        mask = torch.tensor([[0] * (width - len(x)) + [1] * len(x) for x in prefixes], device="cuda")
        # Token budget from the real clip length: a runaway sample cannot hold the whole batch for 1400 steps.
        budget = min(1400, cfg["max_length"] - width,
                     int(max(p["duration"] for _, p in batch) * AUDIO_TOKENS_PER_SECOND * 1.6) + 70)
        torch.manual_seed(1000 * batch[0][0] + b)
        with torch.inference_mode():
            out = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=budget, do_sample=True,
                                 temperature=0.6, top_p=0.95, repetition_penalty=1.1,
                                 eos_token_id=SPECIAL["end_speech"], pad_token_id=SPECIAL["pad"], use_cache=True)
        for row, (seed, probe) in zip(out[:, width:].tolist(), batch):
            target = state.root / f"synth/seed{seed}/{probe['id']}.flac"
            target.parent.mkdir(parents=True, exist_ok=True)
            meta = dict(seed=seed, budget=budget)
            try:
                codes, status = decode_frames(row)
                with torch.inference_mode():
                    audio = codec.decode([torch.tensor([c], device="cuda") for c in codes]).squeeze().float().cpu().numpy()
                meta.update(status, seconds=len(audio) / 24000)
            except ValueError as exc:  # no valid frame: keep a silent placeholder so the sentence scores as wrong
                audio = [0.0] * 2400
                meta.update(error=str(exc)[:200])
            sf.write(target.with_suffix(".tmp"), audio, 24000, format="FLAC")
            target.with_suffix(".tmp").replace(target)
            atomic_json(target.with_suffix(".json"), meta)
        LOG.info("Generated %d/%d", min(b + batch_size, len(pending)), len(pending))
        state.push("mining: generation progress", force=False)
    del model, codec
    torch.cuda.empty_cache()
    shutil.rmtree(adapter, ignore_errors=True)
    return finished


def transcribe(state, jobs, deadline, batch_size=16):
    """jobs: [(key, path)] -> appends {"key", "text"} lines to asr.jsonl (resumable)."""
    import numpy as np
    import soundfile as sf
    import torch
    from math import gcd
    from scipy.signal import resample_poly
    from transformers import pipeline
    out = state.root / "asr.jsonl"
    seen = set()
    if out.is_file():
        seen = {json.loads(line)["key"] for line in out.read_text(encoding="utf-8").splitlines() if line.strip()}
    jobs = [j for j in jobs if j[0] not in seen]
    if not jobs:
        return True
    asr = pipeline("automatic-speech-recognition", model=WHISPER, torch_dtype=torch.float16, device=0)
    finished = True
    for start in range(0, len(jobs), 64):
        if time.monotonic() > deadline:
            finished = False
            break
        chunk = jobs[start:start + 64]
        inputs = []
        for _, path in chunk:
            signal, sr = sf.read(path, dtype="float32")
            if signal.ndim > 1:
                signal = signal.mean(axis=1)
            d = gcd(sr, 16000)
            inputs.append(dict(raw=resample_poly(signal, 16000 // d, sr // d).astype(np.float32), sampling_rate=16000))
        results = asr(inputs, batch_size=batch_size, generate_kwargs=dict(language="urdu", task="transcribe"))
        with out.open("a", encoding="utf-8") as f:
            for (key, _), r in zip(chunk, results):
                f.write(json.dumps(dict(key=key, text=r["text"]), ensure_ascii=False) + "\n")
        LOG.info("Transcribed %d/%d", min(start + 64, len(jobs)), len(jobs))
        state.push("mining: ASR progress", force=False)
    del asr
    torch.cuda.empty_cache()
    return finished


def cer(reference, hypothesis):
    from eval_pronunciation import distance
    ref, hyp = normalize(reference), normalize(hypothesis)
    return distance(ref, hyp) / max(1, len(ref))


def score(state, probes, rows, seeds, options):
    asr = {}
    for line in (state.root / "asr.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            asr[r["key"]] = r["text"]
    items = [dict(reference=p["text"], real=asr[f"real/{p['id']}"],
                  synth=[asr[f"synth{s}/{p['id']}"] for s in range(seeds)]) for p in probes]
    stats = score_words(items, **options)
    hard = {w: s for w, s in stats.items() if s["weight"] > 1.0}
    aslp_cer = {r["id"]: round(cer(r["text"], asr[f"real/{r['id']}"]), 4)
                for r in rows["train"] if r["source"] == "aslp" and f"real/{r['id']}" in asr}
    synth_cer = [cer(p["text"], asr[f"synth{s}/{p['id']}"]) for p in probes for s in range(seeds)]
    real_cer = [cer(p["text"], asr[f"real/{p['id']}"]) for p in probes]
    eligible = sum(s["trials"] for s in stats.values())
    summary = dict(probe_sentences=len(probes), seeds=seeds, distinct_words=len(stats), hard_words=len(hard),
                   unknown_occurrences=sum(s["unknown"] for s in stats.values()),
                   unstable_occurrences=sum(s["unstable"] for s in stats.values()),
                   word_error_rate_on_eligible=sum(s["errors"] for s in stats.values()) / max(1, eligible),
                   mean_synth_cer=sum(synth_cer) / len(synth_cer), mean_real_cer=sum(real_cer) / len(real_cer),
                   aslp_clips_scored=len(aslp_cer), aslp_unreliable_over_0_6=sum(v > 0.6 for v in aslp_cer.values()))
    atomic_json(state.root / "score.json", dict(summary=summary, options=options, hard_words=hard,
                                                all_words=stats, aslp_cer=aslp_cer))
    LOG.info("Score summary: %s", summary)
    return summary


def align(state, rows, hard, deadline):
    """CTC forced alignment (MMS-1b-all, Urdu adapter) -> hard-word time spans per UrduTTS clip."""
    import numpy as np
    import soundfile as sf
    import torch
    from math import gcd
    from scipy.signal import resample_poly
    from transformers import AutoProcessor, Wav2Vec2ForCTC
    out = state.root / "align.jsonl"
    seen = set()
    if out.is_file():
        seen = {json.loads(line)["id"] for line in out.read_text(encoding="utf-8").splitlines() if line.strip()}
    todo = [r for r in rows["train"] if r["source"] == "urdutts" and r["id"] not in seen
            and any(w in hard for w in words(r["text"]))]
    if not todo:
        return True
    processor = AutoProcessor.from_pretrained(MMS, target_lang=MMS_LANG)
    model = Wav2Vec2ForCTC.from_pretrained(MMS, target_lang=MMS_LANG, ignore_mismatched_sizes=True,
                                           torch_dtype=torch.float16).cuda().eval()
    tok = processor.tokenizer
    unk, blank, sep = tok.unk_token_id, tok.pad_token_id, tok.convert_tokens_to_ids("|")
    finished = True
    with out.open("a", encoding="utf-8") as f:
        for n, r in enumerate(todo):
            if time.monotonic() > deadline:
                finished = False
                break
            ws = words(r["text"])
            tokens, token_word = [], []
            for i, w in enumerate(ws):
                ids = [t for t in (tok.convert_tokens_to_ids(ch) for ch in w) if t not in (None, unk)]
                if tokens and ids:
                    tokens.append(sep)
                    token_word.append(-1)
                tokens += ids
                token_word += [i] * len(ids)
            signal, sr = sf.read(r["path"], dtype="float32")
            d = gcd(sr, 16000)
            signal = resample_poly(signal, 16000 // d, sr // d).astype(np.float32)
            values = processor(signal, sampling_rate=16000, return_tensors="pt").input_values.half().cuda()
            with torch.inference_mode():
                log_probs = model(values).logits[0].float().log_softmax(-1).cpu().numpy()
            frames = ctc_align(log_probs, tokens, blank) if tokens else None
            spans = []
            if frames is not None:
                seconds = len(signal) / 16000 / len(log_probs)
                for w, span in zip(ws, word_spans(frames, token_word, seconds)):
                    if span and w in hard:
                        spans.append([round(span[0], 3), round(span[1], 3), round(hard[w]["weight"], 3)])
            f.write(json.dumps(dict(id=r["id"], aligned=frames is not None, spans=spans), ensure_ascii=False) + "\n")
            if n % 500 == 499:
                f.flush()
                LOG.info("Aligned %d/%d", n + 1, len(todo))
                state.push("mining: alignment progress", force=False)
    del model
    torch.cuda.empty_cache()
    return finished


def publish(state, rows, adapter_spec, seed):
    scored = read_json(state.root / "score.json")
    spans, failed = {}, 0
    for line in (state.root / "align.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            failed += not r["aligned"]
            if r["spans"]:
                spans[r["id"]] = r["spans"]
    hard = scored["hard_words"]
    atomic_json(state.root / "loss_weights.json", dict(
        version=1, adapter=adapter_spec, created=time.strftime("%Y-%m-%d %H:%M:%S"),
        summary=dict(scored["summary"], clips_with_spans=len(spans), alignment_failures=failed),
        options=scored["options"], hard_words=hard, aslp_cer=scored["aslp_cer"], spans=spans))
    # Held-out evaluation prompts: 100 random UrduTTS validation sentences + 50 that contain hard words.
    pool = [r for r in rows["validation"] if r["source"] == "urdutts" and 2.0 <= r["duration"] <= 10.0]
    random.Random(seed).shuffle(pool)
    general = pool[:100]
    rest = sorted(pool[100:], key=lambda r: -sum(hard[w]["weight"] - 1 for w in words(r["text"]) if w in hard))
    targeted = [r for r in rest if any(w in hard for w in words(r["text"]))][:50]
    with (state.root / "eval_prompts.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, ["text", "set"])
        w.writeheader()
        w.writerows([dict(text=r["text"], set="general") for r in general] +
                     [dict(text=r["text"], set="hard_words") for r in targeted])
    state.mark("publish", clips_with_spans=len(spans), eval_prompts=len(general) + len(targeted))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True)
    p.add_argument("--adapter", required=True, help="repo:run_id:checkpoint of the checkpoint to probe")
    p.add_argument("--repo", default="hassan-31x/orpheus-urdu-stage2-checkpoints", help="Where state is mirrored")
    p.add_argument("--work", type=Path, default=Path("/kaggle/working/mining"))
    p.add_argument("--data-work", type=Path, default=Path("/kaggle/working/orpheus-stage2"))
    p.add_argument("--sentences", type=int, default=1200)
    p.add_argument("--seeds", type=int, default=2)
    p.add_argument("--batch", type=int, default=12)
    p.add_argument("--hours", type=float, default=11.0, help="Stop cleanly (state saved) after this long")
    p.add_argument("--seed", type=int, default=3411)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    deadline = time.monotonic() + args.hours * 3600
    repo_root = Path(__file__).resolve().parent
    token = os.environ.get("HF_TOKEN")
    state = State(args.work / args.name, args.repo, args.name, token)
    state.pull()
    sys.path.insert(0, str(repo_root))
    import stage2_plan
    cfg = stage2_plan.epoch_config(4, stage2_plan.load_base(repo_root))
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    cfg["model_revision"] = api.model_info(cfg["model_id"], revision=cfg["model_revision"]).sha
    cfg["codec_revision"] = api.model_info(CODEC_ID, revision=cfg["codec_revision"]).sha
    rows = load_rows(repo_root, args.data_work)

    probes_path = state.root / "probes.json"
    if not probes_path.is_file():
        atomic_json(probes_path, select(rows, args.sentences, args.seed))
    probes = read_json(probes_path)
    if not state.done("generate"):
        if not generate(state, probes, args.adapter, args.seeds, args.batch, deadline, cfg):
            state.push("mining: paused in generation")
            return LOG.warning("Session budget reached during generation; rerun the same command to continue")
        free_model(cfg["model_id"])
        state.mark("generate", sentences=len(probes), seeds=args.seeds)
    if not state.done("asr"):
        jobs = [(f"synth{s}/{p['id']}", str(state.root / f"synth/seed{s}/{p['id']}.flac"))
                for s in range(args.seeds) for p in probes]
        by_id = {r["id"]: r for r in rows["train"]}
        jobs += [(f"real/{p['id']}", by_id[p["id"]]["path"]) for p in probes]
        jobs += [(f"real/{r['id']}", r["path"]) for r in rows["train"] if r["source"] == "aslp"]
        if not transcribe(state, jobs, deadline):
            state.push("mining: paused in ASR")
            return LOG.warning("Session budget reached during ASR; rerun the same command to continue")
        free_model(WHISPER)
        state.mark("asr", transcripts=len(jobs))
    options = dict(min_trials=2, min_error_rate=0.25, scale=2.0, cap=3.0)
    if not state.done("score"):
        score(state, probes, rows, args.seeds, options)
        state.mark("score")
    hard = read_json(state.root / "score.json")["hard_words"]
    if not state.done("align"):
        if not align(state, rows, hard, deadline):
            state.push("mining: paused in alignment")
            return LOG.warning("Session budget reached during alignment; rerun the same command to continue")
        free_model(MMS)
        state.mark("align")
    if not state.done("publish"):
        publish(state, rows, args.adapter, args.seed)
    info = state.api.list_repo_commits(args.repo)[0]
    print(json.dumps(dict(read_json(state.root / "loss_weights.json")["summary"],
                          loss_weights=f"{args.repo}:{state.prefix}/loss_weights.json",
                          commit=info.commit_id), indent=2))


if __name__ == "__main__":
    main()
