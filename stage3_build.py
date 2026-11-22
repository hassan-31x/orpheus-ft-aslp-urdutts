#!/usr/bin/env python3
"""Stage-3 data build (dataset v4): add the UAT read-speech corpus, rebalance, publish a pinned private dataset.

  python stage3_build.py --phase all --work /kaggle/working/stage3 --repo hassan-31x/urdu-orpheus-stage3-data

Phases (each can be re-run; the UAT pool lives in --work, the old ASLP/UrduTTS pools are read from the Hub):
  uat     mahwizzzz/UAT -> 24 kHz mono FLAC + raw metrics (DNSMOS, VAD, loudness), no thresholds    [CPU workers]
  align   MMS-1b-all Urdu CTC on every UAT clip: how well the transcript matches the audio          [GPU]
  select  apply the v4 rules, write train/validation parquet, labels/v4.jsonl and eval prompts,
          upload them, then publish the dataset pointer that stage2_plan.py reads for epochs 6+

Why UAT needs the alignment screen: its transcripts look partly phonetic or machine-made (e.g. سادت for سعادت),
and a clip whose text does not match its audio teaches a wrong spelling-to-sound mapping.
The ASLP and UrduTTS pools were already measured by build_stage2.py; they are only re-selected here.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import io
import json
import logging
import math
import os
import random
import re
import shutil
import sys
import tempfile
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import build_stage2 as b2
import stage2_labels

LOG = logging.getLogger("stage3")
UAT_ID, UAT_REV = "mahwizzzz/UAT", "da45a8fce76781c4b8e9609f256d737d5dc5f91d"
POOL_REPO, POOL_REV = "hassan-31x/urdu-orpheus-stage2-data", "feda4a51ead6a93511759aa7b7ff8d8f7833173c"
NEW_REPO = "hassan-31x/urdu-orpheus-stage3-data"
CKPT_REPO = "hassan-31x/orpheus-urdu-stage2-checkpoints"
MINING_PATH = "mining/v1/loss_weights.json"
MINING_REV = "f3e9fee0f656ee9b66fd03b34481aefb6f7c94e6"
POINTER_PATH = "datasets/stage3-v4.json"
MMS, MMS_LANG = "facebook/mms-1b-all", "urd-script_arabic"
UAT_VOICE = "uat_narrator | unknown | audiobook | neutral"
ALIGN_COLUMNS = ("id", "align_nll", "align_cer", "unk_fraction", "asr_text", "align_ok")

V4 = dict(
    # An epoch of the 44 h v2 mixture took ~8.5 h plus ~1 h of first-time encoding; 40 h leaves a safe margin in one 12 h session.
    target_train_hours=40.0, minimum_train_hours=36.0, maximum_train_hours=44.0,
    # ASLP: same gates, holdout and DNSMOS tier as the stage-2 build, but a larger per-speaker cap and no acted-speech
    # cap (v3 capped drama/comedy/film at 7 h; v4 wants every conversational hour it can get).
    aslp_speaker_max_minutes=45.0,
    min_urdutts_hours=10.0, urdutts_hard_fraction=0.75,
    # UAT: drop the worst-aligned clips, plus anything whose greedy CTC transcript is far from the text.
    uat_min_seconds=1.5, uat_max_seconds=20.0, uat_dnsmos=dict(sig=2.8, bak=2.6, ovrl=2.4),
    uat_align_drop_fraction=0.15, uat_max_cer=0.5, uat_max_unk_fraction=0.2,
    uat_validation_clips=250, uat_eval_prompts=150, uat_validation_max_cer=0.2,
    seed=3407,
)
LATIN = re.compile(r"[A-Za-z]")


# ---------------------------------------------------------------- small helpers
def clean_uat_text(text):
    return " ".join(unicodedata.normalize("NFC", text or "").split())


class PoolSource:
    """Read-only view of a pinned Hub dataset revision with the Hub interface existing_parts() expects."""
    enabled = True

    def __init__(self, repo, revision, token, max_parts=0):
        from huggingface_hub import HfApi
        self.repo, self.revision, self.token, self.max_parts = repo, revision, token, max_parts
        self.api = HfApi(token=token)

    def names(self, prefix=""):
        found = sorted(f for f in self.api.list_repo_files(self.repo, repo_type="dataset", revision=self.revision)
                       if f.startswith(prefix))
        return found[:self.max_parts] if self.max_parts else found

    def get(self, remote, local_dir):
        from huggingface_hub import hf_hub_download
        return hf_hub_download(self.repo, remote, repo_type="dataset", revision=self.revision, token=self.token,
                               local_dir=local_dir)


def quantile(values, q):
    values = sorted(values)
    if not values:
        return math.inf
    return values[min(len(values) - 1, max(0, math.ceil(q * len(values)) - 1))]


# ---------------------------------------------------------------- phase 1: UAT ingest
def uat_worker(args):
    """Convert one UAT clip to 24 kHz mono and measure it. Never applies a threshold."""
    row, model_dir, scratch, shard, index = args
    from urdutts_data import convert_audio
    stem = Path(row["name"]).stem.lower()
    out = dict(id=f"uat-{stem}", source="uat", pool_split="train", category="narration", speaker="uat_narrator",
               text=clean_uat_text(row["text"]),
               provenance=json.dumps(dict(dataset=UAT_ID, revision=UAT_REV, shard=shard, row=index,
                                          file_name=row["name"], source_text=row["text"]), ensure_ascii=False))
    wav = Path(scratch) / f"{out['id']}.wav"
    try:
        conv = convert_audio(dict(bytes=row["bytes"]), wav)
        out["duration"] = out["metadata_duration"] = conv["output_frames"] / 24000
        if not V4["uat_min_seconds"] <= out["duration"] <= V4["uat_max_seconds"]:
            return out  # rejected by duration in select; no need to spend ~1 s of DNSMOS/VAD on it
        out["wav_sha256"] = b2.sha256_bytes(wav.read_bytes())
        metrics, _ = b2.scorer(model_dir).acoustic(wav)
        for k in ("dnsmos_sig", "dnsmos_bak", "dnsmos_ovrl", "vad_fraction", "rms_db", "peak", "clipped_fraction"):
            out[k] = metrics.get(k)
        out["flac"] = b2.to_flac(wav)
    except Exception as exc:
        out["error"] = "convert_or_score:" + type(exc).__name__ + ":" + str(exc)[:120]
    finally:
        wav.unlink(missing_ok=True)
    return out


def uat_shard_names(token):
    from huggingface_hub import HfApi
    names = sorted(f for f in HfApi(token=token).list_repo_files(UAT_ID, repo_type="dataset", revision=UAT_REV)
                   if f.startswith("data/train-") and f.endswith(".parquet"))
    if not names:
        raise RuntimeError("No UAT shards found at the pinned revision")
    return names


def uat_rows(path, limit=0):
    import pyarrow.parquet as pq
    n = 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=64):
        for r in batch.to_pylist():
            yield n, dict(name=r["file_name"]["path"], bytes=r["file_name"]["bytes"], text=r["text"])
            n += 1
            if limit and n >= limit:
                return


def phase_uat(a, hub):
    from huggingface_hub import hf_hub_download
    work = Path(a.work)
    done_shards = set()
    progress = work / "pool/uat-progress.json"
    if progress.is_file():
        done_shards = set(json.loads(progress.read_text())["shards_done"])
    shards = uat_shard_names(a.token)[: (1 if a.limit else None)]
    LOG.info("UAT: %d shards, %d already done", len(shards), len(done_shards))
    existing = sorted((work / "pool/uat").glob("part-*.parquet")) if (work / "pool/uat").is_dir() else []
    writer = b2.PartWriter(hub, work, "pool/uat", start_index=len(existing), every_rows=10 ** 9)
    b2.scorer(str(work / "dnsmos"))  # download + checksum the DNSMOS model once, before forking workers
    b2._SCORER = None
    scratch = tempfile.mkdtemp(prefix="uat-", dir=work)
    stats, started = Counter(), time.time()

    def fetch(name):
        cache = tempfile.mkdtemp(prefix="uat-shard-", dir=work)
        for attempt in range(5):
            try:
                return cache, hf_hub_download(UAT_ID, name, repo_type="dataset", revision=UAT_REV, cache_dir=cache,
                                              token=a.token or None)
            except Exception as exc:
                LOG.warning("UAT shard %s attempt %d: %s", name, attempt + 1, type(exc).__name__)
                time.sleep(10 * 2 ** attempt)
        raise RuntimeError(f"could not download UAT shard {name}")

    todo = [s for s in shards if s not in done_shards]
    with cf.ThreadPoolExecutor(1) as downloader, cf.ProcessPoolExecutor(a.workers) as pool:
        future = downloader.submit(fetch, todo[0]) if todo else None
        for i, name in enumerate(todo):
            cache, path = future.result()
            future = downloader.submit(fetch, todo[i + 1]) if i + 1 < len(todo) else None
            jobs = [(r, str(work / "dnsmos"), scratch, name, n) for n, r in uat_rows(path, a.limit)]
            for out in pool.map(uat_worker, jobs, chunksize=8):
                stats["error" if out.get("error") else "ok"] += 1
                stats["hours"] += (out.get("duration") or 0) / 3600
                writer.buffer.append(out)
            done_shards.add(name)
            writer.flush(progress=dict(phase="uat", shards_done=sorted(done_shards), **stats))
            shutil.rmtree(cache, ignore_errors=True)
            LOG.info("UAT shard %d/%d done: %s  %.1f min elapsed", i + 1, len(todo), dict(stats),
                     (time.time() - started) / 60)
    shutil.rmtree(scratch, ignore_errors=True)
    LOG.info("UAT phase complete: %s", dict(stats))


# ---------------------------------------------------------------- phase 2: text/audio agreement (GPU)
def ctc_scores(log_probs, token_ids, blank, id_to_char):
    """Per-character CTC negative log-likelihood of the transcript, and the greedy-decode CER against it.

    log_probs: (frames, vocab) numpy array. Returns (nll_per_char, greedy_text_chars) with nll inf when the
    transcript cannot be aligned (more tokens than frames).
    """
    import numpy as np
    import torch
    frames = log_probs.shape[0]
    nll = math.inf
    if token_ids and frames >= len(token_ids):
        loss = torch.nn.functional.ctc_loss(
            torch.from_numpy(np.ascontiguousarray(log_probs, dtype="float32"))[:, None, :],
            torch.tensor([token_ids], dtype=torch.long), torch.tensor([frames]), torch.tensor([len(token_ids)]),
            blank=blank, reduction="sum", zero_infinity=False)
        nll = float(loss) / len(token_ids) if torch.isfinite(loss) else math.inf
    best = log_probs.argmax(axis=-1).tolist()
    chars, previous = [], None
    for t in best:
        if t != previous and t != blank:
            chars.append(id_to_char.get(t, ""))
        previous = t
    return nll, "".join(chars)


def phase_align(a, hub):
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    import soundfile as sf
    import torch
    from math import gcd
    from scipy.signal import resample_poly
    from transformers import AutoProcessor, Wav2Vec2ForCTC
    from pron_words import normalize
    from quality import edit_distance
    work = Path(a.work)
    parts = sorted((work / "pool/uat").glob("part-*.parquet"))
    if not parts:
        raise RuntimeError("No UAT pool parts; run --phase uat first")
    processor = AutoProcessor.from_pretrained(MMS, target_lang=MMS_LANG)
    model = Wav2Vec2ForCTC.from_pretrained(MMS, target_lang=MMS_LANG, ignore_mismatched_sizes=True,
                                           torch_dtype=torch.float16).cuda().eval()
    model.load_adapter(MMS_LANG)  # the Urdu head: from_pretrained only warns about the 154 -> 127 size change
    tok = processor.tokenizer
    unk, blank, sep = tok.unk_token_id, tok.pad_token_id, tok.convert_tokens_to_ids("|")
    id_to_char = {i: ("" if t.startswith("<") else (" " if t == "|" else t)) for t, i in tok.get_vocab().items()}
    out_dir = work / "pool/uat-align"
    out_dir.mkdir(parents=True, exist_ok=True)
    started, total = time.time(), 0
    for part in parts:
        target = out_dir / part.name
        if target.is_file():
            continue
        rows = []
        for r in pq.read_table(part, columns=["id", "text", "flac", "error"]).to_pylist():
            if r.get("error") or not r.get("flac"):
                continue
            text = normalize(r["text"])
            tokens, dropped = [], 0
            for i, w in enumerate(text.split()):
                ids = [t for t in (tok.convert_tokens_to_ids(ch) for ch in w) if t not in (None, unk)]
                dropped += len(w) - len(ids)
                if tokens and ids:
                    tokens.append(sep)
                tokens += ids
            unk_fraction = dropped / max(1, len(text.replace(" ", "")))
            signal, sr = sf.read(io.BytesIO(r["flac"]), dtype="float32")
            d = gcd(sr, 16000)
            signal = resample_poly(signal, 16000 // d, sr // d).astype(np.float32) if sr != 16000 else signal
            values = processor(signal, sampling_rate=16000, return_tensors="pt").input_values.half().cuda()
            with torch.inference_mode():
                logits = model(values).logits
            if logits.shape[-1] != len(tok):
                raise RuntimeError(f"MMS head has {logits.shape[-1]} outputs, tokenizer {len(tok)}: adapter not loaded")
            log_probs = logits[0].float().log_softmax(-1).cpu().numpy()
            nll, greedy = ctc_scores(log_probs, tokens, blank, id_to_char)
            hyp = normalize(greedy)
            ref = text.replace(" ", "")
            cer = edit_distance(ref, hyp.replace(" ", "")) / max(1, len(ref))
            rows.append(dict(id=r["id"], align_nll=nll if math.isfinite(nll) else 1e9, align_cer=cer,
                             unk_fraction=unk_fraction, asr_text=hyp, align_ok=bool(tokens) and math.isfinite(nll)))
        pq.write_table(pa.table({c: [r[c] for r in rows] for c in ALIGN_COLUMNS}), target, compression="zstd")
        total += len(rows)
        LOG.info("aligned %s (%d clips, %d total, %.1f min)", part.name, len(rows), total, (time.time() - started) / 60)
    del model
    torch.cuda.empty_cache()


# ---------------------------------------------------------------- phase 3: selection (pure functions)
def uat_gate(r, rules):
    """Reason a UAT clip fails the non-alignment rules, else None."""
    if r.get("error"):
        return "error"
    d = r["duration"]
    if not rules["uat_min_seconds"] <= d <= rules["uat_max_seconds"]:
        return "duration"
    if not r["text"] or LATIN.search(r["text"]):
        return "latin_or_empty_text"
    cps = len(r["text"].replace(" ", "")) / d
    if not b2.RULES["min_chars_per_second"] <= cps <= b2.RULES["max_chars_per_second"]:
        return "speaking_rate_vs_transcript"
    if r["clipped_fraction"] > b2.RULES["max_clipped_fraction"]:
        return "clipping"
    if not b2.RULES["min_rms_db"] <= r["rms_db"] <= b2.RULES["max_rms_db"]:
        return "loudness"
    if r.get("vad_fraction") is not None and r["vad_fraction"] < b2.RULES["min_vad_fraction"]:
        return "insufficient_speech"
    if b2.estimated_tokens(r) > b2.RULES["max_sequence_tokens"]:
        return "sequence_length"
    if not b2.dns_ok(r, rules["uat_dnsmos"]):
        return "dnsmos_below_tier"
    return None


def choose_uat(uat, align, rules, rng, banned_texts):
    """-> (train, validation, report). `align` maps clip id -> alignment row; clips without one are rejected."""
    reasons, gated = Counter(), []
    for r in sorted(uat, key=lambda r: r["id"]):
        why = uat_gate(r, rules)
        a = align.get(r["id"])
        if not why and a is None:
            why = "no_alignment"
        elif not why and (not a["align_ok"] or a["unk_fraction"] > rules["uat_max_unk_fraction"]):
            why = "unalignable_text"
        elif not why and b2.norm_text(r["text"]) in banned_texts:
            why = "text_in_other_source"
        if why:
            reasons[why] += 1
        else:
            gated.append(dict(r, **{k: a[k] for k in ("align_nll", "align_cer")}))
    seen_audio, seen_text, unique = set(), set(), []
    for r in sorted(gated, key=lambda r: (r["align_nll"], r["id"])):  # the best-aligned copy of a duplicate wins
        text = b2.norm_text(r["text"])
        if r["wav_sha256"] in seen_audio or text in seen_text:
            reasons["duplicate"] += 1
            continue
        seen_audio.add(r["wav_sha256"])
        seen_text.add(text)
        unique.append(r)
    cut = quantile([r["align_nll"] for r in unique], 1.0 - rules["uat_align_drop_fraction"])
    kept = []
    for r in unique:
        if r["align_nll"] > cut or r["align_cer"] > rules["uat_max_cer"]:
            reasons["text_audio_mismatch"] += 1
        else:
            kept.append(r)
    # Validation: clips whose text matches the audio very well, so held-out loss and eval prompts are clean.
    median = quantile([r["align_nll"] for r in kept], 0.5)
    clean = sorted((r for r in kept if r["align_nll"] <= median and r["align_cer"] <= rules["uat_validation_max_cer"]),
                   key=lambda r: r["id"])
    rng.shuffle(clean)
    val = clean[: rules["uat_validation_clips"]]
    val_ids = {r["id"] for r in val}
    train = [r for r in kept if r["id"] not in val_ids]
    report = dict(pool=len(uat), rejected=dict(reasons), align_nll_cut=round(cut, 4) if math.isfinite(cut) else None,
                  train_clips=len(train), train_hours=round(b2.hours(train), 3), validation_clips=len(val))
    return train, val, report


def choose_aslp_v4(aslp, rng, rules):
    saved = b2.RULES["speaker_max_minutes"]
    b2.RULES["speaker_max_minutes"] = rules["aslp_speaker_max_minutes"]
    try:
        return b2.choose_aslp(aslp, rng)
    finally:
        b2.RULES["speaker_max_minutes"] = saved


def hard_priority(spans):
    """Sum of (weight - 1) over the clip's hard-word spans: how much pronunciation signal the clip carries."""
    return sum(max(0.0, s[2] - 1.0) for s in spans or [])


def pick_urdutts(candidates, spans, hours_wanted, rules):
    """candidates: UrduTTS rows sorted best-DNSMOS first. Mostly clips with hard words, the rest cleanest-first."""
    hard = sorted((r for r in candidates if spans.get(r["id"])), key=lambda r: (-hard_priority(spans[r["id"]]), r["id"]))
    easy = [r for r in candidates if not spans.get(r["id"])]
    picked, seconds = [], 0.0
    for r in hard:
        if seconds >= rules["urdutts_hard_fraction"] * hours_wanted * 3600:
            break
        picked.append(r)
        seconds += r["duration"]
    for r in easy:
        if seconds >= hours_wanted * 3600:
            break
        picked.append(r)
        seconds += r["duration"]
    for r in hard[len(picked):]:  # not enough easy clips: top up with the remaining hard ones
        if seconds >= hours_wanted * 3600:
            break
        if r not in picked:
            picked.append(r)
            seconds += r["duration"]
    return picked


def select_v4(aslp, tts, uat, align, spans, rules=V4):
    """Pure selection over pool metadata rows. Returns (train, validation, report)."""
    rng = random.Random(b2.RULES["seed"])
    aslp_train, aslp_val, tier, aslp_reasons, holdout = choose_aslp_v4(aslp, rng, rules)
    tts_val_pool = sorted((r for r in tts if not r.get("error") and r["pool_split"] == "validation"),
                          key=lambda r: r["id"])
    rng.shuffle(tts_val_pool)  # same call order as the stage-2 build, so the validation clips are identical
    tts_val = tts_val_pool[: b2.RULES["urdutts_validation_clips"]]
    banned = {b2.norm_text(r["text"]) for r in aslp_train + aslp_val + tts_val}
    uat_train, uat_val, uat_report = choose_uat(uat, align, rules, random.Random(rules["seed"]), banned)
    banned |= {b2.norm_text(r["text"]) for r in uat_train + uat_val}
    tts_ok = [r for r in tts if not r.get("error") and r["pool_split"] == "train"
              and b2.estimated_tokens(r) <= b2.RULES["max_sequence_tokens"] and b2.norm_text(r["text"]) not in banned]
    tts_ok.sort(key=lambda r: -(r["dnsmos_ovrl"] + 0.1 * r["dnsmos_bak"]))
    seen, tts_unique = set(), []
    for r in tts_ok:
        if r["wav_sha256"] not in seen:
            seen.add(r["wav_sha256"])
            tts_unique.append(r)
    fixed = b2.hours(aslp_train) + b2.hours(uat_train)
    fill = min(max(rules["target_train_hours"] - fixed, rules["min_urdutts_hours"]),
               rules["maximum_train_hours"] - fixed)
    tts_train = pick_urdutts(tts_unique, spans, max(0.0, fill), rules)
    train = aslp_train + uat_train + tts_train
    val = aslp_val + uat_val + tts_val
    assert not ({r["id"] for r in train} & {r["id"] for r in val}), "train/validation overlap"
    assert not ({r["speaker"] for r in aslp_train} & {r["speaker"] for r in aslp_val}), "ASLP speaker leak"
    train_h = b2.hours(train)
    hard_clips = sum(1 for r in tts_train if spans.get(r["id"]))
    report = dict(
        status="ok" if rules["minimum_train_hours"] <= train_h <= rules["maximum_train_hours"] else "out_of_range",
        rules=rules, chosen_aslp_dnsmos_tier=tier,
        train=dict(clips=len(train), hours=round(train_h, 3), aslp_hours=round(b2.hours(aslp_train), 3),
                   uat_hours=round(b2.hours(uat_train), 3), urdutts_hours=round(b2.hours(tts_train), 3),
                   conversational_fraction=round(b2.hours(aslp_train) / train_h, 3) if train_h else 0,
                   urdutts_clips_with_hard_words=hard_clips, composition=b2.summarize(train)),
        validation=dict(clips=len(val), hours=round(b2.hours(val), 3), composition=b2.summarize(val),
                        aslp_heldout_speakers=len(holdout)),
        uat=uat_report, aslp_rejections=dict(aslp_reasons))
    return train, val, report


def verify_final(out):
    """Fail loudly before anything is uploaded if the dataset the trainer will read is inconsistent."""
    import io
    import pyarrow.parquet as pq
    import soundfile as sf
    out = Path(out)
    labels = {}
    for line in (out / "labels/v4.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        labels[r["id"]] = r
    seen, hours = {}, Counter()
    for split in ("train", "validation"):
        shards = sorted((out / split).glob("*.parquet"))
        if not shards:
            raise RuntimeError(f"No {split} shards were written")
        for path in shards:
            for batch in pq.ParquetFile(path).iter_batches(batch_size=64):
                for r in batch.to_pylist():
                    audio = (r.get("audio") or {}).get("bytes")
                    if not audio or r.get("error"):
                        raise RuntimeError(f"{r.get('id')} has no usable audio")
                    info = sf.info(io.BytesIO(audio))
                    if info.samplerate != 24000 or info.channels != 1 or abs(info.duration - r["duration"]) > 0.05:
                        raise RuntimeError(f"{r['id']}: audio is {info.samplerate} Hz x{info.channels}, "
                                           f"{info.duration:.2f}s vs manifest {r['duration']:.2f}s")
                    if r["id"] in seen:
                        raise RuntimeError(f"{r['id']} appears twice")
                    if labels.get(r["id"], {}).get("split") != split or not labels[r["id"]]["voice"].strip():
                        raise RuntimeError(f"{r['id']} has no matching label")
                    if not r["text"].strip():
                        raise RuntimeError(f"{r['id']} has empty text")
                    seen[r["id"]] = split
                    hours[(split, r["source"])] += r["duration"] / 3600
    for split in ("train", "validation"):
        listed = {json.loads(l)["id"] for l in (out / f"manifests/{split}.jsonl").read_text(encoding="utf-8").splitlines()}
        if listed != {i for i, sp in seen.items() if sp == split}:
            raise RuntimeError(f"{split} manifest and shards disagree")
    if set(labels) != set(seen):
        raise RuntimeError("labels and shards disagree")
    return {f"{k[0]}/{k[1]}": round(v, 2) for k, v in sorted(hours.items())}


def make_labels(train, val):
    """v4 labels: same voice tags as v3 for ASLP/UrduTTS, one new narrator tag for UAT. Everything is kept."""
    out = []
    for split, rows in (("train", train), ("validation", val)):
        for r in sorted(rows, key=lambda r: r["id"]):
            voice = UAT_VOICE if r["source"] == "uat" else stage2_labels.voice(r)
            out.append(dict(id=r["id"], split=split, keep=True, voice=voice))
    return out


def conversational_voices(train, minimum_minutes=5):
    """Full voice tags of non-acted ASLP speakers with enough neutral audio: candidates for the default eval voice."""
    seconds = Counter()
    for r in train:
        if r["source"] == "aslp" and r["category"] not in ("drama", "comedy_show", "film"):
            tag = stage2_labels.voice(r)
            if tag.endswith("| neutral"):
                seconds[tag] += r["duration"]
    return [dict(voice=k, minutes=round(v / 60, 1)) for k, v in seconds.most_common(8) if v >= minimum_minutes * 60]


# ---------------------------------------------------------------- phase 3: IO
def read_align(work):
    import pyarrow.parquet as pq
    rows = {}
    for p in sorted((Path(work) / "pool/uat-align").glob("part-*.parquet")):
        for r in pq.read_table(p).to_pylist():
            rows[r["id"]] = r
    return rows


def load_spans(token, mining=None):
    if mining is None:
        from huggingface_hub import hf_hub_download
        mining = json.loads(Path(hf_hub_download(CKPT_REPO, MINING_PATH, revision=MINING_REV, token=token)).read_text())
    return mining.get("spans", {})


def phase_select(a, hub):
    import pyarrow as pa
    import pyarrow.parquet as pq
    work = Path(a.work)
    meta_cols = [c for c in b2.POOL_COLUMNS if c != "flac"]
    old = PoolSource(POOL_REPO, POOL_REV, a.token, max_parts=a.pool_parts)
    old_work = work / "oldpool"

    def unique_by_id(rows):
        seen, out = set(), []
        for r in rows:
            if r["id"] not in seen:
                seen.add(r["id"])
                out.append(r)
        return out
    aslp = unique_by_id(b2.read_parts(b2.existing_parts(old, old_work, "pool/aslp"), columns=meta_cols))
    tts = unique_by_id(b2.read_parts(b2.existing_parts(old, old_work, "pool/urdutts"), columns=meta_cols))
    uat_parts = sorted((work / "pool/uat").glob("part-*.parquet"))
    uat = unique_by_id(b2.read_parts(uat_parts, columns=meta_cols))
    align = read_align(work)
    LOG.info("pools: aslp %d, urdutts %d, uat %d (aligned %d)", len(aslp), len(tts), len(uat), len(align))
    train, val, report = select_v4(aslp, tts, uat, align, load_spans(a.token, a.mining))
    out = work / "final"
    shutil.rmtree(out, ignore_errors=True)
    for sub in ("train", "validation", "manifests", "labels", "eval"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    labels = make_labels(train, val)
    report["conversational_voice_candidates"] = conversational_voices(train)
    (out / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    LOG.info("SELECTION %s", json.dumps({k: report[k] for k in ("status", "train", "validation")}, ensure_ascii=False))
    if report["status"] != "ok" and not a.allow_out_of_range:
        raise SystemExit(f"Train hours {report['train']['hours']} outside the required range; pools kept, rerun select")
    with open(out / "labels/v4.jsonl", "w", encoding="utf-8") as f:
        for r in labels:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    for name, rows in (("train", train), ("validation", val)):
        with open(out / "manifests" / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in sorted(rows, key=lambda r: r["id"]):
                f.write(json.dumps({k: r.get(k) for k in meta_cols}, ensure_ascii=False) + "\n")
    with open(out / "eval/uat_heldout_prompts.csv", "w", encoding="utf-8", newline="") as f:
        import csv
        w = csv.writer(f)
        w.writerow(["text", "set"])
        for r in sorted((r for r in val if r["source"] == "uat"), key=lambda r: r["id"])[: V4["uat_eval_prompts"]]:
            w.writerow([r["text"], "uat_heldout"])
    # Pull audio for the chosen ids from the pool parts and write ~450 MB parquet shards.
    split_of = {r["id"]: "train" for r in train} | {r["id"]: "validation" for r in val}
    order = {r["id"]: i for i, r in enumerate(train + val)}
    buckets, placed = defaultdict(list), set()
    for part in sorted(old_work.glob("pool/aslp/part-*.parquet")) + sorted(old_work.glob("pool/urdutts/part-*.parquet")) + uat_parts:
        for r in pq.read_table(part).to_pylist():
            if r["id"] in split_of and r["id"] not in placed:
                placed.add(r["id"])
                buckets[split_of[r["id"]]].append(r)
    missing = set(split_of) - placed
    if missing:
        raise RuntimeError(f"{len(missing)} selected clips have no audio in the pools, e.g. {sorted(missing)[:3]}")
    uploads = [(out / p, p) for p in ("report.json", "labels/v4.jsonl", "manifests/train.jsonl",
                                      "manifests/validation.jsonl", "eval/uat_heldout_prompts.csv")]
    for split, rows in buckets.items():
        rng = random.Random(b2.RULES["seed"] + (split == "validation"))
        rows.sort(key=lambda r: order[r["id"]])
        rng.shuffle(rows)  # mixed sources within every shard
        n = max(1, math.ceil(sum(len(r["flac"]) for r in rows) / 450e6))
        for k in range(n):
            chunk = rows[k::n]
            table = pa.table({"audio": [dict(bytes=r["flac"], path=r["id"] + ".flac") for r in chunk],
                              **{c: [r.get(c) for r in chunk] for c in meta_cols}})
            name = f"{split}/{split}-{k:05d}-of-{n:05d}.parquet"
            pq.write_table(table, out / name, compression="none")
            uploads.append((out / name, name))
    (out / "README.md").write_text(DATASET_CARD.format(report=json.dumps(report["train"], indent=1, ensure_ascii=False)))
    uploads.append((out / "README.md", "README.md"))
    LOG.info("Verified dataset (hours by split/source): %s", verify_final(out))
    if a.no_hub:
        LOG.info("--no-hub: final dataset left in %s (%d files)", out, len(uploads))
        return out
    for i in range(0, len(uploads), 6):
        hub.put(uploads[i:i + 6], f"stage-3 dataset files {i // 6 + 1}")
    revision = hub.api.dataset_info(hub.repo).sha
    pointer = dict(dataset_id=hub.repo, dataset_revision=revision, stage2_labels="labels/v4.jsonl",
                   train_hours=report["train"]["hours"], validation_clips=report["validation"]["clips"],
                   created=time.strftime("%Y-%m-%d %H:%M:%S"))
    path = work / "pointer.json"
    path.write_text(json.dumps(pointer, indent=1))
    from huggingface_hub import HfApi
    HfApi(token=a.token).upload_file(path_or_fileobj=str(path), path_in_repo=POINTER_PATH, repo_id=CKPT_REPO,
                                     repo_type="model", commit_message="Pin the stage-3 (v4) dataset revision")
    LOG.info("Published %s", pointer)
    return out


DATASET_CARD = """---
configs:
- config_name: default
  data_files:
  - split: train
    path: train/*.parquet
  - split: validation
    path: validation/*.parquet
---
# Urdu Orpheus stage-3 mixture (private, course project)

24 kHz mono FLAC. ASLP-lab/UrduSpeech conversational clips, kaab4321/UrduTTS audiobook clips and
mahwizzzz/UAT read speech (screened for text/audio agreement with MMS CTC alignment). The UAT dataset has no
declared licence: keep this repository private and do not redistribute its audio or models trained on it
without the author's permission. See report.json for rules, rejections and composition.

Train summary:
```
{report}
```
"""


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--phase", choices=["uat", "align", "select", "all"], default="all")
    p.add_argument("--work", default="/kaggle/working/stage3")
    p.add_argument("--repo", default=NEW_REPO)
    p.add_argument("--workers", type=int, default=max(2, (os.cpu_count() or 4)))
    p.add_argument("--limit", type=int, default=0, help="smoke test: first N UAT rows of the first shard")
    p.add_argument("--pool-parts", type=int, default=0, help="smoke test: read only N parts of each old pool")
    p.add_argument("--no-hub", action="store_true", help="never upload; leave the dataset in --work/final")
    p.add_argument("--allow-out-of-range", action="store_true")
    a = p.parse_args()
    a.token = os.environ.get("HF_TOKEN", "")
    a.mining = None
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
    Path(a.work).mkdir(parents=True, exist_ok=True)
    hub = b2.Hub(a.repo, a.token, enabled=not a.no_hub)
    # The UAT pool stays on local disk: 2 GB more private Hub storage would only help a session that dies mid-build.
    local = b2.Hub(a.repo, a.token, enabled=False)
    phases = dict(uat=(phase_uat, local), align=(phase_align, local), select=(phase_select, hub))
    for phase in (["uat", "align", "select"] if a.phase == "all" else [a.phase]):
        function, target = phases[phase]
        function(a, target)


if __name__ == "__main__":
    main()
