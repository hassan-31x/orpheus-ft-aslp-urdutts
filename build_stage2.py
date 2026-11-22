"""Stage-2 Urdu Orpheus data build: ASLP conversational + best UrduTTS, CPU only.

Phases (each resumable, partial progress pushed to the HF dataset repo):
  aslp     download -> 24 kHz mono (repo convert_audio) -> raw metrics (repo DNSMOS + VAD), no thresholds
  urdutts  extract ranked pool rows from pinned shards, verify against parent manifest sha256
  select   apply the stage-2 rules to the saved raw metrics and write the final train/validation parquet

Decisions are only made in `select`; the pool keeps every raw metric so rules can be re-run offline.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import io
import json
import logging
import math
import os
import random
import re
import sys
import tempfile
import threading
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

LOG = logging.getLogger("stage2")
ASLP_ID, ASLP_REV = "ASLP-lab/UrduSpeech", "16dd380cfd9049a3db7f06a98e878086916bf833"
TTS_ID, TTS_REV = "kaab4321/UrduTTS", "a012374328f140d2647e9138e6ceae85d4ee9f2c"
POOL_COLUMNS = ("id", "source", "pool_split", "category", "speaker", "text", "duration", "metadata_duration",
                "dnsmos_sig", "dnsmos_bak", "dnsmos_ovrl", "vad_fraction", "rms_db", "peak", "clipped_fraction",
                "wav_sha256", "parent_sha256", "provenance", "error", "flac")

# ---------------------------------------------------------------- selection rules
RULES = dict(
    target_train_hours=45.0, minimum_train_hours=40.0, maximum_train_hours=50.0,
    min_seconds=1.5, max_seconds=20.0,
    # Metadata durations are approximate; reject only gross disagreement (likely wrong transcript).
    duration_tolerance_seconds=1.0, duration_tolerance_fraction=0.25,
    min_chars_per_second=4.0, max_chars_per_second=25.0,
    max_clipped_fraction=0.005, min_rms_db=-42.0, max_rms_db=-8.0, min_vad_fraction=0.4,
    # Orpheus max_length 2048: ~86 SNAC tokens/s plus Urdu text (conservative 0.7 token/char) plus specials.
    max_sequence_tokens=2040, audio_tokens_per_second=86.0, text_tokens_per_char=0.7,
    # Strictest DNSMOS tier that still yields this many ASLP training hours wins.
    aslp_dnsmos_tiers=[dict(sig=3.0, bak=3.0, ovrl=2.7), dict(sig=2.9, bak=2.8, ovrl=2.55),
                       dict(sig=2.8, bak=2.6, ovrl=2.4)],
    aslp_tier_target_hours=20.0,
    speaker_max_minutes=30.0,
    aslp_validation_fraction=0.03, urdutts_validation_clips=250,
    min_urdutts_hours=15.0, seed=3407,
)


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def norm_text(text):
    text = unicodedata.normalize("NFKC", text).translate(str.maketrans("كيىۀة", "کییہہ"))
    return " ".join("".join(c if unicodedata.category(c)[0] in "LN" else " " for c in text).split())


# ---------------------------------------------------------------- HF persistence
class Hub:
    def __init__(self, repo, token, enabled=True):
        from huggingface_hub import HfApi
        self.repo, self.token, self.enabled = repo, token, enabled
        self.api = HfApi(token=token)
        if enabled:
            self.api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)

    def names(self, prefix=""):
        if not self.enabled:
            return []
        return [f for f in self.api.list_repo_files(self.repo, repo_type="dataset") if f.startswith(prefix)]

    def put(self, items, message):
        if not self.enabled:
            return
        from huggingface_hub import CommitOperationAdd
        ops = [CommitOperationAdd(path_in_repo=remote, path_or_fileobj=str(local)) for local, remote in items]
        for attempt in range(6):
            try:
                self.api.create_commit(self.repo, ops, commit_message=message, repo_type="dataset")
                return
            except Exception as exc:  # network/rate limit: back off, never lose the local copy
                LOG.warning("upload attempt %d failed: %s", attempt + 1, type(exc).__name__)
                time.sleep(min(300, 15 * 2 ** attempt))
        raise RuntimeError(f"upload failed after retries: {message}")

    def get(self, remote, local_dir):
        from huggingface_hub import hf_hub_download
        return hf_hub_download(self.repo, remote, repo_type="dataset", token=self.token, local_dir=local_dir)


def write_part(rows, path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    table = pa.table({c: [r.get(c) for r in rows] for c in POOL_COLUMNS})
    pq.write_table(table, path, compression="zstd")


def read_parts(paths, columns=None):
    import pyarrow.parquet as pq
    rows = []
    for p in sorted(paths):
        rows += pq.read_table(p, columns=columns).to_pylist()
    return rows


class PartWriter:
    """Buffers pool rows and pushes a parquet part (audio included) every N rows / T seconds."""

    def __init__(self, hub, work, prefix, start_index, every_rows=2500, every_seconds=600, tag=""):
        self.hub, self.work, self.prefix, self.tag = hub, Path(work), prefix, tag
        self.index, self.buffer, self.last = start_index, [], time.time()
        self.every_rows, self.every_seconds = every_rows, every_seconds
        (self.work / prefix).mkdir(parents=True, exist_ok=True)

    def add(self, row):
        self.buffer.append(row)
        if len(self.buffer) >= self.every_rows or time.time() - self.last > self.every_seconds:
            self.flush()

    def flush(self, progress=None):
        if not self.buffer:
            return
        name = f"{self.prefix}/part-{self.tag}{self.index:05d}.parquet"
        write_part(self.buffer, self.work / name)
        items = [(self.work / name, name)]
        if progress is not None:
            p = self.work / f"{self.prefix}-progress.json"
            p.write_text(json.dumps(progress, indent=1))
            items.append((p, f"progress/{self.prefix.split('/')[-1]}.json"))
        self.hub.put(items, f"pool {name} ({len(self.buffer)} rows)")
        LOG.info("pushed %s (%d rows)", name, len(self.buffer))
        self.index, self.buffer, self.last = self.index + 1, [], time.time()


def existing_parts(hub, work, prefix):
    if not hub.enabled:
        return sorted((Path(work) / prefix).glob("part-*.parquet"))
    names = sorted(n for n in hub.names(prefix + "/") if n.endswith(".parquet"))
    local = []
    for n in names:
        target = Path(work) / n
        if not target.is_file():
            hub.get(n, work)
        local.append(target)
    return local


# ---------------------------------------------------------------- audio helpers (repo code)
def to_flac(wav_path):
    import soundfile as sf
    wave, rate = sf.read(wav_path, dtype="int16")
    buffer = io.BytesIO()
    sf.write(buffer, wave, rate, format="FLAC", subtype="PCM_16")
    return buffer.getvalue()


_SCORER = None


def scorer(model_dir):
    global _SCORER
    if _SCORER is None:
        from quality import QualityScorer
        permissive = dict(minimum_seconds=0, maximum_seconds=1e9, maximum_clipped_fraction=1.0,
                          minimum_rms_db=-1e9, maximum_rms_db=1e9, minimum_vad_fraction=0.0,
                          minimum_dnsmos_sig=-1e9, minimum_dnsmos_bak=-1e9, minimum_dnsmos_ovrl=-1e9,
                          asr_model="none", asr_revision="none")
        _SCORER = QualityScorer(permissive, model_dir)
    return _SCORER


def aslp_worker(args):
    """Download, convert and measure one ASLP clip. Never applies a threshold."""
    row, token, model_dir, scratch = args
    import requests
    from huggingface_hub import hf_hub_url
    from urdutts_data import convert_audio
    out = dict(id=row["id"], source="aslp", pool_split="train", category=row["category"], speaker=row["speaker"],
               text=row["text"], metadata_duration=row["metadata_duration"],
               provenance=json.dumps(dict(dataset=ASLP_ID, revision=ASLP_REV, source_path=row["source_path"],
                                          confidence=row["confidence"], accuracy=row["accuracy"],
                                          emotion=row.get("emotion"), gender=row.get("gender")), ensure_ascii=False))
    url = hf_hub_url(ASLP_ID, row["source_path"], repo_type="dataset", revision=ASLP_REV)
    payload = None
    for attempt in range(5):
        try:
            r = requests.get(url, headers={"Authorization": f"Bearer {token}"}, timeout=(20, 120))
            if r.status_code == 404:
                out["error"] = "missing_audio"
                return out
            r.raise_for_status()
            payload = r.content
            break
        except Exception as exc:
            if attempt == 4:
                out["error"] = "download:" + type(exc).__name__
                return out
            time.sleep(3 * 2 ** attempt)
    wav = Path(scratch) / f"{row['id']}.wav"
    try:
        conv = convert_audio(dict(bytes=payload), wav)
        out["duration"] = conv["output_frames"] / 24000
        out["wav_sha256"] = sha256_bytes(wav.read_bytes())
        metrics, _ = scorer(model_dir).acoustic(wav)
        for k in ("dnsmos_sig", "dnsmos_bak", "dnsmos_ovrl", "vad_fraction", "rms_db", "peak", "clipped_fraction"):
            out[k] = metrics.get(k)
        out["flac"] = to_flac(wav)
    except Exception as exc:
        out["error"] = "convert_or_score:" + type(exc).__name__ + ":" + str(exc)[:120]
    finally:
        wav.unlink(missing_ok=True)
    return out


def phase_aslp(a, hub):
    work = Path(a.work)
    tag = (("r" if a.reverse else "s") + f"{a.shard}-") if a.shards > 1 else ""
    mine = [p for p in existing_parts(hub, work, "pool/aslp") if p.name.startswith("part-" + tag)]
    done = {r["id"] for r in read_parts(mine, columns=["id"])}
    cands = [json.loads(l) for l in open(a.inputs / "aslp_candidates.jsonl")]
    if a.limit:
        cands = cands[:a.limit]
    cands = cands[a.shard::a.shards]
    if a.reverse:  # extra kernels start from the far end; overlap is dropped by id in `select`
        cands = cands[::-1]
    todo = [c for c in cands if c["id"] not in done]
    LOG.info("ASLP: %d candidates, %d already in pool, %d to process", len(cands), len(done), len(todo))
    writer = PartWriter(hub, work, "pool/aslp", start_index=len(mine), every_rows=1500, every_seconds=420, tag=tag)
    scratch = tempfile.mkdtemp(prefix="aslp-", dir=work)
    global _SCORER
    scorer(str(work / "dnsmos"))  # download + checksum the DNSMOS model once, before forking workers
    _SCORER = None
    stats, started = Counter(), time.time()
    with cf.ProcessPoolExecutor(a.workers) as pool:
        jobs = ((c, a.token, str(work / "dnsmos"), scratch) for c in todo)
        for i, out in enumerate(pool.map(aslp_worker, jobs, chunksize=8), 1):
            stats["error" if out.get("error") else "ok"] += 1
            stats["hours"] += (out.get("duration") or 0) / 3600
            writer.add(out)
            if i % 500 == 0:
                rate = i / (time.time() - started)
                LOG.info("ASLP %d/%d  %.1f clips/s  ok=%d err=%d  %.2fh  eta %.0f min", i, len(todo), rate,
                         stats["ok"], stats["error"], stats["hours"], (len(todo) - i) / rate / 60)
    writer.flush(progress=dict(phase="aslp", done=True, **stats))
    LOG.info("ASLP phase complete: %s", dict(stats))


# ---------------------------------------------------------------- UrduTTS
def phase_urdutts(a, hub):
    import pyarrow as pa
    from huggingface_hub import hf_hub_download
    from urdutts_data import convert_audio
    work = Path(a.work)
    done = {r["id"] for r in read_parts(existing_parts(hub, work, "pool/urdutts"), columns=["id"])}
    want = defaultdict(dict)
    for name, split in (("urdutts_pool.jsonl", "train"), ("urdutts_validation_pool.jsonl", "validation")):
        rows = [json.loads(l) for l in open(a.inputs / name)]
        if a.limit:
            rows = rows[:a.limit]
        for r in rows:
            r["pool_split"] = split
            r["id"] = "tts-" + r["parent_audio"].split("/")[-1][:-4] + ("" if split == "train" else "-val")
            if r["id"] not in done:
                want[r["source_shard"]][r["source_row"]] = r
    LOG.info("UrduTTS: %d shards, %d rows to extract (%d already pooled)", len(want),
             sum(map(len, want.values())), len(done))
    writer = PartWriter(hub, work, "pool/urdutts", start_index=len(list((work / "pool/urdutts").glob("*.parquet")))
                        if (work / "pool/urdutts").is_dir() else 0, every_rows=3000)
    stats = Counter()

    def fetch(shard):
        cache = tempfile.mkdtemp(prefix="shard-", dir=work)
        for attempt in range(5):
            try:
                return cache, hf_hub_download(TTS_ID, shard, repo_type="dataset", revision=TTS_REV,
                                              cache_dir=cache, token=a.token)
            except Exception as exc:
                LOG.warning("shard %s attempt %d: %s", shard, attempt + 1, type(exc).__name__)
                time.sleep(10 * 2 ** attempt)
        raise RuntimeError(f"could not download {shard}")

    shards = sorted(want)
    with cf.ThreadPoolExecutor(2) as downloader:  # prefetch next shard while converting this one
        futures = {s: downloader.submit(fetch, s) for s in shards[:2]}
        for i, shard in enumerate(shards):
            cache, path = futures.pop(shard).result()
            if i + 2 < len(shards):
                futures[shards[i + 2]] = downloader.submit(fetch, shards[i + 2])
            rows = want[shard]
            with pa.memory_map(path, "r") as source, pa.ipc.open_stream(source) as reader:
                index = 0
                for batch in reader:
                    hits = [j for j in range(batch.num_rows) if index + j in rows]
                    for j in hits:
                        src = batch.slice(j, 1).to_pylist()[0]
                        r = rows[index + j]
                        out = dict(id=r["id"], source="urdutts", pool_split=r["pool_split"], category="audiobook",
                                   speaker=None, text=r["text"], duration=r["duration"], metadata_duration=r["duration"],
                                   dnsmos_sig=r["dnsmos_sig"], dnsmos_bak=r["dnsmos_bak"], dnsmos_ovrl=r["dnsmos_ovrl"],
                                   vad_fraction=r["vad_fraction"], rms_db=r["rms_db"],
                                   clipped_fraction=r["clipped_fraction"], parent_sha256=r["parent_sha256"],
                                   provenance=json.dumps(dict(dataset=TTS_ID, revision=TTS_REV, source_shard=shard,
                                                              source_row=index + j, parent_audio=r["parent_audio"],
                                                              source_text=src.get("transcription")), ensure_ascii=False))
                        wav = work / f"{r['id']}.wav"
                        try:
                            conv = convert_audio(src["audio"], wav)
                            out["duration"] = conv["output_frames"] / 24000
                            out["wav_sha256"] = sha256_bytes(wav.read_bytes())
                            out["flac"] = to_flac(wav)
                            stats["sha_match" if out["wav_sha256"] == r["parent_sha256"] else "sha_mismatch"] += 1
                        except Exception as exc:
                            out["error"] = "convert:" + type(exc).__name__
                            stats["error"] += 1
                        finally:
                            wav.unlink(missing_ok=True)
                        writer.add(out)
                    index += batch.num_rows
            import shutil
            shutil.rmtree(cache, ignore_errors=True)
            LOG.info("UrduTTS shard %d/%d %s done: %s", i + 1, len(shards), shard, dict(stats))
    writer.flush(progress=dict(phase="urdutts", done=True, **stats))


# ---------------------------------------------------------------- selection
def estimated_tokens(row):
    return RULES["audio_tokens_per_second"] * row["duration"] + RULES["text_tokens_per_char"] * len(row["text"]) + 16


def aslp_gate(r):
    """Return None when the clip passes the non-DNSMOS rules, else the reason."""
    if r.get("error"):
        return "error"
    d = r["duration"]
    if not RULES["min_seconds"] <= d <= RULES["max_seconds"]:
        return "duration"
    if abs(d - r["metadata_duration"]) > max(RULES["duration_tolerance_seconds"],
                                              RULES["duration_tolerance_fraction"] * r["metadata_duration"]):
        return "metadata_duration_mismatch"
    cps = len(r["text"].replace(" ", "")) / d
    if not RULES["min_chars_per_second"] <= cps <= RULES["max_chars_per_second"]:
        return "speaking_rate_vs_transcript"
    if r["clipped_fraction"] > RULES["max_clipped_fraction"]:
        return "clipping"
    if not RULES["min_rms_db"] <= r["rms_db"] <= RULES["max_rms_db"]:
        return "loudness"
    if r.get("vad_fraction") is not None and r["vad_fraction"] < RULES["min_vad_fraction"]:
        return "insufficient_speech"
    if estimated_tokens(r) > RULES["max_sequence_tokens"]:
        return "sequence_length"
    return None


def dns_ok(r, tier):
    return r["dnsmos_sig"] >= tier["sig"] and r["dnsmos_bak"] >= tier["bak"] and r["dnsmos_ovrl"] >= tier["ovrl"]


def hours(rows):
    return sum(r["duration"] for r in rows) / 3600


def summarize(rows):
    by = defaultdict(lambda: [0, 0.0])
    for r in rows:
        by[f"{r['source']}/{r['category']}"][0] += 1
        by[f"{r['source']}/{r['category']}"][1] += r["duration"] / 3600
    return {k: dict(clips=v[0], hours=round(v[1], 3)) for k, v in sorted(by.items())}


def choose_aslp(aslp, rng):
    gated, reasons = [], Counter()
    for r in aslp:
        why = aslp_gate(r)
        if why:
            reasons[why] += 1
        else:
            gated.append(r)
    # Exact duplicate audio and repeated same-speaker transcripts: keep the best scored copy.
    gated.sort(key=lambda r: -(r["dnsmos_ovrl"] + 0.1 * r["dnsmos_bak"]))
    seen_audio, seen_text, unique = set(), set(), []
    for r in gated:
        key = (r["speaker"], norm_text(r["text"]))
        if r["wav_sha256"] in seen_audio or key in seen_text:
            reasons["duplicate"] += 1
            continue
        seen_audio.add(r["wav_sha256"]); seen_text.add(key); unique.append(r)
    # Speaker-disjoint holdout per category, chosen before DNSMOS tiers so it is stable across tiers.
    by_cat = defaultdict(lambda: defaultdict(list))
    for r in unique:
        by_cat[r["category"]][r["speaker"]].append(r)
    holdout = set()
    for cat, speakers in sorted(by_cat.items()):
        names = sorted(speakers)
        rng.shuffle(names)
        goal, got = RULES["aslp_validation_fraction"] * hours([x for s in names for x in speakers[s]]), 0.0
        for s in names:
            if got >= goal:
                break
            if hours(speakers[s]) > 2 * goal + 0.05:  # do not hold out a dominant speaker
                continue
            holdout.add(s); got += hours(speakers[s])
    chosen_tier, result = None, None
    for tier in RULES["aslp_dnsmos_tiers"]:
        passing = [r for r in unique if dns_ok(r, tier)]
        train, val, per_speaker = [], [], Counter()
        for r in passing:
            if r["speaker"] in holdout:
                val.append(r)
            elif per_speaker[r["speaker"]] + r["duration"] <= RULES["speaker_max_minutes"] * 60:
                per_speaker[r["speaker"]] += r["duration"]; train.append(r)
        result, chosen_tier = (train, val), tier
        if hours(train) >= RULES["aslp_tier_target_hours"]:
            break
    train, val = result
    dns_rejected = sum(1 for r in unique if not dns_ok(r, chosen_tier))
    reasons["dnsmos_below_tier"] = dns_rejected
    return train, val, chosen_tier, reasons, holdout


def phase_select(a, hub):
    import pyarrow as pa
    import pyarrow.parquet as pq
    work = Path(a.work)
    meta_cols = [c for c in POOL_COLUMNS if c != "flac"]
    def unique_by_id(rows):
        seen, out = set(), []
        for r in rows:
            if r["id"] not in seen:
                seen.add(r["id"]); out.append(r)
        return out
    aslp = unique_by_id(read_parts(existing_parts(hub, work, "pool/aslp"), columns=meta_cols))
    tts = unique_by_id(read_parts(existing_parts(hub, work, "pool/urdutts"), columns=meta_cols))
    rng = random.Random(RULES["seed"])
    aslp_train, aslp_val, tier, reasons, holdout = choose_aslp(aslp, rng)
    aslp_texts = {norm_text(r["text"]) for r in aslp_train + aslp_val}
    tts_ok = [r for r in tts if not r.get("error") and r["pool_split"] == "train"
              and estimated_tokens(r) <= RULES["max_sequence_tokens"] and norm_text(r["text"]) not in aslp_texts]
    tts_ok.sort(key=lambda r: -(r["dnsmos_ovrl"] + 0.1 * r["dnsmos_bak"]))
    seen, tts_unique = set(), []
    for r in tts_ok:
        if r["wav_sha256"] not in seen:
            seen.add(r["wav_sha256"]); tts_unique.append(r)
    fill = min(max(RULES["target_train_hours"] - hours(aslp_train), RULES["min_urdutts_hours"]),
               RULES["maximum_train_hours"] - hours(aslp_train))
    tts_train, acc = [], 0.0
    for r in tts_unique:
        if acc >= fill * 3600:
            break
        tts_train.append(r); acc += r["duration"]
    tts_val_pool = sorted((r for r in tts if not r.get("error") and r["pool_split"] == "validation"),
                          key=lambda r: r["id"])
    rng.shuffle(tts_val_pool)
    tts_val = tts_val_pool[:RULES["urdutts_validation_clips"]]
    train, val = aslp_train + tts_train, aslp_val + tts_val
    assert not ({r["id"] for r in train} & {r["id"] for r in val})
    assert not ({r["speaker"] for r in aslp_train} & {r["speaker"] for r in aslp_val})
    train_h = hours(train)
    report = dict(
        status="ok" if RULES["minimum_train_hours"] <= train_h <= RULES["maximum_train_hours"] else "out_of_range",
        rules=RULES, chosen_aslp_dnsmos_tier=tier,
        train=dict(clips=len(train), unique_hours=round(train_h, 3), aslp_hours=round(hours(aslp_train), 3),
                   urdutts_hours=round(hours(tts_train), 3),
                   aslp_fraction=round(hours(aslp_train) / train_h, 3) if train_h else 0,
                   aslp_speakers=len({r["speaker"] for r in aslp_train}), composition=summarize(train)),
        validation=dict(clips=len(val), hours=round(hours(val), 3), composition=summarize(val),
                        aslp_heldout_speakers=len(holdout)),
        pool=dict(aslp_clips=len(aslp), aslp_errors=sum(1 for r in aslp if r.get("error")),
                  urdutts_clips=len(tts), urdutts_sha_mismatch=sum(1 for r in tts if r.get("wav_sha256")
                                                                    and r["wav_sha256"] != r["parent_sha256"]),
                  aslp_rejections=dict(reasons)),
        metric_coverage=dict(dnsmos="all selected clips", whisper_wer="not measured for new clips; ASLP uses "
                             "human 'Highly Accurate' transcripts gated by metadata-duration and speaking-rate checks"),
    )
    out = work / "final"
    (out / "data").mkdir(parents=True, exist_ok=True)
    (out / "manifests").mkdir(exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    LOG.info("SELECTION %s", json.dumps({k: report[k] for k in ("status", "chosen_aslp_dnsmos_tier", "train", "validation")},
                                        ensure_ascii=False))
    if report["status"] != "ok" and not a.allow_out_of_range:
        hub.put([(out / "report.json", a.out_prefix + "report.json")], "selection report (out of range; no data written)")
        raise SystemExit(f"Train hours {train_h:.2f} outside required range; pool kept for re-selection")
    for name, rows in (("train", train), ("validation", val)):
        with open(out / "manifests" / f"{name}.jsonl", "w") as handle:
            for r in sorted(rows, key=lambda r: r["id"]):
                handle.write(json.dumps({k: r.get(k) for k in meta_cols}, ensure_ascii=False) + "\n")
    # Pull audio for the chosen ids from the pool parts and write ~500 MB parquet shards.
    wanted = {r["id"]: ("train" if i < len(train) else "validation") for i, r in enumerate(train + val)}
    order = {r["id"]: i for i, r in enumerate(train + val)}
    buckets, placed = defaultdict(list), set()
    for part in sorted((work / "pool").glob("*/part-*.parquet")):
        for r in pq.read_table(part).to_pylist():
            if r["id"] in wanted and r["id"] not in placed:
                placed.add(r["id"])
                buckets[wanted[r["id"]]].append(r)
    uploads = [(out / "report.json", a.out_prefix + "report.json"),
               (out / "manifests/train.jsonl", a.out_prefix + "manifests/train.jsonl"),
               (out / "manifests/validation.jsonl", a.out_prefix + "manifests/validation.jsonl")]
    for split, rows in buckets.items():
        rng2 = random.Random(RULES["seed"] + (split == "validation"))
        rows.sort(key=lambda r: order[r["id"]])
        rng2.shuffle(rows)  # mixed sources within every shard
        size = sum(len(r["flac"]) for r in rows)
        n = max(1, math.ceil(size / 450e6))
        for k in range(n):
            chunk = rows[k::n]
            table = pa.table({
                "audio": [dict(bytes=r["flac"], path=r["id"] + ".flac") for r in chunk],
                **{c: [r.get(c) for r in chunk] for c in meta_cols},
            })
            name = f"data/{split}-{k:05d}-of-{n:05d}.parquet"
            pq.write_table(table, out / name, compression="none")
            uploads.append((out / name, a.out_prefix + name))
    card = out / "README.md"
    card.write_text(DATASET_CARD.format(prefix=a.out_prefix, report=json.dumps(report["train"], indent=1, ensure_ascii=False),
                                        tier=json.dumps(tier)))
    uploads.append((card, a.out_prefix + "README.md"))
    for i in range(0, len(uploads), 8):
        hub.put(uploads[i:i + 8], f"stage-2 dataset files {i // 8 + 1}")
    LOG.info("Final dataset uploaded: %d train / %d validation clips", len(train), len(val))


DATASET_CARD = """---
configs:
- config_name: default
  data_files:
  - split: train
    path: {prefix}data/train-*.parquet
  - split: validation
    path: {prefix}data/validation-*.parquet
dataset_info:
  features:
  - name: audio
    dtype:
      audio:
        sampling_rate: 24000
---
# Urdu Orpheus stage-2 mixture (private)

24 kHz mono FLAC, Standard Pakistani Urdu. Conversational ASLP-lab/UrduSpeech US-Std clips
(drama, film, comedy show, roadside interview, interview, podcast, vlogs, news, food, YouTube review;
poetry recitation and prose narration excluded) plus the cleanest kaab4321/UrduTTS training clips
by saved DNSMOS. Validation: speaker-disjoint ASLP holdout + UrduTTS clips from its original eval split.
Every row keeps provenance, raw DNSMOS/loudness/VAD metrics and the converted-WAV sha256.
See report.json for rules, rejections and composition. ASLP DNSMOS tier used: {tier}

Train summary:
```
{report}
```
"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=["aslp", "urdutts", "select", "all"], default="all")
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--work", default="/tmp/stage2")
    p.add_argument("--repo", default="hassan-31x/urdu-orpheus-stage2-data")
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--limit", type=int, default=0, help="smoke test: first N rows per input")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--reverse", action="store_true")
    p.add_argument("--out-prefix", default="")
    p.add_argument("--no-hub", action="store_true")
    p.add_argument("--allow-out-of-range", action="store_true")
    a = p.parse_args()
    a.token = os.environ.get("HF_TOKEN", "")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
    Path(a.work).mkdir(parents=True, exist_ok=True)
    hub = Hub(a.repo, a.token, enabled=not a.no_hub)
    phases = ["aslp", "urdutts", "select"] if a.phase == "all" else [a.phase]
    if a.phase == "all":
        # UrduTTS extraction is mostly network/arrow work; run it beside the ASLP pool.
        failure = []

        def run_urdutts():
            try:
                phase_urdutts(a, hub)
            except BaseException as exc:  # surfaced below; never select from a partial UrduTTS pool
                LOG.exception("UrduTTS phase failed")
                failure.append(exc)
        t = threading.Thread(target=run_urdutts, name="urdutts")
        t.start()
        phase_aslp(a, hub)
        t.join()
        if failure:
            raise RuntimeError("UrduTTS phase failed; ASLP pool is saved, rerun --phase urdutts then select")
        phase_select(a, hub)
        return
    for phase in phases:
        globals()["phase_" + phase](a, hub)


if __name__ == "__main__":
    main()
