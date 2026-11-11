"""Stage-2 training data: the curated ASLP + UrduTTS mixture published as a private Hub dataset.

The dataset (`hassan-31x/urdu-orpheus-stage2-data`) already holds the selected, deduplicated,
quality-screened clips as 24 kHz mono PCM16 FLAC with separate train/validation shards. This module
materialises it as the `audio/<split>/<id>.flac` + `<split>.csv` layout the training pipeline audits
and encodes. Nothing is re-screened or re-ranked here, so every epoch trains on identical data;
the Trainer reshuffles it each epoch from the run seed.
"""
from __future__ import annotations

import csv
import logging
import os
import re
import shutil
import tempfile
from pathlib import Path

from orpheus_utils import atomic_json, fingerprint, read_json, sha256
from pipeline_recovery import retry_io

LOG = logging.getLogger("orpheus_urdu")
RECIPE_VERSION = 1
FIELDS = ("audio", "text", "source", "category")
VOICE_COLUMN = "voice"  # "speaker | gender | style | emotion" prompt prefix from a pinned labels file


def validate_config(cfg):
    from urdutts_data import validate_config as validate_tts
    validate_tts(dict(cfg, speaker_column=None))  # pinned 40-char dataset revision, owner/repo id, no manual manifests
    if bool(cfg.get("stage2_labels")) != (cfg["speaker_column"] == VOICE_COLUMN):
        raise ValueError(f"stage2_labels and speaker_column={VOICE_COLUMN!r} go together (tagged prompts need labels)")
    if cfg.get("mixture"):
        raise ValueError("stage2_dataset runs use the curated dataset, not the on-the-fly mixture")
    if any(cfg.get(k) is not None and cfg[k] < 2 for k in ("test_train_clips", "test_validation_clips")):
        raise ValueError("Stage-2 smoke subsets need at least two clips to represent both sources")
    if cfg.get("init_from_run") and cfg.get("init_source"):
        raise ValueError("Choose init_from_run OR init_source")
    if cfg.get("init_source") and cfg["init_source"]["repo_id"] == cfg["hf_repo_id"]:
        raise ValueError("The UrduTTS parent and the stage-2 destination must be separate repositories")
    if cfg["sampling"] != "random":
        raise ValueError("Stage-2 training shuffles with the Trainer's seeded random sampler")


def recipe_for(cfg):
    subset = None
    if cfg.get("test_train_clips") or cfg.get("test_validation_clips"):
        subset = dict(train=cfg.get("test_train_clips"), validation=cfg.get("test_validation_clips"))
    recipe = dict(version=RECIPE_VERSION, dataset_id=cfg["dataset_id"], revision=cfg["dataset_revision"],
                  audio="flac-24000-mono-pcm16", subset=subset)
    if cfg.get("stage2_labels"):
        recipe["labels"] = cfg["stage2_labels"]  # older runs keep their recipe identity unchanged
    return recipe


def load_labels(cfg, token):
    """id -> (keep, voice) from the labels file pinned at the same dataset revision."""
    import json
    from huggingface_hub import hf_hub_download
    path = retry_io(lambda: hf_hub_download(cfg["dataset_id"], cfg["stage2_labels"], repo_type="dataset",
                                           revision=cfg["dataset_revision"], token=token), attempts=5)
    labels = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if not r["voice"].strip():
                raise RuntimeError(f"Empty voice tag for {r['id']}")
            labels[r["id"]] = (r["keep"], r["voice"].strip())
    return labels


def write_manifest(path, rows):
    fields = FIELDS + ((VOICE_COLUMN,) if rows and VOICE_COLUMN in rows[0] else ())
    temp = path.with_suffix(".tmp.csv")
    with temp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: row[k] for k in fields} for row in rows)
    temp.replace(path)


def _shards(api, cfg, split):
    names = sorted(f for f in api.list_repo_files(cfg["dataset_id"], repo_type="dataset", revision=cfg["dataset_revision"])
                   if f.startswith(split + "/") and f.endswith(".parquet"))
    if not names:
        raise RuntimeError(f"No {split} shards in {cfg['dataset_id']}@{cfg['dataset_revision'][:8]}")
    return names


def _materialise(cfg, root, split, files, limit, token, scratch, labels=None):
    """Write one split's FLAC files and return its manifest rows (atomic per file; safe to repeat)."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    rows, seconds = [], 0.0
    for name in files:
        with tempfile.TemporaryDirectory(prefix="stage2-", dir=scratch) as cache:
            LOG.info("Fetching stage-2 shard %s", name)
            path = retry_io(lambda: hf_hub_download(cfg["dataset_id"], name, repo_type="dataset",
                                                   revision=cfg["dataset_revision"], token=token, cache_dir=cache),
                            attempts=5)
            for batch in pq.ParquetFile(path).iter_batches(batch_size=256):
                for row in batch.to_pylist():
                    if limit and len(rows) >= limit:
                        break
                    if labels is not None:
                        if row["id"] not in labels:
                            raise RuntimeError(f"Clip {row['id']} has no label; labels and dataset revision disagree")
                        if not labels[row["id"]][0]:
                            continue
                    if row.get("error") or not row["audio"]["bytes"]:
                        raise RuntimeError(f"Curated dataset row {row['id']} is unusable; rebuild the dataset")
                    relative = f"audio/{split}/{row['id']}.flac"
                    target = root / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if not target.is_file() or target.stat().st_size != len(row["audio"]["bytes"]):
                        temp = target.with_suffix(".tmp")
                        temp.write_bytes(row["audio"]["bytes"])
                        temp.replace(target)
                    rows.append(dict(audio=relative, text=row["text"], source=row["source"],
                                     category=row["category"], duration=row["duration"], id=row["id"],
                                     **({VOICE_COLUMN: labels[row["id"]][1]} if labels is not None else {})))
                    seconds += row["duration"]
                if limit and len(rows) >= limit:
                    break
        if limit and len(rows) >= limit:
            break
    return rows


def prepare_dataset(cfg, work):
    validate_config(cfg)
    recipe = recipe_for(cfg)
    root = Path(work) / "datasets" / fingerprint(recipe)
    root.mkdir(parents=True, exist_ok=True)
    marker = root / "dataset_source.json"
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN must have read access to the private stage-2 dataset " + cfg["dataset_id"])
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    if marker.is_file():
        report = read_json(marker)
        if report["recipe"] != recipe:
            raise RuntimeError("Stage-2 preparation identity changed")
        missing = [r for split in ("train", "validation")
                   for r in report["rows"][split] if not (root / r["audio"]).is_file()]
        if not missing:
            return root.resolve()
        LOG.warning("%d stage-2 audio files missing (cleanup or new session); re-fetching pinned shards", len(missing))
    scratch = Path(work) / "source_shards"
    scratch.mkdir(parents=True, exist_ok=True)
    selected, splits = {}, {}
    labels = load_labels(cfg, token) if cfg.get("stage2_labels") else None
    for split in ("train", "validation"):
        limit = None
        if recipe["subset"]:
            # Smoke runs read only the first (source-mixed) shard and sample from the rows they keep.
            limit = max(40, 4 * (recipe["subset"][split] or 10))
        files = _shards(api, cfg, split)
        files = files[:1] if limit else files
        rows = _materialise(cfg, root, split, files, limit, token, scratch, labels)
        if not rows:
            raise RuntimeError(f"No {split} rows materialised")
        write_manifest(root / f"{split}.csv", rows)
        selected[split] = rows
        by = {}
        for r in rows:
            by.setdefault(f"{r['source']}/{r['category']}", [0, 0.0])
            by[f"{r['source']}/{r['category']}"][0] += 1
            by[f"{r['source']}/{r['category']}"][1] += r["duration"] / 3600
        splits[split] = dict(clips=len(rows), hours=sum(r["duration"] for r in rows) / 3600,
                             composition={k: dict(clips=v[0], hours=round(v[1], 3)) for k, v in sorted(by.items())},
                             manifest_sha256=sha256(root / f"{split}.csv"))
    ids = [r["id"] for rows in selected.values() for r in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Curated dataset contains a clip in more than one place")
    atomic_json(marker, dict(recipe=recipe, splits=splits, conversion_exclusions=[], rows=selected,
                             limitations=("Curated by DNSMOS/loudness/VAD gates and transcript-length checks; ASLP transcripts "
                                          "are dataset-provided and were not ASR-verified. UrduTTS transcripts are ASR-generated.")))
    LOG.info("Stage-2 data ready: %s", {k: (v["clips"], round(v["hours"], 2)) for k, v in splits.items()})
    return root.resolve()
