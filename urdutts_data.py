"""Read the public UrduTTS save_to_disk Arrow release, one shard at a time.

No datasets.load_dataset(), Torch audio decoder, Drive input, or full Hub snapshot
is needed. Native embedded WAVs are converted to the existing ASLP manifest API.
"""
from __future__ import annotations

import csv
import io
import logging
import math
import re
from pathlib import Path

from orpheus_utils import atomic_json, contained, fingerprint, read_json, sha256
from pipeline_recovery import retry_io, set_activity

LOG = logging.getLogger("orpheus_urdu")
RECIPE_VERSION = 1
SPLITS = {"train": "train", "eval": "validation", "test": "test"}


def validate_config(cfg):
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", cfg["dataset_id"]):
        raise ValueError("dataset_id must be owner/repository")
    if not re.fullmatch(r"[0-9a-f]{40}", cfg["dataset_revision"]):
        raise ValueError("dataset_revision must be a fixed 40-character Hub commit SHA")
    if cfg["dataset_text_column"] != "transcription":
        raise ValueError("This Urdu-script adaptation requires dataset_text_column=transcription")
    if not isinstance(cfg["dataset_include_test"], bool):
        raise ValueError("dataset_include_test must be true or false")
    if cfg.get("data_dir"):
        raise ValueError("UrduTTS downloads directly from the Hub; leave data_dir=null")
    if any(cfg.get(k) for k in ("train_manifest", "validation_manifest", "test_manifest", "speaker_column")):
        raise ValueError("Use the released UrduTTS splits and text-only prompts; leave manifest/speaker overrides null")


def fetch_json(cfg, filename):
    import requests
    from huggingface_hub import hf_hub_url
    url = hf_hub_url(cfg["dataset_id"], filename, repo_type="dataset", revision=cfg["dataset_revision"])
    def fetch():
        with requests.get(url, timeout=(30, 120)) as response:
            response.raise_for_status()
            return response.json()
    return retry_io(fetch)


def stream_rows(cfg, filename, limit=None):
    """Stream IPC record batches over HTTPS; never retain a source shard on disk.

    Arrow still reads an entire record batch (up to ~500 MB in this release).
    Smoke mode stops after its requested rows rather than visiting every shard.
    Network errors propagate to the shard retry; malformed IPC is a hard failure.
    """
    import pyarrow as pa
    import requests
    from urllib3.exceptions import HTTPError as TransportError
    from huggingface_hub import hf_hub_url
    url = hf_hub_url(cfg["dataset_id"], filename, repo_type="dataset", revision=cfg["dataset_revision"])
    try:
        with requests.get(url, stream=True, timeout=(30, 180)) as response:
            response.raise_for_status()
            response.raw.decode_content = True
            with pa.ipc.open_stream(response.raw) as reader:
                if not {"audio", cfg["dataset_text_column"]} <= set(reader.schema.names):
                    raise ValueError(f"{filename}: missing audio/transcription columns: {reader.schema}")
                count = 0
                for batch in reader:
                    # Slice BEFORE converting to Python: avoid copying all embedded WAVs.
                    take = batch.num_rows if limit is None else min(batch.num_rows, limit - count)
                    for row in batch.slice(0, take).to_pylist():
                        yield count, row
                        count += 1
                    if limit is not None and count >= limit:
                        return
    except TransportError as exc:
        # raw.read() exposes urllib3 errors instead of requests' wrapped errors.
        # Retry the shard, never classify a broken connection as a bad audio row.
        raise ConnectionError(f"Interrupted source transfer: {filename}") from exc


def convert_audio(audio, destination):
    """Preserve full clips; average channels and polyphase-resample to SNAC 24 kHz."""
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly
    payload = audio.get("bytes") if isinstance(audio, dict) else None
    if not payload:
        # Released paths refer to the creator's machine; never try opening them.
        raise ValueError("audio.bytes is missing; the release must contain embedded audio")
    wave, rate = sf.read(io.BytesIO(payload), dtype="float32", always_2d=True)
    if rate <= 0 or not len(wave) or not np.isfinite(wave).all() or not np.any(wave):
        raise ValueError("Empty, silent or nonfinite source audio")
    channels, source_frames = wave.shape[1], len(wave)
    mono = wave.mean(axis=1)
    if rate != 24000:
        divisor = math.gcd(rate, 24000)
        mono = resample_poly(mono, 24000 // divisor, rate // divisor).astype("float32")
    # Do not silently saturate PCM16 if a different release has out-of-range audio.
    if not np.isfinite(mono).all() or np.max(np.abs(mono)) > 1.0001:
        raise ValueError("Resampled signal exceeds PCM16 range; review the source clip")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_suffix(".tmp.wav")
    sf.write(temp, mono, 24000, subtype="PCM_16", format="WAV")
    temp.replace(destination)
    return dict(source_rate=rate, source_channels=channels, source_frames=source_frames,
                output_frames=len(mono), duration_seconds=source_frames / rate)


def valid_part(root, marker, identity):
    try:
        part = read_json(marker)
        if part["identity"] != identity:
            return None
        for row in part["rows"]:
            audio = contained(root, row["audio"])
            if not audio.is_file() or sha256(audio) != row["sha256"]:
                return None
        return part
    except (OSError, ValueError, KeyError, TypeError):
        return None


def write_manifest(path, rows):
    temp = path.with_suffix(".tmp.csv")
    with temp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["audio", "text"])
        writer.writeheader()
        writer.writerows({key: row[key] for key in ("audio", "text")} for row in rows)
    temp.replace(path)


def prepare_dataset(cfg, work, *, metadata=None, row_reader=None):
    """Resume completed shards, verify WAV hashes and emit train/validation CSVs.

    Dataset revision, conversion source/packages and smoke limits form an immutable
    preparation key. Exclusions are recorded; failed network reads never count as
    bad training rows. Test remains undownloaded unless explicitly requested.
    Optional callables enable CPU fixture tests with the real Arrow/audio formats.
    """
    import importlib.metadata
    validate_config(cfg)
    metadata = metadata or (lambda filename: fetch_json(cfg, filename))
    row_reader = row_reader or (lambda filename, limit: stream_rows(cfg, filename, limit))
    recipe = dict(version=RECIPE_VERSION, dataset_id=cfg["dataset_id"], revision=cfg["dataset_revision"],
                  text_column=cfg["dataset_text_column"], target_rate=24000, subtype="PCM_16",
                  channels="mean", resampling="scipy.signal.resample_poly", trimming=False,
                  train_limit=cfg.get("test_train_clips"), eval_limit=cfg.get("test_validation_clips"),
                  include_test=cfg["dataset_include_test"], policy=cfg["invalid_row_policy"],
                  source_sha256=sha256(Path(__file__)),
                  packages={name: importlib.metadata.version(name) for name in
                            ("numpy", "scipy", "soundfile", "pyarrow")})
    root = Path(work) / "datasets" / fingerprint(recipe)
    root.mkdir(parents=True, exist_ok=True)
    released = metadata("dataset_dict.json")["splits"]
    required = ["train", "eval"] + (["test"] if cfg["dataset_include_test"] else [])
    if not set(required) <= set(released):
        raise ValueError(f"Expected released splits {required}; found {released}")
    report = dict(recipe=recipe, splits={}, conversion_exclusions=[],
                  test_downloaded=cfg["dataset_include_test"],
                  limitations="Released utterance splits; no speaker labels. Transcripts are ASR-generated.")
    for source_split in required:
        split = SPLITS[source_split]
        state = metadata(f"{source_split}/state.json")
        files = [f"{source_split}/{item['filename']}" for item in state["_data_files"]]
        # Validate all paths before constructing remote requests or local markers.
        for filename in files:
            contained(root, filename)
            if not filename.endswith(".arrow"):
                raise ValueError(f"Unexpected shard filename {filename}")
        limit = cfg.get("test_train_clips") if split == "train" else cfg.get("test_validation_clips")
        rows, consumed = [], 0
        for shard_index, filename in enumerate(files):
            remaining = None if limit is None else limit - consumed
            if remaining is not None and remaining <= 0:
                break
            marker = root / "parts" / f"{split}-{shard_index:05d}.json"
            identity = dict(recipe=fingerprint(recipe), filename=filename, limit=remaining)
            part = valid_part(root, marker, identity)
            if part is None:
                def process_shard():
                    accepted, rejected, count = [], [], 0
                    set_activity(f"downloading/converting UrduTTS {source_split} shard {shard_index+1}/{len(files)}")
                    for index, row in row_reader(filename, remaining):
                        count += 1
                        relative = f"audio/{split}/{shard_index:05d}-{index:06d}.wav"
                        text = row.get(cfg["dataset_text_column"])
                        try:
                            if not isinstance(text, str) or not re.search(r"[\u0600-\u06ff]", text):
                                raise ValueError("Missing Urdu-script transcription")
                            properties = convert_audio(row.get("audio"), root / relative)
                            accepted.append(dict(audio=relative, text=text, sha256=sha256(root / relative),
                                                 source_shard=filename, source_row=index, **properties))
                        except (ValueError, RuntimeError) as exc:
                            # Disk/permission/network errors must stop, not delete rows.
                            if cfg["invalid_row_policy"] == "strict":
                                raise
                            rejected.append(dict(split=split, source_shard=filename, source_row=index,
                                                 error=str(exc), stage="source_conversion"))
                    if count == 0:
                        raise ValueError(f"Empty source shard: {filename}")
                    result = dict(identity=identity, consumed=count, rows=accepted, excluded=rejected)
                    atomic_json(marker, result)
                    return result
                part = retry_io(process_shard)
            consumed += part["consumed"]
            rows.extend(part["rows"])
            report["conversion_exclusions"].extend(part["excluded"])
            LOG.info("UrduTTS %s shard %d/%d: %d source rows, %d converted", source_split,
                     shard_index+1, len(files), consumed, len(rows))
        write_manifest(root / f"{split}.csv", rows)
        report["splits"][split] = dict(source_split=source_split, source_rows=consumed, clips=len(rows),
                                      hours=sum(r["output_frames"] for r in rows)/24000/3600,
                                      native_rates=sorted({r["source_rate"] for r in rows}),
                                      manifest_sha256=sha256(root / f"{split}.csv"))
    atomic_json(root / "dataset_source.json", report)
    if not all(report["splits"][s]["clips"] for s in ("train", "validation")):
        raise ValueError(f"No usable train/validation rows; see {root / 'dataset_source.json'}")
    if report["conversion_exclusions"]:
        LOG.warning("Source conversion excluded %d rows; see dataset_source.json", len(report["conversion_exclusions"]))
    return root.resolve()
