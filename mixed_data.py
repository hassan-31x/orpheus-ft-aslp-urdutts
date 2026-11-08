"""Pinned-source selection with explicit exclusions, bounded disk use and replay."""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import io
import json
import logging
from pathlib import Path
import re
import shutil

from orpheus_utils import atomic_json, clean_text, contained, fingerprint, read_json, sha256
from pipeline_recovery import retry_io
from urdutts_data import convert_audio, fetch_json, stream_rows
from quality import DNS_REVISION, QualityScorer, metadata_problem, quality_rank, text_problem
from mixture import best_with_budget, group_is_validation, select_aslp, source_summary, validate_training_mix

LOG = logging.getLogger("orpheus_urdu")
CATEGORIES = ("interview", "podcast", "drama", "news", "roadside_interview")


def write_manifest(path, rows):
    temp = path.with_suffix(".tmp.csv")
    fields = ("audio", "text", "source", "category")
    with temp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: row[k] for k in fields} for row in rows)
    temp.replace(path)


def validate_config(cfg):
    from urdutts_data import validate_config as validate_tts
    validate_tts(cfg)
    mix = cfg["mixture"]
    if any(cfg.get(k) is not None and cfg[k] < 2 for k in ("test_train_clips", "test_validation_clips")):
        raise ValueError("Mixed smoke subsets need at least two clips to represent both sources")
    if not re.fullmatch(r"[0-9a-f]{40}", mix["aslp_revision"]):
        raise ValueError("Pin the ASLP dataset commit")
    if not re.fullmatch(r"[0-9a-f]{40}", mix["asr_revision"]):
        raise ValueError("Pin the ASR model commit")
    if cfg.get("init_from_run") and cfg.get("init_source"):
        raise ValueError("Choose init_from_run OR init_source")
    if cfg.get("init_source") and cfg["init_source"]["repo_id"] == cfg["hf_repo_id"]:
        raise ValueError("The initial UrduTTS source and mixed-data destination must be separate")
    if cfg["sampling"] != "random":
        raise ValueError("Mixed training uses its recorded curriculum, not length grouping")
    if not .5 <= mix["aslp_fraction"] <= .75:
        raise ValueError("aslp_fraction must be in [0.5, 0.75]")
    fractions = mix["curriculum_aslp_fractions"]
    if len(fractions) != 3 or sorted(fractions) != fractions or not all(.3 <= f <= .75 for f in fractions):
        raise ValueError("Use three increasing curriculum fractions, each retaining at least 25% replay")
    if abs(sum(fractions)/len(fractions) - mix["aslp_fraction"]) > 1e-8:
        raise ValueError("The curriculum's mean fraction must match the selected audio-hour mixture")
    if not .6 <= mix["minimum_core_fraction"] <= 1:
        raise ValueError("Interview/podcast must remain a majority of ASLP")
    if set(mix["category_caps"]) != {"drama", "news", "roadside_interview"} or any(
            not 0 <= v <= .25 for v in mix["category_caps"].values()):
        raise ValueError("Only small explicit drama/news/roadside caps are supported")
    if sum(mix["category_caps"].values()) > 1-mix["minimum_core_fraction"] + 1e-8:
        raise ValueError("Fallback category caps exceed the non-core ASLP budget")
    if not 0 < mix["validation_fraction"] < .3 or mix["quality_device"] not in ("cpu", "cuda"):
        raise ValueError("Invalid validation fraction or screening device")
    if mix["tts_scan_limit"] is not None and not cfg.get("test_train_clips"):
        raise ValueError("Full runs must scan all UrduTTS training shards; tts_scan_limit is smoke-only")


def recipe_for(cfg):
    import importlib.metadata
    return dict(dataset_id=cfg["dataset_id"], revision=cfg["dataset_revision"],
                aslp_id=cfg["mixture"]["aslp_id"], aslp_revision=cfg["mixture"]["aslp_revision"],
                mixture=cfg["mixture"], seed=cfg["seed"], test_train_clips=cfg["test_train_clips"],
                test_validation_clips=cfg["test_validation_clips"], dns_revision=DNS_REVISION,
                source_sha256={n: sha256(Path(__file__).parent/n) for n in
                    ("mixed_data.py", "quality.py", "mixture.py", "urdutts_data.py")},
                packages={n: importlib.metadata.version(n) for n in
                    ("numpy", "scipy", "soundfile", "pyarrow", "onnxruntime", "faster-whisper", "ctranslate2")})


def remote_bytes(repo, revision, path, missing_ok=False):
    import requests
    from huggingface_hub import hf_hub_url
    def fetch():
        response = requests.get(hf_hub_url(repo, path, repo_type="dataset", revision=revision), timeout=(30, 180))
        if missing_ok and response.status_code == 404:
            return None
        response.raise_for_status()
        return response.content
    return retry_io(fetch)


def jsonl_metadata(payload, name):
    if payload is None:
        return None
    rows = [json.loads(line) for line in payload.decode("utf-8-sig").splitlines() if line.strip()]
    by_clip, conflicts = {}, set()
    for row in rows:
        clip = row["Audio_Clip"]
        if Path(clip).name != clip or not re.fullmatch(r"[\w.-]+\.wav", clip, re.I):
            raise ValueError("Unsafe ASLP Audio_Clip in " + name)
        # audio_id is not unique in roadside annotations; join on full Audio_Clip.
        if clip in by_clip and by_clip[clip] != row:
            conflicts.add(clip)
        by_clip[clip] = row
    for clip in conflicts:
        del by_clip[clip]
    return by_clip


def aslp_candidates(cfg, report, fetch=remote_bytes):
    mix = cfg["mixture"]
    result = defaultdict(list)
    for category in CATEGORIES:
        for length in ("short", "long"):
            prefix = f"corpus/US-Std/{length}/{category}/"
            trans_name, para_name = (prefix+category+suffix for suffix in
                                     ("_final_transcription.jsonl", "_paralinguistic.jsonl"))
            trans = jsonl_metadata(fetch(mix["aslp_id"], mix["aslp_revision"], trans_name, missing_ok=True), trans_name)
            para = jsonl_metadata(fetch(mix["aslp_id"], mix["aslp_revision"], para_name, missing_ok=True), para_name)
            report["metadata_files"].append(dict(category=category, length=length,
                transcripts=len(trans or {}), paralinguistics=len(para or {}),
                missing_transcripts=trans is None, missing_paralinguistics=para is None))
            if trans is None or para is None:
                continue
            for clip, row in sorted(trans.items()):
                labels = para.get(clip, {}).get("model_paralinguistic")
                problem = metadata_problem(row, labels, mix)
                if problem:
                    report["metadata_exclusions"][problem] += 1
                    report.setdefault("metadata_rejections", []).append(dict(source_path=prefix+"audio/"+clip, reason=problem))
                    continue
                path = prefix + "audio/" + clip
                identifier = fingerprint(dict(dataset=mix["aslp_id"], revision=mix["aslp_revision"], path=path))[:24]
                # Speaker IDs are not reliable global identities across categories.
                group = category + "/" + str(row["Speaker_id"])
                result[category].append(dict(id=identifier, source="aslp", category=category, group=group,
                    source_path=path, text=clean_text(row["Transcription"]),
                    metadata_duration=float(row["Duration_seconds"]), labels=labels))
    selected = []
    for category, candidates in result.items():
        # Metadata scores are almost always tied at .98; seeded hashing avoids a first-file bias.
        candidates.sort(key=lambda r: fingerprint([cfg["seed"], r["id"]]))
        limit = mix["candidate_limits"][category]
        if cfg.get("test_train_clips"):
            limit = max(20, cfg["test_train_clips"]*4)
        report["candidate_limits"][category] = dict(eligible=len(candidates), screened=min(limit, len(candidates)))
        selected += candidates[:limit]
    return selected


def convert_payload(payload, path):
    return convert_audio(dict(bytes=payload), path)


def screen_one(row, path, scorer, cache, acoustic_only=False):
    identity = fingerprint(dict(id=row["id"], sha256=sha256(path), text=row["text"], acoustic_only=acoustic_only))
    marker = cache / (identity + ".json")
    if marker.is_file():
        return read_json(marker)
    quality, problem = scorer.acoustic(path)
    if problem is None and not acoustic_only:
        alignment, problem = scorer.alignment(path, row["text"])
        quality.update(alignment)
    result = dict(quality=quality, rejection=problem, identity=identity, source=row["source"], id=row["id"],
                  provenance={k:row[k] for k in ("source_path", "source_shard", "source_row") if k in row})
    atomic_json(marker, result)
    return result


def screen_aslp(cfg, root, scorer, report):
    cache = root / "quality_cache"
    cache.mkdir(exist_ok=True)
    rows = []
    for index, row in enumerate(aslp_candidates(cfg, report)):
        path = root / "screening" / ("aslp-"+row["id"]+".wav")
        conversion = path.with_suffix(".conversion.json")
        if not path.is_file():
            payload = remote_bytes(cfg["mixture"]["aslp_id"], cfg["mixture"]["aslp_revision"], row["source_path"], missing_ok=True)
            if payload is None:
                report["screening_exclusions"]["aslp/missing_audio"] += 1
                continue
            try:
                atomic_json(conversion, convert_payload(payload, path))
            except (ValueError, RuntimeError):
                report["screening_exclusions"]["aslp/invalid_audio"] += 1
                continue
        properties = read_json(conversion)
        measured = properties["output_frames"]/24000
        properties["duration_seconds"] = measured
        if abs(measured-row["metadata_duration"]) > max(.5, .05*measured):
            report["screening_exclusions"]["aslp/duration_mismatch"] += 1
            path.unlink(missing_ok=True)
            continue
        score = screen_one(row, path, scorer, cache)
        if score["rejection"]:
            report["screening_exclusions"]["aslp/"+score["rejection"]] += 1
            path.unlink(missing_ok=True)
            continue
        rows.append(dict(**row, quality=score["quality"],
                         screening_path=path.relative_to(root).as_posix(), sha256=sha256(path), **properties))
        if (index+1) % 25 == 0:
            LOG.info("ASLP screened %d; accepted %d", index+1, len(rows))
    return rows


def scan_tts(cfg, root, scorer, target_seconds, report):
    """Score every training clip acoustically; ASR-check a bounded top shortlist.

    Only shortlisted WAVs persist. A per-shard cursor + hashes resumes screening.
    The released eval split supplies validation only; test is never downloaded.
    """
    mix, pool = cfg["mixture"], []
    state_path = root / "tts_scan_state.json"
    state = read_json(state_path) if state_path.is_file() else dict(done=[], pool=[], exclusions={}, scanned=0)
    if not all((root/r["screening_path"]).is_file() and sha256(root/r["screening_path"]) == r["sha256"]
               for r in state["pool"]):
        raise RuntimeError("UrduTTS shortlist damaged; use a new preparation directory")
    pool = state["pool"]
    budget = target_seconds*mix["tts_shortlist_multiplier"]
    for split in ("train", "eval"):
        source = fetch_json(cfg, split+"/state.json")
        consumed = sum(1 for r in pool if r.get("source_split") == split)
        for shard_index, file in enumerate(source["_data_files"]):
            filename = split+"/"+file["filename"]
            contained(root, filename)
            if filename in state["done"]:
                continue
            # Smoke scans enough candidates to exercise screening without the whole 64 h release.
            limit = mix["tts_scan_limit"] if split == "train" else mix["validation_clips_per_source"]*10
            if cfg.get("test_train_clips"):
                limit = max(100, cfg["test_train_clips"]*10)
            for index, original in stream_rows(cfg, filename, limit):
                state["scanned"] += 1
                text = original.get(cfg["dataset_text_column"])
                problem = text_problem(text, mix)
                if problem:
                    state["exclusions"][problem] = state["exclusions"].get(problem, 0)+1
                    continue
                identifier = fingerprint(dict(revision=cfg["dataset_revision"], shard=filename, row=index))[:24]
                row = dict(id=identifier, source="urdutts", category="read_speech", text=clean_text(text),
                           source_shard=filename, source_row=index, source_split=split)
                path = root / "screening" / ("urdutts-"+identifier+".wav")
                try:
                    properties = convert_audio(original.get("audio"), path)
                except (ValueError, RuntimeError):
                    state["exclusions"]["invalid_audio"] = state["exclusions"].get("invalid_audio", 0)+1
                    continue
                score = screen_one(row, path, scorer, root/"quality_cache", acoustic_only=True)
                if score["rejection"]:
                    state["exclusions"][score["rejection"]] = state["exclusions"].get(score["rejection"], 0)+1
                    path.unlink(missing_ok=True)
                    continue
                row.update(**properties, quality=score["quality"], sha256=sha256(path),
                           screening_path=path.relative_to(root).as_posix())
                pool.append(row)
                # Bounded shortlist by audio seconds for train; bounded clip count for eval.
                partition = [r for r in pool if r["source_split"] == split]
                keep = best_with_budget(partition, budget if split == "train" else float("inf"),
                                       max_clips=mix["validation_clips_per_source"]*3 if split == "eval" else None)
                keep_ids = {r["id"] for r in keep}
                # Stale checkpoint references survive until the next atomic cursor write.
                pool = [r for r in pool if r["source_split"] != split or r["id"] in keep_ids]
                if index % 250 == 0:
                    LOG.info("UrduTTS acoustic scan %s row %d; shortlist %d clips", filename, index, len(pool))
            state.update(pool=pool)
            state["done"].append(filename)
            atomic_json(state_path, state)
            # Delete only registered generated candidates after saving the new shortlist.
            protected = {r["screening_path"] for r in pool}
            for path in (root/"screening").glob("urdutts-*.wav"):
                if path.relative_to(root).as_posix() not in protected:
                    path.unlink()
            if limit is not None:
                break
    accepted = []
    for row in pool:
        result = screen_one(row, root/row["screening_path"], scorer, root/"quality_cache")
        if result["rejection"]:
            report["screening_exclusions"]["urdutts/"+result["rejection"]] += 1
        else:
            row["quality"] = result["quality"]
            accepted.append(row)
    report["tts_acoustic_scan"] = dict(scanned=state["scanned"], exclusions=state["exclusions"],
        completed_shards=state["done"], asr_shortlist_clips=len(pool), asr_accepted_clips=len(accepted),
        scope="Full training acoustic scan; ASR only on top bounded shortlist. Eval is bounded; test untouched.")
    return accepted


def regenerate(root, report, cfg):
    """Recreate cleaned-up selected WAVs without repeating quality selection."""
    by_shard = defaultdict(list)
    for split, rows in report["selected"].items():
        for row in rows:
            path = root/row["audio"]
            if path.is_file() and sha256(path) == row["sha256"]:
                continue
            if row["source"] == "aslp":
                convert_payload(remote_bytes(cfg["mixture"]["aslp_id"], cfg["mixture"]["aslp_revision"], row["source_path"]), path)
                if sha256(path) != row["sha256"]:
                    raise RuntimeError("Regenerated ASLP waveform differs")
            else:
                by_shard[row["source_shard"]].append(row)
    for shard, rows in by_shard.items():
        lookup = {r["source_row"]: r for r in rows}
        for index, original in stream_rows(cfg, shard, max(lookup)+1):
            if index in lookup:
                row = lookup.pop(index)
                convert_audio(original["audio"], root/row["audio"])
                if sha256(root/row["audio"]) != row["sha256"]:
                    raise RuntimeError("Regenerated UrduTTS waveform differs")
            if not lookup:
                break
        if lookup:
            raise RuntimeError("Missing UrduTTS source rows on regeneration")


def restore_selection(cfg, work, report_path):
    """Restore selected rows/manifest hashes; an optimizer resume must not rerank."""
    recipe = recipe_for(cfg)
    report = read_json(report_path)
    if report["recipe"] != recipe:
        raise RuntimeError("Saved mixed-data selection does not match the preparation recipe")
    root = Path(work)/"datasets"/fingerprint(recipe)
    root.mkdir(parents=True, exist_ok=True)
    for split, rows in report["selected"].items():
        write_manifest(root/f"{split}.csv", rows)
        if sha256(root/f"{split}.csv") != report["splits"][split]["manifest_sha256"]:
            raise RuntimeError("Restored selection manifest checksum differs")
    atomic_json(root/"parts/selected.json", dict(identity=dict(recipe=root.name),
                rows=[r for rows in report["selected"].values() for r in rows]))
    atomic_json(root/"dataset_source.json", report)
    return root


def prepare_dataset(cfg, work, scorer_factory=QualityScorer):
    validate_config(cfg)
    recipe = recipe_for(cfg)
    root = Path(work)/"datasets"/fingerprint(recipe)
    root.mkdir(parents=True, exist_ok=True)
    if (root/"dataset_source.json").is_file():
        report = read_json(root/"dataset_source.json")
        if report["recipe"] != recipe:
            raise RuntimeError("Preparation identity changed")
        regenerate(root, report, cfg)
        return root.resolve()
    (root/"screening").mkdir(exist_ok=True)
    (root/"quality_cache").mkdir(exist_ok=True)
    report = dict(recipe=recipe, metadata_files=[], metadata_exclusions=Counter(), candidate_limits={},
                  screening_exclusions=Counter(), conversion_exclusions=[], splits={}, test_downloaded=False)
    scorer = scorer_factory(cfg["mixture"], Path(work)/"quality_models")
    aslp = screen_aslp(cfg, root, scorer, report)
    report["aslp_quality_accepted"] = source_summary(aslp)
    atomic_json(root/"selection_progress.json", report)
    mix = cfg["mixture"]
    train_aslp = select_aslp([r for r in aslp if not group_is_validation(r["group"], mix["validation_fraction"], cfg["seed"])], mix)
    val_aslp = best_with_budget([r for r in aslp if group_is_validation(r["group"], mix["validation_fraction"], cfg["seed"])],
                               float("inf"), max_clips=mix["validation_clips_per_source"])
    if not cfg.get("test_train_clips") and sum(r["duration_seconds"] for r in train_aslp)/3600 < mix["minimum_aslp_hours"]:
        raise ValueError("Too little qualifying ASLP; inspect quality reports rather than adding noisy clips")
    aslp_seconds = sum(r["duration_seconds"] for r in train_aslp)
    target_tts = aslp_seconds*(1-mix["aslp_fraction"])/mix["aslp_fraction"]
    tts = scan_tts(cfg, root, scorer, target_tts, report)
    atomic_json(root/"selection_progress.json", report)
    train_tts = best_with_budget([r for r in tts if r["source_split"] == "train"], target_tts)
    val_tts = best_with_budget([r for r in tts if r["source_split"] == "eval"], float("inf"),
                               max_clips=mix["validation_clips_per_source"])
    if not all((train_aslp, train_tts, val_aslp, val_tts)):
        raise ValueError("Both sources must retain train AND validation clips after all quality gates")
    # Protect held-out text/audio before assembling training; remove duplicates conservatively.
    selected = {"validation": val_aslp+val_tts, "train": train_aslp+train_tts}
    seen_text, seen_audio = set(), set()
    for split in ("validation", "train"):
        unique = []
        for row in selected[split]:
            key = clean_text(row["text"])
            if key in seen_text or row["sha256"] in seen_audio:
                report["screening_exclusions"]["duplicate_text_or_audio"] += 1
                continue
            seen_text.add(key)
            seen_audio.add(row["sha256"])
            unique.append(row)
        selected[split] = unique
    # Rebalance after dedup without replacement or relaxing any quality threshold.
    train_aslp = select_aslp([r for r in selected["train"] if r["source"] == "aslp"], mix)
    train_tts = [r for r in selected["train"] if r["source"] == "urdutts"]
    actual_tts = sum(r["duration_seconds"] for r in train_tts)
    rebalance_cfg = dict(mix, max_aslp_hours=min(sum(r["duration_seconds"] for r in train_aslp),
                         actual_tts*mix["aslp_fraction"]/(1-mix["aslp_fraction"]))/3600)
    train_aslp = select_aslp(train_aslp, rebalance_cfg)
    selected["train"] = train_aslp+best_with_budget(train_tts,
                            sum(r["duration_seconds"] for r in train_aslp)*(1-mix["aslp_fraction"])/mix["aslp_fraction"])
    for split, limit in (("train", cfg.get("test_train_clips")), ("validation", cfg.get("test_validation_clips"))):
        if limit:
            # Smoke mode remains source-stratified; it never seeds the full-data run.
            fraction = mix["aslp_fraction"]
            a_count = max(1, min(limit-1, round(limit*fraction)))
            selected[split] = best_with_budget([r for r in selected[split] if r["source"] == "aslp"], float("inf"), max_clips=a_count) + \
                              best_with_budget([r for r in selected[split] if r["source"] == "urdutts"], float("inf"), max_clips=limit-a_count)
    for split, rows in selected.items():
        if {r["source"] for r in rows} != {"aslp", "urdutts"}:
            raise ValueError("Empty source partition after deduplication")
        for row in rows:
            row["audio"] = f"audio/{split}/{row['source']}-{row['id']}.wav"
            target = root/row["audio"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root/row["screening_path"], target)
        write_manifest(root/f"{split}.csv", rows)
        report["splits"][split] = dict(clips=len(rows), hours=sum(r["duration_seconds"] for r in rows)/3600,
                                      composition=source_summary(rows), manifest_sha256=sha256(root/f"{split}.csv"))
    validate_training_mix(selected["train"], mix, smoke=bool(cfg.get("test_train_clips")))
    report["selected"] = selected
    report["limitations"] = ("Automated quality proxies, not certified Urdu MOS or verified speaker diarization. "
        "ASLP holdout groups are category+speaker ID; global speaker disjointness is unverified. "
        "UrduTTS eval is utterance-held-out and has already been used for epoch-1 monitoring. "
        "No emotion labels or speaker-control tokens are introduced.")
    (root/"parts").mkdir(exist_ok=True)
    atomic_json(root/"parts/selected.json", dict(identity=dict(recipe=root.name),
                rows=[r for rows in selected.values() for r in rows]))
    atomic_json(root/"dataset_source.json", report)
    # Reproducible, registered screening candidates are not required by training.
    for path in (root/"screening").glob("*.wav"):
        if re.fullmatch(r"(?:aslp|urdutts)-[a-f0-9]{24}\.wav", path.name):
            path.unlink()
    LOG.info("Mixed data selected: %s", report["splits"])
    return root.resolve()
