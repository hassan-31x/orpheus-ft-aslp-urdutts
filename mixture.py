"""Selection by measured seconds and a deterministic one-pass curriculum."""
from collections import defaultdict
import hashlib
import random

from quality import quality_rank


def group_is_validation(group, fraction, seed):
    value = int(hashlib.sha256(f"{seed}:{group}".encode()).hexdigest()[:16], 16) / 2**64
    return value < fraction


def best_with_budget(rows, budget, speaker_cap=float("inf"), max_clips=None):
    kept, seconds, speakers = [], 0., defaultdict(float)
    for row in sorted(rows, key=lambda r: (-quality_rank(r), r["id"])):
        duration = row["duration_seconds"]
        group = row.get("group", row["id"])
        if seconds + duration > budget or speakers[group] + duration > speaker_cap:
            continue
        kept.append(row)
        seconds += duration
        speakers[group] += duration
        if max_clips and len(kept) >= max_clips:
            break
    return kept


def select_aslp(rows, cfg):
    """Core conversational speech determines scale; never fill with narration."""
    core = [r for r in rows if r["category"] in ("interview", "podcast")]
    core = best_with_budget(core, cfg["max_aslp_hours"]*3600, cfg["speaker_max_seconds"])
    core_seconds = sum(r["duration_seconds"] for r in core)
    maximum = min(cfg["max_aslp_hours"]*3600, core_seconds / cfg["minimum_core_fraction"])
    selected = list(core)
    for category, fraction in cfg["category_caps"].items():
        selected += best_with_budget([r for r in rows if r["category"] == category],
                                     min(maximum*fraction, maximum-sum(r["duration_seconds"] for r in selected)),
                                     cfg["speaker_max_seconds"])
    if not selected:
        raise ValueError("No high-quality ASLP interview/podcast core; thresholds are not relaxed automatically")
    # Absolute caps use actual final seconds, not the larger provisional budget.
    # Keep all core rows; choose a feasible total from available category capacities.
    for _ in range(12):
        total = sum(r["duration_seconds"] for r in selected)
        reduced = list(core)
        for category, fraction in cfg["category_caps"].items():
            reduced += best_with_budget([r for r in selected if r["category"] == category],
                                         total*fraction, cfg["speaker_max_seconds"])
        if len(reduced) == len(selected):
            break
        selected = reduced
    return selected


def curriculum_order(rows, seed, fractions=(.5, .6, .7)):
    """Unique clips only; three equal-duration sections with increasing ASLP share.

    A deterministic sampler visits this order without Trainer reshuffling it.
    Clip granularity can cause small deviations, reported for every section.
    """
    queues = {s: [i for i, row in enumerate(rows) if row["source"] == s] for s in ("aslp", "urdutts")}
    rng = random.Random(seed)
    for values in queues.values():
        rng.shuffle(values)
    seconds = sum(r["duration_seconds"] for r in rows)
    order, report, position = [], [], {s: 0 for s in queues}
    elapsed = 0.
    for stage, fraction in enumerate(fractions):
        goal = seconds*(stage+1)/len(fractions)
        used = {s: 0. for s in queues}
        while elapsed < goal or (stage == len(fractions)-1 and len(order) < len(rows)):
            available = [s for s in queues if position[s] < len(queues[s])]
            if not available:
                break
            total = sum(used.values())
            preferred = "aslp" if used["aslp"] <= fraction*total else "urdutts"
            source = preferred if preferred in available else available[0]
            index = queues[source][position[source]]
            position[source] += 1
            order.append(index)
            duration = rows[index]["duration_seconds"]
            used[source] += duration
            elapsed += duration
        report.append(dict(stage=stage+1, target_aslp_fraction=fraction, seconds=used,
                           actual_aslp_fraction=used["aslp"]/max(1, sum(used.values()))))
    if len(order) != len(rows) or len(set(order)) != len(rows):
        raise RuntimeError("Curriculum must include every selected clip exactly once")
    return order, report


def source_summary(rows):
    seconds = defaultdict(float)
    counts = defaultdict(int)
    for r in rows:
        key = r["source"] + "/" + r.get("category", "read_speech")
        seconds[key] += r["duration_seconds"]
        counts[key] += 1
    total = sum(seconds.values())
    return {k: dict(clips=counts[k], hours=v/3600, fraction=v/max(total, 1)) for k, v in sorted(seconds.items())}


def validate_training_mix(rows, cfg, smoke=False):
    aslp = [r for r in rows if r["source"] == "aslp"]
    tts = [r for r in rows if r["source"] == "urdutts"]
    if not aslp or not tts:
        raise ValueError("Both ASLP and UrduTTS must remain in training")
    a = sum(r["duration_seconds"] for r in aslp)
    t = sum(r["duration_seconds"] for r in tts)
    if smoke:
        return
    if a/3600 < cfg["minimum_aslp_hours"]:
        raise ValueError("Too little qualifying conversational adaptation data after exclusions")
    if abs(a/(a+t)-cfg["aslp_fraction"]) > .03:
        raise ValueError("Actual audio-hour mixture drifted by more than 3 percentage points")
    core = sum(r["duration_seconds"] for r in aslp if r["category"] in ("interview", "podcast"))
    if core/a+1e-9 < cfg["minimum_core_fraction"]:
        raise ValueError("Interview/podcast core fell below its minimum after exclusions")
    for category, cap in cfg["category_caps"].items():
        if sum(r["duration_seconds"] for r in aslp if r["category"] == category)/a > cap+1e-9:
            raise ValueError("ASLP category cap exceeded after exclusions: " + category)
