#!/usr/bin/env python3
"""Compare adapters on pronunciation: same prompts and seeds, Whisper large-v3 transcripts, CER/WER.

  python eval_pronunciation.py --output /kaggle/working/pron-eval \
      --adapter epoch1=hassan-31x/orpheus-urdutts-checkpoints:urdutts-r32-lr1e4-seed3407-epoch1:checkpoint-5029 \
      --adapter epoch2=hassan-31x/orpheus-urdu-stage2-checkpoints:stage2-r32-lr1e4-seed3407-epoch2:checkpoint-3382

Each adapter is a checkpoint snapshot in a checkpoint repo (runs/<run>/snapshots/<checkpoint>-*/<checkpoint>/).
Adapters trained with a speaker column are prompted with --voice; plain adapters get plain text.
Whisper also mis-hears real Urdu speech (~25% CER on human clips), so compare adapters with each other,
not against zero.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

from orpheus_utils import atomic_json, read_json
from pron_words import matched_words, normalize

DEFAULT_VOICE = "urdutts_narrator | unknown | audiobook | neutral"


def distance(a, b):
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (x != y)))
        previous = current
    return previous[-1]


def summarize(per, repeats):
    """Accuracy plus seed consistency; per is ordered repeat-major (all prompts, then all again)."""
    cers = sorted(r["cer"] for r in per)
    n = len(per) // repeats
    groups = [[per[k * n + i] for k in range(repeats)] for i in range(n)]
    both = total = 0
    gaps = []
    for g in groups:
        ref = g[0]["reference_normalized"].split()
        ok = [matched_words(ref, r["hypothesis_normalized"].split()) for r in g]
        both += sum(all(o[k] for o in ok) for k in range(len(ref)))
        total += len(ref)
        gaps.append(max(r["cer"] for r in g) - min(r["cer"] for r in g))
    return dict(samples=len(per),
                corpus_cer=sum(r["char_errors"] for r in per) / sum(r["ref_chars"] for r in per),
                corpus_wer=sum(r["word_errors"] for r in per) / sum(r["ref_words"] for r in per),
                median_cer=cers[len(cers) // 2],
                samples_cer_under_20pct=sum(c < 0.2 for c in cers) / len(cers),
                words_correct_in_every_seed=both / max(1, total),
                mean_seed_cer_gap=sum(gaps) / max(1, len(gaps)),
                stopped_cleanly=sum(bool(r["stopped"]) for r in per) / len(per),
                mean_seconds=sum(r["seconds"] for r in per) / len(per))


def fetch_adapter(spec, root, token=None):
    """spec = label=repo:run_id:checkpoint -> local directory with weights and run identity."""
    from huggingface_hub import HfApi, hf_hub_download
    label, rest = spec.split("=", 1)
    repo, run_id, checkpoint = rest.split(":")
    prefix = f"runs/{run_id}/snapshots/{checkpoint}-"
    names = [f for f in HfApi(token=token).list_repo_files(repo) if f.startswith(prefix)]
    wanted = {}
    for name in names:  # snapshots/<checkpoint>-<hash>/<checkpoint>/<file>
        parts = name.split("/")
        if len(parts) == 6 and parts[4] == checkpoint and parts[5] in (
                "adapter_config.json", "adapter_model.safetensors", "run_identity.json", "COMPLETE.json"):
            wanted.setdefault(parts[3], {})[parts[5]] = name
    complete = [files for files in wanted.values() if len(files) == 4]
    if not complete:
        raise RuntimeError(f"No complete {checkpoint} snapshot for {run_id} in {repo}")
    target = Path(root) / label
    target.mkdir(parents=True, exist_ok=True)
    for base, name in complete[0].items():
        source = hf_hub_download(repo, name, token=token)
        (target / base).write_bytes(Path(source).read_bytes())
        Path(source).resolve().unlink(missing_ok=True)  # drop the Hub-cache copy too; adapters are not reused
    return label, target


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--adapter", action="append", required=True, help="label=repo:run_id:checkpoint")
    p.add_argument("--prompts", type=Path, default=Path(__file__).parent / "benchmark_prompts.csv")
    p.add_argument("--repeats", type=int, default=2, help="Samples per prompt (different seeds)")
    p.add_argument("--voice", default=DEFAULT_VOICE, help="Prompt prefix for voice-tagged adapters")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--asr-model", default="openai/whisper-large-v3")
    args = p.parse_args()
    repo = Path(__file__).resolve().parent
    args.output.mkdir(parents=True, exist_ok=True)
    with args.prompts.open(encoding="utf-8-sig") as f:
        prompt_rows = list(csv.DictReader(f))
    texts = [r["text"] for r in prompt_rows]
    sets = [r.get("set") or "all" for r in prompt_rows]
    adapters = [(spec.split("=", 1)[0], None) for spec in args.adapter]

    for spec in args.adapter:
        label = spec.split("=", 1)[0]
        out = args.output / "audio" / label
        if (out / "generated.csv").is_file() and len(list(out.glob("utterance-*.json"))) == len(texts) * args.repeats:
            continue  # resumable: this adapter is already generated
        # One adapter at a time: each is ~2 GB, and eleven together overflow Kaggle's 20 GB working disk.
        label, adapter = fetch_adapter(spec, args.output / "adapters")
        tagged = bool(read_json(adapter / "run_identity.json")["config"].get("speaker_column"))
        manifest = args.output / f"prompts-{label}.csv"
        with manifest.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, ["text", "speaker"])
            w.writeheader()
            for _ in range(args.repeats):
                w.writerows(dict(text=t, speaker=args.voice if tagged else "") for t in texts)
        subprocess.run([sys.executable, str(repo / "synthesize.py"), "--adapter", str(adapter),
                        "--manifest", str(manifest), "--output", str(out), "--seed", "3407"], check=True, cwd=repo)
        shutil.rmtree(adapter)

    # ASR after all generation, so Orpheus and Whisper never share the GPU.
    import numpy as np
    import soundfile as sf
    import torch
    from math import gcd
    from scipy.signal import resample_poly
    from transformers import pipeline
    asr = pipeline("automatic-speech-recognition", model=args.asr_model, torch_dtype=torch.float16, device=0)
    records, summary = [], {}
    for label, _ in adapters:
        out = args.output / "audio" / label
        with (out / "generated.csv").open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        per = []
        for row in rows:
            signal, sr = sf.read(out / row["audio"], dtype="float32")
            d = gcd(sr, 16000)
            signal = resample_poly(signal, 16000 // d, sr // d).astype(np.float32)
            hyp = asr(dict(raw=signal, sampling_rate=16000), generate_kwargs=dict(language="urdu", task="transcribe"))["text"]
            meta = read_json((out / row["audio"]).with_suffix(".json"))
            ref, h = normalize(row["text"]), normalize(hyp)
            rec = dict(adapter=label, audio=row["audio"], reference=row["text"], hypothesis=hyp,
                       prompt_set=sets[len(per) % len(texts)], reference_normalized=ref, hypothesis_normalized=h,
                       char_errors=distance(ref, h), ref_chars=len(ref),
                       word_errors=distance(ref.split(), h.split()), ref_words=len(ref.split()),
                       seconds=meta["duration_seconds"], stopped=meta.get("ended_with_speech_stop"),
                       hit_token_budget=meta.get("reached_token_budget"))
            rec.update(cer=rec["char_errors"] / rec["ref_chars"], wer=rec["word_errors"] / rec["ref_words"])
            per.append(rec)
        records += per
        summary[label] = summarize(per, args.repeats)
        if len(set(sets)) > 1:
            n = len(texts)
            summary[label]["by_set"] = {name: summarize([r for i, r in enumerate(per) if sets[i % n] == name], args.repeats)
                                        for name in sorted(set(sets))}
        print(label, json.dumps(summary[label]), flush=True)
    atomic_json(args.output / "per_utterance.json", records)
    atomic_json(args.output / "summary.json", dict(asr_model=args.asr_model, prompts=len(texts),
                                                   repeats=args.repeats, adapters=summary))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
