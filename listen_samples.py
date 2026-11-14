#!/usr/bin/env python3
"""Generate neutral listening samples for several adapters/checkpoints (no ASR) for judging by ear.

  python listen_samples.py --output /kaggle/working/listen \
      --adapter "epoch1 step 5029=hassan-31x/orpheus-urdutts-checkpoints:urdutts-r32-lr1e4-seed3407-epoch1:checkpoint-5029" ...

Same sentences and seeds for every adapter. Voice-tagged adapters get --voice; plain adapters get plain text.
Writes <output>/index.json (label, sentence, take, audio path) which the notebook renders as audio players.
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

from eval_pronunciation import DEFAULT_VOICE, fetch_adapter
from orpheus_utils import atomic_json, read_json


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--adapter", action="append", required=True, help="'display label=repo:run_id:checkpoint'")
    p.add_argument("--prompts", type=Path, default=Path(__file__).parent / "listening_prompts.csv")
    p.add_argument("--repeats", type=int, default=2, help="Takes per sentence (different seeds)")
    p.add_argument("--voice", default=DEFAULT_VOICE, help="Prompt prefix for voice-tagged adapters")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    repo = Path(__file__).resolve().parent
    args.output.mkdir(parents=True, exist_ok=True)
    with args.prompts.open(encoding="utf-8-sig") as f:
        texts = [r["text"] for r in csv.DictReader(f)]
    index = dict(sentences=texts, repeats=args.repeats, voice=args.voice, samples=[])
    for n, spec in enumerate(args.adapter):
        display, rest = spec.split("=", 1)
        label, adapter = fetch_adapter(f"a{n}={rest}", args.output / "adapters")
        out = args.output / "audio" / label
        tagged = bool(read_json(adapter / "run_identity.json")["config"].get("speaker_column"))
        manifest = args.output / f"prompts-{label}.csv"
        with manifest.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, ["text", "speaker"])
            w.writeheader()
            for _ in range(args.repeats):
                w.writerows(dict(text=t, speaker=args.voice if tagged else "") for t in texts)
        subprocess.run([sys.executable, str(repo / "synthesize.py"), "--adapter", str(adapter),
                        "--manifest", str(manifest), "--output", str(out), "--seed", "3407"], check=True, cwd=repo)
        for i in range(args.repeats * len(texts)):
            meta = read_json(out / f"utterance-{i:05d}.json")
            index["samples"].append(dict(label=display, tagged=tagged, sentence=i % len(texts), take=i // len(texts) + 1,
                                         audio=str((out / f"utterance-{i:05d}.wav").relative_to(args.output)),
                                         seconds=round(meta["duration_seconds"], 1),
                                         stopped=bool(meta.get("ended_with_speech_stop"))))
        atomic_json(args.output / "index.json", index)  # partial results survive a crash


if __name__ == "__main__":
    main()
