"""Stage-2 v3 labels: per-clip voice tags (speaker | gender | style | emotion) and the epoch-3 selection.

Reads the pinned v2 manifests (manifests/{train,validation}.jsonl) and writes labels/v3.jsonl with
id, split, keep, voice. Emotion is mapped from ASLP's free-text annotation to a fixed vocabulary.
"""
import json, re, sys, collections
from pathlib import Path

EMOTIONS = [  # first keyword hit (by position in the description) wins
    ("angry", "angry anger agitated frustrat irritat annoy exasperat confront furious indignant accus defian hostil aggress scold outrag resent bitter"),
    ("sad", "sad sorrow melanchol grief mournful regret somber sombre dejected disappoint nostalg wistful hurt despair resign heartbr tearful lament"),
    ("fearful", "fear afraid anxious anxiety worried worry nervous scared panic apprehens distress alarm uneasy tense"),
    ("surprised", "surpris shock astonish incredul disbelief amazed stunned"),
    ("excited", "excit enthusias energetic triumph exuberan elat passionate animated thrill eager"),
    ("happy", "amus cheer joy happy playful lighthearted light-hearted humor humour jovial delight pleased warm affection teas laugh friendly content"),
    ("serious", "serious assert firm stern contemplat reflect thoughtful devotion reveren solemn matter-of-fact informat explan formal earnest persua authorit didactic instruct"),
    ("neutral", "neutral calm inquisit curious conversational casual polite"),
]
STYLE = {"urdutts": "audiobook"}
# Clips per category, epoch 3: keep all speech-like categories, cap acted dialogue to the cleanest.
ACTED = {"drama", "comedy_show", "film"}
ACTED_HOURS = 7.0
TERMINATORS = re.compile(r"[۔؟!?.]")


def emotion(text):
    if not text:
        return "neutral"
    t = text.lower()
    best = None
    for label, words in EMOTIONS:
        for w in words.split():
            i = t.find(w)
            if i >= 0 and (best is None or i < best[0]):
                best = (i, label)
    return best[1] if best else "neutral"


def gender(g):
    g = (g or "").strip().lower()
    return g if g in ("male", "female") else "unknown"


def voice(row):
    if row["source"] == "urdutts":
        return "urdutts_narrator | unknown | audiobook | neutral"
    p = json.loads(row["provenance"])
    spk = row["speaker"].replace("aslp-", "aslp_").replace("-SPEAKER_", "_")
    return f"{spk} | {gender(p.get('gender'))} | conversational | {emotion(p.get('emotion'))}"


def main(src, out):
    src = Path(src)
    out_rows, stats = [], collections.Counter()
    for split in ("train", "validation"):
        rows = [json.loads(l) for l in open(src / f"manifests/{split}.jsonl")]
        acted = [r for r in rows if r["source"] == "aslp" and r["category"] in ACTED]
        keep_acted = set()
        if split == "train":
            # Single-turn, transcript-plausible acted clips first (fewer sentence breaks = fewer speaker turns),
            # then cleanest audio, until the acted-speech budget is spent.
            def rank(r):
                turns = len(TERMINATORS.findall(r["text"].rstrip("۔؟!?. ")))
                cps = len(r["text"]) / max(r["duration"], 1e-3)
                return (turns > 1, not 6 <= cps <= 18, -r["dnsmos_ovrl"])
            budget = ACTED_HOURS * 3600
            for r in sorted(acted, key=rank):
                if budget <= 0:
                    break
                keep_acted.add(r["id"]); budget -= r["duration"]
        else:
            keep_acted = {r["id"] for r in acted}
        for r in rows:
            keep = r["source"] == "urdutts" or r["category"] not in ACTED or r["id"] in keep_acted
            v = voice(r)
            out_rows.append(dict(id=r["id"], split=split, keep=keep, voice=v))
            if keep:
                stats[(split, r["source"])] += r["duration"] / 3600
                stats[(split, "emotion", v.rsplit(" | ", 1)[1])] += 1
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    for k, v in sorted(stats.items(), key=str):
        print(k, round(v, 2))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
