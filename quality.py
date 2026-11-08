"""Conservative speech screening, not a calibrated Urdu naturalness metric."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
import re
import unicodedata

from orpheus_utils import atomic_json, read_json, sha256
from pipeline_recovery import retry_io

DNS_REVISION = "591184a9fcb2cbdec02520fed81a32bbbf9d73ff"
DNS_URL = f"https://raw.githubusercontent.com/microsoft/DNS-Challenge/{DNS_REVISION}/DNSMOS/DNSMOS/sig_bak_ovr.onnx"
DNS_SHA256 = "269fbebdb513aa23cddfbb593542ecc540284a91849ac50516870e1ac78f6edd"


def normalize_asr(text):
    """Comparison only: preserve original training text, pauses and repetitions."""
    text = unicodedata.normalize("NFKC", text).translate(str.maketrans("كيىۀة", "کییہہ"))
    text = "".join(c if unicodedata.category(c)[0] in "LN" else " " for c in text
                   if unicodedata.category(c) != "Mn")
    return " ".join(text.split())


def edit_distance(a, b):
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j-1] + (x != y)))
        previous = current
    return previous[-1]


def transcript_errors(reference, hypothesis):
    reference, hypothesis = normalize_asr(reference), normalize_asr(hypothesis)
    a, b = reference.replace(" ", ""), hypothesis.replace(" ", "")
    return dict(cer=edit_distance(a, b) / max(1, len(a)),
                wer=edit_distance(reference.split(), hypothesis.split()) / max(1, len(reference.split())))


def text_problem(text, cfg):
    if not isinstance(text, str) or not text.strip():
        return "missing_text"
    letters = [c for c in text if unicodedata.category(c).startswith("L")]
    urdu = sum("\u0600" <= c <= "\u06ff" for c in letters)
    if not letters or urdu / len(letters) < cfg["minimum_urdu_letter_fraction"]:
        return "script_or_codeswitch"
    if len(text) > cfg["maximum_chars"]:
        return "text_length"
    if re.search(r"https?://|\[[^]]+\]|<[^>]+>|(?:transcription|subtitle)", text, re.I):
        return "transcript_annotation"
    words = text.split()
    if len(words) < 3:
        return "too_few_words"
    # Preserve conversational repetitions; reject only pathological loops.
    if len(words) >= 12 and len(set(words)) <= 2:
        return "transcript_loop"
    return None


def metadata_problem(row, para, cfg):
    problem = text_problem(row.get("Transcription"), cfg)
    if problem:
        return problem
    try:
        duration, confidence = float(row["Duration_seconds"]), float(row["Confidence_score"])
    except (KeyError, ValueError, TypeError):
        return "missing_numeric_metadata"
    if not math.isfinite(confidence) or confidence < cfg["minimum_confidence"]:
        return "transcription_confidence"
    if not math.isfinite(duration) or not cfg["minimum_seconds"] <= duration <= cfg["maximum_seconds"]:
        return "duration"
    if not isinstance(para, dict):
        return "missing_paralinguistics"
    accent = str(para.get("accent", "")).strip().lower().rstrip(".")
    if accent != "standard pakistani urdu":
        return "accent"
    # Narrow positive acceptance of standard accent; no inferred regional relabelling.
    description = " ".join(str(para.get(k, "")) for k in
                           ("speed", "rhythm", "tone", "texture", "paralinguistic", "contextual"))
    # Remove explicitly negated noise statements before detecting unwanted events.
    description = re.sub(r"\b(?:no|without)\s+(?:noticeable\s+|audible\s+)?(?:background\s+)?"
                         r"(?:noise|music|overlap|distortion)\b", "", description, flags=re.I)
    if re.search(r"\b(sung|singing|song|musical|recitation|poetry|poetic|monologue|narration|narrator|"
                 r"shouting|screaming|yelling|overlapping|distorted|muffled)\b|background music|"
                 r"multiple speakers|two speakers|speakers talking", description, re.I):
        return "unsuitable_speech_style"
    if not re.search(r"clear|articulat|enunciat|crisp|distinct", str(para.get("pronunciation", "")), re.I):
        return "uncertain_pronunciation"
    return None


def to_16k(wave, rate):
    import numpy as np
    from scipy.signal import resample_poly
    divisor = math.gcd(rate, 16000)
    return resample_poly(wave, 16000 // divisor, rate // divisor).astype(np.float32) if rate != 16000 else wave


def acoustic_metrics(wave, rate):
    import numpy as np
    if wave.ndim != 1 or not len(wave) or not np.isfinite(wave).all():
        raise ValueError("Invalid mono waveform")
    return dict(duration_seconds=len(wave)/rate, peak=float(np.max(np.abs(wave))),
                rms_db=float(20*np.log10(max(1e-12, np.sqrt(np.mean(wave**2))))),
                clipped_fraction=float(np.mean(np.abs(wave) >= .999)))


class QualityScorer:
    def __init__(self, cfg, model_dir):
        import onnxruntime as ort
        import requests
        self.cfg = cfg
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        model = model_dir / "sig_bak_ovr.onnx"
        if not model.is_file():
            def download():
                response = requests.get(DNS_URL, timeout=(30, 180))
                response.raise_for_status()
                temp = model.with_suffix(".partial")
                temp.write_bytes(response.content)
                temp.replace(model)
            retry_io(download)
        if sha256(model) != DNS_SHA256:
            raise RuntimeError("DNSMOS model checksum mismatch")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        self.dns = ort.InferenceSession(str(model), sess_options=options, providers=["CPUExecutionProvider"])
        atomic_json(model_dir / "quality_models.json", dict(dnsmos_url=DNS_URL, dnsmos_sha256=sha256(model),
                    asr_id=cfg["asr_model"], asr_revision=cfg["asr_revision"],
                    dnsmos_variant="non-personalized P.835 SIG/BAK/OVRL; not P.808"))
        self.asr = None

    def acoustic(self, path):
        import numpy as np
        import soundfile as sf
        wave, rate = sf.read(path, dtype="float32")
        result = acoustic_metrics(wave, rate)
        cfg = self.cfg
        if not cfg["minimum_seconds"] <= result["duration_seconds"] <= cfg["maximum_seconds"]:
            return result, "duration"
        if result["clipped_fraction"] > cfg["maximum_clipped_fraction"]:
            return result, "clipping"
        if not cfg["minimum_rms_db"] <= result["rms_db"] <= cfg["maximum_rms_db"]:
            return result, "loudness"
        wave = to_16k(wave, rate)
        from faster_whisper.vad import get_speech_timestamps, VadOptions
        segments = get_speech_timestamps(wave, VadOptions(min_silence_duration_ms=500), sampling_rate=16000)
        result["vad_fraction"] = sum(s["end"]-s["start"] for s in segments) / len(wave)
        if result["vad_fraction"] < cfg["minimum_vad_fraction"]:
            return result, "insufficient_speech"
        window = int(9.01*16000)
        # Reference DNSMOS repeats short clips for scoring; saved training WAV is unchanged.
        while len(wave) < window:
            wave = np.concatenate([wave, wave])
        scores = []
        for start in range(0, len(wave)-window+1, 16000):
            scores.append(self.dns.run(None, {self.dns.get_inputs()[0].name: wave[None, start:start+window]})[0][0])
        raw = np.asarray(scores)
        sig = np.polyval([-.08397278, 1.22083953, .0052439], raw[:, 0]).mean()
        bak = np.polyval([-.13166888, 1.60915514, -.39604546], raw[:, 1]).mean()
        ovrl = np.polyval([-.06766283, 1.11546468, .04602535], raw[:, 2]).mean()
        result.update(dnsmos_sig=float(sig), dnsmos_bak=float(bak), dnsmos_ovrl=float(ovrl))
        if not all(np.isfinite(v) for v in result.values()):
            raise RuntimeError("Nonfinite quality score")
        for name in ("sig", "bak", "ovrl"):
            if result[f"dnsmos_{name}"] < cfg[f"minimum_dnsmos_{name}"]:
                return result, "dnsmos_" + name
        return result, None

    def alignment(self, path, text):
        import soundfile as sf
        from faster_whisper import WhisperModel
        from huggingface_hub import snapshot_download
        if self.asr is None:
            model = retry_io(lambda: snapshot_download(self.cfg["asr_model"], revision=self.cfg["asr_revision"],
                             allow_patterns=["config.json", "model.bin", "preprocessor_config.json", "tokenizer.json", "vocabulary.*"]))
            device = self.cfg["quality_device"]
            self.asr = WhisperModel(model, device=device, compute_type="int8_float16" if device == "cuda" else "int8",
                                    cpu_threads=4, num_workers=1)
        wave, rate = sf.read(path, dtype="float32")
        segments, _ = self.asr.transcribe(to_16k(wave, rate), language="ur", task="transcribe", beam_size=5,
                                         condition_on_previous_text=False, vad_filter=False, temperature=0)
        segments = list(segments)
        hypothesis = " ".join(s.text.strip() for s in segments)
        errors = transcript_errors(text, hypothesis)
        errors["asr_text"] = hypothesis
        problem = None if hypothesis and errors["cer"] <= self.cfg["maximum_cer"] and errors["wer"] <= self.cfg["maximum_wer"] else "asr_disagreement"
        return errors, problem


def quality_rank(row):
    q = row["quality"]
    # Prefer clean acoustics and close transcript agreement; all rows passed hard gates.
    return q["dnsmos_ovrl"] + .1*q["dnsmos_bak"] - q.get("cer", 0) - .2*q.get("wer", 0)
