"""Require a completed source and a matching remotely durable GPU smoke result."""
import math
from pathlib import Path

from checkpoint_init import inspect_source
from orpheus_utils import fingerprint, latest_checkpoint, read_json
from pipeline_recovery import retry_io


def recipe_key(cfg, commit):
    omit = {"run_id", "work_dir", "session_hours", "test_train_clips", "test_validation_clips",
            "save_steps", "eval_steps", "logging_steps", "sample_count"}
    return fingerprint(dict(commit=commit, config={k:v for k,v in cfg.items() if k not in omit}))


def source_and_smoke_gate(cfg, test, commit, api=None, download=None):
    from huggingface_hub import HfApi, hf_hub_download
    api, download = api or HfApi(), download or hf_hub_download
    if not test and cfg.get("init_source"):
        # Keep a started run's immutable parent revision across notebook restarts.
        # Only a genuinely missing preparation record permits fresh resolution.
        from huggingface_hub.utils import EntryNotFoundError
        try:
            previous = read_json(download(cfg["hf_repo_id"], "runs/"+cfg["run_id"]+"/preparation/resolved_config.json"))
        except EntryNotFoundError:
            previous = None
        if previous and previous.get("init_source"):
            request, recorded = cfg["init_source"], previous["init_source"]
            if any(request.get(k) != recorded.get(k) for k in ("repo_id", "run_id")):
                raise RuntimeError("Started run has a different parent; choose a new run ID")
            if request.get("revision") and request["revision"] != recorded.get("revision"):
                raise RuntimeError("Started run's parent revision cannot be changed")
            cfg["init_source"] = recorded
    source = cfg.get("init_source")
    if source:
        info, _ = inspect_source(source, api=api, download=download)
        if not info["complete"]:
            raise RuntimeError(f"Finish UrduTTS epoch 1 first: {info['checkpoint']} is {info['status']} at epoch {info['epoch']}")
        cfg["init_source"] = dict(repo_id=info["repo_id"], revision=info["revision"],
                                  run_id=info["run_id"], checkpoint=info["checkpoint"])
    if test:
        return
    name = "workflow/smoke-"+recipe_key(cfg, commit)+".json"
    try:
        receipt = read_json(download(cfg["hf_repo_id"], name))
    except Exception as exc:
        # No bypass when the result is absent or inaccessible.
        raise RuntimeError("No matching durable smoke receipt. Run this commit/settings with TEST=True first.") from exc
    if receipt.get("recipe") != recipe_key(cfg, commit) or receipt.get("status") != "passed":
        raise RuntimeError("Smoke receipt does not match this training configuration")


def publish_smoke_receipt(cfg, commit, run, api=None):
    from huggingface_hub import HfApi
    from orpheus_utils import atomic_json
    status = read_json(Path(run)/"status.json")
    backup = read_json(Path(run)/"backup_status.json")
    loss = status.get("validation", {}).get("full_validation_loss")
    if (status.get("status") != "smoke_complete" or not isinstance(loss, (int, float)) or not math.isfinite(loss)
            or not backup.get("remote_snapshot_current") or latest_checkpoint(run) is None):
        raise RuntimeError("Cannot issue a passing receipt for an incomplete/undurable smoke run")
    receipt = dict(status="passed", recipe=recipe_key(cfg, commit), commit=commit,
                   smoke_run=cfg["run_id"], validation=status["validation"], source=cfg.get("init_source"))
    path = Path(run)/"smoke_receipt.json"
    atomic_json(path, receipt)
    api = api or HfApi()
    retry_io(lambda: api.upload_file(path_or_fileobj=str(path), path_in_repo="workflow/smoke-"+receipt["recipe"]+".json",
                    repo_id=cfg["hf_repo_id"], repo_type="model", commit_message="Verified mixed-data GPU smoke result"))
