"""Import only verified weights from a completed run in another Hub repository."""
from pathlib import Path
import os
import shutil

from orpheus_utils import atomic_json, contained, latest_checkpoint, read_json, sha256
from pipeline_recovery import retry_io

COMPATIBILITY = ("model_id", "model_revision", "codec_revision", "rank", "lora_alpha", "rslora", "precision")


def inspect_source(source, api=None, download=None):
    from huggingface_hub import HfApi, hf_hub_download
    api, download = api or HfApi(), download or hf_hub_download
    repo = source["repo_id"]
    revision = retry_io(lambda: api.model_info(repo, revision=source.get("revision") or "main")).sha
    prefix = "runs/" + source["run_id"] + "/"
    def get(path):
        return Path(retry_io(lambda: download(repo, prefix+path, revision=revision)))
    pointer = read_json(get("latest.json"))
    if pointer.get("format") != "files-v1":
        raise RuntimeError("Source checkpoint must use the verified direct-files snapshot format")
    checkpoint = pointer["checkpoint"]
    if source.get("checkpoint") and source["checkpoint"] != checkpoint:
        raise RuntimeError("Pinned source checkpoint does not match its pinned latest pointer")
    def verified(name):
        entry = pointer["files"][name]
        # Verify pointer paths as well as the downloaded file, before using either.
        contained(Path("/snapshot"), entry["path"])
        path = get(entry["path"])
        if sha256(path) != entry["sha256"]:
            raise RuntimeError("Source snapshot checksum mismatch: " + name)
        return path
    status = read_json(verified("status.json"))
    config = read_json(verified("resolved_config.json"))
    state = read_json(verified(checkpoint + "/trainer_state.json"))
    complete = status.get("status") == "complete" and state["global_step"] >= state["max_steps"]
    return dict(repo_id=repo, revision=revision, run_id=source["run_id"], checkpoint=checkpoint,
                step=state["global_step"], epoch=state.get("epoch"), status=status.get("status"),
                complete=complete, config=config, pointer=pointer), verified


def stage_external_adapter(cfg, run, *, api=None, download=None):
    run = Path(run)
    if latest_checkpoint(run) is not None:
        return None
    target, marker = run / "init_adapter", run / "init_source.json"
    if marker.is_file():
        record = read_json(marker)
        if record["request"] != cfg["init_source"]:
            raise RuntimeError("Source initialization changed; choose a new run ID")
        if all((target / name).is_file() and sha256(target / name) == digest
               for name, digest in record["files"].items()):
            return target
        raise RuntimeError("Staged source weights are corrupt; restore the recorded source before training")
    if download is None:
        from huggingface_hub import hf_hub_download
        download = lambda *a, **kw: hf_hub_download(*a, **kw, cache_dir=run.parent.parent/"source_hub_cache")
    info, verified = inspect_source(cfg["init_source"], api=api, download=download)
    if not info["complete"]:
        raise RuntimeError(f"UrduTTS source is {info['status']} at {info['checkpoint']} (epoch {info['epoch']}); "
                           "finish epoch 1 before starting the mixed-data stage")
    for key in COMPATIBILITY:
        if cfg[key] != info["config"][key]:
            raise RuntimeError(f"Source adapter compatibility mismatch: {key}")
    cp = info["checkpoint"]
    seal = read_json(verified(cp + "/COMPLETE.json"))
    required = {"adapter_config.json", "adapter_model.safetensors", "trainer_state.json",
                "optimizer.pt", "scheduler.pt", "training_args.bin", "rng_state.pth"}
    if not required <= seal.get("files", {}).keys():
        raise RuntimeError("Source checkpoint seal is incomplete")
    for name in ("adapter_config.json", "adapter_model.safetensors", "trainer_state.json"):
        if seal["files"][name] != info["pointer"]["files"][cp+"/"+name]["sha256"]:
            raise RuntimeError("Source checkpoint seal and snapshot pointer disagree")
    adapter_cfg = read_json(verified(cp + "/adapter_config.json"))
    if adapter_cfg["r"] != cfg["rank"] or adapter_cfg["lora_alpha"] != cfg["lora_alpha"]:
        raise RuntimeError("Source PEFT configuration does not match")
    target.mkdir(exist_ok=True)
    files = {}
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        source = verified(cp + "/" + name)
        temp = target / (name + ".tmp")
        temp.unlink(missing_ok=True)
        try:
            os.link(source, temp)
        except OSError:
            shutil.copyfile(source, temp)
        (target / (name + ".tmp")).replace(target / name)
        files[name] = sha256(target / name)
    atomic_json(marker, dict(request=cfg["init_source"], files=files,
                            **{k: v for k, v in info.items() if k not in ("pointer", "config")}))
    atomic_json(target/"run_identity.json", dict(config=info["config"]))
    return target
