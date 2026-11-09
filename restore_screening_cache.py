"""Restore screening caches only across the audited transport-only change."""
import argparse
import copy
import json
from pathlib import Path
import re
import shutil
import sys

OLD_TRANSPORT = "80872442d9be92f08181e34e0d8c67ac40ffa3a9ed2ff76ffa2d239a195f8d89"
NEW_TRANSPORT = "38cdecf7b519f0c6ecb82bf7ea66e73ab5d707cc29db4f40c112ada47f42ab4f"


def restore(cfg, inputs, work, recipe_for, fingerprint, atomic_json):
    expected = recipe_for(cfg)
    if expected["source_sha256"]["urdutts_data.py"] != NEW_TRANSPORT:
        raise RuntimeError("Cache restoration is limited to the verified transport fix")
    destination = Path(work)/"datasets"/fingerprint(expected)
    if (destination/"dataset_source.json").exists():
        return {"status": "selected_dataset_already_exists"}
    for progress in sorted(Path(inputs).glob("**/orpheus-mixed/datasets/*/selection_progress.json")):
        report = json.loads(progress.read_text())
        original = report.get("recipe", {})
        previous = copy.deepcopy(original)
        if previous.get("source_sha256", {}).get("urdutts_data.py") != OLD_TRANSPORT:
            continue
        previous["source_sha256"]["urdutts_data.py"] = NEW_TRANSPORT
        # Includes data revisions, quality code/thresholds, split seed and package versions.
        if previous != expected:
            continue
        copied = {"quality_records": 0, "aslp_files": 0}
        source = progress.parent
        for folder, pattern, counter in (
            ("quality_cache", r"[0-9a-f]{64}\.json", "quality_records"),
            ("screening", r"aslp-[0-9a-f]{24}(?:\.wav|\.conversion\.json)", "aslp_files"),
        ):
            target = destination/folder
            target.mkdir(parents=True, exist_ok=True)
            for path in sorted((source/folder).glob("*")):
                if not re.fullmatch(pattern, path.name) or not path.is_file() or path.is_symlink():
                    continue
                result = target/path.name
                if not result.exists():
                    shutil.copyfile(path, result)
                    copied[counter] += 1
        evidence = dict(status="compatible_cache_restored", source=str(source),
                        destination=str(destination), previous_recipe=original, recipe=expected, **copied)
        atomic_json(destination/"screening_cache_restore.json", evidence)
        run = Path(work)/"runs"/cfg["run_id"]
        run.mkdir(parents=True, exist_ok=True)
        atomic_json(run/"screening_cache_restore.json", evidence)
        return evidence
    return {"status": "no_matching_preparation_cache"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--inputs", default="/kaggle/input")
    args = parser.parse_args()
    sys.path.insert(0, args.project)
    from mixed_data import recipe_for
    from orpheus_utils import fingerprint, atomic_json
    cfg = json.loads(Path(args.config).read_text())
    result = restore(cfg, args.inputs, cfg["work_dir"], recipe_for, fingerprint, atomic_json)
    print("Preparation cache restore:", {k:v for k,v in result.items() if "recipe" not in k})
