"""Publish a stage-3 notebook to the authenticated Kaggle account (the account whose token the Kaggle CLI uses).

  python kaggle_submit_stage3.py data  --no-run     # build + publish the v4 dataset
  python kaggle_submit_stage3.py train --no-run     # smoke test then the full epoch 6, in one run

Kaggle clones this repository at --ref (default: local HEAD, which must be pushed), so keep one ref for both notebooks.
API-started sessions cannot read the HF_TOKEN Kaggle secret: push with --no-run, then on kaggle.com open the notebook,
attach the secret under Add-ons > Secrets and choose Save Version > Save & Run All (this is the only manual step).
"""
import argparse
import json
import subprocess
import tempfile
from pathlib import Path

NOTEBOOKS = {"data": "kaggle_stage3_data.ipynb", "train": "kaggle_stage3_train.ipynb"}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("action", choices=sorted(NOTEBOOKS))
    p.add_argument("--ref", help="40-character commit pushed to GitHub (default: local HEAD)")
    p.add_argument("--epoch", type=int, default=6, help="train: first epoch to run (6, 7 or 8)")
    p.add_argument("--smoke-only", action="store_true", help="data: run only the ~10 minute smoke test, publish nothing")
    p.add_argument("--run-mode", choices=["smoke_then_full", "smoke", "full"], default="smoke_then_full",
                   help="train: 'smoke' runs only the short smoke test (publishes the receipt that lets a later run skip it)")
    p.add_argument("--kernel", help="owner/slug (default: <you>/orpheus-urdu-stage3-<action>)")
    p.add_argument("--no-run", action="store_true")
    args = p.parse_args()
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    user = api.get_config_value(api.CONFIG_NAME_USER)
    repo = Path(__file__).resolve().parent
    ref = args.ref or subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    if len(ref) != 40:
        raise SystemExit("--ref must be a full 40-character commit")
    pushed = subprocess.run(["git", "branch", "-r", "--contains", ref], cwd=repo, capture_output=True, text=True).stdout
    if not pushed.strip():
        raise SystemExit(f"Commit {ref} is not on a pushed remote branch; push it first (Kaggle clones from GitHub)")
    notebook = json.loads((repo / NOTEBOOKS[args.action]).read_text(encoding="utf-8"))
    replacements = {"REPO_REF = ": f'REPO_REF = "{ref}"'}
    if args.action == "train":
        replacements["EPOCH = "] = f"EPOCH = {args.epoch}"
        replacements["RUN_MODE = "] = f'RUN_MODE = "{args.run_mode}"'
    else:
        replacements["SMOKE_ONLY = "] = f"SMOKE_ONLY = {args.smoke_only}"
    changed = set()
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        lines = "".join(cell["source"]).splitlines(keepends=True)
        for i, line in enumerate(lines):
            for prefix, new in replacements.items():
                if line.startswith(prefix):
                    comment = line.split("  #", 1)[1].rstrip("\n") if "  #" in line else ""
                    lines[i] = new + ("  #" + comment if comment else "") + "\n"
                    changed.add(prefix)
        cell["source"] = lines
    if changed != set(replacements):
        raise SystemExit(f"Notebook lines not found: {sorted(set(replacements) - changed)}")
    kernel = args.kernel or f"{user}/orpheus-urdu-stage3-{args.action}"
    if not kernel.startswith(user + "/"):
        raise SystemExit("Use a notebook owned by the authenticated Kaggle account")
    code_file = NOTEBOOKS[args.action]
    meta = dict(id=kernel, title=kernel.split("/")[1].replace("-", " "), code_file=code_file, language="python",
                kernel_type="notebook", is_private=True, enable_gpu=True, enable_internet=True, enable_tpu=False,
                dataset_sources=[], competition_sources=[], kernel_sources=[], model_sources=[])
    with tempfile.TemporaryDirectory(prefix="stage3-kaggle-") as d:
        (Path(d) / code_file).write_text(json.dumps(notebook, ensure_ascii=False, indent=1))
        (Path(d) / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
        subprocess.run(["kaggle", "kernels", "push", "-p", d, "--accelerator", "NvidiaTeslaT4"]
                       + (["--no-run"] if args.no_run else []), check=True)
    print(f"Notebook: https://www.kaggle.com/code/{kernel} | action={args.action} ref={ref[:10]}")


if __name__ == "__main__":
    main()
