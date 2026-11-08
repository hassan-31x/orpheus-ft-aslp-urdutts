"""Publish private Kaggle notebooks using credentials configured outside this repo.

New notebooks still need HF_TOKEN enabled in Kaggle's Secrets UI. This script
never embeds a token in code, notebook outputs or a Kaggle dataset.
"""
import argparse
import json
from pathlib import Path
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["resume", "draft", "smoke", "full"])
    parser.add_argument("--existing-kernel", help="Submit the resume to an existing UrduTTS notebook; API versions still require a UI secret grant")
    parser.add_argument("--no-run", action="store_true", help="Publish without executing")
    args = parser.parse_args()
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    username = api.get_config_value(api.CONFIG_NAME_USER)
    repo = Path(__file__).resolve().parent
    resume = args.action == "resume"
    path = repo/("resume_urdutts_epoch1.ipynb" if resume else "kaggle_run.ipynb")
    notebook = json.loads(path.read_text())
    if args.action == "full":
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                cell["source"] = "".join(cell["source"]).replace("TEST = True", "TEST = False").splitlines(keepends=True)
    kernel = args.existing_kernel if resume and args.existing_kernel else username+"/"+(
        "orpheus-urdutts-epoch1-resume" if resume else "orpheus-ft-aslp-urdutts")
    if not kernel.startswith(username+"/"):
        raise ValueError("Use a notebook owned by the authenticated Kaggle account")
    metadata = dict(id=kernel, title="Orpheus UrduTTS epoch 1 resume" if resume else "Orpheus conversational Urdu ASLP + UrduTTS",
        code_file=path.name, language="python", kernel_type="notebook", is_private=True, enable_gpu=True,
        enable_internet=True, enable_tpu=False, dataset_sources=[], competition_sources=[], kernel_sources=[], model_sources=[])
    with tempfile.TemporaryDirectory(prefix="orpheus-kaggle-") as d:
        directory = Path(d)
        (directory/path.name).write_text(json.dumps(notebook,ensure_ascii=False,indent=1))
        (directory/"kernel-metadata.json").write_text(json.dumps(metadata,indent=2))
        result = api.kernels_push(str(directory), acc="nvidiaTeslaT4", no_run=args.no_run or args.action == "draft")
        print("Kaggle notebook:", "https://www.kaggle.com/code/"+kernel)
        # Do not dump arbitrary API payloads that might include account configuration.
        print("Submitted:", getattr(result,"version_number",None), "| execution requested:", not (args.no_run or args.action=="draft"))


if __name__ == "__main__":
    main()
