"""Back up preparation separately from training, without changing trainer code.

The notebook embeds this supervisor while retaining the smoke-tested trainer commit.
Snapshots contain the preparation recipe, quality records and live shortlist WAVs.
"""
import argparse
import json
import logging
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import time

LOG = logging.getLogger('screening_recovery')


def compatible(root, recipe):
    for name in ('dataset_source.json', 'selection_progress.json', 'recovery_recipe.json'):
        path = root/name
        if path.is_file():
            data = json.loads(path.read_text())
            return data.get('recipe', data) == recipe
    return False


def validate_pool(root):
    from orpheus_utils import contained, sha256
    if (root/'dataset_source.json').is_file():
        return  # Final selection supersedes the disposable screening shortlist.
    state = root/'tts_scan_state.json'
    if state.exists():
        for row in json.loads(state.read_text()).get('pool', []):
            path = contained(root, row['screening_path'])
            if not path.is_file() or sha256(path) != row['sha256']:
                raise RuntimeError('Shortlist changed during backup; retry next interval')


def copy_cache(source, destination, recipe):
    """Copy only owned preparation files; never environments, secrets or raw shards."""
    from orpheus_utils import atomic_json
    destination.mkdir(parents=True, exist_ok=True)
    # Read cursor first, then its referenced audio. Validation catches concurrent deletion.
    names = ['tts_scan_state.json', 'selection_progress.json', 'dataset_source.json',
             'train.csv', 'validation.csv', 'parts', 'quality_cache', 'screening', 'audio']
    for name in names:
        path = source/name
        if path.is_symlink():
            raise ValueError('Symlink in preparation cache')
        if path.is_dir():
            for item in path.rglob('*'):
                if item.is_symlink():
                    raise ValueError('Symlink in preparation cache')
                if item.is_file() and '.tmp' not in item.name:
                    target = destination/item.relative_to(source)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        shutil.copyfile(item, target)
                    except FileNotFoundError:
                        # A discarded candidate is optional; live cursor references checked below.
                        pass
        elif path.is_file():
            shutil.copyfile(path, destination/name)
    atomic_json(destination/'recovery_recipe.json', {'recipe': recipe})
    validate_pool(destination)


def restore_inputs(cfg, recipe, destination, inputs):
    """Kaggle's previous-version output is the bridge for the already-running job."""
    from orpheus_utils import atomic_json
    if (destination/'dataset_source.json').exists() or (destination/'recovery_recipe.json').exists():
        return False
    candidates = sorted(Path(inputs).glob('**/orpheus-mixed/datasets/*'))
    for source in candidates:
        if source.is_dir() and compatible(source, recipe):
            with tempfile.TemporaryDirectory(dir=destination.parent) as tmp:
                stage = Path(tmp)/'cache'
                copy_cache(source, stage, recipe)
                shutil.copytree(stage, destination, dirs_exist_ok=True)
            evidence = dict(source=str(source), completed_shards=len(json.loads(
                (destination/'tts_scan_state.json').read_text()).get('done', []))
                if (destination/'tts_scan_state.json').exists() else 0)
            atomic_json(destination/'recovery_import.json', evidence)
            LOG.info('Previous-session screening restored: %s', evidence)
            return True
    return False


def publish(cfg, recipe, root, api=None):
    from huggingface_hub import HfApi, CommitOperationAdd
    from orpheus_utils import fingerprint, sha256
    if not root.exists():
        return
    prefix = 'runs/'+cfg['run_id']+'/screening_recovery/'
    with tempfile.TemporaryDirectory(dir=Path(cfg['work_dir'])) as tmp:
        stage = Path(tmp)/'cache'
        copy_cache(root, stage, recipe)
        archive = Path(tmp)/'cache.tar.gz'
        with tarfile.open(archive, 'w:gz', compresslevel=1) as handle:
            handle.add(stage, arcname='cache')
        digest = sha256(archive)
        # Immutable blob and pointer are committed together; failed uploads leave old pointer intact.
        relative = prefix+'archives/'+digest+'.tar.gz'
        pointer = Path(tmp)/'latest.json'
        pointer.write_text(json.dumps(dict(recipe=recipe, recipe_id=fingerprint(recipe),
            archive=relative, sha256=digest, timestamp=time.time())))
        (api or HfApi()).create_commit(repo_id=cfg['hf_repo_id'], repo_type='model',
            operations=[CommitOperationAdd(path_in_repo=relative,path_or_fileobj=str(archive)),
                        CommitOperationAdd(path_in_repo=prefix+'latest.json',path_or_fileobj=str(pointer))],
            commit_message='Save resumable screening progress')
        LOG.info('Screening backup committed (%d MiB)', archive.stat().st_size//2**20)


def restore_remote(cfg, recipe, destination):
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError
    from orpheus_utils import sha256
    pointer = 'runs/'+cfg['run_id']+'/screening_recovery/latest.json'
    try:
        data = json.loads(Path(hf_hub_download(cfg['hf_repo_id'],pointer)).read_text())
    except EntryNotFoundError:
        return False
    if data['recipe'] != recipe:
        raise RuntimeError('Remote screening recipe differs; refusing cache reuse')
    with tempfile.TemporaryDirectory(dir=destination.parent) as tmp:
        archive = Path(hf_hub_download(cfg['hf_repo_id'],data['archive'],local_dir=tmp))
        if sha256(archive) != data['sha256']:
            raise RuntimeError('Screening archive checksum mismatch')
        unpack = Path(tmp)/'unpack'
        with tarfile.open(archive) as handle:
            for member in handle.getmembers():
                path = Path(member.name)
                if path.is_absolute() or '..' in path.parts or not path.parts or path.parts[0] != 'cache' or not (member.isfile() or member.isdir()):
                    raise ValueError('Unsafe screening archive member')
            handle.extractall(unpack, filter='data')
        validate_pool(unpack/'cache')
        # Copy complete snapshot, preserving no stale cursor from a different snapshot.
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(unpack/'cache',destination)
    LOG.info('Restored screening progress from Hugging Face')
    return True


def supervise(cfg, command, inputs='/kaggle/input', interval=600, hours=None, require_cache=False):
    from mixed_data import recipe_for
    from orpheus_utils import fingerprint
    recipe = recipe_for(cfg)
    parent = Path(cfg['work_dir'])/'datasets'
    parent.mkdir(parents=True,exist_ok=True)
    root = parent/fingerprint(recipe)
    if not root.exists():
        if not restore_remote(cfg,recipe,root):
            imported = restore_inputs(cfg,recipe,root,inputs)
            if require_cache and not imported:
                raise RuntimeError("Previous screening outputs unavailable; refusing to start from scratch. Wait for the source notebook to finish and save outputs.")
    # Commit imported progress BEFORE spending any GPU time screening again.
    if root.exists():
        publish(cfg,recipe,root)
    started = time.monotonic()
    process = subprocess.Popen(command,start_new_session=True)
    last = started
    stopped = False
    try:
        while process.poll() is None:
            if time.monotonic()-last >= interval:
                try:
                    publish(cfg,recipe,root)
                except Exception as exc:
                    LOG.error('Screening backup failed (%s); local progress retained, retry in next interval',type(exc).__name__)
                last = time.monotonic()
            if hours and time.monotonic()-started >= hours*3600:
                LOG.warning('Session deadline: stopping process group, then saving screening progress')
                os.killpg(process.pid,signal.SIGINT)
                try:
                    process.wait(timeout=120)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGTERM)
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid,signal.SIGKILL);process.wait()
                stopped = True
                break
            time.sleep(5)
    finally:
        if process.poll() is None:
            os.killpg(process.pid,signal.SIGTERM)
            process.wait(timeout=30)
        # Failure is deliberately visible: never claim a successful durable save without one.
        publish(cfg,recipe,root)
    return 0 if stopped else process.returncode


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    parser.add_argument('--hours',type=float)
    parser.add_argument('--inputs',default='/kaggle/input')
    parser.add_argument('--require-cache',action='store_true')
    parser.add_argument('command',nargs=argparse.REMAINDER)
    args=parser.parse_args()
    command=args.command[1:] if args.command[:1]==['--'] else args.command
    if not command: parser.error('Missing child command')
    sys.exit(supervise(json.loads(Path(args.config).read_text()),command,args.inputs,hours=args.hours,require_cache=args.require_cache))
