"""Install into a dedicated venv without changing Kaggle's notebook packages."""
import argparse
import importlib.metadata as metadata
import os
from pathlib import Path
import re
import subprocess
import sys
import venv

CUDA_PACKAGES = ('torch', 'torchvision', 'torchaudio', 'triton', 'xformers')


def cuda_xformers_command(python, torch_version):
    """Use the official CUDA wheel, never the CUDA-less PyPI wheel.

    xFormers 0.0.35 uses the Torch 2.10+ stable ABI. Its cu130 wheel
    explicitly includes sm_75 (T4); FlashAttention itself need not support T4.
    --no-deps preserves the host's Torch/CUDA stack.
    """
    from packaging.version import Version
    version = Version(torch_version)
    cuda = re.fullmatch(r'cu(126|128|130)', version.local or '')
    if not (Version('2.10') <= version < Version('2.13')) or not cuda:
        raise RuntimeError('This GPU bootstrap requires Torch 2.10–2.12 with a '
                           'cu126/cu128/cu130 build; found ' + torch_version)
    return [python, '-m', 'pip', 'install', '--no-cache-dir', '--progress-bar', 'off',
            '--force-reinstall', '--no-deps', '--only-binary=:all:',
            '--index-url', 'https://download.pytorch.org/whl/' + version.local,
            'xformers==0.0.35']


def cuda_constraints(version=metadata.version):
    constraints = []
    for package in CUDA_PACKAGES:
        try:
            value = version(package)
        except metadata.PackageNotFoundError:
            continue
        constraints.append(f'{package}=={value}')
    if not any(item.startswith('torch==') for item in constraints):
        raise RuntimeError('Kaggle must provide Torch; setup will not download a replacement CUDA stack')
    return constraints


def create_environment(target, evidence):
    """Prefer stdlib venv; Debian images may lack the ensurepip component."""
    try:
        venv.EnvBuilder(system_site_packages=True, with_pip=True).create(target)
    except (subprocess.CalledProcessError, OSError) as exc:
        print(f'Stdlib venv unavailable ({type(exc).__name__}); using virtualenv', flush=True)
        bootstrap = evidence / 'bootstrap'
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--target', str(bootstrap),
                        'virtualenv>=20.26,<21'], check=True)
        env = os.environ.copy()
        env['PYTHONPATH'] = str(bootstrap)
        subprocess.run([sys.executable, '-m', 'virtualenv', '--system-site-packages',
                        str(target)], env=env, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--venv', type=Path, default=Path('/kaggle/working/orpheus-env'))
    parser.add_argument('--gpu', action='store_true', help='Also run CUDA imports and attention backward')
    args = parser.parse_args()
    repo = Path(__file__).resolve().parent
    target = args.venv.resolve()
    if Path(sys.prefix).resolve() == target:
        raise RuntimeError('Run this bootstrap using the notebook Python, not the training venv')
    # Inherit CUDA wheels to avoid several GB of duplicate downloads. Pip installs
    # shadowing project packages into the venv; the host kernel stays unchanged.
    locked = cuda_constraints()
    if args.gpu:
        locked = [item for item in locked if not item.startswith('xformers==')]
        locked.append('xformers==0.0.35')
    target.mkdir(parents=True, exist_ok=True)
    evidence = target.parent / (target.name + '-setup')
    evidence.mkdir(parents=True, exist_ok=True)
    constraints = evidence / 'cuda-constraints.txt'
    constraints.write_text('\n'.join(locked) + '\n')
    create_environment(target, evidence)
    python = str(target / 'bin' / 'python')
    # Resolve before installation so unsupported images fail without package changes.
    attention_install = cuda_xformers_command(python, metadata.version('torch')) if args.gpu else None
    command = [python, '-m', 'pip', 'install', '--no-cache-dir', '--progress-bar', 'off', '--constraint', str(constraints), '--report',
               str(evidence / 'installation.json'), '-r', str(repo / 'requirements-kaggle.txt')]
    # Stream output to both Kaggle and a persistent setup log, preserving failure.
    with (evidence / 'installation.log').open('a') as log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1)
        for line in process.stdout:
            print(line, end='', flush=True)
            log.write(line)
            log.flush()
        if process.wait():
            raise RuntimeError(f'Installation failed. See {evidence}/installation.log. '
                               'CUDA wheels are locked; do not upgrade them blindly.')
    if attention_install:
        print('Installing xFormers CUDA operators from the matching official Torch index', flush=True)
        subprocess.run(attention_install, check=True)
        subprocess.run([python, '-m', 'xformers.info'], check=True)
    freeze = subprocess.check_output([python, '-m', 'pip', 'freeze'], text=True)
    (evidence / 'resolved-requirements.txt').write_text(freeze)
    command = [python, str(repo / 'check_environment.py'), '--report',
               str(evidence / ('gpu-check.json' if args.gpu else 'cpu-check.json'))]
    if args.gpu:
        command.append('--gpu')
    subprocess.run(command, check=True)
    print(f'TRAIN_PYTHON = {python!r}')


if __name__ == '__main__':
    main()
