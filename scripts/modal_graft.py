"""Run a command from this repository on a Modal GPU: the test suite, or a whole `graft rebase`.

    modal run scripts/modal_graft.py --cmd "python -m unittest discover -s tests -t ."
    GRAFT_GPU=A100-80GB modal run scripts/modal_graft.py --workdir /runs/x --cmd "graft rebase --target new-base"

The repository is mounted at /repo as it is on disk (uncommitted changes included) and `graft` runs from it.
Two Modal volumes persist between runs: graft-hf-cache (downloaded model weights, so each is fetched once)
and graft-runs (project folders; copy one in with `modal volume put graft-runs <folder> /<name>`, read results
with `modal volume get`). Billing is per second while the command runs; nothing is billed when idle.
Output streams to this terminal and the command's exit code is this script's exit code.
"""
import os
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parent.parent
GPU = os.environ.get("GRAFT_GPU", "L4")   # chosen when the app is defined: L4, A10G, L40S, A100-80GB, H100 ...
app = modal.App("graft-runner")
hf_cache = modal.Volume.from_name("graft-hf-cache", create_if_missing=True)
runs = modal.Volume.from_name("graft-runs", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    # the versions the CPU tests on the Linux server ran with, so results compare
    .pip_install("torch==2.15.0", "transformers==5.23.0", "peft==0.22.0", "accelerate==1.16.0",
                 "safetensors==0.8.0", "huggingface_hub==2.4.0")
    .run_commands("printf '#!/bin/sh\\nexec python -m graft \"$@\"\\n' > /usr/local/bin/graft",
                  "chmod +x /usr/local/bin/graft")
    .env({"HF_HOME": "/cache/hf", "PYTHONPATH": "/repo", "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false",
          "HF_HUB_DISABLE_TELEMETRY": "1"})
    .add_local_dir(REPO, remote_path="/repo",
                   ignore=[".git", ".graft", "**/__pycache__", "**/.venv", "**/out", "*.pt", "*.safetensors"])
)


@app.function(image=image, gpu=GPU, volumes={"/cache": hf_cache, "/runs": runs}, timeout=3 * 3600)
def run(cmd: str, workdir: str = "/repo") -> int:
    import subprocess
    import time
    start = time.time()
    subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], check=False)
    code = subprocess.run(cmd, shell=True, cwd=workdir, check=False).returncode
    hf_cache.commit()
    runs.commit()
    print(f"\n[modal_graft] exit {code} after {time.time() - start:.0f}s on the GPU", flush=True)
    return code


@app.local_entrypoint()
def main(cmd: str = "python -m unittest discover -s tests -t .", workdir: str = "/repo"):
    code = run.remote(cmd, workdir)
    raise SystemExit(code)
