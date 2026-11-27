"""Run `graft rebase` on a Modal GPU, from any machine with the Modal client: a Mac included.

The project goes up as an archive (the genome, graft.toml, local files it names, and the files of a rebase in
progress), the same graft runs there with the GPU asked for, its output streams to this terminal, and what it
changed under .graft comes back, so `graft show`, `graft export` and `--continue` work here as if it had run here.
Every run folder is also kept whole on the Modal volume graft-runs, with the adapters of all its rounds.

Needs `pip install 'graft-sh[modal]'` and a Modal account (`modal token new`). graft sends no credentials:
for a gated model on Hugging Face, create a Modal secret with HF_TOKEN in it and name it in [modal] secrets.
"""
from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path, PurePosixPath
from typing import Any

from graft import rebase
from graft.config import Project

# The versions the CPU tests and the measured GPU runs used, so results compare. [modal] packages overrides them.
PACKAGES = ["torch==2.15.0", "transformers==5.23.0", "peft==0.22.0", "accelerate==1.16.0", "safetensors==0.8.0",
            "huggingface_hub==2.4.0"]
GPUS = {"none", "T4", "L4", "A10G", "L40S", "A100", "A100-40GB", "A100-80GB", "H100", "H200", "B200"}
SKIP_DIRS = {".git", "__pycache__", ".venv", "venv", ".ruff_cache", ".pytest_cache", ".mypy_cache"}
SKIP_SUFFIXES = {".zip", ".gguf"}
WARN_BYTES = 2 << 30
RESULT = "result.tar.gz"
PENDING = "modal-run.json"   # under .graft: the remote run this project last started


class RemoteError(Exception):
    pass


def settings(project: Project, gpu: str | None) -> dict[str, Any]:
    table = project.raw.get("modal", {})
    if not isinstance(table, dict):
        raise RemoteError("[modal] must be a table")
    chosen = gpu or str(table.get("gpu", "L4"))
    if chosen not in GPUS:
        raise RemoteError(f"unknown GPU {chosen!r}; one of {', '.join(sorted(GPUS))}")
    secrets = table.get("secrets", [])
    packages = table.get("packages", PACKAGES)
    if not (isinstance(secrets, list) and all(isinstance(s, str) for s in secrets)):
        raise RemoteError("[modal] secrets must be a list of Modal secret names")
    if not (isinstance(packages, list) and all(isinstance(s, str) for s in packages)):
        raise RemoteError("[modal] packages must be a list of pip requirements")
    return {"gpu": chosen, "timeout_hours": float(table.get("timeout_hours", 6)), "secrets": secrets,
            "packages": packages, "volume": str(table.get("volume", "graft-runs")),
            "cache": str(table.get("cache_volume", "graft-hf-cache"))}


def check_reachable(project: Project, names: list[str]) -> None:
    """Models on this machine's localhost cannot be reached from Modal; say so before anything is uploaded."""
    for name in names:
        endpoint = str(project.model(name).get("endpoint", ""))
        if any(host in endpoint for host in ("://127.", "://localhost", "://0.0.0.0", "://[::1]")):
            raise RemoteError(f"[models.{name}] is served at {endpoint}, which a Modal container cannot reach; "
                              f"use weights (path = ...) or a public endpoint, or run the rebase here")


def _keep(project: Project, path: Path) -> bool:
    rel = path.relative_to(project.root)
    if any(part in SKIP_DIRS for part in rel.parts) or path.suffix in SKIP_SUFFIXES:
        return False
    parts = rel.parts
    # adapters of finished runs stay here: a rebase in progress needs only its own
    if len(parts) > 6 and parts[0] == ".graft" and parts[1] == "grafts" and parts[4] == "lora":
        current = rebase.in_progress(project)
        if not current or (parts[2], parts[3]) != (current["target"], current["run"]):
            return False
    return True


def pack(project: Project) -> tuple[bytes, dict[str, list[int]]]:
    """The project as a gzipped tar, and the size and mtime of each file under .graft that went up."""
    buffer, sent = io.BytesIO(), {}
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for path in sorted(project.root.rglob("*")):
            if path.is_symlink() or not path.is_file() or not _keep(project, path):
                continue
            rel = path.relative_to(project.root).as_posix()
            archive.add(path, arcname=rel, recursive=False)
            if rel.startswith(".graft/"):
                stat = path.stat()
                sent[rel] = [stat.st_size, stat.st_mtime_ns]
    return buffer.getvalue(), sent


def _safe(name: str) -> bool:
    path = PurePosixPath(name)
    return bool(name) and not path.is_absolute() and ".." not in path.parts and path.parts[0] == ".graft"


def unpack(project: Project, changes: bytes, deleted: list[str]) -> int:
    """Write back what the remote run changed under .graft; returns the number of files written."""
    written = 0
    with tarfile.open(fileobj=io.BytesIO(changes), mode="r:gz") as archive:
        for member in archive.getmembers():
            if not member.isfile() or not _safe(member.name):
                raise RemoteError(f"refusing an unexpected file from the remote run: {member.name!r}")
            target = project.root / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            assert source is not None
            target.write_bytes(source.read())
            written += 1
    for name in deleted:
        if not _safe(name):
            raise RemoteError(f"refusing to delete {name!r}")
        (project.root / name).unlink(missing_ok=True)
    return written


def collect(root: Path, sent: dict[str, list[int]]) -> tuple[bytes, list[str]]:
    """On the remote side: what changed under .graft since upload, as a gzipped tar, and what was deleted.
    Adapters of rounds that were not kept stay out (they are on the volume); the kept one comes back."""
    kept = set()
    for state_path in (root / ".graft" / "grafts").glob("*/*/state.json"):
        adapter = (json.loads(state_path.read_text(encoding="utf-8")).get("graft") or {}).get("adapter")
        if adapter:
            kept.add(PurePosixPath(Path(adapter).as_posix()))
    buffer, now = io.BytesIO(), set()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for path in sorted((root / ".graft").rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(root).as_posix()
            now.add(rel)
            stat = path.stat()
            if sent.get(rel) == [stat.st_size, stat.st_mtime_ns]:
                continue
            parts = PurePosixPath(rel).parts
            if "lora" in parts and "adapter" in parts:
                folder = PurePosixPath(*parts[:parts.index("adapter") + 1])
                if folder not in kept:
                    continue
            archive.add(path, arcname=rel, recursive=False)
    return buffer.getvalue(), sorted(set(sent) - now)


def restore_times(work: Path, sent: dict[str, list[int]]) -> None:
    """A tar keeps whole-second mtimes; put back the exact ones, so files the run did not touch compare equal."""
    for rel, (size, mtime) in sent.items():
        path = work / rel
        if path.is_file() and path.stat().st_size == size:
            os.utime(path, ns=(mtime, mtime))


def remote_rebase(archive: bytes, sent: dict[str, list[int]], argv: list[str], keep_as: str,
                  volumes: list[str]) -> dict[str, Any]:
    """Runs in the Modal container."""
    work = Path("/tmp/project")
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
        tar.extractall(work, filter="data")
    restore_times(work, sent)
    import shutil
    if shutil.which("nvidia-smi"):
        subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], check=False)
    start = time.time()
    # the graft that was shipped (beside this module), whatever the working folder
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [str(Path(__file__).resolve().parent.parent),
                                                                      os.environ.get("PYTHONPATH")]))}
    code = subprocess.run([sys.executable, "-m", "graft", "rebase", *argv], cwd=work, env=env,
                          check=False).returncode
    print(f"\n[graft on modal] exit {code} after {time.time() - start:.0f}s", flush=True)
    # every run folder, whole, on the volume
    saved = Path("/runs") / keep_as
    for run_dir in (work / ".graft" / "grafts").glob("*/*"):
        destination = saved / run_dir.relative_to(work / ".graft" / "grafts")
        shutil.copytree(run_dir, destination, dirs_exist_ok=True)
    try:
        import modal
        for name in volumes:
            modal.Volume.from_name(name).commit()
    except Exception as err:   # the volumes also commit when the container stops
        print(f"[graft on modal] volume commit deferred: {err}", flush=True)
    changes, deleted = collect(work, sent)
    # the result also stays on the volume, for `graft rebase --fetch` when the client was gone by now
    saved.mkdir(parents=True, exist_ok=True)
    (saved / RESULT).write_bytes(changes)
    (saved / "result.json").write_text(json.dumps({"code": code, "deleted": deleted}), encoding="utf-8")
    try:
        import modal
        modal.Volume.from_name(volumes[0]).commit()
    except Exception as err:
        print(f"[graft on modal] volume commit deferred: {err}", flush=True)
    return {"code": code, "changes": changes, "deleted": deleted, "saved": str(saved)}


def run(project: Project, argv: list[str], gpu: str | None, models: list[str]) -> int:
    opts = settings(project, gpu)
    check_reachable(project, models)
    try:
        import modal
    except ImportError:
        raise RemoteError("--runner modal needs the Modal client: pip install 'graft-sh[modal]', "
                          "then `modal token new`") from None
    archive, sent = pack(project)
    size = len(archive)
    print(f"uploading the project ({size / 1e6:.1f} MB) to Modal, GPU {opts['gpu']}", flush=True)
    if size > WARN_BYTES:
        print("warning: that is large; local weights named by path are uploaded with it", file=sys.stderr)
    image = (modal.Image.debian_slim(python_version=f"{sys.version_info.major}.{sys.version_info.minor}")
             .pip_install(*opts["packages"])
             .env({"HF_HOME": "/cache/hf", "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false",
                   "HF_HUB_DISABLE_TELEMETRY": "1"})
             .add_local_python_source("graft"))
    app = modal.App("graft-rebase")
    volumes = {"/cache": modal.Volume.from_name(opts["cache"], create_if_missing=True),
               "/runs": modal.Volume.from_name(opts["volume"], create_if_missing=True)}
    function = app.function(image=image, gpu=None if opts["gpu"] == "none" else opts["gpu"], volumes=volumes,
                            timeout=int(opts["timeout_hours"] * 3600),
                            secrets=[modal.Secret.from_name(name) for name in opts["secrets"]],
                            serialized=True)(remote_rebase)
    keep_as = f"{project.root.name}-{time.strftime('%Y%m%d-%H%M%S')}"
    pending = project.state_dir / PENDING
    project.state_dir.mkdir(exist_ok=True)
    pending.write_text(json.dumps({"id": keep_as, "volume": opts["volume"], "argv": argv,
                                   "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}) + "\n", encoding="utf-8")
    # detached: the run goes on if this terminal closes or the laptop sleeps; --fetch brings it back later
    with modal.enable_output(), app.run(detach=True):
        call = function.spawn(archive, sent, argv, keep_as, [opts["volume"], opts["cache"]])
        try:
            result = call.get()
        except KeyboardInterrupt:
            print(f"\nstopped watching; the rebase goes on on Modal. Bring it back when it is done with "
                  f"`graft rebase --fetch` (run {keep_as})", file=sys.stderr)
            return 130
    return _apply(project, result["changes"], result["deleted"], int(result["code"]), keep_as, opts["volume"])


def _apply(project: Project, changes: bytes, deleted: list[str], code: int, keep_as: str, volume: str) -> int:
    written = unpack(project, changes, deleted)
    (project.state_dir / PENDING).unlink(missing_ok=True)
    print(f"\n{written} files came back into .graft; every run folder, all rounds' adapters included, is on the "
          f"Modal volume {volume} under /{keep_as}")
    return code


def fetch(project: Project, wanted: str | None) -> int:
    """Bring back the result of a remote rebase that finished while this machine was not watching."""
    pending = project.state_dir / PENDING
    known = json.loads(pending.read_text(encoding="utf-8")) if pending.is_file() else {}
    run_id = wanted or known.get("id")
    if not run_id:
        raise RemoteError("no remote rebase to fetch: none was started from this project, and none was named")
    if not all(c.isalnum() or c in "-_." for c in run_id):
        raise RemoteError(f"bad run id {run_id!r}")
    volume_name = known.get("volume") if known.get("id") == run_id else None
    volume_name = volume_name or settings(project, None)["volume"]
    try:
        import modal
    except ImportError:
        raise RemoteError("--fetch needs the Modal client: pip install 'graft-sh[modal]'") from None
    volume = modal.Volume.from_name(volume_name)
    try:
        status = json.loads(b"".join(volume.read_file(f"{run_id}/result.json")))
        changes = b"".join(volume.read_file(f"{run_id}/{RESULT}"))
    except Exception:
        raise RemoteError(f"{run_id} has no result on the volume {volume_name} yet: it is still running, or it "
                          f"stopped before the end (see `modal app logs` for graft-rebase)") from None
    return _apply(project, changes, status["deleted"], int(status["code"]), run_id, volume_name)
