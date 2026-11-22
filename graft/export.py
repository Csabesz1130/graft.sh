"""Package a verified rebase without including the run's recordings or training set."""
from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from zipfile import ZipFile

from graft import certificate, genome, rebase
from graft.config import Project


def bundle(project: Project, run: str | None, output: Path, target_format: str, base_model: str | None) -> None:
    state = rebase.find_run(project, run, for_graft=True)
    if state.get("status") != "passed":
        raise rebase.RebaseError("only a passed rebase can be exported")
    folder = rebase.run_dir_of(project, state)
    if not folder.resolve().is_relative_to((project.state_dir / "grafts").resolve()):
        raise rebase.RebaseError("run path is outside the graft store")
    cert_bytes = (folder / "certificate.json").read_bytes()
    cert = json.loads(cert_bytes)
    graft = json.loads((folder / "graft.json").read_bytes())
    sealed = cert["graft"]
    if (not certificate.intact(cert) or cert["passed"] is not True or cert["target"]["name"] != state["target"]
            or cert["run"] != state["run"] or graft.get("kind") not in {"none", "prompt", "lora"}
            or graft["kind"] != sealed["kind"] or genome.digest({k: v for k, v in graft.items() if k != "adapter"})
            != sealed["sha256"] or graft.get("adapter") != sealed.get("adapter")):
        raise rebase.RebaseError("the certificate or graft no longer matches this run")
    if output.exists():
        raise rebase.RebaseError(f"output already exists: {output}")
    kind = graft["kind"]
    with tempfile.TemporaryDirectory(dir=output.parent) as tmp:
        stage = Path(tmp) / "bundle"
        stage.mkdir()
        (stage / "certificate.json").write_bytes(cert_bytes)
        portable = dict(graft)
        if kind == "lora":
            source = rebase.adapter_path(project.root, graft)
            if source is None or not source.is_dir() or not source.resolve().is_relative_to(project.root.resolve()):
                raise rebase.RebaseError("local adapter folder is missing or outside the project")
            if (stage.resolve().is_relative_to(source.resolve()) or source.is_symlink()
                    or any(p.is_symlink() or not (p.is_file() or p.is_dir()) for p in source.rglob("*"))):
                raise rebase.RebaseError("adapter must contain regular files and be separate from the export")
            shutil.copytree(source, stage / "adapter")
            if genome.tree_digest(stage / "adapter") != sealed.get("adapter_sha256"):
                raise rebase.RebaseError("adapter contents do not match the certified digest")
            portable["adapter"] = "adapter"
        if graft.get("system"):
            (stage / "system.txt").write_text(graft["system"], encoding="utf-8")
        manifest = {"format_version": 1, "kind": kind, "certificate_digest": cert["digest"],
                    "target": cert["target"]["name"], "graft": portable}
        if target_format == "goethe":
            if kind == "lora":
                raise rebase.RebaseError("Goethe export supports prompt and fast-forward runs; use vllm for LoRA")
            if not base_model:
                raise rebase.RebaseError("--base-model is required for a Goethe export")
            manifest.update(serve_argv=["goethe", "load", base_model], request_model=base_model)
        if target_format == "vllm":
            if not base_model:
                raise rebase.RebaseError("--base-model is required for a vLLM export")
            argv = ["vllm", "serve", base_model]
            model = base_model
            if kind == "lora":
                if not all((stage / "adapter" / f).is_file()
                           for f in ("adapter_config.json", "adapter_model.safetensors")):
                    raise rebase.RebaseError("vLLM requires a PEFT adapter config and safetensors weights")
                argv.extend(["--enable-lora", "--lora-modules", "grafted=./adapter"])
                model = "grafted"
            manifest.update(serve_argv=argv, request_model=model)
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        archive = Path(tmp) / "export.zip"
        with ZipFile(archive, "w") as package:
            for src in sorted(stage.rglob("*")):
                if src.is_file():
                    package.write(src, src.relative_to(stage))
        os.link(archive, output)  # publish complete bytes without replacing another caller's bundle
