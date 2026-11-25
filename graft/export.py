"""Package a verified rebase without including the run's recordings or training set."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from zipfile import ZipFile

from graft import certificate, genome, rebase, signing
from graft.config import Project

# GGUF types convert_hf_to_gguf.py writes itself; the rest go through llama-quantize from a bf16 file
DIRECT = {"f32", "f16", "bf16", "q8_0"}


def bundle(project: Project, run: str | None, output: Path, target_format: str,
           base_model: str | None, quant: str = "Q8_0", llama_cpp: Path | None = None) -> list[str]:
    """Write the bundle; returns the names (path, model id) the certified base goes by."""
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
        sig = signing.load_signature(folder / signing.NAME)
        if sig is not None:
            if sig.get("certificate_digest") != cert["digest"]:
                raise rebase.RebaseError(f"{signing.NAME} in the run signs another certificate; sign it again")
            (stage / signing.NAME).write_text(json.dumps(sig, indent=2) + "\n", encoding="utf-8")
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
        if sig is not None:
            manifest["signed_by"] = sig.get("key_id")
        # What the certificate was checked with besides the graft: the exact base, and the chat template arguments
        # (Qwen3's enable_thinking, for one). Served without them, the model is not the one that was certified.
        fingerprint = cert["target"].get("fingerprint") or {}
        base = {k: fingerprint[k] for k in ("path", "model", "revision", "config_sha256") if fingerprint.get(k)}
        if base:
            manifest["certified_base"] = base
        template = fingerprint.get("template_args") or {}
        if template:
            manifest["template_args"] = template
            manifest["request_extra_body"] = {"chat_template_kwargs": template}
        if target_format == "goethe":
            if kind == "lora":
                raise rebase.RebaseError("Goethe serves no adapters: for a trained graft use --format gguf "
                                         "(the adapter merged in) or vllm")
            if not base_model:
                raise rebase.RebaseError("--base-model is required for a Goethe export")
            manifest.update(serve_argv=["goethe", "load", base_model], request_model=base_model)
        if target_format == "vllm":
            if not base_model:
                raise rebase.RebaseError("--base-model is required for a vLLM export")
            argv = ["vllm", "serve", base_model]
            if template:   # the server default; request_extra_body says the same per request
                argv.extend(["--default-chat-template-kwargs", json.dumps(template, sort_keys=True)])
            model = base_model
            if kind == "lora":
                if not all((stage / "adapter" / f).is_file()
                           for f in ("adapter_config.json", "adapter_model.safetensors")):
                    raise rebase.RebaseError("vLLM requires a PEFT adapter config and safetensors weights")
                argv.extend(["--enable-lora", "--lora-modules", "grafted=./adapter"])
                model = "grafted"
            manifest.update(serve_argv=argv, request_model=model)
        if target_format == "gguf":
            if not base_model:
                raise rebase.RebaseError("--base-model is required for a GGUF export: the weights the graft was "
                                         "certified on, as a local folder or a Hugging Face id")
            name = _gguf(stage, Path(tmp), kind, base_model, quant, llama_cpp)
            portable.pop("adapter", None)
            if kind == "lora":
                shutil.rmtree(stage / "adapter")       # merged into the GGUF
            argv = ["llama-server", "-m", name, "--jinja", "--alias", "grafted"]
            if template:
                argv.extend(["--chat-template-kwargs", json.dumps(template, sort_keys=True)])
            digest = hashlib.sha256()
            with open(stage / name, "rb") as weights:
                for block in iter(lambda: weights.read(1 << 20), b""):
                    digest.update(block)
            manifest.update(graft=portable, serve_argv=argv, request_model="grafted",
                            gguf={"file": name, "type": quant, "sha256": digest.hexdigest(),
                                  "adapter_merged": kind == "lora"},
                            served_as_certified=False,
                            check=("The certificate was earned by the unquantized weights. Serve this file, add it "
                                   "to graft.toml as a model, and run `graft verify <model> --against <this bundle>` "
                                   "before you trust it."))
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        archive = Path(tmp) / "export.zip"
        with ZipFile(archive, "w") as package:
            for src in sorted(stage.rglob("*")):
                if src.is_file():
                    package.write(src, src.relative_to(stage))
        os.link(archive, output)  # publish complete bytes without replacing another caller's bundle
    return [str(base[k]) for k in ("path", "model") if base.get(k)]


def _tool(llama_cpp: Path | None, name: str) -> list[str]:
    """How to run one of llama.cpp's tools: from --llama-cpp (a checkout, a build, or an install), else PATH."""
    if llama_cpp is None and os.environ.get("GRAFT_LLAMA_CPP"):
        llama_cpp = Path(os.environ["GRAFT_LLAMA_CPP"])
    if name.endswith(".py"):
        for folder in ([llama_cpp] if llama_cpp else []):
            if (folder / name).is_file():
                return [sys.executable, str(folder / name)]
        raise rebase.RebaseError(f"{name} not found: pass --llama-cpp with a llama.cpp checkout "
                                 f"(or set GRAFT_LLAMA_CPP); it also needs `pip install gguf`")
    for folder in ([llama_cpp, llama_cpp / "build" / "bin", llama_cpp / "bin"] if llama_cpp else []):
        if (folder / name).is_file():
            return [str(folder / name)]
    found = shutil.which(name)
    if not found:
        raise rebase.RebaseError(f"{name} not found in --llama-cpp or on PATH; build llama.cpp, or export as "
                                 f"--quant Q8_0, f16 or bf16, which need no build")
    return [found]


def _run(argv: list[str], what: str) -> None:
    done = subprocess.run(argv, capture_output=True, text=True, check=False)
    if done.returncode:
        tail = (done.stderr or done.stdout).strip().splitlines()[-5:]
        raise rebase.RebaseError(f"{what} failed ({done.returncode}): " + " | ".join(tail))


def _gguf(stage: Path, tmp: Path, kind: str, base_model: str, quant: str, llama_cpp: Path | None) -> str:
    """One GGUF file in the stage folder with the graft's adapter merged in; returns its name."""
    if not quant or not all(c.isalnum() or c == "_" for c in quant):
        raise rebase.RebaseError(f"bad quantization type {quant!r}")
    convert = _tool(llama_cpp, "convert_hf_to_gguf.py")
    quantize = None if quant.lower() in DIRECT else _tool(llama_cpp, "llama-quantize")
    local = Path(base_model).expanduser()
    if kind == "lora" or not (local / "config.json").is_file():
        merged = tmp / "merged"
        argv = [sys.executable, "-m", "graft.merge_hf", "--base", base_model, "--output", str(merged)]
        if kind == "lora":
            argv[4:4] = ["--adapter", str(stage / "adapter")]
        _run(argv, "merging the adapter into the base" if kind == "lora" else "fetching the base")
    else:
        merged = local
    name = f"model-{quant}.gguf"
    direct = quant.lower() if quantize is None else "bf16"
    first = stage / name if quantize is None else tmp / "model-bf16.gguf"
    _run([*convert, str(merged), "--outfile", str(first), "--outtype", direct], "convert_hf_to_gguf.py")
    if quantize is not None:
        _run([*quantize, str(first), str(stage / name), quant], "llama-quantize")
        first.unlink()
    if not (stage / name).is_file():
        raise rebase.RebaseError("the GGUF conversion wrote no file")
    return name
