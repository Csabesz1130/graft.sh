"""Does the certified Qwen3-8B graft survive GGUF quantization under llama.cpp?

1. export (CPU): `graft export --format gguf` on the certified run, Q8_0 and Q4_K_M, with llama.cpp's own converter
   and quantizer. The adapter is merged into the base first.
2. serve_and_verify (L4): llama-server from the official llama.cpp CUDA image, started with the bundle's serve_argv,
   then `graft verify served --against <bundle>` three times.

    modal run --detach scripts/modal_gguf.py [--quants Q8_0,Q4_K_M] [--parallel 4]
"""
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parent.parent
app = modal.App("graft-gguf-verify")
hf_cache = modal.Volume.from_name("graft-hf-cache")
runs = modal.Volume.from_name("graft-runs")
IGNORE = [".git", ".graft", "**/__pycache__"]

convert_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "cmake", "build-essential")
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cpu")
    .pip_install("transformers>=4.51", "peft", "accelerate", "safetensors", "sentencepiece", "protobuf", "numpy")
    .run_commands("git clone --depth 1 https://github.com/ggml-org/llama.cpp /llama.cpp",
                  "pip install /llama.cpp/gguf-py",
                  "cmake -S /llama.cpp -B /llama.cpp/build -DGGML_NATIVE=OFF -DLLAMA_CURL=OFF "
                  "&& cmake --build /llama.cpp/build --target llama-quantize -j 8",
                  "cd /llama.cpp && git rev-parse HEAD > /llama.cpp/COMMIT")
    .env({"HF_HOME": "/cache/hf", "PYTHONPATH": "/repo", "PYTHONUNBUFFERED": "1", "HF_HUB_DISABLE_TELEMETRY": "1"})
    .add_local_dir(REPO, remote_path="/repo", ignore=IGNORE)
)

serve_image = (
    modal.Image.from_registry("ghcr.io/ggml-org/llama.cpp:server-cuda", add_python="3.12")
    .entrypoint([])
    .env({"PYTHONPATH": "/repo", "PYTHONUNBUFFERED": "1", "LD_LIBRARY_PATH": "/app"})
    .add_local_dir(REPO, remote_path="/repo", ignore=IGNORE)
)


@app.function(image=convert_image, cpu=8, memory=40 * 1024, volumes={"/cache": hf_cache, "/runs": runs},
              timeout=3 * 3600, retries=0)
def export(project: str, target: str, base_model: str, quants: list[str]) -> list[str]:
    import shutil
    import subprocess
    import sys
    print("llama.cpp", Path("/llama.cpp/COMMIT").read_text().strip(), flush=True)
    work = Path("/tmp/work")
    shutil.copytree(project, work / "project", ignore=shutil.ignore_patterns("evals", "round-[1-7]"))
    out = Path("/runs/gguf")
    out.mkdir(exist_ok=True)
    made = []
    for quant in quants:
        bundle = out / f"{target}-{quant}.zip"
        if bundle.exists():
            print("already exported:", bundle, flush=True)
            made.append(str(bundle))
            continue
        local = work / f"{quant}.zip"
        done = subprocess.run([sys.executable, "-m", "graft", "export", target, "--format", "gguf", "--base-model",
                               base_model, "--quant", quant, "--llama-cpp", "/llama.cpp", "--out", str(local)],
                              cwd=work / "project", capture_output=True, text=True)
        print(f"export {quant}: exit {done.returncode}", done.stdout.strip(), done.stderr.strip()[-3000:], flush=True)
        if done.returncode:
            continue
        shutil.copy(local, bundle)
        local.unlink()
        runs.commit()
        made.append(str(bundle))
    return made


@app.function(image=serve_image, gpu="L4", volumes={"/runs": runs}, timeout=2 * 3600, retries=0)
def serve_and_verify(project: str, bundle_path: str, tool_check_passes: int = 3, parallel: int = 4) -> dict:
    import json
    import shutil
    import subprocess
    import sys
    import tempfile
    import time
    import urllib.request
    import zipfile
    work = Path(tempfile.mkdtemp(prefix="serve-"))
    shutil.copytree(project, work / "project", ignore=shutil.ignore_patterns("evals", "round-*", "lora"))
    proj = work / "project"
    bundle = work / "bundle"
    zipfile.ZipFile(bundle_path).extractall(bundle)
    manifest = json.loads((bundle / "manifest.json").read_text())
    argv = ["/app/llama-server", *manifest["serve_argv"][1:], "--host", "127.0.0.1", "--port", "8000",
            "-ngl", "999", "-c", str(8192 * parallel), "--parallel", str(parallel), "--temp", "0"]
    print("serving:", " ".join(argv), flush=True)
    log = open(work / "server.log", "w")
    server = subprocess.Popen(argv, cwd=bundle, stdout=log, stderr=subprocess.STDOUT)
    for _ in range(360):
        try:
            urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=3).close()
            break
        except OSError:
            if server.poll() is not None:
                return {"bundle": bundle_path, "error": "server exited",
                        "log": (work / "server.log").read_text()[-3000:]}
            time.sleep(5)
    else:
        server.kill()
        return {"bundle": bundle_path, "error": "server did not come up",
                "log": (work / "server.log").read_text()[-3000:]}
    # an ordinary OpenAI client: no adapter, no template arguments; the bundle's serve_argv carries them
    with open(proj / "graft.toml", "a") as out:
        out.write('\n[models.served]\nendpoint = "http://127.0.0.1:8000/v1"\nmodel = "grafted"\n'
                  f'params = {{ max_tokens = 256 }}\nconcurrency = {parallel}\n')
    passes = []
    for attempt in range(1, tool_check_passes + 1):
        done = subprocess.run([sys.executable, "-m", "graft", "verify", "served", "--against", str(bundle_path),
                               "--json"], cwd=proj, capture_output=True, text=True)
        try:
            report = json.loads(done.stdout)
        except json.JSONDecodeError:
            server.terminate()
            return {"bundle": bundle_path, "error": "verify failed", "stdout": done.stdout[-2000:],
                    "stderr": done.stderr[-2000:]}
        passes.append({"passed": report["passed"], "of": report["certified_cases"],
                       "failures": [{"case": f["case"], "rule": f.get("rule"), "split": f.get("split"),
                                     "problem": (f.get("problems") or ["?"])[0],
                                     "reply": json.dumps((f.get("replies") or [{}])[0])[:300]}
                                    for f in report["failures"]]})
        print(f"{Path(bundle_path).name} pass {attempt}: {report['passed']}/{report['certified_cases']}", flush=True)
        for f in passes[-1]["failures"]:
            print(f"   FAIL {f['rule']}/{f['case']} [{f['split']}]: {f['problem']}\n        {f['reply']}", flush=True)
    server.terminate()
    return {"bundle": bundle_path, "gguf": manifest["gguf"], "passes": passes}


@app.local_entrypoint()
def main(project: str = "/runs/qwen3-8b", target: str = "qwen3", base_model: str = "Qwen/Qwen3-8B",
         quants: str = "Q8_0,Q4_K_M", parallel: int = 4):
    import json
    bundles = export.remote(project, target, base_model, quants.split(","))
    print("bundles:", bundles)
    results = list(serve_and_verify.map([project] * len(bundles), bundles, [3] * len(bundles),
                                        [parallel] * len(bundles)))
    for r in results:
        r["parallel"] = parallel
    print(json.dumps(results, indent=2))
    Path(f"modal_gguf_results_p{parallel}.json").write_text(json.dumps(results, indent=2))
