# Changelog

## 0.3.0

- `graft rebase --runner modal [--gpu L4]`: runs the rebase on a Modal GPU from any machine with the Modal client,
  streams its output, and brings what it changed under `.graft` back, so `show`, `verify`, `export` and `--continue`
  work locally. Every run folder is kept whole on the Modal volume `graft-runs`. `[modal]` sets the GPU, secrets,
  timeout and packages. Needs `graft-sh[modal]`.
- The remote rebase is detached: Ctrl-C or a closed laptop does not stop it, and graft prints the
  `graft rebase --fetch <run>` and `graft rebase --cancel` (or `modal app stop <app>`) commands. `--fetch` with no
  run brings back the one started here, writes only under `.graft`, refuses files changed here while the job ran,
  checks sha256 before and after writing, then removes the result bundle from the volume (`--keep-remote` keeps it).

## 0.2.0

- The license is now the Functional Source License, Version 1.1, ALv2 Future License (FSL-1.1-ALv2). Each version
  becomes Apache-2.0 two years after its release; 0.1.1 and earlier stay Apache-2.0. The README explains in plain
  words what the license allows.
- Added TRADEMARKS.md, a contributor license agreement (CLA.md) with a CLA check on pull requests, and CONTRIBUTING.md.
- `graft capture` derives structured checks by default: tool names, order and count, numbers, ids and codes, enum values and values copied from the request are pinned; other strings only have to be non-empty; a JSON-only reply must stay JSON-only. `--checks exact` keeps 0.1's behaviour. The summary line counts the loosened values.
- `graft log` shows the score of the round a run kept, not of its last round (they differ when a later lora round did worse). Runs record the stage of their graft; for older runs it is read from the adapter path.
- `graft verify <model> --against <run|certificate|bundle>`: checks a deployment (another engine, a quantization, a served or merged adapter) on every case the certificate counted, and fails if one fails now. `--passes N` repeats it and fails if a case fails in any pass.
- `graft export --format gguf`: merges a trained adapter into its base and converts it with llama.cpp's tools (`--quant`, `--llama-cpp`), for llama.cpp, LM Studio, Ollama or Goethe on a Mac. The manifest asks for `graft verify --against` before the file is trusted.
- `graft keygen`, `graft sign` and `graft check-certificate`: Ed25519 signatures over the certificate digest, carried in export bundles, and a deploy gate that checks a certificate is untouched, passed and signed by a trusted key. Checking needs only the standard library; signing needs `graft-sh[sign]`.
- `python -m graft.merge_hf`: fold an adapter into its base as a plain Hugging Face model.
- README: the certified Qwen3-8B run as Q8_0 and Q4_K_M GGUF files on llama.cpp, checked with `verify --against` (`scripts/modal_gguf.py`).
- CI runs the install, unit tests and demo on macOS (Apple Silicon). The demo server no longer waits 35 s for a host-name lookup on a Mac.
- A release workflow builds the wheel and sdist, writes SHA256SUMS and attests them with GitHub build provenance.

## 0.1.1

- The license is now Apache-2.0 (0.1.0 was MIT).
- README: a terminal recording of real commands on two saved runs.
- README: the hosted-to-own recipe run end to end on real models, with its result (a conflict on two held-out cases).
- Added `graft export` for portable bundles and serving manifests for vLLM and Goethe.
- An export records the certified base and the model's chat template arguments, and passes them to vLLM.
- User guide: a tested recipe for carrying recorded traffic from a hosted model onto a model of your own.
- A second example genome, `examples/sql-conventions`: text-to-SQL checked by execution, with the results of rebasing it (all conflicts on held-out cases).

## 0.1.0

First version.

- **Genome.** Rules in TOML, cases in JSONL, counterfactual pairs, per-rule gates, capability suites. The genome id is a hash over the whole folder.
- **Verifiers.** `contains`, `not_contains`, `regex`, `not_regex`, `equals`, `length`, `json` (schema, subset, equals), `tool_call`, `no_tool_call`, `similar`, `command`, and `all`, `any`, `not`. All deterministic.
- **Capture.** `graft capture goethe` reads a Goethe trace folder and rebuilds each turn's history; `graft capture jsonl` reads chat fine-tuning files.
- **Runners.** OpenAI-compatible endpoints, Hugging Face weights in process with an optional adapter, and your own program.
- **Rebase.** A ladder of `zero-shot`, `prompt` and `lora` rungs, with conflicts, `--continue` and `--abort`. The rebase steers by training cases only and counts every look at the held-out ones.
- **Training.** Datasets built from verified replies: the new model's own, sampled attempts the verifiers accept, then the source's reference. A built-in LoRA trainer, or any command.
- **Certificate.** Per-rule results, an exact binomial bound on held-out cases, paired suite comparisons with a sign test, and a digest.
- **Commands.** `init`, `status`, `rule add`, `capture`, `check`, `verify`, `models`, `rebase`, `log`, `show`.
