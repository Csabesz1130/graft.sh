# Changelog

## 0.2.0 (unreleased)

- The license is now the Functional Source License, Version 1.1, ALv2 Future License (FSL-1.1-ALv2). Each version
  becomes Apache-2.0 two years after its release; 0.1.1 and earlier stay Apache-2.0. The README explains in plain
  words what the license allows.
- Added TRADEMARKS.md, a contributor license agreement (CLA.md) with a CLA check on pull requests, and CONTRIBUTING.md.
- `graft capture` derives structured checks by default: tool names, order and count, numbers, ids and codes, enum values and values copied from the request are pinned; other strings only have to be non-empty; a JSON-only reply must stay JSON-only. `--checks exact` keeps 0.1's behaviour. The summary line counts the loosened values.
- `graft log` shows the score of the round a run kept, not of its last round (they differ when a later lora round did worse). Runs record the stage of their graft; for older runs it is read from the adapter path.

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
