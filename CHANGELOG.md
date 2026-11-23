# Changelog

## 0.1.1

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
