# graft command reference

Every `graft` command and its options. For a walkthrough, see the [user guide](user-guide.md).

```text
graft <command> [options]
graft <command> --help
```

## Conventions

- graft works in the folder that holds `graft.toml`, or any folder below it.
- Models are named as in `graft.toml`: `[models.prod]` is `prod`.
- A rebase is named `<target>/<run>`, for example `new-base/0002-4f2a9c1e`. The run id is a counter and the first 8 characters of the genome id. Where a command takes a run, the target alone means that target's latest run, and nothing means the latest run of all.
- Commands that list things accept `--json` for scripts.

Exit codes:

| Code | Meaning |
|---|---|
| `0` | Success: the command ran, and every gate it checked holds. `graft` with no command prints this help and exits 0 |
| `1` | An error, a failing gate, or a rebase that ended in a conflict |
| `2` | Wrong usage: an unknown command or option, or a malformed argument such as a rule id or gate share |
| `3` | A rebase is paused and waits for you: a trained adapter has to be loaded into the endpoint |

## Contents

- [Genome](#genome): `init`, `status`, `rule add`, `capture`
- [Checking](#checking): `check`, `verify`, `models`
- [Rebasing](#rebasing): `rebase`, `log`, `show`

## Genome

### `graft init [folder]`

Starts a genome: writes `graft.toml`, creates `genome/` with its subfolders, and adds `.graft/` to `.gitignore`. If `graft.toml` is already there it changes nothing and exits 1.

### `graft status`

The genome id, each rule with its number of cases, pairs and weakly verified cases, and any rebase in progress.

| Option | Meaning |
|---|---|
| `--json` | The same as JSON, with each rule's statement |

### `graft rule add <id> <statement>`

Adds `genome/rules/<id>.toml`. The statement is the behavior in a sentence or two. An id holds letters, digits, dot, dash and underscore; any other id is wrong usage (exit 2) and writes nothing.

| Option | Meaning |
|---|---|
| `--gate <share>` | The share of the rule's cases that must pass, above 0 and at most 1. Default `1.0` |

### `graft capture goethe <folder>`

Reads a [Goethe](https://github.com/meshapplied/goethe) trace folder and adds one case per chat-completions turn that finished cleanly. The request is the turn's full history, rebuilt through its parent turns; the reference is the recorded reply. Verifiers are derived from the reply. By default (`--checks structured`) they pin what the reply decided: the tools called, their order and number, numbers, booleans, ids and codes, values from a tool's `enum`, and values copied from the request. Other strings, prose such as an escalation's reason or a ticket's summary, only have to be present and non-empty, and a reply that was only JSON must stay only JSON. `--checks exact` pins every value, prose included. Plain text gets `similar` either way. A regenerated reply replaces the earlier one for the same request. Turns from other APIs, failed or cut-off turns, and turns whose history is gone are skipped and counted.

| Option | Meaning |
|---|---|
| `--rule <id>` | The rule the cases go under. Default `recorded`; it is created if missing |
| `--model <id>` | Only turns served by this model id |
| `--since <time>` | Only turns that started at or after this UTC time, for example `2026-11-01` |
| `--tools-only` | Only turns whose reply calls a tool |
| `--limit <n>` | Add at most this many cases |
| `--dry-run` | Say what would be added; write nothing |
| `--checks structured` or `exact` | What the derived verifiers pin, as above. Default `structured` |

Running it again adds only cases that are not there yet.

### `graft capture jsonl <file>`

Reads a chat fine-tuning file: one `{"messages": [...], "tools": [...]}` object per line. The last assistant message is the reference and everything before it the request.

| Option | Meaning |
|---|---|
| `--every-turn` | A case for every assistant turn, not just the last |
| `--checks structured` or `exact` | What the derived verifiers pin; see `capture goethe`. Default `structured` |
| `--rule`, `--limit`, `--dry-run` | As for `capture goethe` |

## Checking

### `graft check`

Checks the genome against the model that has the behavior today. A case that fails here has a wrong verifier, or describes behavior the source never had; a rebase leaves such cases out. Exits 1 when a case fails.

| Option | Meaning |
|---|---|
| `--model <name>` | Ask this model. Default: check each case's recorded reference against its verifiers, which needs no model |
| `--json` | The failing cases with their replies |

### `graft verify <model>`

Runs every gate against a model and exits 0 only when all hold. Prints each rule with its pass count and the first failing cases.

| Option | Meaning |
|---|---|
| `--graft [run]` | Apply the graft of a rebase: its prompt block, or its trained adapter. Without a run, or with only a target, the latest rebase that was not aborted |
| `--rule <id>` | Only this rule. Repeatable |
| `--json` | Every gate and every failing case with its reply |

With `--graft`, a trained adapter whose files changed since the rebase is refused (exit 1): the run recorded its contents' sha256.

Use it in CI to gate any change to a model, a prompt or the serving stack.

#### `graft verify <model> --against <certified>`

Checks a deployment against a certificate: a run (`target/run`), a `certificate.json` or an export bundle `.zip`. It runs the cases the certificate counted as passing, sends the certified prompt graft if there is one (a trained adapter is the server's part: served, merged, or quantized into a GGUF), and exits 1 if any of them fails now. Use it whenever the serving side changes: another engine, a quantization, a chat template, a GPU.

The genome must be the one the certificate checked; `--graft` and `--rule` cannot be combined with it. `--json` reports `certified_cases`, `passed`, `ok` and every failure with its reply.

### `graft models`

Lists the models in `graft.toml` and sends each endpoint one short request to see whether it answers. Models loaded in process are listed but not loaded. Exits 1 when a model does not answer.

## Rebasing

### `graft rebase --target <model>`

Carries the genome onto a new base model. It checks the genome against the source, then climbs the ladder and stops at the first rung where every gate holds. See [the user guide](user-guide.md#rebase) for what each rung does.

| Option | Meaning |
|---|---|
| `--target <model>` | The new base |
| `--source <model>` | The model that has the behavior today. Default: `[rebase] source`; without one, the recorded references stand in |
| `--ladder <rungs>` | Rungs to try, comma-separated: `zero-shot`, `prompt`, `lora`. Default: `[rebase] ladder`, else `zero-shot,prompt` |
| `--rounds <n>` | Attempts per rung. Default: `[rebase] rounds`, else 3 |
| `--continue` | Pick up the rebase in progress |
| `--abort` | Drop the rebase in progress. Nothing it trained or recorded is deleted |
| `--enforce-all` | Hold the new base to every case, also those the source fails. By default those are left out: a rebase carries over behavior the source has. Default: `[rebase] enforce_all`. Can be turned on with `--continue`, which checks the source again |

One rebase can be in progress at a time. It ends in one of three ways:

- **Passed** (exit 0). Every gate holds. The certificate is written and the rebase is no longer in progress.
- **Conflict** (exit 1). The ladder is used up and cases still fail, or a suite gate fails. A certificate marked as not passed lists them. Change the genome, the ladder or the trainer, then `graft rebase --continue`: it checks the current graft against the genome as it is now, and carries on from the rung it stopped at. `--ladder` and `--rounds` can be given with `--continue`.
- **Paused** (exit 3). An adapter is trained and the endpoint has to load it. Load it under the model's `adapter_model` name, then `graft rebase --continue`.

If the target stops answering mid-run, the rebase stops with an error and stays in progress; `--continue` resumes it.

### `graft log`

Past rebases, newest first: run, status, outcome, the last pass count, start time and source.

### `graft show [run]`

Prints a rebase's certificate as a readable report and checks its digest. Exits 1 if the file was changed after graft wrote it, or if the trained adapter it names no longer has the contents it was certified with.

| Option | Meaning |
|---|---|
| `--json` | The certificate itself |
| `--graft` | Print the graft instead: the system prompt block of a prompt graft, or the adapter path of a trained one. Without a run, the latest rebase that was not aborted; an aborted run named exactly exits 1 |

A run's files are in `.graft/grafts/<target>/<run>/`: `certificate.json`, `REPORT.md`, `graft.json`, the evaluation of every step under `evals/`, and each training round's dataset, log and adapter under `lora/`.

## Signing

### `graft keygen`

Makes an Ed25519 key: the private key at `~/.config/graft/signing-key.pem` (mode 0600; `--key` or `$GRAFT_SIGNING_KEY` for another place) and the public key next to it as `.pub`. It never overwrites a key. Needs `pip install 'graft-sh[sign]'`.

### `graft sign [run]`

Signs a run's certificate: `certificate.sig` next to it, over the certificate's digest, which covers every field. `graft show` and `graft export` carry the signature along. Signing a certificate that was changed after graft wrote it is refused.

### `graft check-certificate <path>`

The gate for a deploy pipeline. `<path>` is a `certificate.json`, a run folder or an export bundle `.zip`. It exits 0 only when the certificate is untouched, says the rebase passed, and, with `--key`, is signed by one of the keys given.

| Option | Meaning |
|---|---|
| `--key <public key>` | A trusted public key: a `.pub` file or its base64 key. Repeatable. Without one the signature is not checked |
| `--allow-conflict` | Accept a certificate that did not pass |

Checking needs only the standard library, so the core install can run it in CI.

## Export a verified run

`graft export [target/run] --out graft.zip` writes a portable certificate, manifest and any required adapter or system prompt. Extract the archive before serving. A certified prompt can contain worked examples from captures and the certificate includes model fingerprints. Review the bundle before sharing it.

For vLLM use `graft export --format vllm --base-model MODEL --out serving.zip`. From the extracted directory, run the argument list in `manifest.json` and use its `request_model` in an OpenAI-compatible client. If the bundle includes `system.txt`, append that text to your client’s system message. The manifest also records what the certificate was checked with: `certified_base` (the base model and its revision) and, when the model has `template_args`, the same arguments as `template_args`, in the vLLM command (`--default-chat-template-kwargs`) and as `request_extra_body` for clients. Serve another base, or leave the template arguments out, and the model is not the one that was certified. If your genome uses tools, add your engine's tool-call options yourself (for vLLM, `--enable-auto-tool-choice` and the `--tool-call-parser` for your model). A LoRA export requires PEFT safetensors compatible with your base model. The portable format also supports custom trainer artifacts.

For llama.cpp, LM Studio, Ollama, or Goethe on a Mac, pass `--format gguf --base-model MODEL --llama-cpp DIR`. A trained adapter is merged into the base (this needs `graft-sh[train]`), and the result converted with llama.cpp's own `convert_hf_to_gguf.py` (`pip install gguf`). `--quant` is `Q8_0` by default; `f16` and `bf16` also need no build, and other types such as `Q4_K_M` go through `llama-quantize`. The manifest's `serve_argv` starts `llama-server` with the template arguments (`--chat-template-kwargs`); add `-ngl 999` to use a GPU. The certificate was earned by the unquantized weights, so the manifest says `served_as_certified: false`: serve the file and run `graft verify <model> --against <bundle>` before you trust it.

For a prompt or fast-forward run on Goethe, pass `--format goethe --base-model MODEL`. The manifest records the `goethe load` arguments and request model. Send requests to the operator's configured OpenAI-compatible endpoint, including the exported system text when present.
