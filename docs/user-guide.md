# graft user guide

Move to a new base model and keep what your old one got right.

graft keeps your model's behavior as a genome: rules, the cases that pin each rule down, and deterministic checks. `graft rebase` carries the genome onto a new base model on your own hardware and writes a certificate of what held.

## Contents

- [Install](#install)
- [The idea in one page](#the-idea-in-one-page)
- [Build a genome](#build-a-genome)
- [Name your models](#name-your-models)
- [Rebase](#rebase)
- [Training](#training)
- [The certificate](#the-certificate)
- [Use the graft](#use-the-graft)
- [Keep the genome alive](#keep-the-genome-alive)
- [Settings](#settings)
- [Troubleshooting](#troubleshooting)

## Install

```bash
curl -fsSL https://raw.githubusercontent.com/csabesz1130/graft.sh/main/install.sh | bash
```

This installs the `graft` command with uv, pipx or pip, whichever is there. It needs Python 3.11 or newer and nothing else: the core uses the standard library only.

The built-in trainer and models loaded in process need PyTorch, Transformers and PEFT:

```bash
GRAFT_EXTRAS=train bash install.sh         # or: pip install 'graft-sh[train]'
```

You do not need them when your models sit behind endpoints and you bring your own trainer.

## The idea in one page

What a fine-tune knows is locked into weights for one architecture. When a better base model comes out, the adapter does not load, and the prompts tuned for the old model no longer steer the new one the same way. What you can keep is a description of the behavior that is precise enough to check:

- A **rule** is one sentence: "Refunds above $500 go to a human."
- A **case** is a request and the checks its reply must pass. Cases are what make a rule testable.
- A **gate** is the share of a rule's cases that must pass, 100% unless you say otherwise.

That is the genome. It lives in your repository, and nothing in it depends on a model.

A rebase then works like `git rebase`. Behavior the new model already has is like a commit that is already upstream: nothing to do. The rest is carried over with the cheapest means that works: first by stating the rules in the prompt, then by training an adapter. If something cannot be carried over, the rebase stops with a conflict and names the cases, and you decide.

## Build a genome

```bash
graft init
```

This writes `graft.toml` and an empty `genome/` folder.

**Start from rules you can say out loud.** Add each one with a sentence, then write a handful of cases for it:

```bash
graft rule add refund-limit "Refunds above \$500 are never issued directly. Escalate them to the refunds queue."
```

`genome/cases/refund-limit.jsonl`, one case per line:

```json
{"id": "refund-750", "messages": [{"role": "user", "content": "Refund $750 for order #1005."}], "tools": [...], "verify": [{"type": "tool_call", "name": "escalate", "arguments": {"queue": "refunds"}}]}
```

The [genome reference](genome.md) lists every verifier. Three habits make a genome worth having:

- **Write pairs at every boundary.** One request at $480 and one at $520, each with its own expected outcome. A pair passes only when both sides do, so it cannot be passed by a model that always escalates.
- **Check the decision, not the wording.** A tool call, a JSON field, a phrase that must or must not appear. Wording changes between models; decisions should not.
- **Vary the cases.** Ten phrasings of one request protect a rule less than five different situations. The certificate can only speak for cases like the ones you wrote.

**Add what production already does.** Recorded traffic is the cheapest source of cases:

```bash
graft capture goethe ~/.goethe/traces --model acme/support-v3 --tools-only
graft capture jsonl finetune-data.jsonl
```

Each assistant turn becomes a case with the recorded reply as its reference. Replies that call tools or return JSON get exact verifiers. Plain-text replies get a similarity check, which is weak: it is a placeholder until you write a real verifier. Captured cases go under the rule `recorded` unless you pass `--rule`. Read what you capture: recorded traffic includes the model's mistakes, and holds whatever your users typed.

**Check the genome against its own source.**

```bash
graft check                 # each recorded reference against its verifiers; needs no model
graft check --model prod    # ask the model that has the behavior today
```

A case that fails here is wrong, or describes behavior the source never had. Fix it or delete it. A rebase leaves such cases out and says so in the certificate.

## Name your models

Models go in `graft.toml`. Any of three kinds:

```toml
# An OpenAI-compatible endpoint: Goethe, vLLM, SGLang, llama.cpp, a hosted API
[models.prod]
endpoint = "http://127.0.0.1:8443/v1"
model = "acme/support-v3"
api_key_env = "GOETHE_API_KEY"          # the name of the environment variable that holds the key

# Hugging Face weights loaded in this process
[models.new-base]
path = "/models/new-base-30b"           # a folder, or a Hugging Face id

# Your own program: one JSON request on stdin, one JSON reply on stdout
[models.custom]
argv = ["python3", "ask_model.py"]
```

| Key | For | Meaning |
|---|---|---|
| `endpoint`, `model` | Endpoints | The base URL ending in `/v1`, and the model id to request |
| `api_key_env` | Endpoints | The environment variable holding the API key. The key itself never goes into the file |
| `params` | Endpoints | Extra request fields, for example `{ max_tokens = 1024 }` |
| `concurrency` | Endpoints | Requests in flight at once. Default 8 |
| `timeout` | Endpoints, programs | Seconds per request. Default 300 for endpoints, 600 for programs |
| `adapter_model` | Endpoints | The model id the endpoint serves a trained adapter under |
| `weights` | Endpoints | Where the model's weights are, for the trainer, when the endpoint id is not a path |
| `path` | Weights | A folder next to `graft.toml`, an absolute path, or a Hugging Face id |
| `max_new_tokens`, `device` | Weights | Reply length limit (default 512), and `cuda` or `cpu` (default: a GPU when there is one) |
| `template_args` | Weights, endpoints | Arguments for the model's chat template, for example `{ enable_thinking = false }`. See below |
| `argv` | Programs | The command. It gets `{"messages", "tools", "temperature", "seed"}` and prints `{"content", "tool_calls"}` |

Some chat templates take arguments. Qwen3 thinks aloud in a `<think>` block before every reply unless its template gets `enable_thinking = false`; the thinking uses up the reply length and would be trained into the adapter. `template_args` sets such arguments for one model:

```toml
[models.qwen3]
path = "Qwen/Qwen3-8B"
template_args = { enable_thinking = false }
```

A model loaded in process passes them to its chat template, and the built-in trainer renders training rows with the same arguments, so the adapter learns replies in the form the model is asked for them. An endpoint gets them in the request as `chat_template_kwargs`, the field vLLM and SGLang read; other servers may ignore it. The arguments are part of the model's fingerprint in the certificate: the same weights with another template are another model.

`graft models` shows whether each endpoint answers. graft asks at temperature 0. For a reproducible certificate, serve the model so that temperature 0 is deterministic.

## Rebase

```bash
graft rebase --target new-base
```

The steps, in order:

1. **Source check.** The genome is checked against the source: the model named by `--source` or `[rebase] source`, or the recorded references when there is none. Cases the source fails are left out: they describe behavior the source never had, so there is nothing to carry over. To hold the new base to them anyway, for rules the old model never followed, use `--enforce-all` (or `[rebase] enforce_all = true`); the certificate then lists those cases as new behavior.
2. **`zero-shot`.** The new model is asked as it is. Often a good share of the genome already passes; that share needs no graft at all. If everything passes, the rebase is a fast-forward and ends here.
3. **`prompt`.** Round 1 states every failing rule in a block appended to the system prompt. Later rounds add worked examples for rules that still fail: up to two more per rule and round, taken from the training cases, failing ones first.
4. **`lora`.** An adapter is trained and the genome checked against the new model with it. See [Training](#training). A later round can do worse than an earlier one; when the rung ends without passing, the rebase keeps the round that passed the most training cases, and `--continue` goes on from it. Held-out cases never take part in that choice.

A rung is tried for `rounds` rounds (default 3), and the next rung starts where the last one stopped. Every round checks the whole genome again, so fixing one rule cannot quietly break another.

The rebase steers by the training cases only. Held-out cases do not pick rules for the prompt, are never used as examples and never reach the trainer. They are consulted only when the training cases all pass, to decide whether the rebase is done.

The default ladder is `zero-shot`, `prompt`. Add `lora` once a trainer is set up:

```toml
[rebase]
source = "prod"
ladder = ["zero-shot", "prompt", "lora"]
```

**When it stops with a conflict**, the failing cases are listed with the reason each one failed. From there:

- A case is wrong or no longer wanted: fix or delete it, then `graft rebase --continue`.
- The rule needs more than a prompt can give: `graft rebase --continue --ladder zero-shot,prompt,lora`.
- More attempts might do it: `graft rebase --continue --rounds 5`.
- Give up on this base: `graft rebase --abort`.

`--continue` checks the current graft against the genome as it is now, so editing the genome between attempts is safe.

## Training

The `lora` rung needs a `[trainer]` table. The built-in trainer fits a LoRA adapter with Transformers and PEFT:

```toml
[trainer]
method = "lora"        # or "full": train all weights, for small models
epochs = 3
lr = 1e-4
rank = 16
alpha = 32
batch_size = 8         # rows per optimizer step
max_length = 8192      # rows longer than this many tokens are skipped
targets = "all-linear" # the layers a LoRA adapts, or a list of module names
gradient_checkpointing = false
```

It reads the base weights from the target's `weights`, `path` or `model`, in that order, and renders the training rows with the target's `template_args`.

**Memory.** `gradient_checkpointing = true` keeps fewer activations and recomputes them in the backward pass: training takes less memory and each step takes longer. Turn it on when the model barely fits, for example a LoRA on Qwen3-8B on one 24 GB GPU. It works with `method = "full"` and with replay distillation as well. A model loaded in process is one full copy of its weights: graft keeps one copy of the new base loaded at a time (as it is, or with an adapter) and frees it before the trainer starts, so the trainer has the GPU to itself.

**What it trains on.** For each training case, one reply that passes the case's verifiers, chosen in this order:

1. The new model's own reply, when it already passes. This keeps the new model's way of saying things.
2. Otherwise, a sampled attempt from the new model (with the prompt graft, if there is one) that passes: up to `samples` attempts at `temperature`. The verifiers act as the reward: only attempts they accept are kept.
3. Otherwise, the source's recorded reference.

Every row of a rule that still fails is weighted up, more in later rounds: its passing replies as well as the corrected ones. Weighting only the corrected replies would tip a boundary rule (refunds up to $500 issued, above that escalated) towards whichever side failed last, and the next round would fail the other side. Training requests do not contain the prompt graft, so what the prompt achieved is learned into the adapter. The certificate reports how many rows came from each origin.

**Keep what the new base already does: replay.** An adapter trained only on the genome learns its rules everywhere, not just where they apply: a support rule against legal advice can turn into refusing arithmetic. A capability suite catches that (the graft costs more than `max_loss`), and replay prevents it. List requests outside the genome, and the new base's own replies to them, asked once at temperature 0, are trained on alongside the genome:

```toml
[trainer]
replay = "genome/replay.jsonl"   # or a list of files; one {"messages": [...], "tools": [...]} per line
replay_weight = 1                # how many times each replayed row counts. Default 1
replay_loss = "kl"               # "kl" (default) or "ce"; see below
```

The replies come from the new base itself, so replay can only keep what it already does; it cannot teach a right answer. The built-in trainer distils replay rows (`replay_loss = "kl"`): on each reply it holds the adapter to the base model's whole next-token distribution, read from the same weights with the adapter switched off, rather than to the one reply it sampled. That keeps how likely each way of answering was, not just one answer. `"ce"` trains on the reply's tokens like any other row; a full fine-tune always does. Use requests like the ones your users send that the genome does not cover, and cover each capability across its range: replay protects only what it shows. If the new base answers easy questions in a line but works hard ones out step by step, replay of easy questions alone teaches the adapter to answer everything in a line, and the hard ones go wrong. On a hybrid model (state-space layers with a few attention layers), adapting only the attention and MLP layers (`targets`) also leaves more of the base intact. A replay request that repeats a genome case or a suite case is refused: a suite the adapter trained on would no longer measure what the graft cost. The certificate counts the replayed rows.

**Your own trainer.** Any command that reads the dataset and writes an adapter folder works:

```toml
[trainer]
command = ["python3", "train.py", "--data", "{dataset}", "--base", "{base}", "--out", "{output}"]
```

`{dataset}` is a chat fine-tuning file, one row per line: `{"messages": [..., assistant], "tools": [...], "weight": n, "case": id, "rule": id, "origin": "own" | "sampled" | "source" | "replay"}`. A replay row's `rule` is `null`. Only the last assistant message is the target. `weight` says how many times the row should count; a fraction such as 0.5 counts partly (the built-in trainer scales that copy's loss). The command must exit 0 and leave something at `{output}`. `{round}` is also available, and `{template_args}` is the target's `template_args` as a JSON object (`{}` when it has none).

**Serving the adapter.** A model loaded in process picks the adapter up by itself. An endpoint has to load it:

```toml
[models.new-base]
endpoint = "http://127.0.0.1:8000/v1"
model = "org/new-base-30b"
weights = "/models/new-base-30b"
adapter_model = "new-base-graft"

[rebase]
after_train = ["./load_adapter.sh", "{adapter}", "{adapter_model}"]
```

`after_train` runs after each training round and should return once the endpoint serves the adapter as `adapter_model`. For an engine with a load-adapter API that is one HTTP call; `examples/support-desk/load_adapter.py` shows the shape. Without `after_train` the rebase pauses with exit code 3 and tells you where the adapter is; load it and run `graft rebase --continue`.

## The certificate

```bash
graft show            # a readable report
graft show --json     # the certificate itself
```

It records the genome id, both models with their fingerprints, the graft, every rule's pass count overall and on held-out cases, every step of the ladder, cases left out, failures, and suite results.

Read it for what it does and does not establish:

- **Exact, for the genome.** Every case was checked by a deterministic verifier. Run `graft verify <target> --graft` and you get the same result, as far as your serving engine is deterministic at temperature 0.
- **Statistical, beyond it.** For held-out cases the report gives a one-sided exact binomial bound: "with 95% confidence, the new model passes at least X% of cases drawn the way these were". With all of 11 held-out cases passing, X is 76%; with all of 300, 99%. The number of held-out cases decides how strong a statement you get. If the rebase consulted the held-out cases more than once, the bound is computed at a stricter level so that it still holds.
- **Nothing, for requests unlike your cases.** No certificate can speak for situations the genome does not contain.

A certificate with `"passed": false` is a conflict report, not a release.

The digest in the certificate covers every other field, and `graft show` warns when it no longer matches. That catches a file edited after the fact. It is not a signature: sign the file yourself if you need to prove who wrote it.

Runs live under `.graft/`, which `graft init` keeps out of git. Copy a certificate out when you want to keep it with a release.

## Use the graft

| Outcome | What to deploy |
|---|---|
| Fast-forward | The new base as it is |
| Prompt graft | The new base, with the block from `graft show --graft` appended to your system prompt |
| Trained adapter | The new base with the adapter at the path `graft show --graft` prints |

Check what you deployed the same way the rebase did: `graft verify <model> --graft`.

## Keep the genome alive

The genome is only useful if it grows with production:

- When the model does something wrong in production, add the request as a case before fixing it. It is a regression test from then on, for this model and every later one.
- Run `graft capture` on new traces regularly. It adds only what is not there yet.
- Run `graft verify` in CI on any change to prompts, serving settings or model versions. It exits 1 when a gate fails.
- When `graft status` reports weakly verified cases, replace `similar` with real verifiers for the ones that matter.

## Settings

All of `graft.toml`:

```toml
[genome]
path = "genome"          # the genome folder
holdout = 0.2            # share of each rule's cases a rebase never learns from

[models.<name>]          # see "Name your models"

[rebase]
source = "prod"          # the model that has the behavior today
ladder = ["zero-shot", "prompt", "lora"]
rounds = 3               # attempts per rung
exemplars = 2            # worked examples added per failing rule and prompt round
samples = 4              # sampled attempts per failing training case, before falling back to the reference
temperature = 0.8        # for those attempts
after_train = [...]      # command that makes the endpoint serve a trained adapter

[trainer]                # see "Training"

[suites.<name>]          # see the genome reference
path = "genome/suites/<name>.jsonl"
min_gain = 0.0
max_loss = 0.02
```

## Troubleshooting

**`no graft.toml here or in any parent folder`.** Run `graft init`, or change into your project.

**`requests to <model> failed`.** The endpoint did not answer or rejected the request; the message has the HTTP status. `graft models` checks every endpoint. A rebase that stopped this way resumes with `graft rebase --continue`.

**The source fails cases in `graft check`.** Look at the reason printed for each. Most often the verifier is stricter than the behavior really is (an exact phrase where several are fine), or the case was captured from a mistake.

**The prompt rung does not help.** Read the block it built with `graft show --graft`. If a rule's statement would not make sense to a person reading it cold, rewrite the statement. Rules about the order of actions across several turns usually need the `lora` rung.

**A rule passes in training cases but fails held-out ones.** The cases are too alike or too few, and the graft learned the cases, not the rule. Add cases that differ in kind, then `graft rebase --continue`.

**Every rule holds, but a suite gate fails (`the graft costs ... above max_loss`).** The adapter learned the genome too broadly. Compare `evals/suite <name> new base.json` with `evals/suite <name> grafted.json` in the run folder to see which answers changed: a refusal or the house tone where a plain answer belongs means the rules leaked. Add `[trainer] replay` (see [Training](#training)), then `graft rebase --continue`.

**Results change between runs.** Something is not deterministic: the serving engine at temperature 0, or a command verifier. Fix that first; a certificate is only worth what its reproducibility is.

**`the built-in trainer needs torch, transformers and peft`.** `pip install 'graft-sh[train]'`, or set `[trainer] command` to your own trainer.
