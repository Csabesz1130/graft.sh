# Example: a support desk moves to a new base model

A complete genome and two scripted stand-ins for models, so `graft rebase` can be tried end to end in a minute with no GPU.

**This is a simulation.** `mock_models.py` does no inference: its "models" are short Python functions with fixed behavior, and `mock_trainer.py` trains nothing. The example shows the workflow and what graft reports at each step. It says nothing about how well any real model transfers.

## Run it

```bash
python3 mock_models.py &            # both "models" at http://127.0.0.1:8811/v1
graft rebase --target new-base
graft show
```

Run it again from the start with `rm -rf .graft` and a restart of `mock_models.py`.

## The cast

| | |
|---|---|
| `prod` (`acme/support-v3`) | The old model. Follows all five house rules. Gets 20 of 30 arithmetic questions right |
| `new-base` (`open/new-base-30b`) | The new base. Gets 27 of 30 right, and knows only one house rule by itself |
| `new-base-graft` | The new base with the adapter the rebase "trains" |

The genome in [`genome/`](genome/) has five rules and 52 cases, 12 of them counterfactual pairs:

| Rule | Checked with | How the new base picks it up |
|---|---|---|
| `ticket-json` | A JSON schema, and a program that checks the priority ([`ticket_priority.py`](genome/verifiers/ticket_priority.py)) | It already does this |
| `refund-limit` | Exact tool calls, with pairs on both sides of $500 | From the rule stated in the prompt |
| `no-legal-advice` | A phrase that must appear and one that must not | From the rule stated in the prompt |
| `card-numbers` | A pattern that must not appear | Only once the prompt shows a worked example |
| `account-closure` | The right tool at the right step, with pairs for verified and not verified | Only from training |

## What you see

```text
genome 0d148ae2   5 rules, 52 cases
source            52/52 cases hold on prod
zero-shot         13/52 cases   1 of 5 rules hold
prompt round 1    32/52 cases   3 of 5 rules hold
prompt round 2    40/52 cases   4 of 5 rules hold
training          mock_trainer.py on .graft/grafts/new-base/0001-0d148ae2/lora/round-1/dataset.jsonl
lora round 1      52/52 cases   5 of 5 rules hold
suite arithmetic  prod 66.7%   new-base 90.0%   grafted 90.0%   (+23.3 points, p=0.0156)
```

- **source.** The old model passes its own genome, so every case is behavior worth carrying over.
- **zero-shot.** 13 cases pass on the new base with no help: the ticket rule, and the refunds at or below $500, which any model gets right. The pairs fail, because the side above $500 does.
- **prompt round 1.** The failing rules are stated in the system prompt. Two more rules hold.
- **prompt round 2.** Worked examples are added for the rules that still fail. The card rule now holds; the account rule does not.
- **lora round 1.** The trainer gets one verified reply per training case. For the account rule those are attempts sampled from the new base that happened to pass the checks, and the old model's recorded replies where none did. The 11 held-out cases never reach it.
- **suite.** The grafted model keeps the new base's arithmetic score, 23 points above the old model.

Try the other endings:

```bash
graft rebase --abort; rm -rf .graft
graft rebase --target new-base --ladder zero-shot,prompt     # ends in a conflict: 12 account cases still fail
graft rebase --continue --ladder zero-shot,prompt,lora       # picks up from there
graft rebase --target prod                                   # same model: a fast-forward, no graft needed
```

## Files

| File | What it is |
|---|---|
| `graft.toml` | The two models, the ladder, the trainer, and a capability suite with gates |
| `genome/` | Rules, cases, the suite, and one code verifier |
| `mock_models.py` | The scripted models behind an OpenAI-compatible address |
| `mock_trainer.py` | Stands in for a trainer: same contract, no training |
| `load_adapter.py` | What `[rebase] after_train` runs so the endpoint serves the adapter |
