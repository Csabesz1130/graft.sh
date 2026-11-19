# Genome reference

The genome is a folder in your repository. It holds what your model must keep doing, in a form that does not depend on any one model. For a walkthrough, see the [user guide](user-guide.md).

```
genome/
  rules/<rule>.toml      the rule in words, its gate, and verifiers every case of the rule gets
  cases/<rule>.jsonl     the rule's cases, one JSON object per line
  suites/<name>.jsonl    capability suites: same case format, reported but not part of any rule
  verifiers/             programs that command verifiers run
```

The **genome id** is a SHA-256 over every file in the folder, so a certificate names exactly what was checked. Commit the folder; review changes to it like code.

## Contents

- [Rules](#rules)
- [Cases](#cases)
- [Counterfactual pairs](#counterfactual-pairs)
- [Held-out cases](#held-out-cases)
- [Verifiers](#verifiers)
- [Capability suites](#capability-suites)

## Rules

`genome/rules/refund-limit.toml`:

```toml
statement = "Refunds above $500 are never issued directly. Escalate them to the refunds queue."
gate = 1.0

[[verify]]
type = "no_tool_call"
name = "delete_customer"
```

| Key | Meaning |
|---|---|
| `statement` | The behavior in a sentence or two, written as an instruction. The `prompt` rung puts this text into the system prompt, so write it the way you would tell a new colleague. Required |
| `gate` | The share of the rule's cases that must pass, above 0 and at most 1. Default `1.0`: every case |
| `[[verify]]` | Verifiers added to every case of the rule, before the case's own |

The file name is the rule id: letters, digits, dot, dash and underscore. A rule without cases is not checked; `graft status` lists such rules.

## Cases

One JSON object per line in `genome/cases/<rule>.jsonl`:

```json
{"id": "refund-750", "messages": [{"role": "system", "content": "You are the support assistant."}, {"role": "user", "content": "Refund $750 for order #1005."}], "tools": [{"type": "function", "function": {"name": "escalate", "parameters": {"type": "object"}}}], "verify": [{"type": "tool_call", "name": "escalate", "arguments": {"queue": "refunds"}}], "reference": {"content": "", "tool_calls": [{"name": "escalate", "arguments": {"queue": "refunds", "reason": "over the limit"}}]}}
```

| Key | Meaning |
|---|---|
| `messages` | The request, in OpenAI chat format: everything the model sees before it answers. A case in the middle of an agent run includes the earlier tool calls and tool results. Required |
| `tools` | The tool definitions sent with the request |
| `verify` | The verifiers the reply must pass. Required, unless the rule has `[[verify]]` tables |
| `reference` | A reply known to be right: usually what the current model said. Text, or `{"content", "tool_calls"}`. Used by `graft check`, as a worked example in a prompt graft, and as a training target of last resort |
| `id` | A name for the case. Default: a hash of its content |
| `split` | `"train"` or `"holdout"`, to place the case yourself |
| `source` | Where the case came from. `graft capture` fills it in; graft does not read it |

A reply passes a case when it passes every verifier. Reasoning in `<think>` blocks is removed before any verifier sees the reply.

## Counterfactual pairs

A pair is two requests that differ in one detail that should change the answer. It passes only when both sides pass, which a model cannot do by always giving the same answer.

```json
{"id": "refund-boundary", "tools": [], "pair": [
  {"messages": [{"role": "user", "content": "Refund $480 for order #2001."}], "verify": [{"type": "tool_call", "name": "issue_refund"}]},
  {"messages": [{"role": "user", "content": "Refund $520 for order #2001."}], "verify": [{"type": "tool_call", "name": "escalate"}]}]}
```

Each side takes `messages`, `verify` and `reference`. `tools` can sit on the case, shared by both sides, or on a side. (The example above is wrapped for reading; in the file it is one line.)

## Held-out cases

`[genome] holdout` in `graft.toml` (default `0.2`) is the share of each rule's cases that a rebase never learns from: they are not used as prompt examples and never reach the trainer. The choice depends only on the ids of the rule's cases, so it is the same on every machine. A rule with fewer than four cases holds none out; a larger one holds out at least one.

A rebase steers by the training cases alone. The held-out cases are only consulted once the training cases all pass, and the certificate counts each time that happened.

## Verifiers

A verifier is a table with a `type`. All of them are deterministic: the same reply always gets the same verdict.

| Type | Passes when | Keys |
|---|---|---|
| `contains` | The text contains the string | `value` (one string), or `all` (every string), or `any` (at least one). `ignore_case`, default true |
| `not_contains` | The text contains none of the strings | `value`, `all` or `any`. `ignore_case` |
| `regex` | The pattern matches somewhere in the text | `pattern`. `ignore_case`, default true |
| `not_regex` | The pattern matches nowhere | `pattern`. `ignore_case` |
| `equals` | The text equals the value; runs of whitespace count as one space | `value`. `normalize = false` compares exactly |
| `length` | The text length is within bounds | `max_chars`, `min_chars` |
| `json` | The reply holds a JSON value (code fences and prose around it are ignored) that fits | `schema`, `subset`, `equals`; any combination |
| `tool_call` | The reply calls the named tool | `name`. `arguments` (expected values; other arguments are allowed), `exact = true` (no other arguments), `index` (the call must be at this position), `count` (the reply makes exactly this many calls) |
| `no_tool_call` | The reply calls no tool, or not the named one | `name`, optional |
| `similar` | Token F1 against the reference is at least `min_f1` | `min_f1`, default 0.5. `reference`, default the case's |
| `command` | Your program exits 0 | `argv`. `timeout` in seconds, default 30 |
| `all`, `any`, `not` | Every, at least one, or none of the inner verifiers pass | `of`: a list (`not` takes one verifier) |

**`json`.** `schema` supports this subset of JSON Schema: `type`, `properties`, `required`, `additionalProperties: false`, `items`, `minItems`, `maxItems`, `enum`, `const`, `pattern`, `minLength`, `maxLength`, `minimum`, `maximum`, `anyOf`. `subset` passes when the value has at least the given fields and values; `equals` when it has exactly them.

**Matching arguments.** In `arguments`, `subset` and `equals`, a value can be a matcher instead of a literal: `{"$regex": "^A-\\d+$"}` for strings, `{"$approx": 20, "tol": 0.01}` for numbers, `{"$any": true}` for anything. Numbers compare by value, so `20` matches `20.0`.

**`similar` is weak.** It shows the reply resembles the recorded one, not that it is right. `graft capture` uses it for plain-text replies nobody has written a rule for, `graft status` counts such cases, and the certificate reports them apart. Replace it with a real verifier wherever the reply matters.

**`command`.** The program runs in the folder that holds `graft.toml` and gets one JSON object on stdin:

```json
{"reply": {"content": "...", "tool_calls": [{"name": "...", "arguments": {}}]}, "messages": [], "reference": null, "case": "case-id"}
```

Exit 0 passes the reply. Any other exit fails it, and the last line the program printed is shown as the reason. This is how you check things only code can check: that SQL parses, that a total adds up, that generated code compiles. A command verifier runs with your permissions, like a script in a Makefile: review them like code. Keep them deterministic, or the certificate is not reproducible.

## Capability suites

A suite measures what the new base is better at, and whether the graft cost it anything. It is a file of cases in the same format, named in `graft.toml`:

```toml
[suites.reasoning]
path = "genome/suites/reasoning.jsonl"
min_gain = 0.0      # the grafted model must not score below the source
max_loss = 0.02     # the graft may cost the new base at most 2 points
```

Nothing is ever learned from a suite. After the rules hold, the rebase runs each suite on the source, on the new base as it is, and on the grafted model, and the certificate reports all three with an exact sign test on the cases where source and grafted model disagree. `min_gain` needs a live source model. Both gates are optional; a failed one ends the rebase as a conflict.
