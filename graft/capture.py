"""Turn recorded production traffic into genome cases.

goethe    a Goethe trace folder (its docs/traces.md, schema 1): chat completions that finished cleanly
jsonl     chat fine-tuning files: one {"messages": [..., assistant], "tools": [...]} object per line

Each assistant turn becomes one case: the history before it is the request, the recorded reply is the
reference, and verifiers are derived from the reply. By default (`structured`) they pin what the reply decided:
tool names, numbers, ids and codes, enum values and values copied from the request; free-text strings such as an
escalation's reason only have to be present. `exact` pins every value, prose included. Plain text gets a similarity
match either way.
"""
from __future__ import annotations

import gzip
import json
import re
from pathlib import Path
from typing import Any

from graft import verifiers
from graft.genome import digest


class CaptureError(Exception):
    pass


CHECKS = ("structured", "exact")
CODE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/#@+-]{0,39}$")   # an id, a code, an enum value: no spaces
PROSE = {"$regex": r"\S"}                                       # present and not empty, any wording
FENCE_OPEN, FENCE_CLOSE = r"^\s*(?:```(?:json)?\s*)?", r"\s*(?:```)?\s*$"
ONLY_JSON = {   # a reply that is one JSON value and nothing else, maybe in a code fence; the inside must look like JSON
    dict: re.compile(FENCE_OPEN + r"\{\s*(?:\"[\s\S]*)?\}" + FENCE_CLOSE),
    list: re.compile(FENCE_OPEN + r"\[\s*(?:(?:[\[{\"\d-]|true|false|null)[\s\S]*)?\]" + FENCE_CLOSE),
}


def _request_text(messages: list[dict[str, Any]]) -> str:
    """Everything the model was given: message text, and the arguments of tool calls already made."""
    parts = []
    for message in messages:
        parts.append(verifiers.text_of(message))
        parts.extend(json.dumps(c["arguments"], ensure_ascii=False) for c in verifiers.calls_of(message))
    return "\n".join(parts)


def _enums(tools: list[dict[str, Any]] | None, name: str) -> dict[str, list[Any]]:
    for tool in tools or []:
        function = tool.get("function", tool)
        if function.get("name") == name:
            properties = (function.get("parameters") or {}).get("properties") or {}
            return {key: spec["enum"] for key, spec in properties.items() if isinstance(spec, dict) and "enum" in spec}
    return {}


def structure(value: Any, request: str, enums: dict[str, list[Any]] | None = None, key: str | None = None,
              loosened: list[int] | None = None, top: bool = True) -> Any:
    """What a recorded value pins: numbers, booleans, null, ids and codes, enum values and strings copied from the
    request stay exact; other strings (prose a second model will word differently) only have to be non-empty."""
    if isinstance(value, dict):   # a tool's enums name its top-level arguments only
        return {k: structure(v, request, enums if top else None, k, loosened, False) for k, v in value.items()}
    if isinstance(value, list):
        return [structure(v, request, None, None, loosened, False) for v in value]
    if not isinstance(value, str):
        return value
    if (enums and key in enums) or CODE.match(value) or value in request:
        return value
    if loosened is not None:
        loosened[0] += 1
    return dict(PROSE)


def structured_verifiers(reply: dict[str, Any], messages: list[dict[str, Any]],
                         tools: list[dict[str, Any]] | None, loosened: list[int] | None = None) -> list[dict[str, Any]]:
    """Verifiers for a recorded reply that a different model can pass: the decisions are pinned, the wording is not.
    Tool calls keep their names, order and count; plain text still gets the weak similarity check."""
    request = _request_text(messages)
    calls = verifiers.calls_of(reply)
    if calls:
        return [{"type": "tool_call", "name": c["name"], "index": i, "count": len(calls),
                 "arguments": structure(c["arguments"], request, _enums(tools, c["name"]), None, loosened)}
                for i, c in enumerate(calls)]
    text = verifiers.text_of(reply)
    try:
        value = verifiers.extract_json(text)
    except (json.JSONDecodeError, ValueError):
        return verifiers.from_reference(reply)
    if isinstance(value, dict | list):
        found = [{"type": "json", "subset": structure(value, request, None, None, loosened)}]
        only = ONLY_JSON[type(value)]
        if only.match(text):   # the recorded reply was JSON and nothing else: so must the new one be
            found.append({"type": "regex", "pattern": only.pattern})
        return found
    return verifiers.from_reference(reply)


def case_from(messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None, reference: dict[str, Any],
              source: dict[str, Any], checks: str = "structured", loosened: list[int] | None = None,
              ) -> dict[str, Any] | None:
    reply = verifiers.normalize(reference)
    if not reply["content"] and not reply["tool_calls"]:
        return None
    verify = (structured_verifiers(reply, messages, tools, loosened) if checks == "structured"
              else verifiers.from_reference(reply))
    case: dict[str, Any] = {"id": digest([messages, tools or []])[:12], "messages": messages,
                            "reference": reply, "verify": verify, "source": source}
    if tools:
        case["tools"] = tools
    return case


def from_jsonl(path: Path, every_turn: bool = False, checks: str = "structured",
               loosened: list[int] | None = None) -> list[dict[str, Any]]:
    cases = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            messages = row["messages"]
        except (json.JSONDecodeError, KeyError, TypeError) as err:
            raise CaptureError(f"{path}:{number}: expected an object with \"messages\"") from err
        turns = [i for i, m in enumerate(messages) if m.get("role") == "assistant"]
        for index in (turns if every_turn else turns[-1:]):
            case = case_from(messages[:index], row.get("tools"), messages[index],
                             {"file": path.name, "line": number}, checks, loosened)
            if case:
                cases.append(case)
    return cases


def goethe_turns(folder: Path) -> dict[str, dict[str, Any]]:
    """Every turn under <folder>/v1 by turn id. A torn last gzip member (a crash mid-write) ends that file."""
    if not (folder / "v1").is_dir():
        raise CaptureError(f"{folder} is not a Goethe trace folder: it has no v1/ inside")
    turns: dict[str, dict[str, Any]] = {}
    for path in sorted((folder / "v1").rglob("*.jsonl.gz")):
        try:
            with gzip.open(path, "rt", encoding="utf-8") as lines:
                for line in lines:
                    turn = json.loads(line)
                    turns[turn["turn_id"]] = turn
        except (EOFError, gzip.BadGzipFile, json.JSONDecodeError, KeyError):
            continue
    return turns


def _chat_reply(turn: dict[str, Any]) -> dict[str, Any] | None:
    body = (turn.get("response") or {}).get("body")
    if not isinstance(body, dict):
        return None
    choices = body.get("choices") or [{}]
    message = choices[0].get("message")
    if not isinstance(message, dict):
        return None
    # what a client would resend: no reasoning, no engine extras
    return {"role": "assistant", **{key: message[key] for key in ("content", "tool_calls") if message.get(key)}}


def _history(turn: dict[str, Any], turns: dict[str, dict[str, Any]]) -> list[dict[str, Any]] | None:
    """The full message list of a turn: the parent's list, the parent's reply, then this turn's new messages."""
    chain = [turn]
    while chain[-1].get("parent"):
        parent = turns.get(chain[-1]["parent"]["turn_id"])
        if parent is None or len(chain) > 10_000:
            return None
        chain.append(parent)
    messages: list[dict[str, Any]] = []
    for index in range(len(chain) - 1, -1, -1):
        current = chain[index]
        if index < len(chain) - 1 and current["parent"].get("with_reply"):
            reply = _chat_reply(chain[index + 1])
            if reply is None:
                return None
            messages.append(reply)
        messages += current.get("messages") or []
    return messages


def _params(turn: dict[str, Any], turns: dict[str, dict[str, Any]]) -> dict[str, Any]:
    seen = 0
    while turn.get("params_from_parent") and turn.get("parent") and seen < 10_000:
        turn = turns.get(turn["parent"]["turn_id"]) or {}
        seen += 1
    return turn.get("params") or {}


def from_goethe(folder: Path, model: str | None = None, since: str | None = None, tools_only: bool = False,
                checks: str = "structured", loosened: list[int] | None = None,
                ) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Cases from a Goethe trace folder, and how many turns were skipped for which reason."""
    turns = goethe_turns(folder)
    skipped = {"other api": 0, "failed or cut off": 0, "other model": 0, "older": 0, "no tool call": 0,
               "history missing": 0, "empty reply": 0}
    by_request: dict[str, dict[str, Any]] = {}
    for turn in sorted(turns.values(), key=lambda t: t.get("started_at", "")):
        response = turn.get("response") or {}
        if turn.get("api") != "chat.completions":
            skipped["other api"] += 1
        elif response.get("status") != 200 or not response.get("complete", True) or response.get("truncated"):
            skipped["failed or cut off"] += 1
        elif model and turn.get("model") != model:
            skipped["other model"] += 1
        elif since and str(turn.get("started_at", "")) < since:
            skipped["older"] += 1
        else:
            reply = _chat_reply(turn)
            finished = ((response.get("body") or {}).get("choices") or [{}])[0].get("finish_reason")
            messages = _history(turn, turns)
            if reply is None or finished == "length":
                skipped["failed or cut off"] += 1
            elif messages is None:
                skipped["history missing"] += 1
            elif tools_only and not reply.get("tool_calls"):
                skipped["no tool call"] += 1
            else:
                source = {"goethe_turn": turn["turn_id"], "model": turn.get("model"),
                          "model_fingerprint": turn.get("model_fingerprint"), "at": turn.get("started_at")}
                case = case_from(messages, _params(turn, turns).get("tools"), reply, source, checks, loosened)
                if case is None:
                    skipped["empty reply"] += 1
                else:
                    by_request[case["id"]] = case       # a regenerated reply replaces the earlier one
    return list(by_request.values()), {reason: count for reason, count in skipped.items() if count}
