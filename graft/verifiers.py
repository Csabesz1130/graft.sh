"""Deterministic verifiers: the same reply always gets the same verdict. No model judges a model here.

A verifier is a table with a "type". A reply is {"content": str, "tool_calls": [{"name", "arguments"}]}.
"""
from __future__ import annotations

import json
import re
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

WEAK = {"similar"}                   # heuristic matches; a certificate counts cases that rest on these alone
THINK = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


class VerifierError(Exception):
    """The verifier itself is wrong (unknown type, missing field), as opposed to a reply that fails it."""


def text_of(reply: dict[str, Any]) -> str:
    content = reply.get("content")
    if isinstance(content, list):   # content blocks
        content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return THINK.sub("", content or "").strip()


def calls_of(reply: dict[str, Any]) -> list[dict[str, Any]]:
    calls = []
    for call in reply.get("tool_calls") or []:
        function = call.get("function", call)
        arguments = function.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments) if arguments.strip() else {}
            except json.JSONDecodeError:
                arguments = {"$unparsed": arguments}
        calls.append({"name": function.get("name", ""), "arguments": arguments})
    return calls


def normalize(reply: dict[str, Any]) -> dict[str, Any]:
    return {"content": text_of(reply), "tool_calls": calls_of(reply)}


def extract_json(text: str) -> Any:
    """The first JSON value in the text; code fences and prose around it are ignored."""
    decoder = json.JSONDecoder()
    for match in re.finditer(r"[\[{]", text):
        try:
            return decoder.raw_decode(text, match.start())[0]
        except json.JSONDecodeError:
            continue
    return json.loads(text.strip())  # scalars; raises on anything else


def _tokens(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def token_f1(a: str, b: str) -> float:
    left, right = Counter(_tokens(a)), Counter(_tokens(b))
    if not left or not right:
        return 1.0 if left == right else 0.0
    overlap = sum((left & right).values())
    if not overlap:
        return 0.0
    precision, recall = overlap / sum(left.values()), overlap / sum(right.values())
    return 2 * precision * recall / (precision + recall)


def matches(expected: Any, actual: Any, exact: bool = False) -> bool:
    """Whether `actual` carries what `expected` asks for. Objects match as subsets unless `exact`."""
    if isinstance(expected, dict):
        if "$any" in expected:
            return True
        if "$regex" in expected:
            return isinstance(actual, str) and re.search(expected["$regex"], actual) is not None
        if "$approx" in expected:
            if isinstance(actual, bool) or not isinstance(actual, int | float):
                return False
            return abs(actual - expected["$approx"]) <= expected.get("tol", 1e-6)
        if not isinstance(actual, dict):
            return False
        if exact and set(expected) != set(actual):
            return False
        return all(key in actual and matches(value, actual[key], exact) for key, value in expected.items())
    if isinstance(expected, list):
        return (isinstance(actual, list) and len(actual) == len(expected)
                and all(matches(e, a, exact) for e, a in zip(expected, actual, strict=True)))
    if isinstance(expected, bool) or isinstance(actual, bool):
        return expected is actual
    if isinstance(expected, int | float) and isinstance(actual, int | float):
        return float(expected) == float(actual)
    return expected == actual


TYPES = {"string": str, "number": (int, float), "integer": int, "boolean": bool, "object": dict, "array": list}


def schema_errors(schema: dict[str, Any], value: Any, path: str = "$") -> list[str]:
    """A small, strict subset of JSON Schema: enough to pin the shape of structured output."""
    errors: list[str] = []
    if "anyOf" in schema:
        if not any(not schema_errors(option, value, path) for option in schema["anyOf"]):
            errors.append(f"{path}: matches none of anyOf")
    if "const" in schema and value != schema["const"]:
        errors.append(f"{path}: expected {schema['const']!r}")
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: {value!r} is not one of {schema['enum']}")
    kind = schema.get("type")
    if kind:
        kinds = kind if isinstance(kind, list) else [kind]
        ok = any((k == "null" and value is None)
                 or (k in TYPES and isinstance(value, TYPES[k])
                     and not (isinstance(value, bool) and k in ("number", "integer")))
                 for k in kinds)
        if not ok:
            return [*errors, f"{path}: expected {' or '.join(kinds)}"]
    if isinstance(value, dict):
        for key in schema.get("required", []):
            if key not in value:
                errors.append(f"{path}: missing {key!r}")
        properties = schema.get("properties", {})
        for key, item in value.items():
            if key in properties:
                errors += schema_errors(properties[key], item, f"{path}.{key}")
            elif schema.get("additionalProperties") is False:
                errors.append(f"{path}: unexpected {key!r}")
    if isinstance(value, list):
        if "items" in schema:
            for index, item in enumerate(value):
                errors += schema_errors(schema["items"], item, f"{path}[{index}]")
        if len(value) < schema.get("minItems", 0):
            errors.append(f"{path}: fewer than {schema['minItems']} items")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            errors.append(f"{path}: more than {schema['maxItems']} items")
    if isinstance(value, str):
        if "pattern" in schema and not re.search(schema["pattern"], value):
            errors.append(f"{path}: does not match {schema['pattern']!r}")
        if len(value) < schema.get("minLength", 0):
            errors.append(f"{path}: shorter than {schema['minLength']}")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            errors.append(f"{path}: longer than {schema['maxLength']}")
    if isinstance(value, int | float) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{path}: below {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{path}: above {schema['maximum']}")
    return errors


def _need(spec: dict[str, Any], *keys: str) -> None:
    if not any(key in spec for key in keys):
        raise VerifierError(f"verifier {spec.get('type')!r} needs {' or '.join(repr(k) for k in keys)}")


def _strings(spec: dict[str, Any]) -> tuple[list[str], bool]:
    """The strings a contains-style verifier looks for, and whether all of them are required."""
    _need(spec, "value", "all", "any")
    if "any" in spec:
        return list(spec["any"]), False
    values = spec.get("all", spec.get("value"))
    return ([values] if isinstance(values, str) else list(values)), True


def _flags(spec: dict[str, Any]) -> int:
    flags = re.DOTALL
    if spec.get("ignore_case", True):
        flags |= re.IGNORECASE
    return flags


def check(spec: dict[str, Any], reply: dict[str, Any], context: dict[str, Any] | None = None) -> str | None:
    """None when the reply satisfies the verifier, otherwise one line saying why not."""
    context = context or {}
    kind = spec.get("type")
    text = text_of(reply)
    calls = calls_of(reply)

    if kind in ("contains", "not_contains"):
        wanted, need_all = _strings(spec)
        fold = (lambda s: s.lower()) if spec.get("ignore_case", True) else (lambda s: s)
        found = [w for w in wanted if fold(w) in fold(text)]
        if kind == "not_contains":
            return f"contains {found[0]!r}" if found else None
        if need_all and len(found) < len(wanted):
            return f"missing {next(w for w in wanted if w not in found)!r}"
        if not need_all and not found:
            return f"contains none of {wanted}"
        return None

    if kind in ("regex", "not_regex"):
        _need(spec, "pattern")
        try:
            found = re.search(spec["pattern"], text, _flags(spec))
        except re.error as err:
            raise VerifierError(f"bad pattern {spec['pattern']!r}: {err}") from err
        if kind == "regex":
            return None if found else f"no match for /{spec['pattern']}/"
        return f"matches /{spec['pattern']}/ at {found.group(0)!r}" if found else None

    if kind == "equals":
        _need(spec, "value")
        squash = (lambda s: " ".join(s.split())) if spec.get("normalize", True) else (lambda s: s)
        return None if squash(text) == squash(str(spec["value"])) else "text differs from the expected reply"

    if kind == "length":
        _need(spec, "max_chars", "min_chars")
        if len(text) > spec.get("max_chars", len(text)):
            return f"{len(text)} characters, above {spec['max_chars']}"
        if len(text) < spec.get("min_chars", 0):
            return f"{len(text)} characters, below {spec['min_chars']}"
        return None

    if kind == "json":
        try:
            value = extract_json(text)
        except (json.JSONDecodeError, ValueError):
            return "reply holds no JSON value"
        if "schema" in spec:
            errors = schema_errors(spec["schema"], value)
            if errors:
                return errors[0]
        if "subset" in spec and not matches(spec["subset"], value):
            return "JSON lacks the expected fields or values"
        if "equals" in spec and not matches(spec["equals"], value, exact=True):
            return "JSON differs from the expected value"
        return None

    if kind == "tool_call":
        _need(spec, "name")
        pool = calls
        if "index" in spec:
            pool = calls[spec["index"]:spec["index"] + 1]
        named = [c for c in pool if c["name"] == spec["name"]]
        if not named:
            made = ", ".join(c["name"] for c in calls) or "none"
            return f"no call to {spec['name']} (calls made: {made})"
        if "arguments" in spec and not any(
                matches(spec["arguments"], c["arguments"], bool(spec.get("exact"))) for c in named):
            return f"{spec['name']} was called with {json.dumps(named[0]['arguments'], sort_keys=True)}"
        if "count" in spec and len(calls) != spec["count"]:
            return f"{len(calls)} tool calls, expected {spec['count']}"
        return None

    if kind == "no_tool_call":
        banned = [c for c in calls if "name" not in spec or c["name"] == spec["name"]]
        return f"called {banned[0]['name']}" if banned else None

    if kind == "similar":
        reference = spec.get("reference") or text_of(context.get("reference") or {})
        if not reference:
            raise VerifierError("verifier 'similar' needs a 'reference' here or on the case")
        score, floor = token_f1(text, reference), float(spec.get("min_f1", 0.5))
        return None if score >= floor else f"token F1 {score:.2f} against the reference, below {floor}"

    if kind in ("all", "any"):
        _need(spec, "of")
        problems = [check(inner, reply, context) for inner in spec["of"]]
        failed = [p for p in problems if p]
        if kind == "all":
            return failed[0] if failed else None
        return None if len(failed) < len(problems) else "none of: " + "; ".join(failed)

    if kind == "not":
        _need(spec, "of")
        return None if check(spec["of"], reply, context) else "the forbidden condition holds"

    if kind == "command":
        _need(spec, "argv")
        payload = {"reply": normalize(reply), "messages": context.get("messages"),
                   "reference": context.get("reference"), "case": context.get("case")}
        try:
            done = subprocess.run([str(a) for a in spec["argv"]], input=json.dumps(payload), text=True,
                                  capture_output=True, timeout=float(spec.get("timeout", 30)),
                                  cwd=context.get("cwd") or Path.cwd(), check=False)
        except FileNotFoundError as err:
            raise VerifierError(f"command verifier: {err}") from err
        except subprocess.TimeoutExpired:
            return f"verifier {spec['argv'][0]} ran past {spec.get('timeout', 30)} s"
        if done.returncode == 0:
            return None
        said = (done.stderr.strip() or done.stdout.strip()).splitlines()
        return said[-1] if said else f"{spec['argv'][0]} exited {done.returncode}"

    raise VerifierError(f"unknown verifier type {kind!r}")


def failures(specs: list[dict[str, Any]], reply: dict[str, Any], context: dict[str, Any] | None = None) -> list[str]:
    return [problem for spec in specs if (problem := check(spec, reply, context))]


def weak_only(specs: list[dict[str, Any]]) -> bool:
    """True when nothing but heuristic matching says the reply is right."""
    def kinds(spec: dict[str, Any]) -> list[str]:
        if spec.get("type") in ("all", "any"):
            return [kind for inner in spec.get("of", []) for kind in kinds(inner)]
        return [str(spec.get("type"))]
    found = [kind for spec in specs for kind in kinds(spec)]
    return any(kind in WEAK for kind in found) and all(kind in WEAK or kind == "no_tool_call" for kind in found)


def from_reference(reference: dict[str, Any]) -> list[dict[str, Any]]:
    """Verifiers for a recorded reply nobody has written rules for yet.

    Tool calls are pinned exactly: same tools, same arguments. Plain text can only be matched
    loosely, and is marked weak until someone writes a real verifier for it.
    """
    calls = calls_of(reference)
    if calls:
        return [{"type": "tool_call", "name": c["name"], "arguments": c["arguments"], "exact": True,
                 "index": i, "count": len(calls)} for i, c in enumerate(calls)]
    text = text_of(reference)
    try:
        value = extract_json(text)
    except (json.JSONDecodeError, ValueError):
        return [{"type": "no_tool_call"}, {"type": "similar", "min_f1": 0.5}]
    if isinstance(value, dict | list):
        return [{"type": "json", "equals": value}]
    return [{"type": "no_tool_call"}, {"type": "similar", "min_f1": 0.5}]
