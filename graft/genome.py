"""The behavioral genome: rules, and the cases that pin each rule down.

On disk, under the genome folder:

    rules/<rule>.toml      the rule in words, its gate, and verifiers every case of the rule gets
    cases/<rule>.jsonl     one case per line
    suites/<name>.jsonl    capability suites: same case format, reported but not part of any rule
    verifiers/             programs that `command` verifiers run

Every file under the folder is part of the genome id, so a certificate names exactly what was checked.
"""
from __future__ import annotations

import hashlib
import json
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class GenomeError(Exception):
    pass


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def tree_digest(folder: Path) -> str:
    """The sha256 of a folder's contents: every file's path relative to the folder, and its bytes."""
    whole = hashlib.sha256()
    for path in sorted(p for p in Path(folder).rglob("*") if p.is_file()):
        part = hashlib.sha256()
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                part.update(block)
        whole.update(f"{path.relative_to(folder).as_posix()}\0{part.hexdigest()}\n".encode())
    return whole.hexdigest()


@dataclass
class Unit:
    """One request and what its reply must satisfy. A plain case has one unit, a counterfactual pair two."""
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] | None
    verify: list[dict[str, Any]]
    reference: dict[str, Any] | None


@dataclass
class Case:
    id: str
    rule: str
    units: list[Unit]
    split: str                       # "train" or "holdout"; empty until the rule's cases are split
    source: dict[str, Any] = field(default_factory=dict)

    @property
    def pair(self) -> bool:
        return len(self.units) > 1


@dataclass
class Rule:
    id: str
    statement: str
    gate: float = 1.0
    verify: list[dict[str, Any]] = field(default_factory=list)
    cases: list[Case] = field(default_factory=list)


@dataclass
class Genome:
    folder: Path
    id: str
    rules: dict[str, Rule]

    @property
    def cases(self) -> list[Case]:
        return [case for rule in self.rules.values() for case in rule.cases]

    def case(self, case_id: str) -> Case:
        for case in self.cases:
            if case.id == case_id:
                return case
        raise GenomeError(f"no case {case_id!r} in the genome")


def assign_splits(cases: list[Case], holdout: float) -> None:
    """Hold out a share of one rule's cases. The choice depends only on the case ids, so it is the same
    on every machine; a case that names its own "split" keeps it."""
    free = [case for case in cases if not case.split]
    want = max(1, int(holdout * len(free) + 0.5)) if holdout > 0 and len(free) >= 4 else 0
    ranked = sorted(free, key=lambda case: hashlib.sha256(f"graft-split:{case.id}".encode()).hexdigest())
    held = {case.id for case in ranked[:want]}
    for case in free:
        case.split = "holdout" if case.id in held else "train"


def genome_id(folder: Path) -> str:
    entries = []
    for path in sorted(p for p in folder.rglob("*") if p.is_file()):
        rel = path.relative_to(folder).as_posix()
        if "__pycache__" in rel or rel.endswith(".pyc"):
            continue
        entries.append((rel, hashlib.sha256(path.read_bytes()).hexdigest()))
    return digest(entries)


def _unit(raw: dict[str, Any], inherited: list[dict[str, Any]], where: str) -> Unit:
    messages = raw.get("messages")
    if not isinstance(messages, list) or not messages:
        raise GenomeError(f"{where}: needs a non-empty \"messages\" list")
    for message in messages:
        if not isinstance(message, dict) or "role" not in message:
            raise GenomeError(f"{where}: every message needs a \"role\"")
    own = raw.get("verify", [])
    if not isinstance(own, list):
        raise GenomeError(f"{where}: \"verify\" must be a list")
    reference = raw.get("reference")
    if isinstance(reference, str):
        reference = {"content": reference}
    if reference is not None and not isinstance(reference, dict):
        raise GenomeError(f"{where}: \"reference\" must be text or a reply object")
    verify = [*inherited, *own]
    if not verify:
        raise GenomeError(f"{where}: nothing verifies this case; add \"verify\" to the case or to its rule")
    tools = raw.get("tools")
    return Unit(messages=messages, tools=tools or None, verify=verify, reference=reference)


def parse_case(raw: dict[str, Any], rule: str, inherited: list[dict[str, Any]], where: str) -> Case:
    if not isinstance(raw, dict):
        raise GenomeError(f"{where}: a case is a JSON object")
    sides = raw.get("pair")
    if sides is not None:
        if not isinstance(sides, list) or len(sides) != 2:
            raise GenomeError(f"{where}: \"pair\" holds exactly two sides")
        shared = {key: raw[key] for key in ("tools",) if key in raw}
        units = [_unit({**shared, **side}, inherited, f"{where} side {n}") for n, side in enumerate(sides, 1)]
    else:
        units = [_unit(raw, inherited, where)]
    case_id = raw.get("id") or digest([[u.messages, u.tools, u.verify] for u in units])[:12]
    if not ID.match(str(case_id)):
        raise GenomeError(f"{where}: case id {case_id!r} may hold letters, digits, dot, dash and underscore only")
    split = raw.get("split") or ""
    if split not in ("", "train", "holdout"):
        raise GenomeError(f"{where}: \"split\" is \"train\" or \"holdout\"")
    return Case(id=str(case_id), rule=rule, units=units, split=split, source=raw.get("source") or {})


def read_cases(path: Path, rule: str, inherited: list[dict[str, Any]], holdout: float) -> list[Case]:
    cases = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        where = f"{path}:{number}"
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as err:
            raise GenomeError(f"{where}: not JSON ({err.msg})") from err
        cases.append(parse_case(raw, rule, inherited, where))
    assign_splits(cases, holdout)
    return cases


def load(folder: Path, holdout: float = 0.2) -> Genome:
    if not folder.is_dir():
        raise GenomeError(f"no genome folder at {folder}; run `graft init`")
    rules: dict[str, Rule] = {}
    for path in sorted((folder / "rules").glob("*.toml")):
        try:
            raw = tomllib.loads(path.read_text(encoding="utf-8"))
        except tomllib.TOMLDecodeError as err:
            raise GenomeError(f"{path}: {err}") from err
        if not ID.match(path.stem):
            raise GenomeError(f"{path}: rule ids may hold letters, digits, dot, dash and underscore only")
        statement = str(raw.get("statement", "")).strip()
        if not statement:
            raise GenomeError(f"{path}: a rule needs a \"statement\": the behavior in one or two sentences")
        gate = float(raw.get("gate", 1.0))
        if not 0.0 < gate <= 1.0:
            raise GenomeError(f"{path}: gate is the share of cases that must pass, above 0 and at most 1")
        verify = raw.get("verify", [])
        if not isinstance(verify, list):
            raise GenomeError(f"{path}: use [[verify]] tables")
        rules[path.stem] = Rule(id=path.stem, statement=statement, gate=gate, verify=verify)
    seen: dict[str, str] = {}
    for path in sorted((folder / "cases").glob("*.jsonl")):
        if path.stem not in rules:
            raise GenomeError(f"{path}: no rule {path.stem!r}; add rules/{path.stem}.toml (`graft rule add`)")
        rule = rules[path.stem]
        rule.cases = read_cases(path, rule.id, rule.verify, holdout)
        for case in rule.cases:
            if case.id in seen:
                raise GenomeError(f"case id {case.id!r} is used in both {seen[case.id]} and {rule.id}")
            seen[case.id] = rule.id
    return Genome(folder=folder, id=genome_id(folder), rules=rules)


def load_suite(path: Path) -> list[Case]:
    """A capability suite: cases under one made-up rule, all held out (nothing ever trains on a suite)."""
    if not path.is_file():
        raise GenomeError(f"no suite file at {path}")
    cases = read_cases(path, f"suite:{path.stem}", [], holdout=0.0)
    for case in cases:
        case.split = "holdout"
    return cases


def append_cases(folder: Path, rule: str, raw_cases: list[dict[str, Any]]) -> int:
    """Add cases to a rule's file, skipping ids that are already there. Returns how many were added."""
    path = folder / "cases" / f"{rule}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    known = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                known.add(json.loads(line).get("id"))
    added = 0
    with path.open("a", encoding="utf-8") as out:
        for raw in raw_cases:
            if raw.get("id") in known:
                continue
            known.add(raw.get("id"))
            out.write(json.dumps(raw, ensure_ascii=False, sort_keys=True) + "\n")
            added += 1
    return added


def write_rule(folder: Path, rule_id: str, statement: str, gate: float = 1.0) -> Path:
    if not ID.match(rule_id):
        raise GenomeError(f"rule id {rule_id!r} may hold letters, digits, dot, dash and underscore only")
    path = folder / "rules" / f"{rule_id}.toml"
    if path.exists():
        raise GenomeError(f"rule {rule_id!r} already exists at {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    body = f"statement = {json.dumps(statement, ensure_ascii=False)}\n"
    if gate != 1.0:
        body += f"gate = {gate}\n"
    path.write_text(body, encoding="utf-8")
    (folder / "cases").mkdir(exist_ok=True)
    return path
