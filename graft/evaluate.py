"""Run cases against a model and decide gates."""
from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from graft import stats, verifiers
from graft.genome import Case, Genome, Rule
from graft.runners import Runner, RunnerError


@dataclass
class CaseResult:
    case: str
    rule: str
    split: str
    ok: bool
    problems: list[str] = field(default_factory=list)       # why it failed, one line per failed check
    replies: list[dict[str, Any]] = field(default_factory=list)
    unit_ok: list[bool] = field(default_factory=list)
    error: bool = False                                      # the model could not be reached or did not answer

    def to_json(self) -> dict[str, Any]:
        return {"case": self.case, "rule": self.rule, "split": self.split, "ok": self.ok,
                "problems": self.problems, "replies": self.replies, "unit_ok": self.unit_ok, "error": self.error}

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> CaseResult:
        return cls(**raw)


def run_case(case: Case, runner: Runner, graft: dict[str, Any] | None, cwd: Path | None) -> CaseResult:
    result = CaseResult(case=case.id, rule=case.rule, split=case.split, ok=True)
    for number, unit in enumerate(case.units, 1):
        side = f"side {number}: " if case.pair else ""
        try:
            reply = runner.reply(unit, graft)
        except RunnerError as err:
            result.error = True
            result.problems.append(f"{side}{err}")
            result.replies.append({})
            result.unit_ok.append(False)
            continue
        context = {"reference": unit.reference, "messages": unit.messages, "case": case.id, "cwd": cwd}
        problems = verifiers.failures(unit.verify, reply, context)
        result.replies.append(reply)
        result.unit_ok.append(not problems)
        result.problems += [f"{side}{p}" for p in problems]
    result.ok = all(result.unit_ok)
    return result


def evaluate(cases: Iterable[Case], runner: Runner, graft: dict[str, Any] | None = None, cwd: Path | None = None,
             tick: Callable[[CaseResult], None] | None = None) -> dict[str, CaseResult]:
    cases = list(cases)
    results: dict[str, CaseResult] = {}
    with ThreadPoolExecutor(max_workers=max(1, runner.concurrency)) as pool:
        for result in pool.map(lambda case: run_case(case, runner, graft, cwd), cases):
            results[result.case] = result
            if tick:
                tick(result)
    return results


@dataclass
class Gate:
    rule: str
    statement: str
    gate: float
    passed: int
    total: int
    holdout_passed: int
    holdout_total: int
    weak: int                    # cases that rest on heuristic matching alone

    @property
    def needed(self) -> int:
        return math.ceil(self.gate * self.total - 1e-9)

    @property
    def ok(self) -> bool:
        return self.passed >= self.needed

    @property
    def train_ok(self) -> bool:
        """The same gate over the training cases alone: what a rebase may steer by."""
        total = self.total - self.holdout_total
        return self.passed - self.holdout_passed >= math.ceil(self.gate * total - 1e-9)

    def to_json(self) -> dict[str, Any]:
        return {"rule": self.rule, "statement": self.statement, "gate": self.gate, "passed": self.passed,
                "total": self.total, "ok": self.ok,
                "holdout": {"passed": self.holdout_passed, "total": self.holdout_total}, "weak": self.weak}


def gate_of(rule: Rule, results: dict[str, CaseResult], excluded: set[str]) -> Gate:
    cases = [c for c in rule.cases if c.id not in excluded and c.id in results]
    held = [c for c in cases if c.split == "holdout"]
    return Gate(rule=rule.id, statement=rule.statement, gate=rule.gate,
                passed=sum(results[c.id].ok for c in cases), total=len(cases),
                holdout_passed=sum(results[c.id].ok for c in held), holdout_total=len(held),
                weak=sum(all(verifiers.weak_only(u.verify) for u in c.units) for c in cases))


def gates(genome: Genome, results: dict[str, CaseResult], excluded: set[str] | None = None) -> list[Gate]:
    excluded = excluded or set()
    return [gate_of(rule, results, excluded) for rule in genome.rules.values()
            if any(c.id not in excluded and c.id in results for c in rule.cases)]


def totals(found: list[Gate], confidence: float = 0.95, looks: int = 1) -> dict[str, Any]:
    """Sums over all rules. `looks` is how many times the held-out results were consulted while a graft was
    built; the bound is computed at 1 - (1 - confidence) / looks so that it still holds after that many looks."""
    passed, total = sum(g.passed for g in found), sum(g.total for g in found)
    held_passed, held_total = sum(g.holdout_passed for g in found), sum(g.holdout_total for g in found)
    looks = max(1, looks)
    bound = stats.lower_bound(held_passed, held_total, 1.0 - (1.0 - confidence) / looks)
    return {"passed": passed, "total": total, "rules_ok": sum(g.ok for g in found), "rules": len(found),
            "weak": sum(g.weak for g in found),
            "holdout": {"passed": held_passed, "total": held_total, "confidence": confidence, "looks": looks,
                        "lower_bound": round(bound, 4)}}


def compare(old: dict[str, CaseResult], new: dict[str, CaseResult]) -> dict[str, Any]:
    """Paired comparison of two models on the same cases."""
    shared = [case for case in old if case in new]
    wins = sum(new[c].ok and not old[c].ok for c in shared)
    losses = sum(old[c].ok and not new[c].ok for c in shared)
    count = len(shared) or 1
    return {"cases": len(shared), "old": sum(old[c].ok for c in shared) / count,
            "new": sum(new[c].ok for c in shared) / count, "wins": wins, "losses": losses,
            "p_value": round(stats.sign_test(wins, losses), 4)}
