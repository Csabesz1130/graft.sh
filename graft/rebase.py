"""graft rebase: carry a genome onto a new base model, one rung at a time.

zero-shot   ask the new model as it is; behavior that already holds needs no work
prompt      state the failing rules in the system prompt, then add worked examples for what still fails
lora        train an adapter on verified replies, sampled from the new model itself where one passes

After every step the whole genome is checked again, so a fix for one rule cannot quietly break another.
The run stops at the first rung where every gate holds and writes a certificate. If no rung gets there,
it stops with the failing cases listed, and `graft rebase --continue` picks up after you change something.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from graft import certificate, evaluate, runners, verifiers
from graft.config import Project
from graft.evaluate import CaseResult
from graft.genome import Case, Genome, Unit, load_suite, tree_digest

HEAD = "REBASE_HEAD"
PASSED, CONFLICT, PAUSED = 0, 1, 3

PREAMBLE = "Standing rules. They apply to every reply and come before your defaults."
EXAMPLES = "Examples of these rules applied:"


class RebaseError(Exception):
    pass


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def head_path(project: Project) -> Path:
    return project.state_dir / HEAD


def in_progress(project: Project) -> dict[str, Any] | None:
    path = head_path(project)
    if not path.is_file():
        return None
    run_dir = project.state_dir / "grafts" / path.read_text(encoding="utf-8").strip()
    state = run_dir / "state.json"
    return json.loads(state.read_text(encoding="utf-8")) if state.is_file() else None


def runs(project: Project) -> list[dict[str, Any]]:
    found = []
    for state in sorted((project.state_dir / "grafts").glob("*/*/state.json")):
        found.append(json.loads(state.read_text(encoding="utf-8")))
    return sorted(found, key=lambda s: s.get("started", ""))


def run_dir_of(project: Project, state: dict[str, Any]) -> Path:
    return project.state_dir / "grafts" / state["target"] / state["run"]


def find_run(project: Project, wanted: str | None, for_graft: bool = False) -> dict[str, Any]:
    """A run by `target`, `target/run`, or run id; the latest one when nothing is named.

    With `for_graft`, a target or nothing skips aborted runs: their graft was given up, and an aborted run
    must not hide the graft of a run before it. A run named exactly is returned whatever its status.
    """
    known = runs(project)
    if not known:
        raise RebaseError("no rebase has run here yet")
    if wanted:
        exact = [s for s in known if wanted in (s["run"], f"{s['target']}/{s['run']}")]
        if exact:
            return exact[-1]
        known = [s for s in known if s["target"] == wanted]
        if not known:
            raise RebaseError(f"no rebase matches {wanted!r}; `graft log` lists them")
    if for_graft:
        known = [s for s in known if s.get("status") != "aborted"]
        if not known:
            onto = f" onto {wanted}" if wanted else ""
            raise RebaseError(f"every rebase{onto} was aborted, so there is no graft to use; "
                              f"name a run to use its graft anyway (`graft log` lists them)")
    return known[-1]


def abort(project: Project) -> dict[str, Any] | None:
    """Drop the rebase in progress. What it trained and recorded stays on disk."""
    state = in_progress(project)
    if state:
        state.update(status="aborted", finished=now())
        path = run_dir_of(project, state) / "state.json"
        path.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    head_path(project).unlink(missing_ok=True)
    return state


def adapter_problem(graft: dict[str, Any]) -> str | None:
    """Why a trained adapter on disk is not the one a rebase recorded, or None when it is (or is not known)."""
    expected, adapter = graft.get("adapter_sha256"), graft.get("adapter")
    if not expected or not adapter:
        return None
    if not Path(adapter).exists():
        return f"the adapter {adapter} is gone"
    if tree_digest(Path(adapter)) != expected:
        return f"the adapter {adapter} was changed after the rebase recorded it (its contents no longer match)"
    return None


def prompt_block(genome: Genome, rules: list[str], exemplars: dict[str, list[str]]) -> str:
    lines = [PREAMBLE, ""]
    lines += [f"{n}. {genome.rules[rule].statement}" for n, rule in enumerate(rules, 1)]
    shown = []
    for n, rule in enumerate(rules, 1):
        for case_id in exemplars.get(rule, []):
            for unit in genome.case(case_id).units:
                if unit.reference is None:
                    continue
                request = runners.render_request(unit.messages)[-1500:]
                shown.append(f"Example (rule {n})\n{request}\nassistant: {runners.render_reply(unit.reference)}")
    if shown:
        lines += ["", EXAMPLES, "", "\n\n".join(shown)]
    return "\n".join(lines)


def assistant_message(reply: dict[str, Any]) -> dict[str, Any]:
    """A reply as the assistant message a chat fine-tuning file holds."""
    message: dict[str, Any] = {"role": "assistant", "content": verifiers.text_of(reply)}
    calls = verifiers.calls_of(reply)
    if calls:
        message["tool_calls"] = [
            {"id": f"call_{n}", "type": "function",
             "function": {"name": c["name"], "arguments": json.dumps(c["arguments"], sort_keys=True)}}
            for n, c in enumerate(calls)]
    return message


class Rebase:
    def __init__(self, project: Project, genome: Genome, state: dict[str, Any],
                 say: Callable[[str], None] = print,
                 tick: Callable[[str, int], Callable[[CaseResult], None]] | None = None):
        self.project, self.genome, self.state, self.say = project, genome, state, say
        self.tick = tick or (lambda label, total: lambda result: None)
        self.dir = run_dir_of(project, state)
        self.options = {**project.rebase, **state.get("options", {})}
        self.target_spec = project.model(state["target"])
        self.target = runners.build(state["target"], self.target_spec, project.root)
        self.last: dict[str, CaseResult] = {}
        self.raw: dict[str, CaseResult] = {}       # the target as it is, before any graft
        self._adapted: dict[str, runners.Runner] = {}

    # ---- state ----

    @classmethod
    def start(cls, project: Project, genome: Genome, target: str, source: str | None,
              ladder: list[str], options: dict[str, Any], **kwargs: Any) -> Rebase:
        project.model(target)
        if source:
            project.model(source)
        if in_progress(project):
            raise RebaseError("a rebase is in progress; finish it with `graft rebase --continue` "
                              "or drop it with `graft rebase --abort`")
        if not genome.cases:
            raise RebaseError("the genome has no cases yet; add some with `graft capture` or by hand")
        number = 1 + len(list((project.state_dir / "grafts" / target).glob("*/state.json")))
        state = {"run": f"{number:04d}-{genome.id[:8]}", "target": target, "source": source,
                 "genome": genome.id, "ladder": ladder, "options": options, "status": "running",
                 "rung": None, "round": 0, "started": now(), "stages": [], "excluded": None,
                 "graft": {"kind": "none"}, "outcome": None, "holdout_looks": 0}
        self = cls(project, genome, state, **kwargs)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.save()
        head_path(project).write_text(f"{target}/{state['run']}\n", encoding="utf-8")
        return self

    @classmethod
    def resume(cls, project: Project, genome: Genome, **kwargs: Any) -> Rebase:
        state = in_progress(project)
        if not state:
            raise RebaseError("no rebase is in progress")
        return cls(project, genome, state, **kwargs)

    def save(self) -> None:
        path = self.dir / "state.json"
        path.write_text(json.dumps(self.state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def opt(self, key: str, default: Any) -> Any:
        value = self.options.get(key)
        return default if value is None else value

    # ---- evaluation ----

    @property
    def excluded(self) -> set[str]:
        return set(self.state["excluded"] or {})

    @property
    def cases(self) -> list[Case]:
        return [c for c in self.genome.cases if c.id not in self.excluded]

    def runner_for(self, graft: dict[str, Any]) -> runners.Runner:
        adapter = graft.get("adapter")
        if not adapter:
            return self.target
        if adapter not in self._adapted:
            try:
                self._adapted[adapter] = runners.build(self.state["target"], self.target_spec,
                                                       self.project.root, adapter=adapter)
            except runners.RunnerError as err:
                raise RebaseError(str(err)) from err
        return self._adapted[adapter]

    def record(self, stage: str, results: dict[str, CaseResult], extra: dict[str, Any] | None = None) -> bool:
        errors = [r for r in results.values() if r.error]
        if errors:
            raise RebaseError(f"{len(errors)} of {len(results)} requests to {self.state['target']} failed "
                              f"({errors[0].problems[0]}); fix that and run `graft rebase --continue`")
        found = evaluate.gates(self.genome, results, self.excluded)
        sums = evaluate.totals(found)
        folder = self.dir / "evals"
        folder.mkdir(exist_ok=True)
        (folder / f"{stage}.json").write_text(
            json.dumps({"stage": stage, "genome": self.genome.id, "graft": self.state["graft"],
                        "results": [r.to_json() for r in results.values()]}, ensure_ascii=False) + "\n",
            encoding="utf-8")
        entry = {"stage": stage, "passed": sums["passed"], "total": sums["total"],
                 "rules_ok": sums["rules_ok"], "rules": sums["rules"], **(extra or {})}
        self.state["stages"] = [s for s in self.state["stages"] if s["stage"] != stage] + [entry]
        self.save()
        self.last = results
        self.say(f"{stage:<18}{sums['passed']}/{sums['total']} cases   "
                 f"{sums['rules_ok']} of {sums['rules']} rules hold")
        # The rebase steers by the training cases alone. The held-out cases only get a say once those all
        # pass, and every such look is counted, because each one tells the search a little about them.
        if not all(g.train_ok for g in found):
            return False
        self.state["holdout_looks"] = self.state.get("holdout_looks", 0) + 1
        self.save()
        return all(g.ok for g in found)

    def check(self, stage: str, graft: dict[str, Any], extra: dict[str, Any] | None = None) -> bool:
        cases = self.cases
        results = evaluate.evaluate(cases, self.runner_for(graft), graft, self.project.root,
                                    self.tick(stage, len(cases)))
        if graft.get("kind") == "none":
            self.raw = results
        return self.record(stage, results, extra)

    def stored(self, stage: str) -> dict[str, CaseResult]:
        """The results of an earlier stage of this run, when it checked the genome as it is now."""
        path = self.dir / "evals" / f"{stage}.json"
        if not path.is_file():
            return {}
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved.get("genome") != self.genome.id:
            return {}
        return {r["case"]: CaseResult.from_json(r) for r in saved["results"]}

    def check_source(self) -> None:
        """Keep only behavior the source really has: a case its own model fails is not something to carry over."""
        name = self.state["source"]
        live = bool(name)
        runner = runners.build(name, self.project.model(name), self.project.root) if live else runners.Replay()
        cases = self.genome.cases
        results = evaluate.evaluate(cases, runner, None, self.project.root, self.tick("source", len(cases)))
        if live and any(r.error for r in results.values()):
            bad = next(r for r in results.values() if r.error)
            raise RebaseError(f"the source model {name} could not answer ({bad.problems[0]})")
        excluded = {r.case: r.problems[0] for r in results.values() if not r.ok and not r.error}
        self.state["excluded"] = excluded
        self.state["source_fingerprint"] = runner.fingerprint()
        self.state["source_checked"] = sum(not r.error for r in results.values())
        self.save()
        kept = len(cases) - len(excluded)
        label = name or "references"
        self.say(f"{'source':<18}{kept}/{len(cases)} cases hold on {label}"
                 + (f"   {len(excluded)} left out (the source fails them)" if excluded else ""))
        if not kept:
            raise RebaseError("the source fails every case, so there is no behavior to carry over; "
                              "`graft check` shows why")

    # ---- rungs ----

    def run(self) -> int:
        state = self.state
        if state["genome"] != self.genome.id:
            self.say(f"the genome changed since this rebase started ({state['genome'][:8]} -> "
                     f"{self.genome.id[:8]}); checking against the new one")
            state["genome"], state["excluded"] = self.genome.id, None
        self.say(f"genome {self.genome.id[:8]}   {len(self.genome.rules)} rules, {len(self.genome.cases)} cases")
        if state["excluded"] is None:
            self.check_source()
        continuing, was_paused = state["rung"] is not None, state["status"] == "paused"
        state["status"] = "running"
        ladder = state["ladder"]
        rung = state["rung"] if state["rung"] in ladder else ladder[0]
        if continuing and not was_paused and self.check("recheck", state["graft"]):
            return self.finish()
        while True:
            if continuing:
                start = state["round"] if was_paused else state["round"] + 1
            else:
                start, state["rung"], state["round"] = 1, rung, 0
                self.save()
            outcome = getattr(self, "rung_" + rung.replace("-", "_"))(start, continuing and was_paused)
            if outcome is True:
                return self.finish()
            if outcome == PAUSED:
                return PAUSED
            continuing = was_paused = False
            if rung == ladder[-1]:
                return self.conflict()
            rung = ladder[ladder.index(rung) + 1]

    def rung_zero_shot(self, start: int, awaiting: bool) -> bool:
        self.state["graft"] = {"kind": "none"}
        self.state["round"] = 1
        return self.check("zero-shot", self.state["graft"])

    def rung_prompt(self, start: int, awaiting: bool) -> bool:
        if not self.last:
            self.check("zero-shot", {"kind": "none"})
        previous = self.state["graft"] if self.state["graft"].get("kind") == "prompt" else {}
        rules: list[str] = list(previous.get("rules", []))
        exemplars: dict[str, list[str]] = {k: list(v) for k, v in previous.get("exemplars", {}).items()}
        per_rule = int(self.opt("exemplars", 2))
        for number in range(start, start + int(self.opt("rounds", 3))):
            failing = [c for c in self.cases if c.split == "train" and not self.last[c.id].ok]
            grew = False
            for rule in dict.fromkeys(c.rule for c in failing):
                if rule not in rules:
                    rules.append(rule)
                    grew = True
                elif number > 1:
                    # worked examples come from the training split only, failing cases first
                    pool = [c for c in self.genome.rules[rule].cases
                            if c.split == "train" and c.id not in self.excluded
                            and c.id not in exemplars.get(rule, []) and all(u.reference for u in c.units)]
                    pool.sort(key=lambda c: self.last[c.id].ok)
                    for case in pool[:per_rule]:
                        exemplars.setdefault(rule, []).append(case.id)
                        grew = True
            if not grew:
                break
            graft = {"kind": "prompt", "rules": rules, "exemplars": exemplars,
                     "system": prompt_block(self.genome, rules, exemplars)}
            self.state["graft"], self.state["round"] = graft, number
            if self.check(f"prompt round {number}", graft):
                return True
        return False

    def targets_for(self, number: int, sampler: runners.Runner, graft: dict[str, Any]) -> list[dict[str, Any]]:
        """Training rows: for each training case, a reply the verifiers accept.

        The new model's own passing reply is preferred (it keeps the new model's voice); for a failing case
        a few sampled attempts are tried, and only then the source's recorded reply is used.

        Weights are per rule, not per reply: every row of a rule that still fails counts `2 ** number` times
        (at most 8), passing rows included. Weighting only the failing replies tips the balance of a rule
        towards whichever side failed last, and on a boundary rule (refunds up to $500 issued, above that
        escalated, pinned by counterfactual pairs) the next round then fails the other side.
        """
        samples, heat = int(self.opt("samples", 4)), float(self.opt("temperature", 0.8))
        weight = min(2 ** number, 8)
        failing = {case.rule for case in self.cases if case.split == "train"
                   and not ((seen := self.last.get(case.id)) and seen.ok)}

        def row(case: Case, index: int, unit: Unit) -> dict[str, Any] | None:
            context = {"reference": unit.reference, "messages": unit.messages, "case": case.id,
                       "cwd": self.project.root}
            seen = self.last.get(case.id)
            times = weight if case.rule in failing else 1
            if seen and seen.unit_ok[index]:
                reply, origin = seen.replies[index], "own"
            else:
                reply, origin = None, "sampled"
                for attempt in range(samples):
                    try:
                        candidate = sampler.reply(unit, graft, temperature=heat, seed=1000 * number + attempt)
                    except runners.RunnerError:
                        break
                    if not verifiers.failures(unit.verify, candidate, context):
                        reply = candidate
                        break
                if reply is None and unit.reference is not None:
                    reply, origin = verifiers.normalize(unit.reference), "source"
                if reply is None:
                    return None
            return {"messages": [*unit.messages, assistant_message(reply)], "tools": unit.tools or [],
                    "weight": times, "case": case.id, "rule": case.rule, "origin": origin}

        jobs = [(case, index, unit) for case in self.cases if case.split == "train"
                for index, unit in enumerate(case.units)]
        with ThreadPoolExecutor(max_workers=max(1, sampler.concurrency)) as pool:
            rows = list(pool.map(lambda job: row(*job), jobs))
        return [r for r in rows if r is not None]

    def train(self, dataset: Path, output: Path, number: int) -> None:
        trainer = self.project.trainer
        base = str(self.target_spec.get("weights") or self.target_spec.get("path") or self.target_spec.get("model"))
        fill = {"dataset": str(dataset), "output": str(output), "base": base, "round": str(number)}
        if trainer.get("command"):
            argv = [str(part).format(**fill) for part in trainer["command"]]
        else:
            argv = [sys.executable, "-m", "graft.train_hf", "--dataset", fill["dataset"],
                    "--output", fill["output"], "--base", base]
            for key in ("method", "epochs", "lr", "rank", "alpha", "batch_size", "max_length", "seed"):
                if key in trainer:
                    argv += [f"--{key.replace('_', '-')}", str(trainer[key])]
        log = output.parent / "train.log"
        name = Path(argv[1]).name if trainer.get("command") and len(argv) > 1 else "graft.train_hf"
        self.say(f"{'training':<18}{name} on {dataset.relative_to(self.project.root)}")
        # the built-in trainer must be importable even when graft runs from a source checkout
        package = str(Path(__file__).resolve().parent.parent)
        env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [package, os.environ.get("PYTHONPATH")]))}
        with log.open("w", encoding="utf-8") as out:
            try:
                done = subprocess.run(argv, stdout=out, stderr=subprocess.STDOUT, cwd=self.project.root,
                                      env=env, check=False)
            except OSError as err:
                raise RebaseError(f"the trainer did not start: {err}") from err
        if done.returncode != 0 or not output.exists():
            tail = "\n".join(log.read_text(encoding="utf-8", errors="replace").splitlines()[-8:])
            raise RebaseError(f"the trainer failed (exit {done.returncode}); the end of {log}:\n{tail}")

    def serve(self, adapter: Path) -> bool:
        """Make the trained adapter answer requests. False when a person has to load it first."""
        if "endpoint" not in self.target_spec:
            return True                               # loaded in this process
        name = self.target_spec.get("adapter_model")
        if not name:
            raise RebaseError(f"[models.{self.state['target']}] needs adapter_model: the name its endpoint "
                              f"serves the trained adapter under")
        hook = self.opt("after_train", None)
        if not hook:
            return False
        argv = [str(part).format(adapter=str(adapter), adapter_model=name) for part in hook]
        done = subprocess.run(argv, capture_output=True, text=True, cwd=self.project.root, check=False)
        if done.returncode != 0:
            raise RebaseError(f"after_train failed (exit {done.returncode}): "
                              f"{(done.stderr or done.stdout).strip()[-300:]}")
        return True

    def rung_lora(self, start: int, awaiting: bool) -> bool | int:
        if not self.project.trainer and not awaiting:
            raise RebaseError("the lora rung needs a [trainer] table in graft.toml (docs/user-guide.md#training)")
        if not self.last and not awaiting:
            self.check("baseline", self.state["graft"])
        sampler, sampling_graft = self.runner_for(self.state["graft"]), self.state["graft"]
        for number in range(start, start + int(self.opt("rounds", 3))):
            folder = self.dir / "lora" / f"round-{number}"
            adapter = folder / "adapter"
            extra: dict[str, Any] = {}
            if not (awaiting and number == start):
                folder.mkdir(parents=True, exist_ok=True)
                rows = self.targets_for(number, sampler, sampling_graft)
                if not rows:
                    raise RebaseError("no training case has a reply to learn from: record references, "
                                      "or lower [genome] holdout so more cases can be used")
                dataset = folder / "dataset.jsonl"
                dataset.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
                extra = {"trained_on": {origin: sum(r["origin"] == origin for r in rows)
                                        for origin in ("own", "sampled", "source")}}
                self.train(dataset, adapter, number)
                if not self.serve(adapter):
                    self.state.update(status="paused", round=number, pending=extra)
                    self.save()
                    name = self.target_spec["adapter_model"]
                    self.say(f"\nThe adapter is trained: {adapter}\n"
                             f"Load it into the endpoint as model {name!r}, then run `graft rebase --continue`.\n"
                             f"Set [rebase] after_train to a command that loads it, and this step runs unattended.")
                    return PAUSED
            graft = {"kind": "lora", "adapter": str(adapter)}
            if adapter.exists():
                graft["adapter_sha256"] = tree_digest(adapter)
            extra = extra or self.state.pop("pending", None) or {}
            self.state["graft"], self.state["round"] = graft, number
            if self.check(f"lora round {number}", graft, extra):
                return True
            sampler, sampling_graft = self.runner_for(graft), graft
        return False

    # ---- endings ----

    def suites(self) -> list[dict[str, Any]]:
        reports = []
        graft = self.state["graft"]
        for name, spec in self.project.suites.items():
            cases = load_suite(self.project.root / str(spec["path"]))
            raw = evaluate.evaluate(cases, self.target, None, self.project.root, self.tick(f"suite {name}", len(cases)))
            grafted = raw if graft.get("kind") == "none" else evaluate.evaluate(
                cases, self.runner_for(graft), graft, self.project.root, self.tick(f"suite {name}", len(cases)))
            if any(r.error for r in grafted.values()) or any(r.error for r in raw.values()):
                raise RebaseError(f"suite {name}: requests to {self.state['target']} failed")
            share = lambda results: sum(r.ok for r in results.values()) / max(1, len(results))  # noqa: E731
            report: dict[str, Any] = {"suite": name, "cases": len(cases), "base": round(share(raw), 4),
                                      "grafted": round(share(grafted), 4), "ok": True}
            line = f"{'suite ' + name:<18}{self.state['target']} {share(raw):.1%}   grafted {share(grafted):.1%}"
            if self.state["source"]:
                source = runners.build(self.state["source"], self.project.model(self.state["source"]),
                                       self.project.root)
                old = evaluate.evaluate(cases, source, None, self.project.root, self.tick(f"suite {name}", len(cases)))
                versus = evaluate.compare(old, grafted)
                report.update(source=round(versus["old"], 4), gain=round(versus["new"] - versus["old"], 4),
                              wins=versus["wins"], losses=versus["losses"], p_value=versus["p_value"])
                line = (f"{'suite ' + name:<18}{self.state['source']} {versus['old']:.1%}   "
                        f"{self.state['target']} {share(raw):.1%}   grafted {versus['new']:.1%}   "
                        f"({(versus['new'] - versus['old']) * 100:+.1f} points, p={versus['p_value']})")
                if "min_gain" in spec and report["gain"] < float(spec["min_gain"]):
                    report["ok"] = False
                    report["why"] = f"gain {report['gain']:+.1%} is below min_gain {float(spec['min_gain']):+.1%}"
            if "max_loss" in spec and report["base"] - report["grafted"] > float(spec["max_loss"]):
                report["ok"] = False
                report["why"] = (f"the graft costs {report['base'] - report['grafted']:.1%} of the new base's "
                                 f"score, above max_loss {float(spec['max_loss']):.1%}")
            self.say(line + ("" if report["ok"] else f"   FAILS: {report['why']}"))
            reports.append(report)
        return reports

    def close(self, passed: bool, suites: list[dict[str, Any]]) -> Path:
        state = self.state
        state["finished"] = now()
        if not self.last:
            self.last = self.stored("recheck") or self.stored("zero-shot")
        cert = certificate.build(state, self.genome, self.last, passed=passed, suites=suites,
                                 target_fingerprint=self.runner_for(state["graft"]).fingerprint(),
                                 raw=self.raw or self.stored("zero-shot") or None)
        path = self.dir / "certificate.json"
        path.write_text(json.dumps(cert, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        (self.dir / "REPORT.md").write_text(certificate.report(cert), encoding="utf-8")
        (self.dir / "graft.json").write_text(json.dumps(state["graft"], indent=2, ensure_ascii=False) + "\n",
                                             encoding="utf-8")
        self.save()
        return path

    def finish(self) -> int:
        state = self.state
        suites = self.suites()
        if not all(s["ok"] for s in suites):
            return self.conflict(suites)
        kind = state["graft"].get("kind")
        state["status"], state["outcome"] = "passed", "fast-forward" if kind == "none" else kind
        path = self.close(True, suites)
        head_path(self.project).unlink(missing_ok=True)
        source = state["source"] or "the recorded behavior"
        self.say({"none": f"\nNo graft needed: {state['target']} already behaves like {source} (fast-forward).",
                  "prompt": f"\nRebased {source} onto {state['target']} as a prompt graft.",
                  "lora": f"\nRebased {source} onto {state['target']} as a trained adapter."}[kind])
        self.say(f"certificate  {path.relative_to(self.project.root)}")
        return PASSED

    def conflict(self, suites: list[dict[str, Any]] | None = None) -> int:
        state = self.state
        state["status"], state["outcome"] = "conflict", "conflict"
        path = self.close(False, suites if suites is not None else [])
        failing = [r for r in self.last.values() if not r.ok and r.case not in self.excluded]
        if failing:
            self.say(f"\nCONFLICT: {len(failing)} cases still fail on {state['target']} after the "
                     f"{state['rung']} rung.")
            for result in failing[:10]:
                self.say(f"  {result.rule}/{result.case}: {result.problems[0] if result.problems else 'failed'}")
            if len(failing) > 10:
                self.say(f"  ... and {len(failing) - 10} more in {path.relative_to(self.project.root)}")
        else:
            self.say("\nCONFLICT: every rule holds, but a capability suite gate fails (above).")
        self.say("\nChange the genome, the ladder or the trainer, then `graft rebase --continue`; "
                 "or `graft rebase --abort`.")
        return CONFLICT
