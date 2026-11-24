"""The graft command."""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from graft import __version__, capture, certificate, config, evaluate, export, genome, rebase, runners, verifiers
from graft.evaluate import CaseResult

PROG = "graft"
ERRORS = (config.ConfigError, genome.GenomeError, runners.RunnerError, verifiers.VerifierError,
          rebase.RebaseError, capture.CaptureError)


def _rule_id(value: str) -> str:
    """A rule id as argparse sees it, so a bad one is a usage error (exit 2) and touches nothing."""
    if not genome.ID.match(value):
        raise argparse.ArgumentTypeError(f"{value!r}: letters, digits, dot, dash and underscore only, "
                                         f"starting with a letter or digit")
    return value


def _share_arg(value: str) -> float:
    try:
        share = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value!r} is not a number") from None
    if not 0.0 < share <= 1.0:
        raise argparse.ArgumentTypeError(f"{value} is not a share between 0 (exclusive) and 1")
    return share


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog=PROG, description="Rebase a model's verified behavior onto a new base model.")
    p.add_argument("--version", action="version", version=f"{PROG} {__version__}")
    sub = p.add_subparsers(dest="command", metavar="command")

    init = sub.add_parser("init", help="start a genome in this folder")
    init.add_argument("folder", nargs="?", default=".", help="default: this folder")

    status = sub.add_parser("status", help="the genome, and any rebase in progress")
    status.add_argument("--json", action="store_true")

    rule = sub.add_parser("rule", help="add a rule")
    rule_sub = rule.add_subparsers(dest="action", metavar="action", required=True)
    add = rule_sub.add_parser("add", help="add a rule: an id and the behavior in a sentence or two")
    add.add_argument("id", type=_rule_id)
    add.add_argument("statement")
    add.add_argument("--gate", type=_share_arg, default=1.0,
                     help="share of the rule's cases that must pass. default 1.0")

    cap = sub.add_parser("capture", help="turn recorded traffic into cases")
    cap_sub = cap.add_subparsers(dest="kind", metavar="kind", required=True)
    goethe = cap_sub.add_parser("goethe", help="a Goethe trace folder")
    goethe.add_argument("folder", type=Path)
    goethe.add_argument("--model", help="only turns served by this model id")
    goethe.add_argument("--since", help="only turns that started at or after this UTC time, e.g. 2026-11-01")
    goethe.add_argument("--tools-only", action="store_true", help="only turns whose reply calls a tool")
    jsonl = cap_sub.add_parser("jsonl", help="a chat fine-tuning file: {\"messages\": [...]} per line")
    jsonl.add_argument("file", type=Path)
    jsonl.add_argument("--every-turn", action="store_true", help="a case per assistant turn, not just the last")
    for parser in (goethe, jsonl):
        parser.add_argument("--rule", default="recorded", help="the rule the cases go under. default: recorded")
        parser.add_argument("--limit", type=int, help="add at most this many cases")
        parser.add_argument("--dry-run", action="store_true", help="say what would be added. write nothing")
        parser.add_argument("--checks", choices=capture.CHECKS, default="structured",
                            help="structured: pin decisions (tools, numbers, ids, enum values, values from the "
                                 "request), require free text to be non-empty. exact: pin every value. "
                                 "default: structured")

    check = sub.add_parser("check", help="check the genome against the model that has the behavior today")
    check.add_argument("--model", help="a model from graft.toml. default: the recorded reference replies")
    check.add_argument("--json", action="store_true")

    verify = sub.add_parser("verify", help="run every gate against a model")
    verify.add_argument("model", help="a model from graft.toml")
    verify.add_argument("--graft", metavar="RUN", nargs="?", const="",
                        help="apply the graft of a rebase (default: the latest one)")
    verify.add_argument("--rule", action="append", default=[], help="only this rule. repeatable")
    verify.add_argument("--json", action="store_true")

    rb = sub.add_parser("rebase", help="carry the genome onto a new base model")
    rb.add_argument("--target", help="the new base: a model from graft.toml")
    rb.add_argument("--source", help="the model that has the behavior today. default: [rebase] source, "
                                     "else the recorded references")
    rb.add_argument("--ladder", help="rungs to try, in order: zero-shot,prompt,lora")
    rb.add_argument("--rounds", type=int, help="attempts per rung. default 3")
    rb.add_argument("--continue", dest="resume", action="store_true", help="pick up the rebase in progress")
    rb.add_argument("--abort", action="store_true", help="drop the rebase in progress")
    rb.add_argument("--enforce-all", action="store_true",
                    help="hold the new base to every case, also those the source fails (new behavior, not carried "
                         "over). default: [rebase] enforce_all, else only what the source already does")

    log = sub.add_parser("log", help="past rebases")
    log.add_argument("--json", action="store_true")

    show = sub.add_parser("show", help="a rebase's certificate")
    show.add_argument("run", nargs="?", help="a target, or target/run. default: the latest rebase")
    show.add_argument("--json", action="store_true")
    show.add_argument("--graft", action="store_true",
                      help="print the graft itself: the system prompt block, or the adapter path")

    exp = sub.add_parser("export", help="bundle a passed rebase for serving")
    exp.add_argument("run", nargs="?", help="a target or target/run; default: latest")
    exp.add_argument("--out", required=True, type=Path, help="new output zip file")
    exp.add_argument("--format", choices=["portable", "vllm", "goethe"], default="portable")
    exp.add_argument("--base-model", help="model id or local path for the serving engine")

    models = sub.add_parser("models", help="the models in graft.toml, and whether each answers")
    models.add_argument("--json", action="store_true")
    return p


def _tick(label: str, total: int) -> Callable[[CaseResult], None]:
    """A one-line progress counter, only when a person is watching."""
    if not sys.stderr.isatty() or total < 2:
        return lambda result: None
    done = [0]

    def tick(result: CaseResult) -> None:
        done[0] += 1
        end = "\r" if done[0] < total else "\r" + " " * 40 + "\r"
        print(f"{label}  {done[0]}/{total}", end=end, file=sys.stderr, flush=True)
    return tick


def _load(project: config.Project) -> genome.Genome:
    return genome.load(project.genome_dir, project.holdout)


def _init(args: argparse.Namespace) -> int:
    root = Path(args.folder).resolve()
    target = root / config.FILE
    if target.exists():
        print(f"{target} already exists; nothing changed")
        return 1
    for folder in ("rules", "cases", "suites", "verifiers"):
        (root / "genome" / folder).mkdir(parents=True, exist_ok=True)
    target.write_text(config.TEMPLATE, encoding="utf-8")
    ignore = root / ".gitignore"
    lines = ignore.read_text(encoding="utf-8").splitlines() if ignore.exists() else []
    if ".graft/" not in lines:
        ignore.write_text("\n".join([*lines, ".graft/"]) + "\n", encoding="utf-8")
    print(f"Started a genome in {root}.\n"
          f"  1. Describe a rule:        graft rule add refund-limit \"Refunds above $500 go to a human.\"\n"
          f"  2. Add cases that pin it:  genome/cases/refund-limit.jsonl, or `graft capture`\n"
          f"  3. Name your models in {config.FILE}, then: graft rebase --target <new-base>")
    return 0


def _status(args: argparse.Namespace) -> int:
    project = config.load()
    found = _load(project)
    running = rebase.in_progress(project)
    rows = [{"rule": r.id, "gate": r.gate, "cases": len(r.cases), "pairs": sum(c.pair for c in r.cases),
             "holdout": sum(c.split == "holdout" for c in r.cases),
             "weak": sum(all(verifiers.weak_only(u.verify) for u in c.units) for c in r.cases),
             "statement": r.statement} for r in found.rules.values()]
    if args.json:
        print(json.dumps({"genome": found.id, "rules": rows, "rebase": running}, indent=2))
        return 0
    cases = found.cases
    print(f"genome {found.id[:8]}   {len(found.rules)} rules, {len(cases)} cases "
          f"({sum(c.split == 'holdout' for c in cases)} held out)")
    width = max([len(r["rule"]) for r in rows] + [4])
    for row in rows:
        notes = [f"{row['cases']} cases"]
        if row["pairs"]:
            notes.append(f"{row['pairs']} pairs")
        if row["weak"]:
            notes.append(f"{row['weak']} weakly verified")
        if row["gate"] != 1.0:
            notes.append(f"gate {row['gate']:.0%}")
        print(f"  {row['rule']:<{width}}  {', '.join(notes)}")
    empty = [r["rule"] for r in rows if not r["cases"]]
    if empty:
        print(f"\nNo cases yet for: {', '.join(empty)}. A rule without cases is not checked.")
    if running:
        print(f"\nRebase in progress onto {running['target']} ({running['status']}, {running['rung']} rung). "
              f"`graft rebase --continue` or `graft rebase --abort`.")
    return 0


def _rule(args: argparse.Namespace) -> int:
    project = config.load()
    path = genome.write_rule(project.genome_dir, args.id, args.statement, args.gate)
    print(f"added {path.relative_to(project.root)}; its cases go in "
          f"{(project.genome_dir / 'cases' / (args.id + '.jsonl')).relative_to(project.root)}")
    return 0


def _capture(args: argparse.Namespace) -> int:
    project = config.load()
    loosened = [0]   # free-text values that only have to be present, counted while cases are made
    if args.kind == "goethe":
        cases, skipped = capture.from_goethe(args.folder, args.model, args.since, args.tools_only, args.checks,
                                             loosened)
    else:
        if not Path(args.file).is_file():
            raise capture.CaptureError(f"{args.file}: no such file")
        cases, skipped = capture.from_jsonl(Path(args.file), args.every_turn, args.checks, loosened), {}
    if args.limit is not None:
        cases = cases[:args.limit]
    weak = sum(verifiers.weak_only(c["verify"]) for c in cases)
    if args.dry_run:
        added = len(cases)
    else:
        if cases and not (project.genome_dir / "rules" / f"{args.rule}.toml").exists():
            genome.write_rule(project.genome_dir, args.rule,
                              "Replies match what the production model did on recorded requests.")
            print(f"added rule {args.rule}; edit its statement in genome/rules/{args.rule}.toml")
        added = genome.append_cases(project.genome_dir, args.rule, cases)
    verb = "would add" if args.dry_run else "added"
    print(f"{verb} {added} cases under {args.rule}" + (f" ({len(cases) - added} were already there)"
                                                        if added < len(cases) else ""))
    if loosened[0]:
        print(f"  {loosened[0]} free-text values (a reason, a summary) only have to be non-empty; decisions, ids and "
              f"values from the request are pinned. --checks exact pins them all")
    for reason, count in skipped.items():
        print(f"  skipped {count}: {reason}")
    if weak:
        print(f"  {weak} cases are plain text with no rule written for them: they are checked by similarity to "
              f"the recorded reply only. Replace `similar` with a real verifier where the reply matters.")
    return 0


def _print_results(found: genome.Genome, results: dict[str, CaseResult], gates: list[evaluate.Gate],
                   limit: int = 3) -> None:
    width = max([len(g.rule) for g in gates] + [4])
    for gate in gates:
        mark = "holds" if gate.ok else "FAILS"
        print(f"  {gate.rule:<{width}}  {f'{gate.passed}/{gate.total}':<7}  {mark}")
        bad = [r for c in found.rules[gate.rule].cases if (r := results.get(c.id)) and not r.ok]
        for result in bad[:limit]:
            print(f"      {result.case}: {result.problems[0] if result.problems else 'failed'}")
        if len(bad) > limit:
            print(f"      ... and {len(bad) - limit} more (--json lists every one)")


def _check(args: argparse.Namespace) -> int:
    project = config.load()
    found = _load(project)
    runner = runners.build(args.model, project.model(args.model), project.root) if args.model else runners.Replay()
    results = evaluate.evaluate(found.cases, runner, None, project.root, _tick("check", len(found.cases)))
    unchecked = [r for r in results.values() if r.error]
    if args.model and unchecked:
        raise runners.RunnerError(f"{args.model} could not answer: {unchecked[0].problems[0]}")
    checked = {case: r for case, r in results.items() if not r.error}
    gates = evaluate.gates(found, checked)
    bad = [r for r in checked.values() if not r.ok]
    if args.json:
        print(json.dumps({"genome": found.id, "against": args.model or "references",
                          "failing": [r.to_json() for r in bad], "unchecked": [r.case for r in unchecked]}, indent=2))
        return 1 if bad else 0
    print(f"genome {found.id[:8]} against {args.model or 'its recorded references'}")
    _print_results(found, checked, gates)
    if unchecked:
        print(f"  {len(unchecked)} cases have no recorded reply, so there was nothing to check them against")
    if bad:
        print(f"\n{len(bad)} cases fail on the source itself. Either the verifier is wrong or the source never "
              f"had this behavior; a rebase leaves them out.")
        return 1
    print("\nEvery checked case holds on the source.")
    return 0


def _verify(args: argparse.Namespace) -> int:
    project = config.load()
    found = _load(project)
    graft: dict[str, Any] = {"kind": "none"}
    if args.graft is not None:
        state = rebase.find_run(project, args.graft or None, for_graft=True)
        graft = state["graft"]
        problem = rebase.adapter_problem(graft, project.root)
        if problem:
            raise rebase.RebaseError(f"{problem}; restore it or run the rebase again")
    adapter = rebase.adapter_path(project.root, graft)
    runner = runners.build(args.model, project.model(args.model), project.root,
                           adapter=str(adapter) if adapter else None)
    cases = [c for c in found.cases if not args.rule or c.rule in args.rule]
    unknown = [r for r in args.rule if r not in found.rules]
    if unknown:
        raise genome.GenomeError(f"no rule named {unknown[0]!r}")
    results = evaluate.evaluate(cases, runner, graft, project.root, _tick("verify", len(cases)))
    errors = [r for r in results.values() if r.error]
    if errors:
        raise runners.RunnerError(f"{len(errors)} of {len(results)} requests to {args.model} failed: "
                                  f"{errors[0].problems[0]}")
    gates = evaluate.gates(found, results)
    sums = evaluate.totals(gates)
    ok = all(g.ok for g in gates)
    if args.json:
        print(json.dumps({"genome": found.id, "model": args.model, "graft": graft.get("kind"), "passed": ok,
                          "gates": [g.to_json() for g in gates], "totals": sums,
                          "failures": [r.to_json() for r in results.values() if not r.ok]}, indent=2))
        return 0 if ok else 1
    applied = {"none": "", "prompt": " with the prompt graft", "lora": " with the trained adapter"}[graft["kind"]]
    print(f"genome {found.id[:8]} against {args.model}{applied}")
    _print_results(found, results, gates)
    held = sums["holdout"]
    print(f"\n{sums['passed']}/{sums['total']} cases pass; {sums['rules_ok']} of {sums['rules']} rules hold."
          + (f" Held out: {held['passed']}/{held['total']}." if held["total"] else ""))
    return 0 if ok else 1


def _rebase(args: argparse.Namespace) -> int:
    project = config.load()
    if args.abort:
        state = rebase.abort(project)
        if not state:
            print("no rebase is in progress")
            return 1
        print(f"dropped the rebase onto {state['target']}; nothing it trained was deleted "
              f"({rebase.run_dir_of(project, state).relative_to(project.root)})")
        return 0
    found = _load(project)
    if args.resume:
        run = rebase.Rebase.resume(project, found, tick=_tick)
        if args.ladder:
            run.state["ladder"] = project.ladder(args.ladder.split(","))
        if args.rounds:
            run.state["options"]["rounds"] = run.options["rounds"] = args.rounds
        if args.enforce_all and not run.opt("enforce_all", False):
            run.state["options"]["enforce_all"] = run.options["enforce_all"] = True
            run.state["excluded"] = None   # the cases the source fails are checked again, and kept this time
    else:
        if not args.target:
            print("graft rebase needs --target <model> (or --continue, or --abort)", file=sys.stderr)
            return 2
        options: dict[str, Any] = {"rounds": args.rounds} if args.rounds else {}
        if args.enforce_all:
            options["enforce_all"] = True
        source = args.source or project.rebase.get("source")
        run = rebase.Rebase.start(project, found, args.target, source,
                                  project.ladder(args.ladder.split(",") if args.ladder else None), options,
                                  tick=_tick)
    return run.run()


def _log(args: argparse.Namespace) -> int:
    project = config.load()
    known = rebase.runs(project)
    if args.json:
        print(json.dumps([{k: s.get(k) for k in ("run", "target", "source", "status", "outcome", "genome",
                                                 "started", "finished")} for s in known], indent=2))
        return 0
    if not known:
        print("no rebase has run here yet")
        return 0
    width = max(len(f"{s['target']}/{s['run']}") for s in known)
    for state in reversed(known):
        last = state["stages"][-1] if state["stages"] else {}
        score = f"{last.get('passed', 0)}/{last.get('total', 0)}" if last else ""
        print(f"{state['target'] + '/' + state['run']:<{width}}  {state['status']:<9} "
              f"{state.get('outcome') or '':<13}{score:<9}{state.get('started', '')}  "
              f"from {state.get('source') or 'references'}")
    return 0


def _show(args: argparse.Namespace) -> int:
    project = config.load()
    state = rebase.find_run(project, args.run, for_graft=args.graft)
    folder = rebase.run_dir_of(project, state)
    if args.graft:
        graft = state["graft"]
        if graft.get("kind") == "prompt":
            print(graft["system"])
        elif graft.get("kind") == "lora":
            print(rebase.adapter_path(project.root, graft))
        elif state.get("status") == "passed":
            print("no graft: the new model already passes as it is", file=sys.stderr)
        elif state.get("status") != "aborted":
            print(f"{state['target']}/{state['run']} is {state['status']}; it has no graft yet", file=sys.stderr)
        else:
            print(f"{state['target']}/{state['run']} is {state['status']}; it has no graft to apply", file=sys.stderr)
            return 1
        return 0
    path = folder / "certificate.json"
    if not path.is_file():
        print(f"{state['target']}/{state['run']} is {state['status']}; it has no certificate yet")
        return 1
    cert = json.loads(path.read_text(encoding="utf-8"))
    if args.json:
        print(json.dumps(cert, indent=2, ensure_ascii=False))
    else:
        print(certificate.report(cert), end="")
    if not certificate.intact(cert):
        print(f"\nWARNING: {path} was changed after graft wrote it: its digest no longer matches.", file=sys.stderr)
        return 1
    problem = rebase.adapter_problem(cert.get("graft") or {}, project.root)
    if problem:
        print(f"\nWARNING: {problem}. This certificate does not describe what is on disk now.", file=sys.stderr)
        return 1
    return 0


def _export(args: argparse.Namespace) -> int:
    try:
        certified = export.bundle(config.load(), args.run, args.out, args.format, args.base_model)
        if args.base_model and certified and args.base_model not in certified:
            print(f"note: the certificate was checked on {' / '.join(certified)}, not on {args.base_model!r}; "
                  f"it holds only if they are the same weights", file=sys.stderr)
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as err:
        raise rebase.RebaseError(f"could not export: {err}") from err
    print(args.out)
    return 0


def _models(args: argparse.Namespace) -> int:
    project = config.load()
    rows = []
    for name, spec in project.models.items():
        row: dict[str, Any] = {"model": name, "ok": False}
        try:
            runner = runners.build(name, spec, project.root)
            row["runner"] = runner.fingerprint().get("runner")
            row["where"] = spec.get("endpoint") or spec.get("path") or " ".join(map(str, spec.get("argv", [])))
            if isinstance(runner, runners.OpenAI):
                runner.retries = 0          # a quick look, not an overnight run
                reply = runner.complete([{"role": "user", "content": "Reply with the word ok."}])
                row["ok"], row["note"] = True, f"answers ({len(reply['content'])} characters)"
            else:
                row["ok"], row["note"] = True, "not contacted: it loads when first used"
        except runners.RunnerError as err:
            row["note"] = str(err)
        rows.append(row)
    if args.json:
        print(json.dumps(rows, indent=2))
    elif not rows:
        print(f"no models in {config.FILE} yet; add a [models.<name>] table")
    else:
        width = max(len(r["model"]) for r in rows)
        for row in rows:
            print(f"{row['model']:<{width}}  {'ok   ' if row['ok'] else 'ERROR'}  {row.get('where', '')}  "
                  f"{row.get('note', '')}")
    return 0 if all(r["ok"] for r in rows) else 1


COMMANDS = {"init": _init, "status": _status, "rule": _rule, "capture": _capture, "check": _check,
            "verify": _verify, "rebase": _rebase, "log": _log, "show": _show, "models": _models, "export": _export}


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 0
    try:
        return COMMANDS[args.command](args)
    except ERRORS as err:
        print(f"error: {err}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\ninterrupted; a rebase in progress keeps its state (`graft rebase --continue`)", file=sys.stderr)
        return 130
    except BrokenPipeError:
        return 0
