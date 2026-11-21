"""The rebase certificate: what was checked, against which model, and what held."""
from __future__ import annotations

from typing import Any

from graft import __version__, evaluate
from graft.evaluate import CaseResult
from graft.genome import Genome, digest


def _totals(found: list[evaluate.Gate], consulted: int) -> dict[str, Any]:
    """Totals with the bound allowing for every look, and the true number of looks for the report."""
    totals = evaluate.totals(found, looks=consulted)
    totals["holdout"]["consulted"] = consulted
    return totals


def build(state: dict[str, Any], genome: Genome, results: dict[str, CaseResult], passed: bool,
          suites: list[dict[str, Any]], target_fingerprint: dict[str, Any],
          raw: dict[str, CaseResult] | None = None) -> dict[str, Any]:
    excluded = state.get("excluded") or {}
    kept = {case: result for case, result in results.items() if case not in excluded}
    found = evaluate.gates(genome, kept, set(excluded))
    graft = state["graft"]
    # the folder an adapter happens to sit in is not part of what the graft is
    described: dict[str, Any] = {"kind": graft.get("kind", "none"),
                                 "sha256": digest({k: v for k, v in graft.items() if k != "adapter"})}
    if graft.get("kind") == "prompt":
        described["rules_stated"] = len(graft.get("rules", []))
        described["examples"] = sum(len(v) for v in graft.get("exemplars", {}).values())
    if graft.get("adapter"):
        described["adapter"] = graft["adapter"]
    if graft.get("adapter_sha256"):
        described["adapter_sha256"] = graft["adapter_sha256"]
    cert: dict[str, Any] = {
        "certificate": 1,
        "graft_version": __version__,
        "run": state["run"],
        "passed": passed,
        "outcome": state.get("outcome"),
        "genome": {"id": genome.id, "rules": len(genome.rules), "cases": len(genome.cases)},
        "source": {"name": state.get("source") or "references", "fingerprint": state.get("source_fingerprint")},
        "target": {"name": state["target"], "fingerprint": target_fingerprint},
        "graft": described,
        "gates": [g.to_json() for g in found],
        "totals": _totals(found, state.get("holdout_looks", 1)),
        "ladder": state.get("stages", []),
        "excluded": [{"case": case, "rule": genome.case(case).rule, "reason": reason}
                     for case, reason in sorted(excluded.items())],
        "failures": [{"case": r.case, "rule": r.rule, "split": r.split, "problems": r.problems}
                     for r in kept.values() if not r.ok],
        "suites": suites,
        "started": state.get("started"),
        "finished": state.get("finished"),
    }
    if raw:
        free = [case for case, result in raw.items() if case not in excluded and result.ok]
        cert["transferred_unaided"] = {"passed": len(free), "total": len(kept)}
    cert["digest"] = digest(cert)
    return cert


def intact(cert: dict[str, Any]) -> bool:
    """Whether the certificate still says what graft wrote: its digest covers every other field."""
    body = {key: value for key, value in cert.items() if key != "digest"}
    return digest(body) == cert.get("digest")


def _share(passed: int, total: int) -> str:
    return f"{passed}/{total}"


def report(cert: dict[str, Any]) -> str:
    """The certificate as a page a person can read."""
    totals, graft = cert["totals"], cert["graft"]
    held = totals["holdout"]
    kinds = {"none": "none needed: the new model already behaves this way",
             "prompt": f"prompt graft ({graft.get('rules_stated', 0)} rules stated, "
                       f"{graft.get('examples', 0)} worked examples)",
             "lora": f"trained adapter at `{graft.get('adapter', '')}`"}
    verdict = ("**PASSED.** Every gate holds on the new model." if cert["passed"]
               else "**CONFLICT.** At least one gate does not hold; this graft is not ready to ship.")
    lines = [
        f"# Rebase certificate: {cert['source']['name']} onto {cert['target']['name']}",
        "",
        verdict,
        "",
        "| | |",
        "|---|---|",
        f"| Genome | `{cert['genome']['id'][:16]}`, {cert['genome']['rules']} rules, {cert['genome']['cases']} cases |",
        f"| Source | {cert['source']['name']} |",
        f"| Target | {cert['target']['name']} |",
        f"| Graft | {kinds.get(graft['kind'], graft['kind'])} |",
        *([f"| Adapter contents | sha256 `{graft['adapter_sha256']}` |"] if graft.get("adapter_sha256") else []),
        f"| Run | {cert['run']}, {cert.get('started')} to {cert.get('finished')} |",
        f"| Digest | `{cert['digest']}` |",
        "",
        "## Rules",
        "",
        "| Rule | Gate | Cases passing | Of those, held out | |",
        "|---|---|---|---|---|",
    ]
    for gate in cert["gates"]:
        lines.append(f"| `{gate['rule']}` | {gate['gate']:.0%} | {_share(gate['passed'], gate['total'])} | "
                     f"{_share(gate['holdout']['passed'], gate['holdout']['total'])} | "
                     f"{'holds' if gate['ok'] else '**fails**'} |")
    lines += ["", f"All rules: {_share(totals['passed'], totals['total'])} cases pass; "
                  f"{totals['rules_ok']} of {totals['rules']} rules hold."]
    lines += ["", "## What this shows, and what it does not", ""]
    lines.append("- Every verdict comes from a deterministic verifier, never from a model's opinion. "
                 "`graft verify` on the same genome and the same model reproduces it, as far as the "
                 "serving engine is deterministic at temperature 0.")
    if "transferred_unaided" in cert:
        free = cert["transferred_unaided"]
        lines.append(f"- {_share(free['passed'], free['total'])} cases already passed on the new model "
                     f"before any graft.")
    if held["total"]:
        looks = held.get("consulted", held.get("looks", 1))
        if looks == 0:
            consulted = ("The training cases never all passed, so the held-out results never steered the "
                         "search.")
        elif looks == 1:
            consulted = "They were consulted once, after the training cases all passed."
        else:
            consulted = (f"They were consulted {looks} times, each time after the training cases all passed; "
                         f"the bound below allows for that.")
        lines.append(f"- {_share(held['passed'], held['total'])} held-out cases pass. No prompt example and no "
                     f"training row was made from them. {consulted} With {held['confidence']:.0%} confidence, "
                     f"the new model passes at least {held['lower_bound']:.1%} of cases drawn the way these "
                     f"were (exact one-sided binomial bound).")
    else:
        lines.append("- No cases were held out, so nothing here measures behavior beyond the cases themselves.")
    if totals["weak"]:
        lines.append(f"- {totals['weak']} cases rest on a similarity match alone (`similar`). They show the "
                     f"reply resembles the recorded one, not that it is right. Write a real verifier for them.")
    lines.append("- It says nothing about requests unlike the genome's cases. A rule is only as well "
                 "protected as its cases are varied.")
    if cert["ladder"]:
        lines += ["", "## How it got there", "", "| Step | Cases passing | Rules holding | Trained on |",
                  "|---|---|---|---|"]
        for stage in cert["ladder"]:
            rows = stage.get("trained_on")
            trained = (f"{rows['own']} own replies, {rows['sampled']} sampled, {rows['source']} from the source"
                       + (f", {rows['replay']} replayed" if rows.get("replay") else "")
                       if rows else "")
            lines.append(f"| {stage['stage']} | {_share(stage['passed'], stage['total'])} | "
                         f"{stage['rules_ok']} of {stage['rules']} | {trained} |")
    if cert["suites"]:
        lines += ["", "## Capability suites", "", "| Suite | Cases | Source | New base | Grafted | Gain | p | |",
                  "|---|---|---|---|---|---|---|---|"]
        for suite in cert["suites"]:
            old = f"{suite['source']:.1%}" if "source" in suite else ""
            gain = f"{suite['gain'] * 100:+.1f} points" if "gain" in suite else ""
            lines.append(f"| `{suite['suite']}` | {suite['cases']} | {old} | {suite['base']:.1%} | "
                         f"{suite['grafted']:.1%} | {gain} | {suite.get('p_value', '')} | "
                         f"{'ok' if suite['ok'] else '**' + suite.get('why', 'fails') + '**'} |")
        lines += ["", "p is the exact two-sided sign test on the cases the two models disagree on."]
    if cert["excluded"]:
        lines += ["", "## Left out", "",
                  "The source itself fails these cases, so they are not behavior to carry over. "
                  "Fix the case or its verifier, or delete it.", ""]
        lines += [f"- `{item['rule']}/{item['case']}`: {item['reason']}" for item in cert["excluded"]]
    if cert["failures"]:
        lines += ["", "## Failures", ""]
        lines += [f"- `{item['rule']}/{item['case']}` ({item['split']}): "
                  f"{item['problems'][0] if item['problems'] else 'failed'}" for item in cert["failures"]]
    return "\n".join(lines) + "\n"
