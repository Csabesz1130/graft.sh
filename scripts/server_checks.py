"""Acceptance checks for scripts/server-test.sh. Uses only the Python standard library."""
from __future__ import annotations

import json
import subprocess
import sys
import unittest
from collections.abc import Iterator
from pathlib import Path


def suite_ids(suite: unittest.TestSuite) -> Iterator[str]:
    for test in suite:
        if isinstance(test, unittest.TestSuite):
            yield from suite_ids(test)
        else:
            yield test.id()


def run_tests(train: bool) -> bool:
    suite = unittest.defaultTestLoader.discover("tests", top_level_dir=".")
    ids = set(suite_ids(suite))
    if not ids:
        print("No tests discovered", file=sys.stderr)
        return False
    trainer_ids = {name for name in ids if name.startswith("tests.test_train_hf.")}
    passed = set()

    class Result(unittest.TextTestResult):
        def addSuccess(self, test):
            super().addSuccess(test)
            if test.id() in trainer_ids:
                passed.add(test.id())

    result = unittest.TextTestRunner(resultclass=Result).run(suite)
    if train and (not trainer_ids or passed != trainer_ids):
        print("Trainer tests missing or not successful:", sorted(trainer_ids - passed), file=sys.stderr)
        return False
    if train and result.skipped:
        print("Tests skipped in --train:", [test.id() for test, _ in result.skipped], file=sys.stderr)
        return False
    return result.wasSuccessful()


def require(condition: bool, description: str) -> None:
    if not condition:
        raise ValueError(description)


def graft_json(*args: str, code: int = 0) -> dict:
    cmd = [str(Path(sys.executable).with_name("graft")), *args, "--json"]
    done = subprocess.run(cmd, capture_output=True, text=True, check=False)
    require(done.returncode == code, f"{' '.join(args)} returned {done.returncode}, expected {code}: {done.stderr}")
    return json.loads(done.stdout)


def check_demo(phase: str) -> None:
    if phase == "before":
        raw = graft_json("verify", "new-base", code=1)
        require(raw["passed"] is False and raw["graft"] == "none", "expected an ungrafted verification failure")
        require((raw["totals"]["passed"], raw["totals"]["total"]) == (13, 52), "expected 13/52 unaided cases")
        return

    # show --json reads the saved certificate and also verifies its digest.
    cert = graft_json("show")
    target = "prod" if phase == "fast-forward" else "new-base"
    require(cert["source"]["name"] == "prod" and cert["target"]["name"] == target, "unexpected source or target")
    totals = cert["totals"]
    require((totals["total"], totals["rules"]) == (52, 5), "expected the complete 52-case, five-rule genome")
    require(len(cert["gates"]) == 5 and not cert["excluded"], "expected all five gates with no excluded cases")
    head = Path(".graft/REBASE_HEAD")
    if phase == "conflict":
        require(head.read_text().strip() == f"{target}/{cert['run']}", "conflict should remain available to continue")
        require(cert["passed"] is False and cert["outcome"] == "conflict", "expected a conflict certificate")
        require(totals["passed"] == 40 and len(cert["failures"]) == 12, "expected 12 unresolved cases")
        return

    kind = "none" if phase == "fast-forward" else "lora"
    outcome = "fast-forward" if kind == "none" else "lora"
    require(not head.exists(), "completed rebase still has a REBASE_HEAD")
    require(cert["passed"] is True and cert["outcome"] == outcome and cert["graft"]["kind"] == kind,
            f"expected completed {outcome} certificate")
    require(totals["passed"] == 52 and totals["rules_ok"] == 5 and not cert["failures"], "expected 52/52 and 5/5")
    require(all(g["ok"] and g["passed"] == g["total"] for g in cert["gates"]), "a rule did not pass")
    held = totals["holdout"]
    require((held["passed"], held["total"]) == (11, 11), "expected all eleven held-out cases to pass")
    if kind == "lora":
        require(cert["transferred_unaided"] == {"passed": 13, "total": 52}, "unexpected ungrafted baseline")
        [arithmetic] = cert["suites"]
        require(arithmetic["suite"] == "arithmetic" and arithmetic["ok"] and arithmetic["cases"] == 30,
                "arithmetic capability gate was missing or failed")
        require(arithmetic["base"] == arithmetic["grafted"] == 0.9, "expected arithmetic to stay at 27/30")
    if phase == "continue":
        require(any(stage["stage"] == "recheck" for stage in cert["ladder"]), "expected the saved run to resume")
    verified = graft_json("verify", target, "--graft")
    require(verified["passed"] is True and verified["graft"] == kind and verified["totals"]["passed"] == 52,
            "installed CLI could not verify the completed graft")


if __name__ == "__main__":
    if sys.argv[1:] in (["tests"], ["tests", "--train"]):
        sys.exit(0 if run_tests("--train" in sys.argv) else 1)
    if len(sys.argv) != 2 or sys.argv[1] not in {"before", "full", "conflict", "continue", "fast-forward"}:
        sys.exit("expected tests [--train] or a demo phase")
    action = sys.argv[1]
    try:
        check_demo(action)
    except (ValueError, KeyError, OSError) as error:
        sys.exit(f"{action}: {error}")
