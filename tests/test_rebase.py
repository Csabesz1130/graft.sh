import json
import unittest

from graft import certificate, rebase
from tests.support import ExampleCase


class Ladder(ExampleCase):
    def run_dir(self):
        return next((self.root / ".graft" / "grafts").glob("*/*"))

    def cert(self):
        return json.loads((self.run_dir() / "certificate.json").read_text())

    def test_full_ladder_ends_in_a_trained_adapter(self):
        code, out, err = self.graft("rebase", "--target", "new-base")
        self.assertEqual(code, 0, out + err)
        for line in ("52/52 cases hold on prod", "zero-shot         13/52", "prompt round 1    32/52",
                     "prompt round 2    40/52", "lora round 1      52/52", "as a trained adapter"):
            self.assertIn(line, out)
        cert = self.cert()
        self.assertTrue(cert["passed"])
        self.assertEqual(cert["outcome"], "lora")
        self.assertTrue(certificate.intact(cert))
        self.assertTrue(all(g["ok"] and g["passed"] == g["total"] for g in cert["gates"]))
        self.assertEqual(cert["transferred_unaided"], {"passed": 13, "total": 52})
        self.assertEqual(cert["totals"]["holdout"]["looks"], 1)
        self.assertEqual(cert["suites"][0]["gain"], round(27 / 30 - 20 / 30, 4))
        self.assertFalse((self.root / ".graft" / "REBASE_HEAD").exists())

        rows = [json.loads(line) for line in
                (self.run_dir() / "lora" / "round-1" / "dataset.jsonl").read_text().splitlines()]
        held = {c["case"] for c in json.loads((self.run_dir() / "evals" / "zero-shot.json").read_text())["results"]
                if c["split"] == "holdout"}
        self.assertTrue(held)
        self.assertFalse(held & {r["case"] for r in rows}, "held-out cases never reach the trainer")
        self.assertEqual({r["origin"] for r in rows}, {"own", "sampled", "source"})
        self.assertTrue(all(r["messages"][-1]["role"] == "assistant" for r in rows))
        before = json.loads((self.run_dir() / "evals" / "prompt round 2.json").read_text())["results"]
        failing = {c["rule"] for c in before if c["split"] == "train" and not c["ok"]}
        self.assertTrue(failing)
        self.assertTrue(all(r["weight"] == (2 if r["rule"] in failing else 1) for r in rows),
                        "every row of a rule that still fails is weighted up, its passing replies too")
        self.assertTrue(any(r["origin"] == "own" and r["weight"] == 2 for r in rows))
        by_case = {}
        for r in rows:
            by_case.setdefault(r["case"], set()).add(r["weight"])
        self.assertTrue(all(len(w) == 1 for w in by_case.values()), "both sides of a pair weigh the same")
        cert = self.cert()
        self.assertEqual(len(cert["graft"]["adapter_sha256"]), 64)
        self.assertIn("Adapter contents", certificate.report(cert))
        graft = json.loads((self.run_dir() / "evals" / "prompt round 2.json").read_text())["graft"]
        shown = {case for cases in graft["exemplars"].values() for case in cases}
        self.assertTrue(shown)
        self.assertFalse(held & shown, "held-out cases never become prompt examples")

        code, out, _ = self.graft("verify", "new-base", "--graft")
        self.assertEqual(code, 0, out)
        self.assertIn("with the trained adapter", out)
        code, out, _ = self.graft("verify", "new-base", "--json")
        self.assertEqual(code, 1)
        self.assertFalse(json.loads(out)["passed"])
        code, out, _ = self.graft("show")
        self.assertEqual(code, 0)
        self.assertIn("**PASSED.**", out)
        code, out, _ = self.graft("log")
        self.assertIn("passed    lora", out)

    def test_replay_trains_on_the_new_base_own_replies(self):
        replay = self.root / "genome" / "replay.jsonl"
        replay.write_text("".join(json.dumps({"messages": [{"role": "user", "content": q}]}) + "\n" for q in
                                  ("What is 31 * 7? Reply with the number only.", "Name a colour of the sky.")))
        self.edit(lambda text: text.replace("[trainer]", '[trainer]\nreplay = "genome/replay.jsonl"'))
        code, out, err = self.graft("rebase", "--target", "new-base")
        self.assertEqual(code, 0, out + err)
        self.assertIn("own replies to 2 requests", out)
        rows = [json.loads(line) for line in
                (self.run_dir() / "lora" / "round-1" / "dataset.jsonl").read_text().splitlines()]
        replayed = [r for r in rows if r["origin"] == "replay"]
        self.assertEqual(len(replayed), 2)
        self.assertTrue(all(r["rule"] is None and r["weight"] == 1 for r in replayed))
        self.assertIn("217", replayed[0]["messages"][-1]["content"], "the new base's own answer, not a label")
        cert = self.cert()
        self.assertEqual(cert["ladder"][-1]["trained_on"]["replay"], 2)
        self.assertIn("2 replayed", certificate.report(cert))
        for kept in ("suite arithmetic new base", "suite arithmetic grafted", "suite arithmetic source"):
            self.assertTrue((self.run_dir() / "evals" / f"{kept}.json").is_file(), kept)

    def test_the_target_template_args_reach_the_trainer(self):
        (self.root / "record_trainer.py").write_text(
            "import pathlib, runpy, sys\n"
            "pathlib.Path('template_args.json').write_text(sys.argv.pop())\n"
            "runpy.run_path('mock_trainer.py', run_name='__main__')\n")
        served = 'adapter_model = "new-base-graft"'
        self.edit(lambda text: text.replace(served, served + "\ntemplate_args = { enable_thinking = false }")
                  .replace('"mock_trainer.py", "{dataset}", "{output}"',
                           '"record_trainer.py", "{dataset}", "{output}", "{template_args}"'))
        code, out, err = self.graft("rebase", "--target", "new-base")
        self.assertEqual(code, 0, out + err)
        self.assertEqual(json.loads((self.root / "template_args.json").read_text()), {"enable_thinking": False})
        self.assertEqual(self.cert()["target"]["fingerprint"]["template_args"], {"enable_thinking": False})

    def test_template_args_must_be_a_table(self):
        self.edit(lambda text: text.replace('adapter_model = "new-base-graft"',
                                            'adapter_model = "new-base-graft"\ntemplate_args = "no thinking"'))
        code, out, err = self.graft("rebase", "--target", "new-base")
        self.assertEqual(code, 1, out + err)
        self.assertIn("[models.new-base] template_args must be a table", err)
        self.assertNotIn("Traceback", err)

    def test_continue_after_a_suite_conflict_trains_again(self):
        self.edit(lambda text: text.replace("max_loss = 0.02", "max_loss = -0.5"))   # a gate nothing can pass
        code, out, _ = self.graft("rebase", "--target", "new-base")
        self.assertEqual(code, 1)
        self.assertIn("capability suite gate fails", out)
        self.assertIn("add [trainer] replay", out)
        code, out, err = self.graft("rebase", "--continue")
        self.assertEqual(code, 1, out + err)
        self.assertIn("lora round 2", out, "a suite conflict is not a dead end: --continue trains again")
        self.assertTrue((self.run_dir() / "lora" / "round-2" / "adapter").exists())
        self.edit(lambda text: text.replace("max_loss = -0.5", "max_loss = 0.02"))
        code, out, err = self.graft("rebase", "--continue")
        self.assertEqual(code, 0, out + err)
        self.assertNotIn("lora round 3", out, "once the suites pass, nothing is trained again")
        self.assertIn("as a trained adapter", out)

    def test_replay_must_not_repeat_a_suite_request(self):
        suite = (self.root / "genome" / "suites" / "arithmetic.jsonl").read_text().splitlines()[0]
        (self.root / "genome" / "replay.jsonl").write_text(suite + "\n")
        self.edit(lambda text: text.replace("[trainer]", '[trainer]\nreplay = "genome/replay.jsonl"'))
        code, out, err = self.graft("rebase", "--target", "new-base")
        self.assertEqual(code, 1)
        self.assertIn("must not repeat a genome or suite request", err)
        self.assertIn("suite arithmetic case", err)

    def test_graft_means_the_latest_passed_run(self):
        self.assertEqual(self.graft("rebase", "--target", "new-base")[0], 0)
        self.assertEqual(self.graft("rebase", "--target", "new-base", "--ladder", "zero-shot")[0], 1)
        self.assertEqual(self.graft("rebase", "--abort")[0], 0)
        code, out, _ = self.graft("verify", "new-base", "--graft")
        self.assertEqual(code, 0, "an aborted run after it does not hide the graft that passed")
        self.assertIn("with the trained adapter", out)
        code, out, _ = self.graft("show", "--graft")
        self.assertEqual(code, 0)
        self.assertIn("adapter", out)
        aborted = next(r for r in json.loads(self.graft("log", "--json")[1]) if r["status"] == "aborted")
        code, _, err = self.graft("show", "--graft", f"{aborted['target']}/{aborted['run']}")
        self.assertEqual(code, 1)
        self.assertIn("aborted; it has no graft to apply", err)

    def test_only_aborted_runs_means_no_graft(self):
        self.assertEqual(self.graft("rebase", "--target", "new-base", "--ladder", "zero-shot")[0], 1)
        self.assertEqual(self.graft("rebase", "--abort")[0], 0)
        code, _, err = self.graft("verify", "new-base", "--graft")
        self.assertEqual(code, 1)
        self.assertIn("was aborted, so there is no graft to use", err)

    def test_a_changed_adapter_is_caught(self):
        self.assertEqual(self.graft("rebase", "--target", "new-base")[0], 0)
        adapter = next(self.run_dir().glob("lora/round-*/adapter"))
        self.assertEqual(self.graft("show")[0], 0)
        (adapter / "learned.json").write_text('{"rules": []}')
        code, _, err = self.graft("show")
        self.assertEqual(code, 1, "show fails when the adapter is not the one the certificate names")
        self.assertIn("was changed after the rebase", err)
        code, _, err = self.graft("verify", "new-base", "--graft")
        self.assertEqual(code, 1)
        self.assertIn("restore it or run the rebase again", err)

    def test_graft_digest_does_not_depend_on_the_folder(self):
        self.assertEqual(self.graft("rebase", "--target", "new-base")[0], 0)
        first = self.cert()["graft"]
        self.assertEqual(self.graft("rebase", "--target", "new-base")[0], 0)
        runs = sorted((self.root / ".graft" / "grafts" / "new-base").glob("*/certificate.json"))
        second = json.loads(runs[-1].read_text())["graft"]
        self.assertNotEqual(first["adapter"], second["adapter"])
        self.assertEqual(first["adapter_sha256"], second["adapter_sha256"])
        self.assertEqual(first["sha256"], second["sha256"])

    def test_conflict_report_says_held_out_cases_were_not_consulted(self):
        code, _, _ = self.graft("rebase", "--target", "new-base", "--ladder", "zero-shot")
        self.assertEqual(code, 1)
        report = certificate.report(self.cert())
        self.assertIn("never steered the search", report)
        self.assertNotIn("after the training cases all passed", report)

    def test_same_model_is_a_fast_forward(self):
        code, out, _ = self.graft("rebase", "--target", "prod", "--source", "prod")
        self.assertEqual(code, 0, out)
        self.assertIn("fast-forward", out)
        self.assertNotIn("prompt round", out)
        self.assertEqual(self.cert()["outcome"], "fast-forward")
        self.assertEqual(self.cert()["graft"]["kind"], "none")
        code, _, err = self.graft("show", "--graft")
        self.assertIn("no graft", err)

    def test_conflict_then_continue_with_a_longer_ladder(self):
        code, out, _ = self.graft("rebase", "--target", "new-base", "--ladder", "zero-shot,prompt")
        self.assertEqual(code, 1)
        self.assertIn("CONFLICT: 12 cases still fail", out)
        self.assertIn("account-closure/closure-first-1041", out)
        cert = self.cert()
        self.assertFalse(cert["passed"])
        self.assertEqual(len(cert["failures"]), 12)
        self.assertEqual(cert["totals"]["holdout"]["looks"], 1, "no look yet: the training cases never all passed")
        code, out, _ = self.graft("status")
        self.assertIn("Rebase in progress onto new-base (conflict, prompt rung)", out)
        code, out, err = self.graft("rebase", "--target", "new-base")
        self.assertEqual(code, 1)
        self.assertIn("a rebase is in progress", err)
        code, out, err = self.graft("show", "--graft")
        self.assertIn("Standing rules.", out)
        self.assertIn("Example (rule", out)

        code, out, err = self.graft("rebase", "--continue", "--ladder", "zero-shot,prompt,lora")
        self.assertEqual(code, 0, out + err)
        self.assertIn("recheck           40/52", out)
        self.assertIn("lora round 1      52/52", out)
        self.assertTrue(self.cert()["passed"])

    def test_continue_after_the_genome_is_fixed(self):
        self.graft("rebase", "--target", "new-base", "--ladder", "zero-shot,prompt")
        (self.root / "genome" / "cases" / "account-closure.jsonl").unlink()
        (self.root / "genome" / "rules" / "account-closure.toml").unlink()
        code, out, err = self.graft("rebase", "--continue")
        self.assertEqual(code, 0, out + err)
        self.assertIn("the genome changed", out)
        self.assertIn("recheck           40/40", out)
        self.assertEqual(self.cert()["outcome"], "prompt")
        self.assertEqual(self.cert()["genome"]["rules"], 4)

    def test_pause_until_the_adapter_is_served(self):
        self.edit(lambda text: "\n".join(line for line in text.splitlines() if not line.startswith("after_train")))
        code, out, _ = self.graft("rebase", "--target", "new-base")
        self.assertEqual(code, rebase.PAUSED)
        self.assertIn("Load it into the endpoint as model 'new-base-graft'", out)
        state = json.loads((self.run_dir() / "state.json").read_text())
        self.assertEqual((state["status"], state["rung"], state["round"]), ("paused", "lora", 1))
        adapter = self.run_dir() / "lora" / "round-1" / "adapter"
        self.load_adapter(adapter, "new-base-graft")
        code, out, err = self.graft("rebase", "--continue")
        self.assertEqual(code, 0, out + err)
        self.assertIn("lora round 1      52/52", out)
        self.assertNotIn("training", out)
        trained = self.cert()["ladder"][-1]["trained_on"]
        rows = (adapter.parent / "dataset.jsonl").read_text().splitlines()
        self.assertEqual(sum(trained.values()), len(rows), "what was trained on survives the pause")
        self.assertGreater(trained["sampled"], 0)

    def test_abort(self):
        self.graft("rebase", "--target", "new-base", "--ladder", "zero-shot")
        code, out, _ = self.graft("rebase", "--abort")
        self.assertEqual(code, 0)
        self.assertIn("dropped the rebase onto new-base", out)
        self.assertEqual(self.graft("rebase", "--abort")[0], 1)
        self.assertEqual(self.graft("rebase", "--continue")[0], 1)
        self.assertIn("aborted", self.graft("log")[1])
        self.assertEqual(self.graft("rebase", "--target", "prod")[0], 0)

    def test_cases_the_source_fails_are_left_out(self):
        path = self.root / "genome" / "cases" / "ticket-json.jsonl"
        extra = {"id": "ticket-never", "messages": [{"role": "user", "content": "File a ticket: hello"}],
                 "verify": [{"type": "json", "subset": {"priority": "urgent"}}]}
        path.write_text(path.read_text() + json.dumps(extra) + "\n")
        code, out, _ = self.graft("check", "--model", "prod")
        self.assertEqual(code, 1)
        self.assertIn("ticket-never", out)
        code, out, err = self.graft("rebase", "--target", "prod")
        self.assertEqual(code, 0, out + err)
        self.assertIn("52/53 cases hold on prod   1 left out", out)
        cert = self.cert()
        self.assertEqual([e["case"] for e in cert["excluded"]], ["ticket-never"])
        self.assertEqual(cert["totals"]["total"], 52)

    def test_a_suite_gate_can_fail_the_rebase(self):
        self.edit(lambda text: text.replace("min_gain = 0.0", "min_gain = 0.5"))
        code, out, _ = self.graft("rebase", "--target", "new-base")
        self.assertEqual(code, 1)
        self.assertIn("is below min_gain", out)
        self.assertIn("a capability suite gate fails", out)
        self.assertFalse(self.cert()["passed"])

    def test_unreachable_target_stops_without_training(self):
        self.edit(lambda text: text.replace('model = "open/new-base-30b"', 'model = "open/not-loaded"'))
        code, out, err = self.graft("rebase", "--target", "new-base")
        self.assertEqual(code, 1)
        self.assertIn("requests to new-base failed", err)
        self.assertIn("is not loaded", err)
        self.assertFalse((self.run_dir() / "lora").exists())
        self.edit(lambda text: text.replace('model = "open/not-loaded"', 'model = "open/new-base-30b"'))
        code, out, err = self.graft("rebase", "--continue")
        self.assertEqual(code, 0, out + err)

    def test_tampered_certificate_is_noticed(self):
        self.graft("rebase", "--target", "prod")
        path = self.run_dir() / "certificate.json"
        cert = json.loads(path.read_text())
        cert["totals"]["passed"] += 1
        path.write_text(json.dumps(cert))
        code, _, err = self.graft("show")
        self.assertEqual(code, 1)
        self.assertIn("digest no longer matches", err)

    def test_usage_errors(self):
        self.assertEqual(self.graft("rebase")[0], 2)
        code, _, err = self.graft("rebase", "--target", "nope")
        self.assertIn("no model named 'nope'", err)
        code, _, err = self.graft("rebase", "--target", "prod", "--ladder", "zero-shot,magic")
        self.assertIn("ladder rungs are", err)


class TrainerArgv(unittest.TestCase):
    fill = {"dataset": "d.jsonl", "output": "out", "base": "Qwen/Qwen3-8B", "round": "1", "template_args": "{}"}

    def test_builtin_trainer_gets_checkpointing_and_template_args(self):
        argv = rebase.trainer_argv({"method": "lora", "gradient_checkpointing": True},
                                   {**self.fill, "template_args": '{"enable_thinking": false}'})
        self.assertEqual(argv[1:3], ["-m", "graft.train_hf"])
        self.assertIn("--gradient-checkpointing", argv)
        self.assertEqual(argv[argv.index("--template-args") + 1], '{"enable_thinking": false}')
        plain = rebase.trainer_argv({"method": "lora", "gradient_checkpointing": False}, self.fill)
        self.assertNotIn("--gradient-checkpointing", plain)
        self.assertNotIn("--template-args", plain, "only passed when the target sets template_args")

    def test_own_trainer_command_can_read_template_args(self):
        argv = rebase.trainer_argv({"command": ["t", "{base}", "{template_args}"]},
                                   {**self.fill, "template_args": '{"enable_thinking": false}'})
        self.assertEqual(argv, ["t", "Qwen/Qwen3-8B", '{"enable_thinking": false}'])


class PromptBlock(unittest.TestCase):
    def test_assistant_message_shape(self):
        message = rebase.assistant_message({"content": "<think>x</think>ok",
                                            "tool_calls": [{"name": "f", "arguments": {"b": 1, "a": 2}}]})
        self.assertEqual(message["content"], "ok")
        self.assertEqual(message["tool_calls"][0]["function"], {"name": "f", "arguments": '{"a": 2, "b": 1}'})
        self.assertNotIn("tool_calls", rebase.assistant_message({"content": "plain"}))


if __name__ == "__main__":
    unittest.main()
