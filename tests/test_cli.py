import json
import tempfile
import unittest
from pathlib import Path

from graft import __version__
from tests.support import ExampleCase, graft


class Fresh(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def graft(self, *argv):
        return graft(*argv, cwd=self.root)

    def test_outside_a_project(self):
        code, _, err = self.graft("status")
        self.assertEqual(code, 1)
        self.assertIn("run `graft init` first", err)

    def test_help_and_version(self):
        code, out, _ = self.graft()
        self.assertEqual(code, 0)
        self.assertIn("rebase", out)
        code, out, _ = self.graft("--version")
        self.assertEqual((code, out.strip()), (0, f"graft {__version__}"))
        self.assertEqual(self.graft("nonsense")[0], 2)

    def test_from_nothing_to_a_checked_genome(self):
        (self.root / ".gitignore").write_text("node_modules/\n")
        code, out, _ = self.graft("init")
        self.assertEqual(code, 0)
        self.assertEqual((self.root / ".gitignore").read_text(), "node_modules/\n.graft/\n")
        self.assertEqual(self.graft("init")[0], 1, "a second init changes nothing")

        code, out, _ = self.graft("rule", "add", "refund-limit", "Refunds above $500 go to a human.", "--gate", "0.9")
        self.assertEqual(code, 0)
        self.assertIn("genome/cases/refund-limit.jsonl", out)
        code, out, _ = self.graft("status")
        self.assertIn("No cases yet for: refund-limit", out)

        rows = [{"messages": [{"role": "user", "content": "Refund $900 please"},
                              {"role": "assistant", "content": None, "tool_calls": [
                                  {"id": "1", "type": "function",
                                   "function": {"name": "escalate", "arguments": "{\"queue\": \"refunds\"}"}}]}],
                 "tools": [{"type": "function", "function": {"name": "escalate"}}]},
                {"messages": [{"role": "user", "content": "thanks"},
                              {"role": "assistant", "content": "You're welcome!"}]}]
        (self.root / "ft.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        code, out, _ = self.graft("capture", "jsonl", "ft.jsonl", "--dry-run")
        self.assertIn("would add 2 cases under recorded", out)
        self.assertFalse((self.root / "genome" / "cases" / "recorded.jsonl").exists())
        code, out, _ = self.graft("capture", "jsonl", "ft.jsonl")
        self.assertIn("added rule recorded", out)
        self.assertIn("added 2 cases under recorded", out)
        self.assertIn("1 cases are plain text", out)
        code, out, _ = self.graft("capture", "jsonl", "ft.jsonl")
        self.assertIn("added 0 cases under recorded (2 were already there)", out)

        code, out, _ = self.graft("status", "--json")
        status = json.loads(out)
        self.assertEqual({r["rule"]: (r["cases"], r["weak"]) for r in status["rules"]},
                         {"recorded": (2, 1), "refund-limit": (0, 0)})
        code, out, _ = self.graft("check")
        self.assertEqual(code, 0, out)
        self.assertIn("Every checked case holds on the source.", out)
        code, out, _ = self.graft("models")
        self.assertIn("no models in graft.toml yet", out)
        code, out, _ = self.graft("log")
        self.assertIn("no rebase has run here yet", out)


class Example(ExampleCase):
    def test_models_check_verify(self):
        code, out, _ = self.graft("models")
        self.assertEqual(code, 0, out)
        self.assertEqual(out.count("ok"), 2)
        code, out, _ = self.graft("check", "--model", "prod")
        self.assertEqual(code, 0, out)
        code, out, _ = self.graft("verify", "new-base")
        self.assertEqual(code, 1)
        self.assertIn("ticket-json      8/8      holds", out)
        self.assertIn("refund-limit     5/16     FAILS", out)
        self.assertIn("... and 8 more", out)
        self.assertIn("13/52 cases pass; 1 of 5 rules hold. Held out: ", out)
        code, out, _ = self.graft("verify", "new-base", "--rule", "ticket-json")
        self.assertEqual(code, 0, out)
        code, _, err = self.graft("verify", "new-base", "--rule", "nope")
        self.assertIn("no rule named 'nope'", err)
        code, _, err = self.graft("verify", "new-base", "--graft")
        self.assertIn("no rebase has run here yet", err)

    def test_a_dead_endpoint_is_reported_by_models(self):
        self.edit(lambda text: text.replace(f"127.0.0.1:{self.port}", "127.0.0.1:9"))
        code, out, _ = self.graft("models", "--json")
        self.assertEqual(code, 1)
        self.assertEqual([row["ok"] for row in json.loads(out)], [False, False])


if __name__ == "__main__":
    unittest.main()
