"""The user guide's recipe: recorded traffic from a hosted model, carried onto a model of your own, no source model."""
import json

from tests.support import ExampleCase, graft

STATEMENT = "Refunds above $500 are never issued directly. Escalate them to the refunds queue."


class HostedToOwn(ExampleCase):
    def test_capture_then_rebase_without_a_source(self):
        # the "hosted model's" traffic, as a chat fine-tuning file: the example's refund requests and recorded replies
        rows = []
        for line in (self.root / "genome" / "cases" / "refund-limit.jsonl").read_text().splitlines():
            case = json.loads(line)
            for unit in case.get("pair") or [case]:
                calls = [{"id": f"c{i}", "type": "function",
                          "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])}}
                         for i, c in enumerate(unit["reference"].get("tool_calls") or [])]
                reply = {"role": "assistant", "content": unit["reference"].get("content") or None, "tool_calls": calls}
                rows.append({"messages": [*unit["messages"], reply], "tools": unit.get("tools") or case.get("tools")})
        mine = self.root.parent / "mine"
        mine.mkdir()
        (mine / "traffic.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))

        def run(*argv):
            return graft(*argv, cwd=mine)

        self.assertEqual(run("init")[0], 0)
        self.assertEqual(run("rule", "add", "refund-limit", STATEMENT)[0], 0)
        code, out, err = run("capture", "jsonl", "traffic.jsonl", "--rule", "refund-limit")
        self.assertEqual(code, 0, out + err)
        self.assertIn(f"added {len(rows)} cases", out)
        self.assertEqual(run("check")[0], 0, "the recorded replies pass the verifiers made from them")
        with open(mine / "graft.toml", "a") as out_file:
            out_file.write(f'\n[models.own]\nendpoint = "{self.endpoint}"\nmodel = "open/new-base-30b"\n')
        code, out, err = run("rebase", "--target", "own")
        self.assertEqual(code, 0, out + err)
        self.assertIn("hold on references", out)
        self.assertIn("zero-shot", out)
        self.assertIn("as a prompt graft", out)
        cert = json.loads(next((mine / ".graft" / "grafts").glob("*/*/certificate.json")).read_text())
        self.assertEqual(cert["source"]["name"], "references")
        self.assertTrue(cert["passed"])
        self.assertEqual(cert["new_behavior"], [])
