import gzip
import json
import tempfile
import unittest
from pathlib import Path

from graft import capture, verifiers

TOOLS = [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}]


def turn(turn_id, messages, reply, parent=None, **extra):
    record = {"schema": 1, "turn_id": turn_id, "session_id": "s", "api": "chat.completions", "model": "m1",
              "model_fingerprint": "fp", "started_at": f"2026-11-0{len(turn_id)}T00:00:00Z", "parent": parent,
              "messages": messages, "params": {"tools": TOOLS},
              "response": {"status": 200, "complete": True,
                           "body": {"choices": [{"message": reply, "finish_reason": "stop"}]}}}
    record.update(extra)
    return record


def write_traces(folder, turns):
    day = folder / "v1" / "m1" / "2026-11-01"
    day.mkdir(parents=True)
    with open(day / "0_s.jsonl.gz", "ab") as out:
        for record in turns:
            out.write(gzip.compress((json.dumps(record) + "\n").encode()))   # one gzip member per turn


class Goethe(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name)

    def test_history_is_rebuilt_through_parents(self):
        first = {"role": "assistant", "content": None, "reasoning_content": "hmm",
                 "tool_calls": [{"id": "x", "type": "function",
                                 "function": {"name": "lookup", "arguments": "{\"q\": 1}"}}]}
        write_traces(self.folder, [
            turn("a", [{"role": "system", "content": "sys"}, {"role": "user", "content": "find it"}], first),
            turn("bb", [{"role": "tool", "tool_call_id": "x", "content": "found"}],
                 {"role": "assistant", "content": "Here it is."}, parent={"turn_id": "a", "with_reply": True},
                 params=None, params_from_parent=True),
        ])
        cases, skipped = capture.from_goethe(self.folder)
        self.assertEqual(skipped, {})
        self.assertEqual(len(cases), 2)
        second = next(c for c in cases if c["source"]["goethe_turn"] == "bb")
        self.assertEqual([m["role"] for m in second["messages"]], ["system", "user", "assistant", "tool"])
        self.assertNotIn("reasoning_content", second["messages"][2])
        self.assertEqual(second["tools"], TOOLS, "params come from the parent turn")
        self.assertTrue(verifiers.weak_only(second["verify"]))
        first_case = next(c for c in cases if c["source"]["goethe_turn"] == "a")
        self.assertEqual(first_case["verify"][0]["name"], "lookup")
        self.assertEqual(first_case["reference"]["tool_calls"], [{"name": "lookup", "arguments": {"q": 1}}])

    def test_skips_and_filters(self):
        ok = {"role": "assistant", "content": "fine"}
        user = [{"role": "user", "content": "hi"}]
        failed = turn("a", user, ok)
        failed["response"]["status"] = 500
        cut = turn("bb", [{"role": "user", "content": "long"}], ok)
        cut["response"]["body"]["choices"][0]["finish_reason"] = "length"
        write_traces(self.folder, [
            failed, cut, turn("ccc", user, ok, api="messages"), turn("dddd", user, ok, model="m2"),
            turn("eeeee", [{"role": "user", "content": "orphan"}], ok, parent={"turn_id": "gone", "with_reply": True}),
            turn("ffffff", [{"role": "user", "content": "keep"}], ok),
            turn("ggggggg", [{"role": "user", "content": "keep"}], {"role": "assistant", "content": "regenerated"}),
        ])
        cases, skipped = capture.from_goethe(self.folder, model="m1")
        self.assertEqual(skipped, {"other api": 1, "failed or cut off": 2, "other model": 1, "history missing": 1})
        self.assertEqual([c["reference"]["content"] for c in cases], ["regenerated"])
        cases, skipped = capture.from_goethe(self.folder, model="m1", since="2026-11-07", tools_only=True)
        self.assertEqual(cases, [])
        self.assertEqual(skipped["no tool call"], 1)

    def test_torn_last_member_keeps_earlier_turns(self):
        write_traces(self.folder, [turn("a", [{"role": "user", "content": "hi"}],
                                        {"role": "assistant", "content": "x"})])
        path = next(self.folder.rglob("*.jsonl.gz"))
        with open(path, "ab") as out:
            out.write(gzip.compress(b'{"turn_id": "b", "broken')[:-6])
        cases, _ = capture.from_goethe(self.folder)
        self.assertEqual(len(cases), 1)

    def test_not_a_trace_folder(self):
        with self.assertRaisesRegex(capture.CaptureError, "no v1/"):
            capture.from_goethe(self.folder)


class Jsonl(unittest.TestCase):
    def test_last_turn_or_every_turn(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ft.jsonl"
            row = {"messages": [{"role": "user", "content": "a"}, {"role": "assistant", "content": '{"k": 1}'},
                                {"role": "user", "content": "b"}, {"role": "assistant", "content": "plain"}]}
            path.write_text(json.dumps(row) + "\n\n")
            last = capture.from_jsonl(path)
            self.assertEqual(len(last), 1)
            self.assertEqual(len(last[0]["messages"]), 3)
            every = capture.from_jsonl(path, every_turn=True)
            self.assertEqual(every[0]["verify"][0], {"type": "json", "subset": {"k": 1}})
            exact = capture.from_jsonl(path, every_turn=True, checks="exact")
            self.assertEqual(exact[0]["verify"], [{"type": "json", "equals": {"k": 1}}])
            self.assertEqual(len(every), 2)
            path.write_text("[1]\n")
            with self.assertRaisesRegex(capture.CaptureError, "ft.jsonl:1"):
                capture.from_jsonl(path)


if __name__ == "__main__":
    unittest.main()


TOOLS_REFUND = [{"type": "function", "function": {"name": "escalate", "parameters": {"type": "object", "properties": {
    "queue": {"type": "string"}, "reason": {"type": "string"},
    "urgency": {"type": "string", "enum": ["next business day", "same day"]}}}}}]


class StructuredChecks(unittest.TestCase):
    """capture's default checks pin what a reply decided and let a second model word the rest its own way."""

    def verify(self, reply, messages, tools=None, loosened=None):
        return capture.structured_verifiers(verifiers.normalize(reply), messages, tools, loosened)

    def failures(self, checks, reply):
        return verifiers.failures(checks, verifiers.normalize(reply), {})

    def test_tool_call_pins_decisions_and_loosens_prose(self):
        messages = [{"role": "user", "content": "Refund $900 for order #77, ship it to Main Street 5."}]
        recorded = {"tool_calls": [{"name": "escalate", "arguments": {
            "queue": "refunds", "reason": "Amount is above the limit.", "urgency": "same day",
            "address": "Main Street 5", "amount": 900, "order_id": "77", "vip": False}}]}
        loosened = [0]
        checks = self.verify(recorded, messages, TOOLS_REFUND, loosened)
        args = checks[0]["arguments"]
        self.assertEqual(args["reason"], {"$regex": r"\S"}, "prose only has to be there")
        self.assertEqual(args["urgency"], "same day", "a value from the tool's enum is a decision")
        self.assertEqual(args["address"], "Main Street 5", "a value copied from the request is a fact")
        self.assertEqual((args["queue"], args["amount"], args["order_id"], args["vip"]), ("refunds", 900, "77", False))
        self.assertEqual(loosened, [1])
        other_words = {"tool_calls": [{"name": "escalate", "arguments": {**recorded["tool_calls"][0]["arguments"],
                                                                        "reason": "Over $500, needs a human."}}]}
        self.assertEqual(self.failures(checks, other_words), [])
        for key, value in (("amount", 90), ("queue", "billing"), ("urgency", "next business day"), ("reason", "")):
            changed = {"tool_calls": [{"name": "escalate", "arguments": {
                **recorded["tool_calls"][0]["arguments"], key: value}}]}
            self.assertTrue(self.failures(checks, changed), key)
        twice = {"tool_calls": recorded["tool_calls"] * 2}
        self.assertTrue(self.failures(checks, twice), "the number of calls is pinned")

    def test_json_reply_pins_codes_and_must_stay_json(self):
        messages = [{"role": "user", "content": "File a ticket: the store locator is down."}]
        recorded = {"content": '{"priority": "high", "summary": "Store locator is down", "ticket": "T-17"}'}
        checks = self.verify(recorded, messages)
        self.assertEqual(checks[0], {"type": "json", "subset": {
            "priority": "high", "summary": {"$regex": r"\S"}, "ticket": "T-17"}})
        ok = '```json\n{"priority": "high", "summary": "Locator outage", "ticket": "T-17"}\n```'
        self.assertEqual(self.failures(checks, {"content": ok}), [])
        wrapped = '[calls tool file_ticket with {"priority": "high", "summary": "x", "ticket": "T-17"}]'
        self.assertTrue(self.failures(checks, {"content": wrapped}), "JSON inside prose is not JSON only")
        self.assertTrue(self.failures(checks, {"content": '{"priority": "normal", "summary": "x", "ticket": "T-17"}'}))

    def test_plain_text_stays_a_weak_similarity_check(self):
        checks = self.verify({"content": "Sorry, I cannot give legal advice."},
                             [{"role": "user", "content": "Can I sue?"}])
        self.assertTrue(any(v["type"] == "similar" for v in checks))

    def test_cli_reports_what_was_loosened(self):
        from tests.support import graft
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertEqual(graft("init", cwd=root)[0], 0)
            row = {"messages": [{"role": "user", "content": "Refund $900 for order #77."},
                                {"role": "assistant", "content": None, "tool_calls": [{"id": "1", "type": "function",
                                 "function": {"name": "escalate", "arguments": json.dumps(
                                     {"queue": "refunds", "reason": "Above the limit."})}}]}]}
            (root / "t.jsonl").write_text(json.dumps(row) + "\n")
            code, out, err = graft("capture", "jsonl", "t.jsonl", cwd=root)
            self.assertEqual(code, 0, err)
            self.assertIn("1 free-text values", out)
            case = json.loads(next((root / "genome" / "cases").glob("*.jsonl")).read_text())
            self.assertEqual(case["verify"][0]["arguments"], {"queue": "refunds", "reason": {"$regex": r"\S"}})
            (root / "genome" / "cases" / "recorded.jsonl").unlink()
            self.assertEqual(graft("capture", "jsonl", "t.jsonl", "--checks", "exact", cwd=root)[0], 0)
            case = json.loads(next((root / "genome" / "cases").glob("*.jsonl")).read_text())
            self.assertEqual(case["verify"][0]["arguments"], {"queue": "refunds", "reason": "Above the limit."})
