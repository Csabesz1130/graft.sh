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
            self.assertEqual(every[0]["verify"], [{"type": "json", "equals": {"k": 1}}])
            self.assertEqual(len(every), 2)
            path.write_text("[1]\n")
            with self.assertRaisesRegex(capture.CaptureError, "ft.jsonl:1"):
                capture.from_jsonl(path)


if __name__ == "__main__":
    unittest.main()
