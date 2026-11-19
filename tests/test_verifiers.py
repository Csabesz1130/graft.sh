import sys
import tempfile
import unittest
from pathlib import Path

from graft import verifiers
from graft.verifiers import check


def text(content):
    return {"content": content}


def call(name, **arguments):
    return {"content": "", "tool_calls": [{"name": name, "arguments": arguments}]}


class Text(unittest.TestCase):
    def test_contains(self):
        self.assertIsNone(check({"type": "contains", "value": "Refund"}, text("a refund is due")))
        self.assertIn("missing", check({"type": "contains", "value": "Refund", "ignore_case": False}, text("refund")))
        self.assertIsNone(check({"type": "contains", "any": ["x", "due"]}, text("a refund is due")))
        self.assertIn("none of", check({"type": "contains", "any": ["x", "y"]}, text("a refund is due")))
        self.assertIn("'b'", check({"type": "contains", "all": ["a", "b"]}, text("a")))
        self.assertIn("contains", check({"type": "not_contains", "value": "sue"}, text("you could sue")))

    def test_regex(self):
        self.assertIsNone(check({"type": "regex", "pattern": r"ending in \d{4}"}, text("card ending in 4242")))
        self.assertIn("no match", check({"type": "regex", "pattern": r"^\d+$"}, text("12a")))
        self.assertIn("4242 4242", check({"type": "not_regex", "pattern": r"\d{4} \d{4}"}, text("4242 4242")))
        with self.assertRaises(verifiers.VerifierError):
            check({"type": "regex", "pattern": "("}, text("x"))

    def test_thinking_is_not_part_of_the_reply(self):
        reply = text("<think>I could say sue.</think>\nI can't give legal advice.")
        self.assertIsNone(check({"type": "not_contains", "value": "sue"}, reply))
        self.assertIsNone(check({"type": "equals", "value": "I can't give legal advice."}, reply))

    def test_equals_and_length(self):
        self.assertIsNone(check({"type": "equals", "value": "a  b"}, text(" a b\n")))
        self.assertIsNotNone(check({"type": "equals", "value": "a  b", "normalize": False}, text("a b")))
        self.assertIn("above 3", check({"type": "length", "max_chars": 3}, text("abcd")))
        self.assertIn("below 9", check({"type": "length", "min_chars": 9}, text("abcd")))

    def test_similar(self):
        context = {"reference": text("Your order ships on Monday by courier.")}
        self.assertIsNone(check({"type": "similar"}, text("The order ships Monday by courier."), context))
        self.assertIn("token F1", check({"type": "similar"}, text("No idea, sorry."), context))
        with self.assertRaises(verifiers.VerifierError):
            check({"type": "similar"}, text("x"), {})


class Json(unittest.TestCase):
    SCHEMA = {"type": "object", "required": ["priority", "summary"], "additionalProperties": False,
              "properties": {"priority": {"enum": ["high", "normal"]},
                             "summary": {"type": "string", "minLength": 1, "maxLength": 10}}}

    def test_json_in_fences_and_prose(self):
        reply = text('Here you go:\n```json\n{"priority": "high", "summary": "outage"}\n```')
        self.assertIsNone(check({"type": "json", "schema": self.SCHEMA}, reply))
        self.assertIsNone(check({"type": "json", "subset": {"priority": "high"}}, reply))
        self.assertIsNotNone(check({"type": "json", "equals": {"priority": "high"}}, reply))
        self.assertIn("no JSON", check({"type": "json"}, text("no json here")))

    def test_schema_errors(self):
        bad = [('{"priority": "low", "summary": "x"}', "not one of"), ('{"priority": "high"}', "missing 'summary'"),
               ('{"priority": "high", "summary": "x", "extra": 1}', "unexpected 'extra'"),
               ('{"priority": "high", "summary": 5}', "expected string"),
               ('{"priority": "high", "summary": "much too long"}', "longer than"), ("[1]", "expected object")]
        for content, why in bad:
            self.assertIn(why, check({"type": "json", "schema": self.SCHEMA}, text(content)), content)

    def test_schema_numbers_arrays_anyof(self):
        schema = {"type": "array", "minItems": 1,
                  "items": {"anyOf": [{"type": "integer", "minimum": 0}, {"type": "null"}]}}
        self.assertEqual(verifiers.schema_errors(schema, [1, None]), [])
        self.assertTrue(verifiers.schema_errors(schema, []))
        self.assertTrue(verifiers.schema_errors(schema, [-1]))
        self.assertTrue(verifiers.schema_errors(schema, [True]), "a boolean is not an integer")


class ToolCalls(unittest.TestCase):
    def test_subset_and_exact(self):
        reply = call("escalate", queue="refunds", reason="over the limit")
        self.assertIsNone(check({"type": "tool_call", "name": "escalate", "arguments": {"queue": "refunds"}}, reply))
        self.assertIsNotNone(check({"type": "tool_call", "name": "escalate", "arguments": {"queue": "refunds"},
                                    "exact": True}, reply))
        self.assertIn("calls made: escalate", check({"type": "tool_call", "name": "issue_refund"}, reply))
        self.assertIn("called with",
                      check({"type": "tool_call", "name": "escalate", "arguments": {"queue": "x"}}, reply))

    def test_openai_shape_and_json_text_arguments(self):
        reply = {"content": None, "tool_calls": [{"id": "c", "type": "function",
                                                  "function": {"name": "issue_refund",
                                                               "arguments": '{"order_id": "7", "amount": 20}'}}]}
        self.assertIsNone(check({"type": "tool_call", "name": "issue_refund",
                                 "arguments": {"order_id": "7", "amount": 20.0}, "count": 1}, reply))
        self.assertIn("expected 2", check({"type": "tool_call", "name": "issue_refund", "count": 2}, reply))

    def test_matchers(self):
        reply = call("issue_refund", order_id="A-77", amount=19.999, note="x")
        ok = {"order_id": {"$regex": r"^A-\d+$"}, "amount": {"$approx": 20, "tol": 0.01}, "note": {"$any": True}}
        self.assertIsNone(check({"type": "tool_call", "name": "issue_refund", "arguments": ok}, reply))
        self.assertIsNotNone(check({"type": "tool_call", "name": "issue_refund",
                                    "arguments": {"amount": {"$approx": 21}}}, reply))
        self.assertFalse(verifiers.matches(1, True), "true is not the number 1")

    def test_no_tool_call_and_index(self):
        two = {"content": "", "tool_calls": [{"name": "a", "arguments": {}}, {"name": "b", "arguments": {}}]}
        self.assertIsNone(check({"type": "no_tool_call"}, text("hello")))
        self.assertEqual(check({"type": "no_tool_call"}, two), "called a")
        self.assertIsNone(check({"type": "no_tool_call", "name": "c"}, two))
        self.assertIsNone(check({"type": "tool_call", "name": "b", "index": 1}, two))
        self.assertIsNotNone(check({"type": "tool_call", "name": "b", "index": 0}, two))


class Combined(unittest.TestCase):
    def test_all_any_not(self):
        yes, no = {"type": "contains", "value": "a"}, {"type": "contains", "value": "z"}
        self.assertIsNone(check({"type": "any", "of": [no, yes]}, text("a")))
        self.assertIn("none of", check({"type": "any", "of": [no, no]}, text("a")))
        self.assertIsNotNone(check({"type": "all", "of": [yes, no]}, text("a")))
        self.assertIsNone(check({"type": "not", "of": no}, text("a")))
        self.assertIsNotNone(check({"type": "not", "of": yes}, text("a")))

    def test_unknown_type_and_missing_field(self):
        with self.assertRaises(verifiers.VerifierError):
            check({"type": "vibes"}, text("a"))
        with self.assertRaises(verifiers.VerifierError):
            check({"type": "contains"}, text("a"))

    def test_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "v.py"
            script.write_text("import json, sys\np = json.load(sys.stdin)\n"
                              "sys.exit(None if 'ok' in p['reply']['content'] else 'reply lacks ok')\n")
            spec = {"type": "command", "argv": [sys.executable, "v.py"]}
            self.assertIsNone(check(spec, text("ok then"), {"cwd": tmp}))
            self.assertEqual(check(spec, text("nope"), {"cwd": tmp}), "reply lacks ok")
            slow = {"type": "command", "argv": [sys.executable, "-c", "import time; time.sleep(5)"], "timeout": 0.2}
            self.assertIn("ran past", check(slow, text("x"), {"cwd": tmp}))
            with self.assertRaises(verifiers.VerifierError):
                check({"type": "command", "argv": ["/no/such/program"]}, text("x"), {"cwd": tmp})


class FromReference(unittest.TestCase):
    def test_tool_calls_are_pinned_exactly(self):
        specs = verifiers.from_reference(call("escalate", queue="refunds"))
        self.assertFalse(verifiers.weak_only(specs))
        self.assertEqual(verifiers.failures(specs, call("escalate", queue="refunds")), [])
        self.assertTrue(verifiers.failures(specs, call("escalate", queue="refunds", extra=1)))
        self.assertTrue(verifiers.failures(specs, text("I escalated it")))

    def test_json_is_pinned_and_text_is_weak(self):
        self.assertEqual(verifiers.from_reference(text('{"a": 1}')), [{"type": "json", "equals": {"a": 1}}])
        specs = verifiers.from_reference(text("Your order ships Monday."))
        self.assertTrue(verifiers.weak_only(specs))
        self.assertFalse(verifiers.weak_only([{"type": "no_tool_call"}]))
        self.assertFalse(verifiers.weak_only(verifiers.from_reference(text('{"a": 1}'))))


if __name__ == "__main__":
    unittest.main()
