import json
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler
from pathlib import Path

from graft import runners
from graft.genome import Unit
from tests.support import mock_module

Server = mock_module().Server   # an HTTP server that skips the slow host-name lookup on macOS


class Scripted(BaseHTTPRequestHandler):
    """Answers with the queued (status, body) pairs in order and records what it was sent."""
    queue: list = []
    seen: list = []

    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).seen.append((self.path, dict(self.headers), body))
        status, answer = type(self).queue.pop(0)
        data = json.dumps(answer).encode()
        self.send_response(status)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def completion(message):
    return {"choices": [{"message": message, "finish_reason": "stop"}]}


class OpenAIRunner(unittest.TestCase):
    def setUp(self):
        Scripted.queue, Scripted.seen = [], []
        self.server = Server(("127.0.0.1", 0), Scripted)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.endpoint = f"http://127.0.0.1:{self.server.server_address[1]}/v1/"

    def runner(self, **kwargs):
        return runners.OpenAI("m", self.endpoint, "org/model", **kwargs)

    def test_request_and_reply_shape(self):
        Scripted.queue = [(200, completion({"role": "assistant", "content": "<think>hm</think>Hello.",
                                            "tool_calls": [{"id": "1", "type": "function",
                                                            "function": {"name": "f", "arguments": "{\"a\": 1}"}}]}))]
        runner = self.runner(api_key="sk-test", params={"max_tokens": 64, "temperature": 0.9})
        reply = runner.complete([{"role": "user", "content": "hi"}], tools=[{"type": "function"}], seed=7)
        self.assertEqual(reply, {"content": "Hello.", "tool_calls": [{"name": "f", "arguments": {"a": 1}}],
                                 "finish_reason": "stop"})
        path, headers, body = Scripted.seen[0]
        self.assertEqual(path, "/v1/chat/completions")
        self.assertEqual(headers["Authorization"], "Bearer sk-test")
        self.assertEqual((body["model"], body["max_tokens"], body["temperature"], body["seed"], body["stream"]),
                         ("org/model", 64, 0.0, 7, False))
        self.assertEqual(body["tools"], [{"type": "function"}])

    def test_template_args_are_sent_as_chat_template_kwargs(self):
        Scripted.queue = [(200, completion({"content": "a"})), (200, completion({"content": "b"}))]
        self.runner().complete([{"role": "user", "content": "hi"}])
        self.assertNotIn("chat_template_kwargs", Scripted.seen[0][2], "not sent unless configured")
        runner = runners.build("m", {"endpoint": self.endpoint, "model": "org/model",
                                     "template_args": {"enable_thinking": False}}, Path.cwd())
        runner.complete([{"role": "user", "content": "hi"}])
        self.assertEqual(Scripted.seen[1][2]["chat_template_kwargs"], {"enable_thinking": False})

    def test_retries_server_errors_but_not_client_errors(self):
        Scripted.queue = [(503, {"error": "busy"}), (200, completion({"content": "ok"}))]
        self.assertEqual(self.runner().complete([{"role": "user", "content": "hi"}])["content"], "ok")
        self.assertEqual(len(Scripted.seen), 2)
        Scripted.queue = [(404, {"error": {"message": "no such model"}})]
        with self.assertRaisesRegex(runners.RunnerError, "HTTP 404.*no such model"):
            self.runner().complete([{"role": "user", "content": "hi"}])
        self.assertEqual(len(Scripted.seen), 3)

    def test_a_reply_that_is_not_a_completion(self):
        Scripted.queue = [(200, {"unexpected": True})]
        with self.assertRaisesRegex(runners.RunnerError, "not a chat completion"):
            self.runner().complete([{"role": "user", "content": "hi"}])

    def test_nothing_listening(self):
        runner = runners.OpenAI("m", "http://127.0.0.1:9/v1", "x", retries=0, timeout=2)
        with self.assertRaises(runners.RunnerError):
            runner.complete([{"role": "user", "content": "hi"}])


class Prompt(unittest.TestCase):
    def test_graft_goes_into_the_system_message(self):
        graft = {"kind": "prompt", "system": "RULES"}
        user = {"role": "user", "content": "hi"}
        self.assertEqual(runners.apply_prompt([user], graft), [{"role": "system", "content": "RULES"}, user])
        merged = runners.apply_prompt([{"role": "system", "content": "Be brief."}, user], graft)
        self.assertEqual(merged, [{"role": "system", "content": "Be brief.\n\nRULES"}, user])
        self.assertEqual(runners.apply_prompt([user], {"kind": "none"}), [user])
        self.assertEqual(runners.apply_prompt([user], None), [user])

    def test_rendering(self):
        reply = {"content": "Sure.", "tool_calls": [{"name": "f", "arguments": {"b": 1, "a": 2}}]}
        self.assertEqual(runners.render_reply(reply), 'Sure.\n[calls tool f with {"a": 2, "b": 1}]')
        text = runners.render_request([{"role": "system", "content": "s"}, {"role": "user", "content": "q"},
                                       {"role": "assistant", **reply}, {"role": "tool", "content": "r"}])
        self.assertEqual(text.splitlines()[0], "user: q")
        self.assertNotIn("system", text)
        self.assertTrue(text.endswith("tool: r"))


class Others(unittest.TestCase):
    def test_replay(self):
        unit = Unit(messages=[{"role": "user", "content": "q"}], tools=None, verify=[], reference={"content": " a "})
        self.assertEqual(runners.Replay().reply(unit), {"content": "a", "tool_calls": []})
        unit.reference = None
        with self.assertRaises(runners.RunnerError):
            runners.Replay().reply(unit)

    def test_command(self):
        echo = ("import json, sys; r = json.load(sys.stdin); "
                "print(json.dumps({'content': r['messages'][-1]['content'].upper()}))")
        runner = runners.build("c", {"argv": [sys.executable, "-c", echo]}, Path.cwd())
        self.assertEqual(runner.complete([{"role": "user", "content": "hi"}])["content"], "HI")
        broken = runners.build("c", {"argv": [sys.executable, "-c", "import sys; sys.exit('boom')"]}, Path.cwd())
        with self.assertRaisesRegex(runners.RunnerError, "boom"):
            broken.complete([])

    def test_generated_text_with_tool_calls(self):
        text = 'Let me check.\n<tool_call>\n{"name": "lookup", "arguments": {"q": "x"}}\n</tool_call>'
        self.assertEqual(runners.parse_generated(text),
                         {"content": "Let me check.", "tool_calls": [{"name": "lookup", "arguments": {"q": "x"}}]})
        self.assertEqual(runners.parse_generated("<tool_call>{broken</tool_call>")["tool_calls"], [])

    def test_hf_messages_turn_argument_text_into_objects(self):
        message = {"role": "assistant", "content": None,
                   "tool_calls": [{"id": "1", "type": "function",
                                   "function": {"name": "f", "arguments": "{\"a\": 1}"}}]}
        self.assertEqual(runners.hf_messages([message])[0]["tool_calls"],
                         [{"type": "function", "function": {"name": "f", "arguments": {"a": 1}}}])

    def test_build(self):
        root = Path.cwd()
        self.assertIsInstance(runners.build("m", {"endpoint": "http://x/v1", "model": "a"}, root), runners.OpenAI)
        self.assertIsInstance(runners.build("m", {"path": "/weights"}, root), runners.HF)
        self.assertIsInstance(runners.build("m", {"runner": "replay"}, root), runners.Replay)
        adapted = runners.build("m", {"endpoint": "http://x/v1", "model": "a", "adapter_model": "a-graft"}, root,
                                adapter="/adapters/1")
        self.assertEqual(adapted.model, "a-graft")
        keyed = runners.build("m", {"endpoint": "http://x/v1", "model": "a", "api_key_env": "K"}, root, env={"K": "s"})
        self.assertEqual(keyed.api_key, "s")
        for spec, why in [({}, "needs an endpoint"), ({"endpoint": "http://x/v1"}, "both endpoint and model"),
                          ({"endpoint": "http://x/v1", "model": "a", "api_key_env": "K"}, r"\$K, which is not set")]:
            with self.assertRaisesRegex(runners.RunnerError, why):
                runners.build("m", spec, root, env={})
        with self.assertRaisesRegex(runners.RunnerError, "adapter_model"):
            runners.build("m", {"endpoint": "http://x/v1", "model": "a"}, root, adapter="/adapters/1")
        for spec in ({"path": "/weights"}, {"endpoint": "http://x/v1", "model": "a"}):
            with self.assertRaisesRegex(runners.RunnerError, r"\[models.m\] template_args must be a table"):
                runners.build("m", {**spec, "template_args": "enable_thinking=false"}, root)


if __name__ == "__main__":
    unittest.main()
