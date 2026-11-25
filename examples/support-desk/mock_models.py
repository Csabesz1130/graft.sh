#!/usr/bin/env python3
"""Two scripted stand-ins for models, behind one OpenAI-compatible address. A simulation, not inference.

    acme/support-v3     the "old" model: follows every house rule, weaker at arithmetic
    open/new-base-30b   the "new base": better at arithmetic, knows none of the house rules by itself.
                        It picks some up from a system prompt, one only from worked examples,
                        and one (the order of tool calls) only from training.
    new-base-graft      the new base with an adapter that mock_trainer.py "trained", once load_adapter.py
                        has loaded it

It exists so `graft rebase` can be tried end to end in seconds, with no GPU.

    python3 mock_models.py [--port 8811]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import socketserver
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

PROD, BASE, GRAFT = "acme/support-v3", "open/new-base-30b", "new-base-graft"
RULES = {"refund-limit", "no-legal-advice", "ticket-json", "card-numbers", "account-closure"}
CARD = re.compile(r"\b(?:\d[ -]?){15}\d\b")


def _bucket(*parts: Any) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


def _text(message: dict[str, Any]) -> str:
    return message.get("content") or ""


def _call(name: str, **arguments: Any) -> dict[str, Any]:
    return {"role": "assistant", "content": None,
            "tool_calls": [{"id": "call_0", "type": "function",
                            "function": {"name": name, "arguments": json.dumps(arguments)}}]}


def _say(text: str) -> dict[str, Any]:
    return {"role": "assistant", "content": text}


def skills(model: str, messages: list[dict[str, Any]], temperature: float, seed: Any,
           adapters: dict[str, set[str]]) -> set[str]:
    """Which house rules this "model" follows for this request."""
    if model == PROD:
        return set(RULES)
    known = {"ticket-json"}
    if model == GRAFT:
        known |= adapters.get(GRAFT, set())
    system = _text(messages[0]) if messages and messages[0].get("role") == "system" else ""
    if "$500" in system:
        known.add("refund-limit")
    if "legal advice" in system:
        known.add("no-legal-advice")
    if "Example (rule" in system and "ending in" in system:
        known.add("card-numbers")              # stating the rule is not enough; it needs to see an example
    user = next((_text(m) for m in reversed(messages) if m.get("role") == "user"), "")
    if temperature > 0 and _bucket(seed, user) % 2 == 0:
        known.add("account-closure")           # when sampling, it sometimes gets the order right
    return known


def reply(model: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None,
          temperature: float = 0.0, seed: Any = None, adapters: dict[str, set[str]] | None = None) -> dict[str, Any]:
    known = skills(model, messages, temperature, seed, adapters or {})
    names = {tool.get("function", {}).get("name") for tool in tools or []}
    last_user = max((i for i, m in enumerate(messages) if m.get("role") == "user"), default=-1)
    user = _text(messages[last_user]) if last_user >= 0 else ""
    results = [m for m in messages[last_user + 1:] if m.get("role") == "tool"]

    if "close_account" in names:
        customer = (re.search(r"C-\d+", user) or re.search(r"C-\d+", "C-0")).group(0)
        if results:
            verified = json.loads(_text(results[-1]) or "{}").get("verified")
            if verified or "account-closure" not in known:
                return _call("close_account", customer_id=customer)
            return _say("I couldn't verify your identity, so I can't close the account. "
                        "Please contact us from the email address on file.")
        if "account-closure" in known:
            return _call("verify_identity", customer_id=customer)
        return _call("close_account", customer_id=customer)

    amount = re.search(r"\$([\d,]+(?:\.\d+)?)", user)
    if "issue_refund" in names and amount:
        value = float(amount.group(1).replace(",", ""))
        order = (re.search(r"#(\d+)", user) or re.search(r"(\d+)", "0")).group(1)
        if value > 500 and "refund-limit" in known:
            return _call("escalate", queue="refunds", reason=f"refund of ${value:g} on order #{order}")
        return _call("issue_refund", order_id=order, amount=value)

    card = CARD.search(user)
    if card:
        digits = re.sub(r"\D", "", card.group(0))
        if "card-numbers" in known:
            return _say(f"Thanks. I've noted the card ending in {digits[-4:]} for this order.")
        return _say(f"Thanks. I've noted card {card.group(0)} for this order.")

    if re.search(r"\b(sue|lawsuit|legal|lawyer)\b", user, re.IGNORECASE):
        if "no-legal-advice" in known:
            return _say("I'm sorry about the trouble. I can't give legal advice, but I can open a formal "
                        "complaint for you and have a manager call you back.")
        return _say("You could sue under consumer protection law; small claims court is usually the fastest route.")

    if user.lower().startswith("file a ticket"):
        summary = user.split(":", 1)[-1].strip()[:80]
        urgent = re.search(r"\b(outage|down|cannot log in|data loss)\b", user, re.IGNORECASE)
        return _say(json.dumps({"priority": "high" if urgent else "normal", "summary": summary}))

    sum_of = re.search(r"What is (\d+) \* (\d+)", user)
    if sum_of:
        product = int(sum_of.group(1)) * int(sum_of.group(2))
        right = _bucket("arithmetic", user) % 10 < (6 if model == PROD else 9)
        return _say(f"{product if right else product + 10}")

    return _say("Happy to help. Could you tell me a bit more?")


class Handler(BaseHTTPRequestHandler):
    adapters: dict[str, set[str]] = {}

    def log_message(self, format: str, *args: Any) -> None:
        pass

    def _send(self, status: int, body: dict[str, Any]) -> None:
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        if self.path.rstrip("/") == "/v1/models":
            ids = [PROD, BASE, *self.adapters]
            self._send(200, {"object": "list", "data": [{"id": i, "object": "model", "owned_by": "mock"} for i in ids]})
        else:
            self._send(404, {"error": {"message": "not found"}})

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        if self.path == "/v1/load_lora_adapter":
            learned = json.loads((Path(body["lora_path"]) / "learned.json").read_text())
            self.adapters[body["lora_name"]] = set(learned["rules"])
            self._send(200, {"loaded": body["lora_name"]})
        elif self.path == "/v1/chat/completions":
            model = body.get("model")
            if model not in (PROD, BASE, *self.adapters):
                self._send(404, {"error": {"message": f"model {model!r} is not loaded"}})
                return
            message = reply(model, body.get("messages", []), body.get("tools"), body.get("temperature") or 0.0,
                            body.get("seed"), self.adapters)
            finish = "tool_calls" if message.get("tool_calls") else "stop"
            self._send(200, {"object": "chat.completion", "model": model,
                             "choices": [{"index": 0, "message": message, "finish_reason": finish}]})
        else:
            self._send(404, {"error": {"message": "not found"}})


class Server(ThreadingHTTPServer):
    def server_bind(self) -> None:
        # HTTPServer.server_bind looks up the host's full name, which can take half a minute on a Mac; the name is
        # never used here
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = "127.0.0.1", self.server_address[1]


def serve(port: int = 8811) -> ThreadingHTTPServer:
    Handler.adapters = {}
    return Server(("127.0.0.1", port), Handler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scripted stand-ins for an old and a new model.")
    parser.add_argument("--port", type=int, default=8811)
    port = parser.parse_args().port
    print(f"mock models at http://127.0.0.1:{port}/v1  ({PROD}, {BASE}); Ctrl-C stops")
    try:
        serve(port).serve_forever()
    except KeyboardInterrupt:
        pass
