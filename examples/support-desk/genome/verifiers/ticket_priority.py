#!/usr/bin/env python3
"""A code verifier: outages and login failures must be filed as high priority, everything else as normal.

graft passes {"reply", "messages", "reference", "case"} as JSON on stdin. Exit 0 means the reply passes;
anything else fails it, and the last line printed is shown as the reason.
"""
import json
import re
import sys

payload = json.load(sys.stdin)
request = next(m["content"] for m in reversed(payload["messages"]) if m["role"] == "user")
ticket = json.loads(payload["reply"]["content"])
urgent = bool(re.search(r"\b(outage|down|cannot log in|data loss)\b", request, re.IGNORECASE))
expected = "high" if urgent else "normal"
if ticket["priority"] != expected:
    sys.exit(f"priority is {ticket['priority']}, but this request is {expected} priority")
