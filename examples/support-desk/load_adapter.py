#!/usr/bin/env python3
"""Tell the mock server to serve a trained adapter, the way [rebase] after_train would tell a real one.

    python3 load_adapter.py <adapter folder> <name to serve it as> [endpoint]
"""
import json
import sys
import urllib.request

adapter, name = sys.argv[1], sys.argv[2]
endpoint = sys.argv[3] if len(sys.argv) > 3 else "http://127.0.0.1:8811/v1"
body = json.dumps({"lora_name": name, "lora_path": adapter}).encode()
request = urllib.request.Request(f"{endpoint}/load_lora_adapter", data=body,
                                 headers={"Content-Type": "application/json"})
with urllib.request.urlopen(request, timeout=30) as response:
    print(response.read().decode())
