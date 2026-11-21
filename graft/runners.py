"""Model runners: how graft asks a model for a reply.

openai    any OpenAI-compatible chat endpoint (Goethe, vLLM, SGLang, llama.cpp, a hosted API)
hf        Hugging Face weights loaded in this process, with an optional LoRA adapter
command   your own program: one JSON request on stdin, one JSON reply on stdout
replay    the recorded reference replies, used to check a genome against itself
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from graft import verifiers
from graft.genome import Unit


class RunnerError(Exception):
    pass


def render_reply(reply: dict[str, Any]) -> str:
    """A reply as plain text, for examples inside a prompt."""
    parts = []
    text = verifiers.text_of(reply)
    if text:
        parts.append(text)
    for call in verifiers.calls_of(reply):
        parts.append(f"[calls tool {call['name']} with {json.dumps(call['arguments'], sort_keys=True)}]")
    return "\n".join(parts)


def render_request(messages: list[dict[str, Any]]) -> str:
    lines = []
    for message in messages:
        role = message.get("role")
        if role == "system":
            continue
        body = render_reply(message) if role == "assistant" else verifiers.text_of(message)
        lines.append(f"{role}: {body}")
    return "\n".join(lines)


def apply_prompt(messages: list[dict[str, Any]], graft: dict[str, Any] | None) -> list[dict[str, Any]]:
    """The request with a prompt graft in its system message. The client's own messages keep their order."""
    block = (graft or {}).get("system")
    if not block:
        return messages
    if messages and messages[0].get("role") == "system":
        first = dict(messages[0])
        first["content"] = f"{verifiers.text_of(first)}\n\n{block}".strip()
        return [first, *messages[1:]]
    return [{"role": "system", "content": block}, *messages]


class Runner:
    name = "model"
    concurrency = 1

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None,
                 temperature: float = 0.0, seed: int | None = None) -> dict[str, Any]:
        raise NotImplementedError

    def reply(self, unit: Unit, graft: dict[str, Any] | None = None,
              temperature: float = 0.0, seed: int | None = None) -> dict[str, Any]:
        return self.complete(apply_prompt(unit.messages, graft), unit.tools, temperature, seed)

    def fingerprint(self) -> dict[str, Any]:
        return {"runner": type(self).__name__}

    def close(self) -> None:
        pass


class Replay(Runner):
    """Answers with each case's recorded reference. A unit without one cannot be replayed."""
    name = "references"
    concurrency = 16

    def reply(self, unit: Unit, graft: dict[str, Any] | None = None,
              temperature: float = 0.0, seed: int | None = None) -> dict[str, Any]:
        if unit.reference is None:
            raise RunnerError("no reference reply recorded for this case")
        return verifiers.normalize(unit.reference)

    def fingerprint(self) -> dict[str, Any]:
        return {"runner": "replay"}


class OpenAI(Runner):
    def __init__(self, name: str, endpoint: str, model: str, api_key: str | None = None,
                 params: dict[str, Any] | None = None, timeout: float = 300.0, concurrency: int = 8,
                 retries: int = 4, template_args: dict[str, Any] | None = None):
        self.name, self.endpoint, self.model = name, endpoint.rstrip("/"), model
        self.api_key, self.params, self.timeout = api_key, params or {}, timeout
        self.concurrency, self.retries = concurrency, retries
        self.template_args = template_args or {}

    def _request(self, path: str, body: dict[str, Any] | None = None) -> Any:
        headers = {"Content-Type": "application/json", "User-Agent": "graft"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        data = json.dumps(body).encode() if body is not None else None
        last = "no attempt made"
        for attempt in range(self.retries + 1):
            request = urllib.request.Request(self.endpoint + path, data=data, headers=headers)
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as err:
                detail = err.read().decode("utf-8", "replace")[:300]
                last = f"HTTP {err.code} from {self.endpoint}{path}: {detail}"
                if err.code not in (408, 409, 429) and err.code < 500:
                    break
            except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError) as err:
                last = f"{self.endpoint}{path}: {getattr(err, 'reason', err)}"
            if attempt < self.retries:
                time.sleep(min(0.5 * 2 ** attempt, 8.0))
        raise RunnerError(last)

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None,
                 temperature: float = 0.0, seed: int | None = None) -> dict[str, Any]:
        body: dict[str, Any] = {**self.params, "model": self.model, "messages": messages,
                                "temperature": temperature, "stream": False}
        if tools:
            body["tools"] = tools
        if seed is not None:
            body["seed"] = seed
        if self.template_args:   # vLLM and SGLang pass these to the chat template
            body["chat_template_kwargs"] = {**(body.get("chat_template_kwargs") or {}), **self.template_args}
        answer = self._request("/chat/completions", body)
        try:
            choice = answer["choices"][0]
            reply = verifiers.normalize(choice["message"])
        except (KeyError, IndexError, TypeError, AttributeError) as err:
            raise RunnerError(f"{self.endpoint}: reply is not a chat completion") from err
        reply["finish_reason"] = choice.get("finish_reason")
        return reply

    def fingerprint(self) -> dict[str, Any]:
        info: dict[str, Any] = {"runner": "openai", "endpoint": self.endpoint, "model": self.model}
        if self.template_args:
            info["template_args"] = self.template_args
        try:
            listed = self._request("/models")
            entry = next((m for m in listed.get("data", []) if m.get("id") == self.model), None)
            if entry:
                info["served"] = {k: entry[k] for k in ("id", "created", "owned_by", "root", "parent", "fingerprint")
                                  if k in entry}
        except (RunnerError, AttributeError):
            pass
        return info


class Command(Runner):
    def __init__(self, name: str, argv: list[str], cwd: Path, timeout: float = 600.0, concurrency: int = 1):
        self.name, self.argv, self.cwd, self.timeout, self.concurrency = name, argv, cwd, timeout, concurrency

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None,
                 temperature: float = 0.0, seed: int | None = None) -> dict[str, Any]:
        request = {"messages": messages, "tools": tools, "temperature": temperature, "seed": seed}
        try:
            done = subprocess.run(self.argv, input=json.dumps(request), text=True, capture_output=True,
                                  timeout=self.timeout, cwd=self.cwd, check=False)
        except (OSError, subprocess.TimeoutExpired) as err:
            raise RunnerError(f"{self.argv[0]}: {err}") from err
        if done.returncode != 0:
            raise RunnerError(f"{self.argv[0]} exited {done.returncode}: {done.stderr.strip()[-300:]}")
        try:
            return verifiers.normalize(json.loads(done.stdout))
        except (json.JSONDecodeError, AttributeError) as err:
            raise RunnerError(f"{self.argv[0]} did not print a JSON reply") from err

    def fingerprint(self) -> dict[str, Any]:
        return {"runner": "command", "argv": self.argv}


TOOL_CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def parse_generated(text: str) -> dict[str, Any]:
    """Generated text as a reply: <tool_call>{"name", "arguments"}</tool_call> blocks become tool calls."""
    calls = []
    for block in TOOL_CALL.findall(text):
        try:
            call = json.loads(block)
        except json.JSONDecodeError:
            continue
        if isinstance(call, dict) and "name" in call:
            calls.append({"name": call["name"], "arguments": call.get("arguments", call.get("parameters", {}))})
    return verifiers.normalize({"content": TOOL_CALL.sub("", text), "tool_calls": calls})


class HF(Runner):
    """Weights loaded in this process. Needs `pip install 'graft-sh[train]'`."""

    def __init__(self, name: str, path: str, adapter: str | None = None, max_new_tokens: int = 512,
                 device: str | None = None, template_args: dict[str, Any] | None = None):
        self.name, self.path, self.adapter = name, path, adapter
        self.max_new_tokens, self.device = max_new_tokens, device
        self.template_args = template_args or {}
        self._lock = threading.Lock()
        self._loaded: tuple[Any, Any, Any] | None = None

    def _load(self) -> tuple[Any, Any, Any]:
        if self._loaded is None:
            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer
            except ImportError as err:
                raise RunnerError("the hf runner needs torch and transformers: "
                                  "pip install 'graft-sh[train]'") from err
            tokenizer = AutoTokenizer.from_pretrained(self.path)
            device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            dtype = {dtype_argument(): torch.bfloat16 if device == "cuda" else torch.float32}
            model = AutoModelForCausalLM.from_pretrained(self.path, **dtype).to(device)
            if self.adapter:
                if (Path(self.adapter) / "adapter_config.json").exists():
                    from peft import PeftModel
                    model = PeftModel.from_pretrained(model, self.adapter)
                else:   # a full fine-tune saved as a model folder
                    model = AutoModelForCausalLM.from_pretrained(self.adapter, **dtype).to(device)
            model.eval()
            self._loaded = (torch, tokenizer, model)
        return self._loaded

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None,
                 temperature: float = 0.0, seed: int | None = None) -> dict[str, Any]:
        with self._lock:
            torch, tokenizer, model = self._load()
            if seed is not None:
                torch.manual_seed(seed)
            prompt = tokenizer.apply_chat_template(hf_messages(messages), tools=tools or None,
                                                   add_generation_prompt=True, tokenize=False,
                                                   **self.template_args)
            inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
            sampling = {"do_sample": True, "temperature": temperature} if temperature > 0 else {"do_sample": False}
            with torch.no_grad():
                output = model.generate(**inputs, max_new_tokens=self.max_new_tokens,
                                        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id, **sampling)
            text = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return parse_generated(text)

    def fingerprint(self) -> dict[str, Any]:
        info: dict[str, Any] = {"runner": "hf", "path": self.path}
        config = Path(self.path) / "config.json"
        if not config.is_file():   # a Hugging Face id: the cached snapshot names the exact revision loaded
            try:
                from huggingface_hub import try_to_load_from_cache
                cached = try_to_load_from_cache(self.path, "config.json")
            except Exception:   # no huggingface_hub, or not a valid repo id: the path alone names the model
                cached = None
            if isinstance(cached, str):
                config = Path(cached)
                info["revision"] = config.parent.name
        if config.is_file():
            info["config_sha256"] = hashlib.sha256(config.read_bytes()).hexdigest()
        if self.adapter:
            info["adapter"] = self.adapter
        if self.template_args:   # another rendering of the same weights answers differently
            info["template_args"] = self.template_args
        return info

    def close(self) -> None:
        self._loaded = None


def dtype_argument() -> str:
    """transformers 5 renamed from_pretrained's torch_dtype to dtype."""
    import transformers
    return "dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"


def hf_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Messages as Hugging Face chat templates expect them: tool-call arguments as objects, not JSON text."""
    out = []
    for message in messages:
        if message.get("role") == "assistant" and message.get("tool_calls"):
            calls = [{"type": "function", "function": call} for call in verifiers.calls_of(message)]
            message = {**message, "content": verifiers.text_of(message), "tool_calls": calls}
        out.append(message)
    return out


def template_args(name: str, spec: dict[str, Any]) -> dict[str, Any] | None:
    """A model's template_args: keyword arguments for its chat template, such as { enable_thinking = false }."""
    value = spec.get("template_args")
    if value is None:
        return None
    if not isinstance(value, dict):
        raise RunnerError(f"[models.{name}] template_args must be a table of chat template arguments, "
                          f"for example {{ enable_thinking = false }}; got {value!r}")
    return dict(value)


def build(name: str, spec: dict[str, Any], root: Path, adapter: str | None = None,
          env: dict[str, str] | None = None) -> Runner:
    """The runner for a [models.<name>] table. With `adapter`, the same model with that adapter applied."""
    env = os.environ if env is None else env
    kind = spec.get("runner") or ("openai" if "endpoint" in spec else "hf" if "path" in spec
                                  else "command" if "argv" in spec else None)
    if kind == "replay":
        return Replay()
    if kind == "openai":
        if "endpoint" not in spec or "model" not in spec:
            raise RunnerError(f"[models.{name}] needs both endpoint and model")
        key = env.get(spec["api_key_env"]) if spec.get("api_key_env") else None
        if spec.get("api_key_env") and not key:
            raise RunnerError(f"[models.{name}] reads its API key from ${spec['api_key_env']}, which is not set")
        model = spec["model"]
        if adapter:
            if not spec.get("adapter_model"):
                raise RunnerError(f"[models.{name}] needs adapter_model: the name its endpoint serves "
                                  f"the trained adapter under")
            model = spec["adapter_model"]
        return OpenAI(name, spec["endpoint"], model, api_key=key, params=spec.get("params"),
                      timeout=float(spec.get("timeout", 300)), concurrency=int(spec.get("concurrency", 8)),
                      template_args=template_args(name, spec))
    if kind == "hf":
        if "path" not in spec:
            raise RunnerError(f"[models.{name}] needs path: a local folder or a Hugging Face id")
        local = root / str(spec["path"])       # a folder next to graft.toml, else a Hugging Face id
        return HF(name, str(local if local.exists() else spec["path"]), adapter=adapter,
                  max_new_tokens=int(spec.get("max_new_tokens", 512)),
                  device=spec.get("device"), template_args=template_args(name, spec))
    if kind == "command":
        if adapter:
            raise RunnerError(f"[models.{name}] is a command runner; it cannot load a trained adapter")
        return Command(name, [str(a) for a in spec["argv"]], root, timeout=float(spec.get("timeout", 600)),
                       concurrency=int(spec.get("concurrency", 1)))
    raise RunnerError(f"[models.{name}] needs an endpoint (with model), a path, or an argv")
