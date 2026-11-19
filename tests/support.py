"""Shared test fixtures: a copy of the support-desk example with its mock models on a free port."""
from __future__ import annotations

import contextlib
import importlib.util
import io
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.request
from pathlib import Path

from graft import cli

REPO = Path(__file__).resolve().parent.parent
EXAMPLE = REPO / "examples" / "support-desk"


def mock_module():
    spec = importlib.util.spec_from_file_location("mock_models", EXAMPLE / "mock_models.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def graft(*argv: str, cwd: Path) -> tuple[int, str, str]:
    """Run the graft command in `cwd`; returns exit code, stdout, stderr."""
    out, err = io.StringIO(), io.StringIO()
    before = Path.cwd()
    os.chdir(cwd)
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                code = cli.main(list(argv))
            except SystemExit as stop:
                code = int(stop.code or 0)
    finally:
        os.chdir(before)
    return code, out.getvalue(), err.getvalue()


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class ExampleCase(unittest.TestCase):
    """A fresh copy of the example per test, served by the mock models in their own process."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "support-desk"
        shutil.copytree(EXAMPLE, self.root, ignore=shutil.ignore_patterns(".graft", "__pycache__"))
        self.port = free_port()
        self.endpoint = f"http://127.0.0.1:{self.port}/v1"
        self.server = subprocess.Popen([sys.executable, str(self.root / "mock_models.py"), "--port", str(self.port)],
                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.addCleanup(self.server.wait)
        self.addCleanup(self.server.kill)
        for _ in range(200):
            try:
                urllib.request.urlopen(f"{self.endpoint}/models", timeout=1).close()
                break
            except OSError:
                time.sleep(0.02)
        else:
            self.fail("the mock models did not start")
        self.edit(lambda text: text.replace("127.0.0.1:8811", f"127.0.0.1:{self.port}")
                  .replace('"python3"', f'"{sys.executable}"'))
        for rule in (self.root / "genome" / "rules").glob("*.toml"):
            rule.write_text(rule.read_text().replace('"python3"', f'"{sys.executable}"'))

    def edit(self, change) -> None:
        path = self.root / "graft.toml"
        path.write_text(change(path.read_text()))

    def graft(self, *argv: str) -> tuple[int, str, str]:
        return graft(*argv, cwd=self.root)

    def load_adapter(self, adapter: Path, name: str) -> None:
        subprocess.run([sys.executable, str(self.root / "load_adapter.py"), str(adapter), name, self.endpoint],
                       check=True, capture_output=True)
