"""The fresh-server runner must not report a skipped or missing CPU trainer as exercised."""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

CHECKS = Path(__file__).resolve().parents[1] / "scripts" / "server_checks.py"
PASS = "import unittest\nclass T(unittest.TestCase):\n    def test_ok(self): pass\n"
SKIP = PASS + "    @unittest.skip('fixture')\n    def test_optional(self): pass\n"


class ServerChecks(unittest.TestCase):
    def test_train_requires_discovered_successes_and_no_skips(self):
        with tempfile.TemporaryDirectory() as tmp:
            tests = Path(tmp) / "tests"
            tests.mkdir()
            (tests / "__init__.py").touch()
            (tests / "test_other.py").write_text(PASS)

            def run(*flags):
                result = subprocess.run([sys.executable, str(CHECKS), "tests", *flags], cwd=tmp,
                                        text=True, capture_output=True, check=False)
                return result.returncode, result.stderr

            self.assertNotEqual(run("--train")[0], 0, "a missing trainer suite must fail")
            trainer = tests / "test_train_hf.py"
            trainer.write_text(SKIP)
            self.assertEqual(run()[0], 0, "the lightweight run allows optional tests to skip")
            self.assertNotEqual(run("--train")[0], 0, "even one skipped trainer test must fail")
            trainer.write_text(PASS)
            (tests / "test_other.py").write_text(SKIP)
            self.assertNotEqual(run("--train")[0], 0, "the full training run requires zero suite skips")
            (tests / "test_other.py").write_text(PASS)
            code, output = run("--train")
            self.assertEqual(code, 0, output)
