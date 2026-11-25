import json
from zipfile import ZipFile

from tests.support import ExampleCase


class Export(ExampleCase):
    def test_certificate_and_portable_adapter(self):
        self.assertEqual(self.graft("rebase", "--target", "new-base")[0], 0)
        code, _, err = self.graft("export", "--out", "release.zip")
        self.assertEqual(code, 0, err)
        with ZipFile(self.root / "release.zip") as z:
            self.assertEqual(set(z.namelist()), {"adapter/learned.json", "manifest.json", "certificate.json"})
            self.assertEqual(json.loads(z.read("manifest.json"))["graft"]["adapter"], "adapter")
        self.assertEqual(self.graft("export", "--out", "release.zip")[0], 1)
        self.assertEqual(self.graft("export", "--format", "vllm", "--base-model", "demo", "--out", "bad.zip")[0], 1)
        next((self.root / ".graft").rglob("learned.json")).write_text("{}")
        self.assertEqual(self.graft("export", "--out", "bad.zip")[0], 1)
        self.assertFalse((self.root / "bad.zip").exists())

    def test_fast_forward_and_tampered_certificate(self):
        self.assertEqual(self.graft("rebase", "--target", "prod")[0], 0)
        self.assertEqual(self.graft("export", "--format", "vllm", "--base-model", "base", "--out", "ship.zip")[0], 0)
        with ZipFile(self.root / "ship.zip") as z:
            self.assertEqual(json.loads(z.read("manifest.json"))["serve_argv"], ["vllm", "serve", "base"])
        code, _, err = self.graft("export", "--format", "goethe", "--base-model", "base", "--out", "goethe.zip")
        self.assertEqual(code, 0, err)
        with ZipFile(self.root / "goethe.zip") as z:
            self.assertEqual(json.loads(z.read("manifest.json"))["serve_argv"], ["goethe", "load", "base"])
        cert = next((self.root / ".graft").rglob("certificate.json"))
        cert.write_text(cert.read_text().replace('"passed": true', '"passed": false', 1))
        self.assertEqual(self.graft("export", "--out", "bad.zip")[0], 1)

    def test_chat_template_arguments_travel_with_the_bundle(self):
        self.edit(lambda text: text.replace('model = "acme/support-v3"',
                                            'model = "acme/support-v3"\ntemplate_args = { enable_thinking = false }'))
        self.assertEqual(self.graft("rebase", "--target", "prod")[0], 0)
        code, _, err = self.graft("export", "--format", "vllm", "--base-model", "acme/support-v3", "--out", "t.zip")
        self.assertEqual(code, 0, err)
        self.assertNotIn("note:", err, "the base named is the certified one")
        with ZipFile(self.root / "t.zip") as z:
            manifest = json.loads(z.read("manifest.json"))
        self.assertEqual(manifest["template_args"], {"enable_thinking": False})
        self.assertEqual(manifest["request_extra_body"], {"chat_template_kwargs": {"enable_thinking": False}})
        self.assertEqual(manifest["serve_argv"][3:], ["--default-chat-template-kwargs", '{"enable_thinking": false}'])
        self.assertEqual(manifest["certified_base"]["model"], "acme/support-v3")
        code, _, err = self.graft("export", "--format", "vllm", "--base-model", "other/model", "--out", "u.zip")
        self.assertEqual(code, 0)
        self.assertIn("not on 'other/model'", err)


FAKE_CONVERT = """import argparse, json, pathlib
p = argparse.ArgumentParser(); p.add_argument("model"); p.add_argument("--outfile"); p.add_argument("--outtype")
a = p.parse_args()
assert (pathlib.Path(a.model) / "config.json").is_file()
pathlib.Path(a.outfile).write_text(json.dumps({"from": a.model, "type": a.outtype}))
"""
FAKE_QUANTIZE = """#!/usr/bin/env python3
import json, pathlib, sys
src, dst, kind = sys.argv[1:4]
body = json.loads(pathlib.Path(src).read_text()); body["quantized"] = kind
pathlib.Path(dst).write_text(json.dumps(body))
"""


class GgufExport(ExampleCase):
    """--format gguf with stand-ins for llama.cpp's converter and quantizer (the real ones run on Modal)."""

    def setUp(self):
        super().setUp()
        self.llama = self.root / "llama.cpp"
        (self.llama / "build" / "bin").mkdir(parents=True)
        (self.llama / "convert_hf_to_gguf.py").write_text(FAKE_CONVERT)
        quantize = self.llama / "build" / "bin" / "llama-quantize"
        quantize.write_text(FAKE_QUANTIZE)
        quantize.chmod(0o755)
        (self.root / "weights").mkdir()
        (self.root / "weights" / "config.json").write_text("{}")
        self.edit(lambda text: text.replace('model = "acme/support-v3"',
                                            'model = "acme/support-v3"\ntemplate_args = { enable_thinking = false }'))
        self.assertEqual(self.graft("rebase", "--target", "prod")[0], 0)

    def export(self, *extra):
        return self.graft("export", "--format", "gguf", "--base-model", "weights", "--llama-cpp", str(self.llama),
                          *extra)

    def test_q8_0_is_converted_directly(self):
        code, _, err = self.export("--out", "q8.zip")
        self.assertEqual(code, 0, err)
        with ZipFile(self.root / "q8.zip") as z:
            manifest = json.loads(z.read("manifest.json"))
            body = json.loads(z.read("model-Q8_0.gguf"))
        self.assertEqual(body["type"], "q8_0")
        self.assertEqual(manifest["serve_argv"], ["llama-server", "-m", "model-Q8_0.gguf", "--jinja", "--alias",
                                                  "grafted", "--chat-template-kwargs", '{"enable_thinking": false}'])
        self.assertFalse(manifest["served_as_certified"])
        self.assertIn("--against", manifest["check"])
        self.assertEqual(len(manifest["gguf"]["sha256"]), 64)

    def test_k_quants_go_through_llama_quantize(self):
        code, _, err = self.export("--quant", "Q4_K_M", "--out", "q4.zip")
        self.assertEqual(code, 0, err)
        with ZipFile(self.root / "q4.zip") as z:
            self.assertEqual(sorted(z.namelist()), ["certificate.json", "manifest.json", "model-Q4_K_M.gguf"])
            body = json.loads(z.read("model-Q4_K_M.gguf"))
        self.assertEqual((body["type"], body["quantized"]), ("bf16", "Q4_K_M"))
        # the bundle can still be checked against, as the manifest tells the user to
        self.assertEqual(self.graft("verify", "prod", "--against", "q4.zip")[0], 0)

    def test_missing_tools_and_bad_types_are_clear_errors(self):
        code, _, err = self.graft("export", "--format", "gguf", "--base-model", "weights", "--out", "x.zip")
        self.assertEqual(code, 1)
        self.assertIn("--llama-cpp", err)
        code, _, err = self.export("--quant", "Q4;rm", "--out", "y.zip")
        self.assertEqual(code, 1)
        self.assertIn("bad quantization type", err)
        self.assertFalse((self.root / "x.zip").exists() or (self.root / "y.zip").exists())
