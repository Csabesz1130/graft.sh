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
