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
