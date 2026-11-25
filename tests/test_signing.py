import base64
import hashlib
import json
import os
import unittest
from pathlib import Path
from zipfile import ZipFile

from graft import signing
from tests.support import ExampleCase

try:
    import cryptography  # noqa: F401
    HAVE_CRYPTO = True
except ImportError:
    HAVE_CRYPTO = False


class Verify(unittest.TestCase):
    """The standard-library verifier, on the RFC 8032 test vectors."""

    VECTORS = [
        ("d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a", "",
         "e5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e065224901555fb8821590a33bacc61e39701cf9b46bd25bf5f05"
         "95bbe24655141438e7a100b"),
        ("3d4017c3e843895a92b70aa74d1b7ebc9c982ccf2ec4968cc0cd55f12af4660c", "72",
         "92a009a9f0d4cab8720e820b5f642540a2b27b5416503f8fb3762223ebdb69da085ac1e43e15996e458f3613d0f11d8c387b2eaeb"
         "4302aeeb00d291612bb0c00"),
    ]

    def test_rfc8032_vectors(self):
        for public, message, signature in self.VECTORS:
            public, message, signature = bytes.fromhex(public), bytes.fromhex(message), bytes.fromhex(signature)
            self.assertTrue(signing.verify(public, message, signature))
            self.assertFalse(signing.verify(public, message + b"x", signature))
            flipped = bytearray(signature)
            flipped[40] ^= 1
            self.assertFalse(signing.verify(public, message, bytes(flipped)))

    def test_rejects_malformed_input(self):
        public, _, signature = (bytes.fromhex(v) for v in self.VECTORS[0])
        self.assertFalse(signing.verify(public[:31], b"", signature))
        self.assertFalse(signing.verify(public, b"", signature[:63]))
        # S at or above the group order is malleable and must be refused
        s = int.from_bytes(signature[32:], "little") + signing._L
        self.assertFalse(signing.verify(public, b"", signature[:32] + s.to_bytes(32, "little")))
        self.assertFalse(signing.verify(b"\xff" * 32, b"", signature), "not a point on the curve")

    @unittest.skipUnless(HAVE_CRYPTO, "needs the cryptography package")
    def test_agrees_with_cryptography(self):
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric import ed25519
        for size in range(0, 60, 7):
            key = ed25519.Ed25519PrivateKey.generate()
            public = key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
            message = os.urandom(size)
            self.assertTrue(signing.verify(public, message, key.sign(message)))


@unittest.skipUnless(HAVE_CRYPTO, "needs the cryptography package (graft-sh[sign])")
class SignAndCheck(ExampleCase):
    def setUp(self):
        super().setUp()
        self.key = self.root / "keys" / "signing-key.pem"

    def keygen(self, key: Path) -> str:
        code, out, err = self.graft("keygen", "--key", str(key))
        self.assertEqual(code, 0, err)
        return str(key.with_suffix(".pub"))

    def test_sign_check_and_export(self):
        public = self.keygen(self.key)
        self.assertEqual(oct(self.key.stat().st_mode & 0o777), "0o600")
        self.assertEqual(self.graft("keygen", "--key", str(self.key))[0], 1, "never overwrites a key")
        self.assertEqual(self.graft("rebase", "--target", "new-base")[0], 0)
        code, out, err = self.graft("sign", "--key", str(self.key))
        self.assertEqual(code, 0, err)
        cert = next((self.root / ".graft").rglob("certificate.json"))
        code, out, _ = self.graft("check-certificate", str(cert), "--key", public)
        self.assertEqual(code, 0, out)
        self.assertIn("(trusted)", out)
        self.assertIn("Signed by ed25519:", self.graft("show")[1])

        # the signature travels in the bundle, and the bundle can be checked as it is
        self.assertEqual(self.graft("export", "--out", "release.zip")[0], 0)
        with ZipFile(self.root / "release.zip") as z:
            self.assertIn(signing.NAME, z.namelist())
            self.assertTrue(json.loads(z.read("manifest.json"))["signed_by"].startswith("ed25519:"))
        self.assertEqual(self.graft("check-certificate", "release.zip", "--key", public)[0], 0)

        # another key is not trusted, and no key means no signature check
        other = self.keygen(self.root / "keys" / "other.pem")
        code, out, _ = self.graft("check-certificate", str(cert), "--key", other)
        self.assertEqual(code, 1)
        self.assertIn("not a trusted key", out)
        code, out, _ = self.graft("check-certificate", str(cert))
        self.assertEqual(code, 0)
        self.assertIn("not checked", out)

    def test_tampering_is_caught(self):
        public = self.keygen(self.key)
        self.assertEqual(self.graft("rebase", "--target", "new-base")[0], 0)
        self.assertEqual(self.graft("sign", "--key", str(self.key))[0], 0)
        cert = next((self.root / ".graft").rglob("certificate.json"))
        sig = cert.with_name(signing.NAME)
        original = cert.read_text()

        # an edited certificate, its digest recomputed to match, is no longer what was signed
        body = json.loads(original)
        body["totals"]["passed"] = body["totals"]["total"] + 1
        body.pop("digest")
        canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        body["digest"] = hashlib.sha256(canonical.encode()).hexdigest()
        cert.write_text(json.dumps(body))
        code, out, _ = self.graft("check-certificate", str(cert), "--key", public)
        self.assertEqual(code, 1)
        self.assertIn("another certificate", out)

        # a forged signature with the right digest
        cert.write_text(original)
        forged = json.loads(sig.read_text())
        forged["signature"] = base64.b64encode(b"\0" * 64).decode()
        sig.write_text(json.dumps(forged))
        code, out, _ = self.graft("check-certificate", str(cert), "--key", public)
        self.assertEqual(code, 1)
        self.assertIn("does not verify", out)

        sig.unlink()
        code, out, _ = self.graft("check-certificate", str(cert), "--key", public)
        self.assertEqual(code, 1)
        self.assertIn("not signed", out)

    def test_a_failed_rebase_is_rejected_unless_allowed(self):
        public = self.keygen(self.key)
        self.assertEqual(self.graft("rebase", "--target", "new-base", "--ladder", "zero-shot")[0], 1)
        self.assertEqual(self.graft("sign", "--key", str(self.key))[0], 0)
        cert = next((self.root / ".graft").rglob("certificate.json"))
        code, out, _ = self.graft("check-certificate", str(cert), "--key", public)
        self.assertEqual(code, 1)
        self.assertIn("did not pass", out)
        self.assertEqual(self.graft("check-certificate", str(cert), "--key", public, "--allow-conflict")[0], 0)
