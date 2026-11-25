"""Ed25519 signatures on certificates, so a pipeline can require that a trusted key signed what it deploys.

The certificate's digest already covers every field; the signature covers the digest. Checking a signature needs
only the standard library (RFC 8032 verification below), so a CI job can enforce it with graft's core install.
Making keys and signing use the `cryptography` package: pip install 'graft-sh[sign]'.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from graft import certificate

FORMAT = "graft-signature/1"
DOMAIN = b"graft certificate signature v1\n"
NAME = "certificate.sig"


class SigningError(Exception):
    pass


def default_key() -> Path:
    if os.environ.get("GRAFT_SIGNING_KEY"):
        return Path(os.environ["GRAFT_SIGNING_KEY"]).expanduser()
    base = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    return base / "graft" / "signing-key.pem"


def key_id(public: bytes) -> str:
    return "ed25519:" + hashlib.sha256(public).hexdigest()[:16]


def _crypto() -> Any:
    try:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric import ed25519
    except ImportError:
        raise SigningError("making keys and signing need the cryptography package: "
                           "pip install 'graft-sh[sign]' (checking signatures does not)") from None
    return ed25519, serialization


def keygen(path: Path) -> bytes:
    """A new private key at path (mode 0600) and its public key at path.pub; returns the public key."""
    ed25519, serialization = _crypto()
    if path.exists() or path.with_suffix(".pub").exists():
        raise SigningError(f"{path} already exists; graft never overwrites a key")
    path.parent.mkdir(parents=True, exist_ok=True)
    private = ed25519.Ed25519PrivateKey.generate()
    pem = private.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                serialization.NoEncryption())
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as out:
        out.write(pem)
    public = private.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    path.with_suffix(".pub").write_text(public_line(public) + "\n", encoding="utf-8")
    return public


def public_line(public: bytes) -> str:
    return f"{key_id(public)} {base64.b64encode(public).decode()}"


def read_public(value: str) -> bytes:
    """A public key given as a .pub file, its line, or the bare base64 key."""
    text = value
    path = Path(value).expanduser()
    if len(value) < 4096 and path.is_file():
        text = path.read_text(encoding="utf-8")
    words = text.split()
    if not words:
        raise SigningError(f"no public key in {value!r}")
    try:
        public = base64.b64decode(words[-1], validate=True)
    except ValueError:
        raise SigningError(f"{value!r} is not a graft public key (a .pub file or its base64 key)") from None
    if len(public) != 32:
        raise SigningError(f"{value!r} is not an Ed25519 public key (32 bytes)")
    if len(words) > 1 and words[0].startswith("ed25519:") and words[0] != key_id(public):
        raise SigningError(f"{value!r}: the key id does not match the key")
    return public


def sign(cert: dict[str, Any], key: Path) -> dict[str, Any]:
    ed25519, serialization = _crypto()
    if not certificate.intact(cert):
        raise SigningError("the certificate was changed after graft wrote it; graft does not sign it")
    try:
        private = serialization.load_pem_private_key(key.read_bytes(), password=None)
    except FileNotFoundError:
        raise SigningError(f"no signing key at {key}; make one with `graft keygen`") from None
    except (ValueError, TypeError) as err:
        raise SigningError(f"{key} is not an unencrypted PEM private key: {err}") from None
    if not isinstance(private, ed25519.Ed25519PrivateKey):
        raise SigningError(f"{key} is not an Ed25519 key")
    public = private.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    signature = private.sign(DOMAIN + cert["digest"].encode())
    return {"format": FORMAT, "algorithm": "ed25519", "key_id": key_id(public),
            "public_key": base64.b64encode(public).decode(), "certificate_digest": cert["digest"],
            "signature": base64.b64encode(signature).decode()}


def problems(cert: dict[str, Any], sig: dict[str, Any] | None, trusted: list[bytes]) -> list[str]:
    """Why this certificate and signature should not be trusted; empty when they can be."""
    found = []
    if not certificate.intact(cert):
        found.append("the certificate was changed after graft wrote it (its digest does not match)")
    if not trusted:
        return found
    if sig is None:
        return found + ["the certificate is not signed"]
    try:
        if sig.get("format") != FORMAT or sig.get("algorithm") != "ed25519":
            raise ValueError("unknown signature format")
        public = base64.b64decode(sig["public_key"], validate=True)
        signature = base64.b64decode(sig["signature"], validate=True)
    except (ValueError, KeyError, TypeError, AttributeError):
        return found + ["the signature file is not a graft signature"]
    if sig.get("certificate_digest") != cert.get("digest"):
        found.append("the signature is for another certificate")
    elif public not in trusted:
        found.append(f"signed by {key_id(public)}, which is not a trusted key")
    elif not verify(public, DOMAIN + str(cert.get("digest")).encode(), signature):
        found.append("the signature does not verify")
    return found


def load_signature(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, UnicodeDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


# ---- Ed25519 verification, RFC 8032 section 5.1.7 ----

_P = 2**255 - 19
_L = 2**252 + 27742317777372353535851937790883648493
_D = -121665 * pow(121666, _P - 2, _P) % _P
_SQRT_M1 = pow(2, (_P - 1) // 4, _P)


def _add(a: tuple[int, ...], b: tuple[int, ...]) -> tuple[int, ...]:
    x = (a[1] - a[0]) * (b[1] - b[0]) % _P
    y = (a[1] + a[0]) * (b[1] + b[0]) % _P
    c = 2 * a[3] * b[3] * _D % _P
    d = 2 * a[2] * b[2] % _P
    e, f, g, h = y - x, d - c, d + c, y + x
    return (e * f % _P, g * h % _P, f * g % _P, e * h % _P)


def _mul(s: int, point: tuple[int, ...]) -> tuple[int, ...]:
    result = (0, 1, 1, 0)
    while s > 0:
        if s & 1:
            result = _add(result, point)
        point = _add(point, point)
        s >>= 1
    return result


def _equal(a: tuple[int, ...], b: tuple[int, ...]) -> bool:
    return (a[0] * b[2] - b[0] * a[2]) % _P == 0 and (a[1] * b[2] - b[1] * a[2]) % _P == 0


def _recover_x(y: int, sign: int) -> int | None:
    if y >= _P:
        return None
    x2 = (y * y - 1) * pow(_D * y * y + 1, _P - 2, _P) % _P
    if x2 == 0:
        return None if sign else 0
    x = pow(x2, (_P + 3) // 8, _P)
    if (x * x - x2) % _P:
        x = x * _SQRT_M1 % _P
    if (x * x - x2) % _P:
        return None
    if (x & 1) != sign:
        x = _P - x
    return x


def _decompress(raw: bytes) -> tuple[int, ...] | None:
    if len(raw) != 32:
        return None
    y = int.from_bytes(raw, "little")
    sign, y = y >> 255, y & ((1 << 255) - 1)
    x = _recover_x(y, sign)
    return None if x is None else (x, y, 1, x * y % _P)


_GY = 4 * pow(5, _P - 2, _P) % _P
_GX = _recover_x(_GY, 0)
assert _GX is not None
_G = (_GX, _GY, 1, _GX * _GY % _P)


def verify(public: bytes, message: bytes, signature: bytes) -> bool:
    if len(public) != 32 or len(signature) != 64:
        return False
    a, r = _decompress(public), _decompress(signature[:32])
    if a is None or r is None:
        return False
    s = int.from_bytes(signature[32:], "little")
    if s >= _L:
        return False
    h = int.from_bytes(hashlib.sha512(signature[:32] + public + message).digest(), "little") % _L
    return _equal(_mul(s, _G), _add(r, _mul(h, a)))
