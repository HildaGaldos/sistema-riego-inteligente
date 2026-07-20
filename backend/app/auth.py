"""Small SQLite-backed auth layer with scrypt password hashes and signed JWTs."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import sqlite3
import time
from pathlib import Path


def _b64(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode()


def _unb64(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, n=2**14, r=8, p=1)
    return f"scrypt${_b64(salt)}${_b64(digest)}"


def verify_password(password: str, stored: str) -> bool:
    try:
        _, salt, digest = stored.split("$")
        expected = hashlib.scrypt(password.encode(), salt=_unb64(salt), n=2**14, r=8, p=1)
        return hmac.compare_digest(expected, _unb64(digest))
    except (ValueError, TypeError):
        return False


class AuthStore:
    def __init__(self, path: str | Path = "data/users.sqlite3"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password_hash TEXT NOT NULL, is_admin INTEGER NOT NULL DEFAULT 0)")
            conn.commit()
        self.ensure_admin()

    def ensure_admin(self) -> None:
        username = os.getenv("IRRIGATION_ADMIN_USER", "admin")
        password = os.getenv("IRRIGATION_ADMIN_PASSWORD", "change-this-password")
        with sqlite3.connect(self.path) as conn:
            row = conn.execute("SELECT username FROM users WHERE username=?", (username,)).fetchone()
            if not row:
                conn.execute("INSERT INTO users VALUES (?, ?, 1)", (username, hash_password(password)))
                conn.commit()

    def authenticate(self, username: str, password: str) -> bool:
        with sqlite3.connect(self.path) as conn:
            row = conn.execute("SELECT password_hash FROM users WHERE username=?", (username,)).fetchone()
        return bool(row and verify_password(password, row[0]))

    def is_admin(self, username: str) -> bool:
        with sqlite3.connect(self.path) as conn:
            row = conn.execute("SELECT is_admin FROM users WHERE username=?", (username,)).fetchone()
        return bool(row and row[0])


def create_token(username: str, secret: str, expires_minutes: int = 60) -> str:
    header = _b64(json.dumps({"alg": "HS256", "typ": "JWT"}, separators=(",", ":")).encode())
    payload = _b64(json.dumps({"sub": username, "exp": int(time.time()) + expires_minutes * 60}, separators=(",", ":")).encode())
    unsigned = f"{header}.{payload}".encode()
    signature = _b64(hmac.new(secret.encode(), unsigned, hashlib.sha256).digest())
    return f"{header}.{payload}.{signature}"


def decode_token(token: str, secret: str) -> dict:
    try:
        header, payload, signature = token.split(".")
        unsigned = f"{header}.{payload}".encode()
        expected = _b64(hmac.new(secret.encode(), unsigned, hashlib.sha256).digest())
        if not hmac.compare_digest(expected, signature):
            raise ValueError("Firma inválida")
        data = json.loads(_unb64(payload))
        if int(data["exp"]) < int(time.time()):
            raise ValueError("Token expirado")
        return data
    except (ValueError, KeyError, json.JSONDecodeError) as exc:
        raise ValueError("Token inválido") from exc
