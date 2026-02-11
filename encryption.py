"""
File encryption at rest for PDFHelper.

Encrypts uploaded PDFs on disk using Fernet (AES-128-CBC).
Only the server with the correct ENCRYPTION_KEY can read them.
"""

import os
import base64
import hashlib

from cryptography.fernet import Fernet


def _get_fernet() -> Fernet:
    """Build a Fernet instance from the ENCRYPTION_KEY env var."""
    key = os.getenv("ENCRYPTION_KEY")
    if not key:
        raise RuntimeError(
            "ENCRYPTION_KEY environment variable is required. "
            "Generate one with: python -c \"from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())\""
        )
    # If the user supplies a plain passphrase instead of a proper Fernet key,
    # derive a valid 32-byte key from it.
    try:
        Fernet(key.encode())
        return Fernet(key.encode())
    except Exception:
        derived = hashlib.sha256(key.encode()).digest()
        fernet_key = base64.urlsafe_b64encode(derived)
        return Fernet(fernet_key)


def encrypt_bytes(data: bytes) -> bytes:
    """Encrypt raw bytes. Returns encrypted blob."""
    return _get_fernet().encrypt(data)


def decrypt_bytes(data: bytes) -> bytes:
    """Decrypt encrypted blob back to raw bytes."""
    return _get_fernet().decrypt(data)


def encrypt_file(source_path: str, dest_path: str) -> None:
    """Read a file, encrypt it, and write to dest_path."""
    with open(source_path, "rb") as f:
        plaintext = f.read()
    with open(dest_path, "wb") as f:
        f.write(encrypt_bytes(plaintext))


def decrypt_file(encrypted_path: str) -> bytes:
    """Read an encrypted file and return decrypted bytes."""
    with open(encrypted_path, "rb") as f:
        return decrypt_bytes(f.read())
