"""Tests for encryption.py — round-trip encrypt/decrypt, key derivation."""

import os
import tempfile
import pytest
from unittest.mock import patch

# Need to clear the lru_cache between tests with different keys
import encryption
from encryption import encrypt_bytes, decrypt_bytes, encrypt_file, decrypt_file, _get_fernet


def _clear_fernet_cache():
    _get_fernet.cache_clear()


class TestEncryption:
    def setup_method(self):
        _clear_fernet_cache()

    def teardown_method(self):
        _clear_fernet_cache()

    @patch.dict(os.environ, {"ENCRYPTION_KEY": ""}, clear=False)
    def test_missing_key_raises(self):
        # Also remove the key entirely if present
        os.environ.pop("ENCRYPTION_KEY", None)
        with pytest.raises(RuntimeError, match="ENCRYPTION_KEY"):
            _get_fernet()

    @patch.dict(os.environ, {"ENCRYPTION_KEY": "my-secret-passphrase"})
    def test_passphrase_key_derivation(self):
        """A plain passphrase should be derived into a valid Fernet key."""
        f = _get_fernet()
        assert f is not None
        # Should be able to encrypt/decrypt
        ct = f.encrypt(b"test")
        assert f.decrypt(ct) == b"test"

    @patch.dict(os.environ, {"ENCRYPTION_KEY": "my-secret-passphrase"})
    def test_roundtrip_bytes(self):
        plaintext = b"Sensitive PDF content here"
        ciphertext = encrypt_bytes(plaintext)
        assert ciphertext != plaintext
        assert decrypt_bytes(ciphertext) == plaintext

    @patch.dict(os.environ, {"ENCRYPTION_KEY": "test-key-123"})
    def test_roundtrip_file(self):
        plaintext = b"File content to encrypt"
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as src:
            src.write(plaintext)
            src_path = src.name

        enc_path = src_path + ".enc"
        try:
            encrypt_file(src_path, enc_path)
            # Encrypted file should exist and differ from plaintext
            with open(enc_path, "rb") as f:
                enc_data = f.read()
            assert enc_data != plaintext

            # Decrypt and verify
            decrypted = decrypt_file(enc_path)
            assert decrypted == plaintext
        finally:
            os.unlink(src_path)
            if os.path.exists(enc_path):
                os.unlink(enc_path)

    @patch.dict(os.environ, {"ENCRYPTION_KEY": "key-a"})
    def test_caching(self):
        """Fernet instance should be cached (same object returned)."""
        f1 = _get_fernet()
        f2 = _get_fernet()
        assert f1 is f2


class TestFernetKeyFormat:
    def setup_method(self):
        _clear_fernet_cache()

    def teardown_method(self):
        _clear_fernet_cache()

    @patch.dict(os.environ, {})
    def test_valid_fernet_key_accepted(self):
        """A proper Fernet key should be used directly (no derivation)."""
        from cryptography.fernet import Fernet
        real_key = Fernet.generate_key().decode()
        os.environ["ENCRYPTION_KEY"] = real_key
        f = _get_fernet()
        ct = f.encrypt(b"data")
        assert f.decrypt(ct) == b"data"
