"""Tests for user authentication: registration, login, JWT, and session isolation."""

import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import (
    _hash_password, _verify_password, _create_jwt, _decode_jwt,
    RegisterRequest, LoginRequest,
)


class TestPasswordHashing:
    def test_hash_and_verify(self):
        pw = "mysecurepassword"
        hashed = _hash_password(pw)
        assert _verify_password(pw, hashed) is True

    def test_wrong_password_fails(self):
        hashed = _hash_password("correct_password")
        assert _verify_password("wrong_password", hashed) is False

    def test_different_hashes_for_same_password(self):
        """Each hash should use a unique salt."""
        h1 = _hash_password("samepassword")
        h2 = _hash_password("samepassword")
        assert h1 != h2  # Different salts
        assert _verify_password("samepassword", h1) is True
        assert _verify_password("samepassword", h2) is True

    def test_malformed_hash_returns_false(self):
        assert _verify_password("password", "not_a_valid_hash") is False
        assert _verify_password("password", "") is False

    def test_hash_format(self):
        hashed = _hash_password("test")
        # Format should be "salt:hash_hex"
        parts = hashed.split(":")
        assert len(parts) == 2
        assert len(parts[0]) == 32  # 16 bytes hex = 32 chars
        assert len(parts[1]) == 64  # SHA-256 hex = 64 chars


class TestJWT:
    def test_create_and_decode(self):
        token = _create_jwt("user-123", "testuser", is_admin=False)
        payload = _decode_jwt(token)
        assert payload is not None
        assert payload["sub"] == "user-123"
        assert payload["username"] == "testuser"
        assert payload["admin"] is False

    def test_admin_flag(self):
        token = _create_jwt("admin-1", "admin", is_admin=True)
        payload = _decode_jwt(token)
        assert payload["admin"] is True

    def test_invalid_token_returns_none(self):
        assert _decode_jwt("not.a.valid.token") is None
        assert _decode_jwt("") is None

    def test_tampered_token_returns_none(self):
        token = _create_jwt("user-1", "user")
        # Tamper with the token
        tampered = token[:-5] + "XXXXX"
        assert _decode_jwt(tampered) is None


class TestRegisterRequestValidation:
    def test_valid_username(self):
        r = RegisterRequest(username="john_doe", password="securepass123")
        assert r.username == "john_doe"

    def test_username_too_short(self):
        with pytest.raises(Exception):
            RegisterRequest(username="ab", password="securepass123")

    def test_username_invalid_chars(self):
        with pytest.raises(Exception):
            RegisterRequest(username="john doe!", password="securepass123")

    def test_password_too_short(self):
        with pytest.raises(Exception):
            RegisterRequest(username="validuser", password="short")

    def test_password_max_length(self):
        # Should not raise
        RegisterRequest(username="validuser", password="a" * 128)

    def test_password_over_max(self):
        with pytest.raises(Exception):
            RegisterRequest(username="validuser", password="a" * 129)


class TestRegisterRequestInviteCode:
    def test_invite_code_optional_by_default(self):
        r = RegisterRequest(username="validuser", password="securepass123")
        assert r.invite_code is None

    def test_invite_code_accepted(self):
        r = RegisterRequest(username="validuser", password="securepass123", invite_code="my-secret")
        assert r.invite_code == "my-secret"

    def test_invite_code_max_length(self):
        # Should not raise at 256 chars
        r = RegisterRequest(username="validuser", password="securepass123", invite_code="a" * 256)
        assert len(r.invite_code) == 256

    def test_invite_code_over_max_length(self):
        with pytest.raises(Exception):
            RegisterRequest(username="validuser", password="securepass123", invite_code="a" * 257)


class TestLoginRequestValidation:
    def test_valid_login(self):
        r = LoginRequest(username="user", password="pass")
        assert r.username == "user"
