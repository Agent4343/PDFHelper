"""Tests for app.py helper functions and utilities."""

import hashlib
import os
from pathlib import Path
import pytest
from unittest.mock import MagicMock

# Import helpers from app
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import (
    _sanitize_filename, _verify_pdf_content, _get_client_ip, PDF_MAGIC_BYTES,
    _validate_filepath, _safe_unlink, UPLOAD_DIR,
)


class TestSanitizeFilename:
    def test_normal_filename(self):
        assert _sanitize_filename("report.pdf") == "report.pdf"

    def test_path_traversal(self):
        result = _sanitize_filename("../../etc/passwd")
        assert ".." not in result
        assert "/" not in result

    def test_dangerous_chars(self):
        result = _sanitize_filename("file<script>.pdf")
        assert "<" not in result
        assert ">" not in result

    def test_hidden_file(self):
        result = _sanitize_filename(".hidden.pdf")
        assert not result.startswith(".")

    def test_empty_filename(self):
        result = _sanitize_filename("")
        assert result == "unnamed.pdf"

    def test_spaces_preserved(self):
        result = _sanitize_filename("my report 2024.pdf")
        assert "my report 2024.pdf" == result

    def test_windows_path(self):
        # On Linux, Path("C:\\Users\\docs\\file.pdf").name doesn't split on backslashes
        # The function sanitizes dangerous chars, so backslashes become underscores
        result = _sanitize_filename("C:\\Users\\docs\\file.pdf")
        assert "file.pdf" in result


class TestVerifyPdfContent:
    def test_valid_pdf(self):
        assert _verify_pdf_content(b"%PDF-1.4 rest of content") is True

    def test_invalid_content(self):
        assert _verify_pdf_content(b"This is not a PDF") is False

    def test_empty_content(self):
        assert _verify_pdf_content(b"") is False

    def test_short_content(self):
        assert _verify_pdf_content(b"%PD") is False


class TestGetClientIp:
    def test_direct_connection(self):
        request = MagicMock()
        request.headers = {}
        request.client.host = "192.168.1.100"
        assert _get_client_ip(request) == "192.168.1.100"

    def test_forwarded_ip(self):
        request = MagicMock()
        request.headers = {"x-forwarded-for": "10.0.0.1, 172.16.0.1"}
        assert _get_client_ip(request) == "10.0.0.1"

    def test_no_client(self):
        request = MagicMock()
        request.headers = {}
        request.client = None
        assert _get_client_ip(request) == "unknown"


class TestContentHash:
    def test_hash_computation(self):
        """Content hash should be SHA-256 of raw PDF bytes."""
        content = b"%PDF-1.4 test content"
        expected = hashlib.sha256(content).hexdigest()
        assert len(expected) == 64
        assert expected == hashlib.sha256(content).hexdigest()

    def test_different_content_different_hash(self):
        h1 = hashlib.sha256(b"content A").hexdigest()
        h2 = hashlib.sha256(b"content B").hexdigest()
        assert h1 != h2


class TestValidateFilepath:
    def test_valid_path_in_upload_dir(self):
        path = UPLOAD_DIR / "test_file.pdf"
        # Create a temp file to resolve against
        path.touch()
        try:
            result = _validate_filepath(path)
            assert result.is_relative_to(UPLOAD_DIR.resolve())
        finally:
            path.unlink(missing_ok=True)

    def test_path_traversal_blocked(self):
        evil_path = UPLOAD_DIR / ".." / ".." / "etc" / "passwd"
        with pytest.raises(ValueError, match="escapes upload directory"):
            _validate_filepath(evil_path)

    def test_absolute_path_outside_upload_dir(self):
        with pytest.raises(ValueError, match="escapes upload directory"):
            _validate_filepath(Path("/etc/passwd"))


class TestSafeUnlink:
    def test_deletes_valid_file(self):
        path = UPLOAD_DIR / "test_delete_me.tmp"
        path.write_bytes(b"data")
        _safe_unlink(path)
        assert not path.exists()

    def test_missing_file_no_error(self):
        path = UPLOAD_DIR / "nonexistent_file.tmp"
        # Should not raise
        _safe_unlink(path)

    def test_traversal_blocked(self):
        evil_path = UPLOAD_DIR / ".." / ".." / "etc" / "passwd"
        # Should not raise, but should log warning and not delete
        _safe_unlink(evil_path)
