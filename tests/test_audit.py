"""Tests for audit.py — structured audit logging."""

import logging
from unittest.mock import patch

from audit import log_upload, log_search, log_delete, log_auth_failure, log_access, audit_log


class TestAuditLogging:
    def test_log_upload(self, caplog):
        with caplog.at_level(logging.INFO, logger="pdfhelper.audit"):
            log_upload("192.168.1.1", "test.pdf", "doc-123", 10)
        assert "UPLOAD" in caplog.text
        assert "192.168.1.1" in caplog.text
        assert "test.pdf" in caplog.text
        assert "doc-123" in caplog.text

    def test_log_search(self, caplog):
        with caplog.at_level(logging.INFO, logger="pdfhelper.audit"):
            log_search("10.0.0.1", "search-1", ["valve"], "isolation query", 5, 10, 2)
        assert "SEARCH" in caplog.text
        assert "search-1" in caplog.text
        assert "10.0.0.1" in caplog.text

    def test_log_delete(self, caplog):
        with caplog.at_level(logging.INFO, logger="pdfhelper.audit"):
            log_delete("10.0.0.2", "doc-456", "old.pdf")
        assert "DELETE" in caplog.text
        assert "doc-456" in caplog.text

    def test_log_auth_failure(self, caplog):
        with caplog.at_level(logging.WARNING, logger="pdfhelper.audit"):
            log_auth_failure("1.2.3.4", "/api/upload")
        assert "AUTH_FAILURE" in caplog.text
        assert "1.2.3.4" in caplog.text

    def test_log_access(self, caplog):
        with caplog.at_level(logging.INFO, logger="pdfhelper.audit"):
            log_access("172.16.0.1", "GET", "/health", 200)
        assert "ACCESS" in caplog.text
        assert "GET" in caplog.text
        assert "200" in caplog.text

    def test_uses_lazy_formatting(self):
        """Verify log functions use %s lazy formatting, not f-strings.

        This is a code-level check — we verify by looking at the source.
        """
        import inspect
        source = inspect.getsource(log_upload)
        # Should use %s formatting
        assert "%s" in source
        # Should NOT use f-string (f" in the log call)
        assert 'f"' not in source and "f'" not in source
