"""Tests for security hardening — input sanitization, length limits, media type validation."""

import pytest
from pydantic import ValidationError

from app import IsolationRequest, SearchRequest, ChatRequest, ChatMessage
from isointel import _sanitize_prompt_input, _ALLOWED_MEDIA_TYPES, _MAX_IMAGE_B64_LEN


# ---------------------------------------------------------------------------
# _sanitize_prompt_input
# ---------------------------------------------------------------------------

class TestSanitizePromptInput:
    def test_normal_input_unchanged(self):
        assert _sanitize_prompt_input("HB-P-1001A") == "HB-P-1001A"

    def test_strips_null_bytes(self):
        result = _sanitize_prompt_input("tag\x00name")
        assert "\x00" not in result
        assert "tagname" == result

    def test_strips_control_chars(self):
        result = _sanitize_prompt_input("line\x01\x02\x03end")
        assert result == "lineend"

    def test_preserves_newlines_and_tabs(self):
        result = _sanitize_prompt_input("line1\nline2\ttab")
        assert "\n" in result
        assert "\t" in result

    def test_truncates_to_max_length(self):
        result = _sanitize_prompt_input("A" * 10000, max_length=100)
        assert len(result) == 100

    def test_empty_string(self):
        assert _sanitize_prompt_input("") == ""

    def test_none_passthrough(self):
        # None should pass through (guard is `if not value`)
        assert _sanitize_prompt_input(None) is None


# ---------------------------------------------------------------------------
# Pydantic model length limits
# ---------------------------------------------------------------------------

class TestRequestModelLimits:
    def test_isolation_equipment_tag_too_long(self):
        with pytest.raises(ValidationError):
            IsolationRequest(
                equipment_tag="X" * 300,
                work_description="test",
                work_type="MAINTENANCE",
            )

    def test_isolation_work_description_too_long(self):
        with pytest.raises(ValidationError):
            IsolationRequest(
                equipment_tag="HB-P-1001",
                work_description="X" * 6000,
                work_type="MAINTENANCE",
            )

    def test_isolation_valid_request(self):
        req = IsolationRequest(
            equipment_tag="HB-P-1001A",
            work_description="Replace pump seals",
            work_type="MAINTENANCE",
        )
        assert req.equipment_tag == "HB-P-1001A"

    def test_search_ai_query_too_long(self):
        with pytest.raises(ValidationError):
            SearchRequest(ai_query="X" * 3000)

    def test_chat_message_too_long(self):
        with pytest.raises(ValidationError):
            ChatRequest(message="X" * 15000)

    def test_chat_valid_request(self):
        req = ChatRequest(message="What does this document say about safety?")
        assert len(req.message) < 10000


# ---------------------------------------------------------------------------
# Media type validation constants
# ---------------------------------------------------------------------------

class TestMediaTypeValidation:
    def test_allowed_types(self):
        assert "image/png" in _ALLOWED_MEDIA_TYPES
        assert "image/jpeg" in _ALLOWED_MEDIA_TYPES

    def test_disallowed_types(self):
        assert "application/pdf" not in _ALLOWED_MEDIA_TYPES
        assert "text/html" not in _ALLOWED_MEDIA_TYPES

    def test_max_image_size_reasonable(self):
        # ~10MB base64 limit
        assert _MAX_IMAGE_B64_LEN > 10_000_000
        assert _MAX_IMAGE_B64_LEN < 20_000_000
