"""Tests for search.py — keyword search, JSON parsing, retry logic."""

import json
import pytest
from unittest.mock import patch, MagicMock

from search import keyword_search, _parse_json_response, _is_retryable_error


# ---------------------------------------------------------------------------
# keyword_search
# ---------------------------------------------------------------------------

class TestKeywordSearch:
    def test_basic_match(self):
        pages = [{"page": 1, "text": "The valve must be closed before maintenance."}]
        results = keyword_search(pages, ["valve"])
        assert len(results) == 1
        assert results[0]["page"] == 1
        assert results[0]["term"] == "valve"
        assert "valve" in results[0]["matched_text"].lower()

    def test_case_insensitive_default(self):
        pages = [{"page": 1, "text": "SAFETY PROCEDURE for isolation"}]
        results = keyword_search(pages, ["safety"])
        assert len(results) == 1

    def test_case_sensitive(self):
        pages = [{"page": 1, "text": "SAFETY PROCEDURE for isolation"}]
        results = keyword_search(pages, ["safety"], case_sensitive=True)
        assert len(results) == 0
        results = keyword_search(pages, ["SAFETY"], case_sensitive=True)
        assert len(results) == 1

    def test_no_match(self):
        pages = [{"page": 1, "text": "Nothing relevant here"}]
        results = keyword_search(pages, ["compliance"])
        assert len(results) == 0

    def test_multiple_terms(self):
        pages = [{"page": 1, "text": "Check the valve and the permit before work."}]
        results = keyword_search(pages, ["valve", "permit"])
        assert len(results) == 2
        terms_found = {r["term"] for r in results}
        assert terms_found == {"valve", "permit"}

    def test_multiple_matches_same_page(self):
        pages = [{"page": 1, "text": "valve A and valve B and valve C"}]
        results = keyword_search(pages, ["valve"])
        assert len(results) == 3

    def test_multiple_pages(self):
        pages = [
            {"page": 1, "text": "First page with valve info"},
            {"page": 2, "text": "Second page with valve data"},
        ]
        results = keyword_search(pages, ["valve"])
        assert len(results) == 2
        assert {r["page"] for r in results} == {1, 2}

    def test_context_window(self):
        """Context should include surrounding text (up to 80 chars each side)."""
        text = "X" * 100 + "KEYWORD" + "Y" * 100
        pages = [{"page": 1, "text": text}]
        results = keyword_search(pages, ["KEYWORD"])
        assert len(results) == 1
        context = results[0]["context"]
        assert "KEYWORD" in context
        # Context has "..." prefix and suffix
        assert context.startswith("...")
        assert context.endswith("...")

    def test_special_regex_chars(self):
        """Search terms with regex special chars should be escaped."""
        pages = [{"page": 1, "text": "Section 3.2.1 (a) reference"}]
        results = keyword_search(pages, ["3.2.1 (a)"])
        assert len(results) == 1

    def test_empty_pages(self):
        results = keyword_search([], ["valve"])
        assert results == []

    def test_empty_terms(self):
        pages = [{"page": 1, "text": "Some content"}]
        results = keyword_search(pages, [])
        assert results == []


# ---------------------------------------------------------------------------
# _parse_json_response (search module)
# ---------------------------------------------------------------------------

class TestSearchParseJson:
    def test_plain_json(self):
        assert _parse_json_response("[]") == []

    def test_code_block(self):
        assert _parse_json_response('```json\n[]\n```') == []

    def test_invalid(self):
        with pytest.raises(json.JSONDecodeError):
            _parse_json_response("nope")


# ---------------------------------------------------------------------------
# _is_retryable_error (search module)
# ---------------------------------------------------------------------------

class TestSearchRetryable:
    def test_retryable(self):
        assert _is_retryable_error(Exception("529 overloaded"))

    def test_not_retryable(self):
        assert not _is_retryable_error(Exception("auth error"))
