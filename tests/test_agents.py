"""Tests for agents.py — text budgeting, chunking, merging, retries, JSON parsing."""

import json
import pytest
from unittest.mock import patch, MagicMock

from agents import (
    _build_page_text,
    _merge_analysis_results,
    _CHUNK_PAGE_LIMIT,
    document_analysis_agent,
    cross_reference_agent,
    compliance_agent,
    summary_report_agent,
    run_full_analysis,
)
from utils import parse_json_response as _parse_json_response, is_retryable_error as _is_retryable_error


# ---------------------------------------------------------------------------
# _build_page_text
# ---------------------------------------------------------------------------

class TestBuildPageText:
    def test_empty_pages(self):
        assert _build_page_text([]) == ""

    def test_single_page_under_budget(self):
        pages = [{"page": 1, "text": "Hello world"}]
        result = _build_page_text(pages, budget=1000)
        assert "Hello world" in result
        assert "PAGE 1" in result

    def test_budget_distributes_evenly(self):
        pages = [
            {"page": 1, "text": "A" * 500},
            {"page": 2, "text": "B" * 500},
        ]
        result = _build_page_text(pages, budget=400)
        # Each page gets ~200 chars of budget; text truncated to that
        a_count = result.count("A")
        b_count = result.count("B")
        # Both should be truncated to roughly equal amounts
        assert abs(a_count - b_count) <= 5
        assert a_count < 500  # definitely truncated
        assert b_count < 500

    def test_surplus_redistribution(self):
        """Short pages donate surplus to long pages."""
        pages = [
            {"page": 1, "text": "A" * 10},    # short — only needs 10
            {"page": 2, "text": "B" * 1000},   # long — needs more
        ]
        result = _build_page_text(pages, budget=600)
        # equal_share = 300; page 1 uses 10, surplus = 290
        # long_limit = 300 + 290 = 590
        a_count = result.count("A")
        b_count = result.count("B")
        assert a_count <= 15  # short page not truncated (10 A's + possible page num)
        assert b_count > 300  # long page gets surplus from short page

    def test_all_short_pages(self):
        """When all pages fit, nothing gets truncated."""
        pages = [
            {"page": 1, "text": "Hi"},
            {"page": 2, "text": "There"},
        ]
        result = _build_page_text(pages, budget=10000)
        assert "Hi" in result
        assert "There" in result

    def test_page_numbers_preserved(self):
        pages = [{"page": 5, "text": "content"}]
        result = _build_page_text(pages)
        assert "PAGE 5" in result


# ---------------------------------------------------------------------------
# _merge_analysis_results
# ---------------------------------------------------------------------------

class TestMergeAnalysisResults:
    def test_merge_takes_first_title(self):
        chunks = [
            {"title": "Doc Title", "type": "procedure", "topics": ["safety"],
             "key_dates": [], "current_revision": "D14", "key_references": [],
             "regulatory_references": [], "sections": [], "roles_identified": [],
             "summary": "First chunk."},
            {"title": "Other Title", "type": "manual", "topics": ["operations"],
             "key_dates": ["2024-01"], "current_revision": "D15", "key_references": [],
             "regulatory_references": [], "sections": [], "roles_identified": [],
             "summary": "Second chunk."},
        ]
        merged = _merge_analysis_results(chunks, "test.pdf")
        assert merged["title"] == "Doc Title"
        assert merged["current_revision"] == "D14"

    def test_merge_deduplicates_lists(self):
        chunks = [
            {"topics": ["safety", "valves"], "key_dates": [], "title": "",
             "type": "", "current_revision": "", "key_references": [],
             "regulatory_references": [], "sections": [], "roles_identified": [],
             "summary": "A"},
            {"topics": ["safety", "permits"], "key_dates": [], "title": "",
             "type": "", "current_revision": "", "key_references": [],
             "regulatory_references": [], "sections": [], "roles_identified": [],
             "summary": "B"},
        ]
        merged = _merge_analysis_results(chunks, "test.pdf")
        assert merged["topics"] == ["safety", "valves", "permits"]

    def test_merge_skips_error_chunks(self):
        chunks = [
            {"error": "API failed"},
            {"title": "Good", "type": "report", "topics": ["X"],
             "key_dates": [], "current_revision": "", "key_references": [],
             "regulatory_references": [], "sections": [], "roles_identified": [],
             "summary": "OK"},
        ]
        merged = _merge_analysis_results(chunks, "test.pdf")
        assert merged["title"] == "Good"

    def test_merge_combines_summaries(self):
        chunks = [
            {"title": "", "type": "", "topics": [], "key_dates": [],
             "current_revision": "", "key_references": [], "regulatory_references": [],
             "sections": [], "roles_identified": [], "summary": "Part 1."},
            {"title": "", "type": "", "topics": [], "key_dates": [],
             "current_revision": "", "key_references": [], "regulatory_references": [],
             "sections": [], "roles_identified": [], "summary": "Part 2."},
        ]
        merged = _merge_analysis_results(chunks, "test.pdf")
        assert "Part 1." in merged["summary"]
        assert "Part 2." in merged["summary"]

    def test_merge_single_summary(self):
        chunks = [
            {"title": "", "type": "", "topics": [], "key_dates": [],
             "current_revision": "", "key_references": [], "regulatory_references": [],
             "sections": [], "roles_identified": [], "summary": "Only one."},
        ]
        merged = _merge_analysis_results(chunks, "test.pdf")
        assert merged["summary"] == "Only one."


# ---------------------------------------------------------------------------
# _parse_json_response
# ---------------------------------------------------------------------------

class TestParseJsonResponse:
    def test_plain_json_array(self):
        assert _parse_json_response('[{"a": 1}]') == [{"a": 1}]

    def test_plain_json_object(self):
        assert _parse_json_response('{"key": "val"}') == {"key": "val"}

    def test_markdown_code_block(self):
        text = '```json\n[{"x": 1}]\n```'
        assert _parse_json_response(text) == [{"x": 1}]

    def test_markdown_no_language(self):
        text = '```\n{"a": "b"}\n```'
        assert _parse_json_response(text) == {"a": "b"}

    def test_invalid_json_raises(self):
        with pytest.raises(json.JSONDecodeError):
            _parse_json_response("not json at all")


# ---------------------------------------------------------------------------
# _is_retryable_error
# ---------------------------------------------------------------------------

class TestIsRetryableError:
    def test_rate_limit(self):
        assert _is_retryable_error(Exception("rate_limit_exceeded")) is True

    def test_overloaded(self):
        assert _is_retryable_error(Exception("Service overloaded")) is True

    def test_server_errors(self):
        assert _is_retryable_error(Exception("502 Bad Gateway")) is True
        assert _is_retryable_error(Exception("503 Service Unavailable")) is True
        assert _is_retryable_error(Exception("500 Internal Server Error")) is True

    def test_timeout(self):
        assert _is_retryable_error(Exception("Connection timeout")) is True

    def test_not_retryable(self):
        assert _is_retryable_error(Exception("Invalid API key")) is False
        assert _is_retryable_error(Exception("Bad request")) is False


# ---------------------------------------------------------------------------
# document_analysis_agent — chunking
# ---------------------------------------------------------------------------

class TestDocumentAnalysisChunking:
    @patch("agents._call_ai")
    def test_small_doc_no_chunking(self, mock_ai):
        mock_ai.return_value = json.dumps({
            "title": "Test", "type": "report", "topics": [],
            "key_dates": [], "current_revision": "", "key_references": [],
            "regulatory_references": [], "sections": [], "roles_identified": [],
            "summary": "A test doc."
        })
        pages = [{"page": i, "text": f"Page {i}"} for i in range(1, 11)]
        result = document_analysis_agent("test.pdf", pages)
        assert result["title"] == "Test"
        assert mock_ai.call_count == 1

    @patch("agents._call_ai")
    def test_large_doc_chunks(self, mock_ai):
        """Documents > 60 pages should be split into chunks."""
        mock_ai.return_value = json.dumps({
            "title": "Big Doc", "type": "manual", "topics": ["topic1"],
            "key_dates": [], "current_revision": "Rev1", "key_references": [],
            "regulatory_references": [], "sections": [], "roles_identified": [],
            "summary": "Chunk summary."
        })
        pages = [{"page": i, "text": f"Page {i}"} for i in range(1, 80)]
        result = document_analysis_agent("big.pdf", pages)
        # 79 pages / 60 = 2 chunks
        assert mock_ai.call_count == 2
        assert result["title"] == "Big Doc"


# ---------------------------------------------------------------------------
# run_full_analysis — orchestration and warnings
# ---------------------------------------------------------------------------

class TestRunFullAnalysis:
    @patch("agents.summary_report_agent")
    @patch("agents.cross_reference_agent")
    @patch("agents.compliance_agent")
    @patch("agents.document_analysis_agent")
    def test_warnings_on_failure(self, mock_doc, mock_comp, mock_cross, mock_summary):
        mock_doc.side_effect = RuntimeError("API down")
        mock_comp.return_value = []
        mock_cross.return_value = []
        mock_summary.return_value = {"executive_summary": "Report"}

        docs = {"test.pdf": [{"page": 1, "text": "hello"}]}
        result = run_full_analysis(docs)

        assert "warnings" in result
        assert any("failed" in w.lower() for w in result["warnings"])

    @patch("agents.summary_report_agent")
    @patch("agents.cross_reference_agent")
    @patch("agents.compliance_agent")
    @patch("agents.document_analysis_agent")
    def test_single_doc_skips_crossref(self, mock_doc, mock_comp, mock_cross, mock_summary):
        mock_doc.return_value = {"title": "T", "summary": "S", "topics": []}
        mock_comp.return_value = []
        mock_summary.return_value = {}

        docs = {"single.pdf": [{"page": 1, "text": "content"}]}
        run_full_analysis(docs)
        mock_cross.assert_not_called()

    @patch("agents.summary_report_agent")
    @patch("agents.cross_reference_agent")
    @patch("agents.compliance_agent")
    @patch("agents.document_analysis_agent")
    def test_successful_run_no_warnings(self, mock_doc, mock_comp, mock_cross, mock_summary):
        mock_doc.return_value = {"title": "T", "type": "report", "summary": "S", "topics": []}
        mock_comp.return_value = []
        mock_cross.return_value = []
        mock_summary.return_value = {"executive_summary": "All good"}

        docs = {"a.pdf": [{"page": 1, "text": "A"}]}
        result = run_full_analysis(docs)
        assert "warnings" not in result
