"""Tests for ocr.py — text extraction, table extraction, OCR decisions."""

from unittest.mock import MagicMock, patch, PropertyMock
import pytest

from ocr import _page_needs_ocr, _extract_page_text, OCR_CHAR_THRESHOLD


# ---------------------------------------------------------------------------
# _page_needs_ocr
# ---------------------------------------------------------------------------

class TestPageNeedsOcr:
    def test_empty_text_needs_ocr(self):
        assert _page_needs_ocr("") is True

    def test_short_text_needs_ocr(self):
        assert _page_needs_ocr("A" * (OCR_CHAR_THRESHOLD - 1)) is True

    def test_sufficient_text_no_ocr(self):
        assert _page_needs_ocr("A" * OCR_CHAR_THRESHOLD) is False

    def test_whitespace_only_needs_ocr(self):
        assert _page_needs_ocr("   \n\t  ") is True


# ---------------------------------------------------------------------------
# _extract_page_text
# ---------------------------------------------------------------------------

class TestExtractPageText:
    def test_plain_text_fallback(self):
        """Pages without find_tables should fall back to get_text()."""
        page = MagicMock(spec=[])  # no find_tables attribute
        page.get_text = MagicMock(return_value="Plain text content")
        result = _extract_page_text(page)
        assert result == "Plain text content"

    def test_no_tables_found(self):
        """If find_tables returns empty, should still return text."""
        page = MagicMock()
        page.get_text.return_value = "Body text"
        tables_result = MagicMock()
        tables_result.tables = []  # no tables
        page.find_tables.return_value = tables_result
        result = _extract_page_text(page)
        assert result == "Body text"

    def test_table_extraction_with_pandas(self):
        """Tables should be extracted and wrapped in [TABLE] markers."""
        page = MagicMock()
        page.get_text.return_value = "Header text"

        # Mock a table that can convert to pandas
        mock_df = MagicMock()
        mock_df.to_string.return_value = "Col1  Col2\nA     B"

        mock_table = MagicMock()
        mock_table.to_pandas.return_value = mock_df

        tables_result = MagicMock()
        tables_result.tables = [mock_table]
        tables_result.__iter__ = lambda self: iter([mock_table])
        page.find_tables.return_value = tables_result

        result = _extract_page_text(page)
        assert "[TABLE]" in result
        assert "[/TABLE]" in result
        assert "Col1" in result
        assert "Header text" in result

    def test_table_extraction_fallback_to_rows(self):
        """When pandas fails, should extract raw rows."""
        page = MagicMock()
        page.get_text.return_value = "Body"

        mock_table = MagicMock()
        mock_table.to_pandas.side_effect = Exception("No pandas")
        mock_table.extract.return_value = [["A", "B"], ["C", None]]

        tables_result = MagicMock()
        tables_result.tables = [mock_table]
        tables_result.__iter__ = lambda self: iter([mock_table])
        page.find_tables.return_value = tables_result

        result = _extract_page_text(page)
        assert "[TABLE]" in result
        assert "A | B" in result
        assert "C | " in result  # None becomes ""

    def test_find_tables_exception_falls_back(self):
        """If find_tables itself throws, fall back to get_text()."""
        page = MagicMock()
        page.get_text.return_value = "Fallback text"
        page.find_tables.side_effect = Exception("Unsupported")

        result = _extract_page_text(page)
        assert result == "Fallback text"
