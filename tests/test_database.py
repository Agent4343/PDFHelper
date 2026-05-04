"""Tests for database.py — model definitions and schema."""

from database import (
    DBDocument, DBSearchResult, DBAnalysisReport,
    DBChatSession, DBChatMessage, DBDrawing, DBIsolationPackage,
)


class TestDatabaseModels:
    def test_document_has_content_hash(self):
        """DBDocument should have a content_hash column for caching."""
        assert hasattr(DBDocument, "content_hash")

    def test_analysis_report_has_cache_key(self):
        """DBAnalysisReport should have an indexed cache_key column."""
        assert hasattr(DBAnalysisReport, "cache_key")

    def test_document_columns(self):
        cols = {c.name for c in DBDocument.__table__.columns}
        assert "id" in cols
        assert "filename" in cols
        assert "filepath" in cols
        assert "page_count" in cols
        assert "text_content" in cols
        assert "content_hash" in cols
        assert "uploaded_at" in cols

    def test_search_result_columns(self):
        cols = {c.name for c in DBSearchResult.__table__.columns}
        assert "search_terms" in cols
        assert "ai_query" in cols
        assert "keyword_results" in cols
        assert "ai_results" in cols
        assert "flagged_for_review" in cols

    def test_chat_session_relationship(self):
        """Chat sessions should have a messages relationship."""
        assert hasattr(DBChatSession, "messages")

    def test_drawing_columns(self):
        cols = {c.name for c in DBDrawing.__table__.columns}
        assert "drawing_number" in cols
        assert "equipment_tags" in cols
        assert "text_content" in cols

    def test_isolation_package_columns(self):
        cols = {c.name for c in DBIsolationPackage.__table__.columns}
        assert "cert_number" in cols
        assert "equipment_tag" in cols
        assert "hazard_classification" in cols
        assert "status" in cols
