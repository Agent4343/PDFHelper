"""
Database models and setup for PDFHelper.

Uses SQLite locally, PostgreSQL on Railway (auto-detected via DATABASE_URL).
"""

import os

from sqlalchemy import Column, String, Integer, Text, DateTime, ForeignKey, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

DATABASE_URL = os.getenv("DATABASE_URL", "").strip() or "sqlite:////tmp/pdfhelper.db"

# Railway PostgreSQL provides postgres:// but SQLAlchemy needs postgresql://
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

_is_sqlite = "sqlite" in DATABASE_URL

engine = create_engine(
    DATABASE_URL,
    # SQLite needs check_same_thread=False for FastAPI
    # PostgreSQL gets a 5s connect timeout so startup doesn't hang
    connect_args={"check_same_thread": False} if _is_sqlite else {"connect_timeout": 5},
    # Auto-reconnect stale PostgreSQL connections
    pool_pre_ping=not _is_sqlite,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class DBDocument(Base):
    __tablename__ = "documents"

    id = Column(String, primary_key=True)
    filename = Column(String, nullable=False)
    filepath = Column(String, nullable=False)
    page_count = Column(Integer, nullable=False)
    text_content = Column(Text, nullable=False)  # JSON of extracted pages
    uploaded_at = Column(DateTime, nullable=False)


class DBSearchResult(Base):
    __tablename__ = "search_results"

    id = Column(String, primary_key=True)
    search_terms = Column(Text, nullable=True)     # JSON list
    ai_query = Column(String, nullable=True)
    keyword_results = Column(Text, nullable=False)  # JSON
    ai_results = Column(Text, nullable=False)       # JSON
    total_keyword_matches = Column(Integer, default=0)
    total_ai_findings = Column(Integer, default=0)
    flagged_for_review = Column(Integer, default=0)
    searched_at = Column(DateTime, nullable=False)


class DBAnalysisReport(Base):
    __tablename__ = "analysis_reports"

    id = Column(String, primary_key=True)
    doc_ids = Column(Text, nullable=False)            # JSON list of document IDs
    compliance_context = Column(Text, nullable=True)   # encrypted
    report_data = Column(Text, nullable=False)         # encrypted JSON of full analysis
    documents_analyzed = Column(Integer, default=0)
    total_issues = Column(Integer, default=0)
    critical_issues = Column(Integer, default=0)
    risk_level = Column(String, nullable=True)
    analyzed_at = Column(DateTime, nullable=False)


class DBChatSession(Base):
    __tablename__ = "chat_sessions"

    id = Column(String, primary_key=True)
    title = Column(String, nullable=True)              # auto-generated from first message
    doc_ids = Column(Text, nullable=False)              # JSON list of document IDs
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)

    messages = relationship("DBChatMessage", back_populates="session",
                            order_by="DBChatMessage.created_at",
                            cascade="all, delete-orphan")


class DBChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(String, primary_key=True)
    session_id = Column(String, ForeignKey("chat_sessions.id"), nullable=False)
    role = Column(String, nullable=False)               # "user" or "assistant"
    content = Column(Text, nullable=False)              # encrypted
    created_at = Column(DateTime, nullable=False)

    session = relationship("DBChatSession", back_populates="messages")
