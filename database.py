"""
Database models and setup for PDFHelper.

Uses SQLite locally, PostgreSQL on Railway (auto-detected via DATABASE_URL).
"""

import os

from sqlalchemy import Column, String, Integer, Text, DateTime, ForeignKey, Boolean, Index, Table, create_engine
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

# Junction tables for many-to-many document references
chat_session_documents = Table(
    "chat_session_documents", Base.metadata,
    Column("session_id", String, ForeignKey("chat_sessions.id", ondelete="CASCADE"), primary_key=True),
    Column("document_id", String, ForeignKey("documents.id", ondelete="CASCADE"), primary_key=True),
)

code_session_documents = Table(
    "code_session_documents", Base.metadata,
    Column("session_id", String, ForeignKey("code_sessions.id", ondelete="CASCADE"), primary_key=True),
    Column("document_id", String, ForeignKey("documents.id", ondelete="CASCADE"), primary_key=True),
)

analysis_report_documents = Table(
    "analysis_report_documents", Base.metadata,
    Column("report_id", String, ForeignKey("analysis_reports.id", ondelete="CASCADE"), primary_key=True),
    Column("document_id", String, ForeignKey("documents.id", ondelete="CASCADE"), primary_key=True),
)

agent_cache_documents = Table(
    "agent_cache_documents", Base.metadata,
    Column("cache_id", String, ForeignKey("agent_cache.id", ondelete="CASCADE"), primary_key=True),
    Column("document_id", String, ForeignKey("documents.id", ondelete="CASCADE"), primary_key=True),
)


class DBUser(Base):
    """Registered user with hashed password."""
    __tablename__ = "users"

    id = Column(String, primary_key=True)
    username = Column(String, nullable=False, unique=True, index=True)
    password_hash = Column(String, nullable=False)
    is_admin = Column(Boolean, default=False)
    created_at = Column(DateTime, nullable=False)


class DBDocument(Base):
    __tablename__ = "documents"

    id = Column(String, primary_key=True)
    filename = Column(String, nullable=False)
    filepath = Column(String, nullable=False)
    page_count = Column(Integer, nullable=False)
    text_content = Column(Text, nullable=False)  # JSON of extracted pages
    content_hash = Column(String, nullable=True, index=True)
    uploaded_at = Column(DateTime, nullable=False, index=True)


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
    compliance_context = Column(Text, nullable=True)   # encrypted
    report_data = Column(Text, nullable=False)         # encrypted JSON of full analysis
    documents_analyzed = Column(Integer, default=0)
    total_issues = Column(Integer, default=0)
    critical_issues = Column(Integer, default=0)
    risk_level = Column(String, nullable=True)
    cache_key = Column(String, nullable=True, index=True)  # SHA-256 of doc hashes + params
    analyzed_at = Column(DateTime, nullable=False)

    documents = relationship("DBDocument", secondary=analysis_report_documents, lazy="selectin")


class DBChatSession(Base):
    __tablename__ = "chat_sessions"

    id = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    title = Column(String, nullable=True)
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False, index=True)

    documents = relationship("DBDocument", secondary=chat_session_documents, lazy="selectin")
    messages = relationship("DBChatMessage", back_populates="session",
                            order_by="DBChatMessage.created_at",
                            cascade="all, delete-orphan")


class DBChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(String, primary_key=True)
    session_id = Column(String, ForeignKey("chat_sessions.id", ondelete="CASCADE"), nullable=False, index=True)
    role = Column(String, nullable=False)               # "user" or "assistant"
    content = Column(Text, nullable=False)              # encrypted
    created_at = Column(DateTime, nullable=False)

    session = relationship("DBChatSession", back_populates="messages")


# ---------------------------------------------------------------------------
# IsoIntel — P&ID Drawings and Isolation Packages
# ---------------------------------------------------------------------------

class DBDrawing(Base):
    """A P&ID drawing uploaded for isolation analysis."""
    __tablename__ = "drawings"

    id = Column(String, primary_key=True)
    filename = Column(String, nullable=False)           # encrypted
    filepath = Column(String, nullable=False)            # encrypted file on disk
    title = Column(String, nullable=True)                # encrypted — drawing title
    drawing_number = Column(String, nullable=True)       # encrypted — e.g. "HEB-PID-1234"
    equipment_tags = Column(Text, nullable=True)         # encrypted — comma-separated tags
    description = Column(Text, nullable=True)            # encrypted — what system this covers
    page_count = Column(Integer, default=1)
    text_content = Column(Text, nullable=True)           # encrypted JSON — OCR extracted text
    uploaded_at = Column(DateTime, nullable=False, index=True)


class DBIsolationPackage(Base):
    """A completed isolation package generated by IsoIntel."""
    __tablename__ = "isolation_packages"

    id = Column(String, primary_key=True)
    cert_number = Column(String, nullable=False, unique=True)  # encrypted
    equipment_tag = Column(String, nullable=False)              # encrypted
    work_description = Column(String, nullable=False)           # encrypted
    work_type = Column(String, nullable=False)                  # encrypted
    fluid_service = Column(String, nullable=True)               # encrypted
    facility = Column(String, nullable=True)                    # encrypted
    regime = Column(String, nullable=True)                      # encrypted
    special_requirements = Column(Text, nullable=True)          # encrypted
    drawing_ids = Column(Text, nullable=False)                  # JSON list of drawing IDs used
    package_data = Column(Text, nullable=False)                 # encrypted JSON — full AI output
    hazard_classification = Column(String, nullable=True)       # HIGH / MEDIUM / LOW
    valve_count = Column(Integer, default=0)
    blind_count = Column(Integer, default=0)
    step_count = Column(Integer, default=0)
    energy_source_count = Column(Integer, default=0)
    status = Column(String, default="draft", index=True)        # draft / approved / closed
    created_at = Column(DateTime, nullable=False, index=True)
    updated_at = Column(DateTime, nullable=False)


# ---------------------------------------------------------------------------
# Doc Updater — Update Sessions
# ---------------------------------------------------------------------------

class DBUpdateSession(Base):
    """A saved Doc Updater session with regulation findings and proposed updates."""
    __tablename__ = "update_sessions"

    id = Column(String, primary_key=True)
    doc_id = Column(String, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True)
    user_id = Column(String, ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    title = Column(String, nullable=True)
    regulation_query = Column(Text, nullable=True)
    regulation_results = Column(Text, nullable=True)       # encrypted
    updates_json = Column(Text, nullable=True)             # encrypted JSON array of update blocks
    accepted_ids = Column(Text, nullable=True)             # JSON array of accepted update IDs
    status = Column(String, default="draft")               # draft / applied
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)


class DBAgentCache(Base):
    """Cached agent results, encrypted at rest."""
    __tablename__ = "agent_cache"

    id = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    cache_key = Column(String, nullable=False, index=True)
    agent_type = Column(String, nullable=False)
    model_used = Column(String, nullable=False)
    result_data = Column(Text, nullable=False)
    params_summary = Column(String, nullable=True)
    created_at = Column(DateTime, nullable=False)
    expires_at = Column(DateTime, nullable=True, index=True)

    documents = relationship("DBDocument", secondary=agent_cache_documents, lazy="selectin")


class DBCodeSession(Base):
    """A code chat session for iterative code generation."""
    __tablename__ = "code_sessions"

    id = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    title = Column(String, nullable=True)
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False, index=True)

    documents = relationship("DBDocument", secondary=code_session_documents, lazy="selectin")
    messages = relationship("DBCodeMessage", back_populates="session",
                            order_by="DBCodeMessage.created_at",
                            cascade="all, delete-orphan")


class DBCodeMessage(Base):
    """A message in a code chat session."""
    __tablename__ = "code_messages"

    id = Column(String, primary_key=True)
    session_id = Column(String, ForeignKey("code_sessions.id", ondelete="CASCADE"), nullable=False, index=True)
    role = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, nullable=False)

    session = relationship("DBCodeSession", back_populates="messages")


class DBPoster(Base):
    """AI-generated poster with editable HTML content."""
    __tablename__ = "posters"

    id = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    title = Column(String, nullable=False)                               # encrypted
    prompt_history = Column(Text, nullable=False)                        # encrypted JSON list of prompts
    html_content = Column(Text, nullable=False)                          # encrypted — full HTML/CSS poster
    thumbnail = Column(Text, nullable=True)                              # optional base64 preview
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)


# ---------------------------------------------------------------------------
# Procedure Builder
# ---------------------------------------------------------------------------

class DBProcedureSession(Base):
    """An AI-guided procedure writing session."""
    __tablename__ = "procedure_sessions"

    id = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    title = Column(String, nullable=True)
    source_doc_id = Column(String, ForeignKey("documents.id", ondelete="SET NULL"), nullable=True)
    status = Column(String, nullable=False, default="gathering")  # gathering, drafting, complete
    gathered_data = Column(Text, nullable=True)     # encrypted JSON — answers collected from user
    style_config = Column(Text, nullable=True)      # encrypted JSON — writing style rules
    template_config = Column(Text, nullable=True)   # encrypted JSON — business template structure
    output_content = Column(Text, nullable=True)    # encrypted — final generated procedure content
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False, index=True)

    messages = relationship("DBProcedureMessage", back_populates="session",
                            order_by="DBProcedureMessage.created_at",
                            cascade="all, delete-orphan")


class DBProcedureMessage(Base):
    """A message in a procedure building conversation."""
    __tablename__ = "procedure_messages"

    id = Column(String, primary_key=True)
    session_id = Column(String, ForeignKey("procedure_sessions.id", ondelete="CASCADE"), nullable=False, index=True)
    role = Column(String, nullable=False)           # "user" or "assistant"
    content = Column(Text, nullable=False)          # encrypted
    created_at = Column(DateTime, nullable=False)

    session = relationship("DBProcedureSession", back_populates="messages")
