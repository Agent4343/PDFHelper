"""Baseline — represents the pre-Alembic schema.

For existing databases: stamp with `alembic stamp 001` to mark as current.
For fresh databases: creates all tables from scratch.

Revision ID: 001
"""
from alembic import op
import sqlalchemy as sa

revision = "001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing = inspector.get_table_names()

    if "documents" in existing:
        return

    op.create_table(
        "users",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("username", sa.String(), nullable=False, unique=True),
        sa.Column("password_hash", sa.String(), nullable=False),
        sa.Column("is_admin", sa.Boolean(), default=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_users_username", "users", ["username"])

    op.create_table(
        "documents",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("filename", sa.String(), nullable=False),
        sa.Column("filepath", sa.String(), nullable=False),
        sa.Column("page_count", sa.Integer(), nullable=False),
        sa.Column("text_content", sa.Text(), nullable=False),
        sa.Column("content_hash", sa.String(), nullable=True),
        sa.Column("uploaded_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_documents_content_hash", "documents", ["content_hash"])
    op.create_index("ix_documents_uploaded_at", "documents", ["uploaded_at"])

    op.create_table(
        "search_results",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("search_terms", sa.Text(), nullable=True),
        sa.Column("ai_query", sa.String(), nullable=True),
        sa.Column("keyword_results", sa.Text(), nullable=False),
        sa.Column("ai_results", sa.Text(), nullable=False),
        sa.Column("total_keyword_matches", sa.Integer(), default=0),
        sa.Column("total_ai_findings", sa.Integer(), default=0),
        sa.Column("flagged_for_review", sa.Integer(), default=0),
        sa.Column("searched_at", sa.DateTime(), nullable=False),
    )

    op.create_table(
        "analysis_reports",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("doc_ids", sa.Text(), nullable=False),
        sa.Column("compliance_context", sa.Text(), nullable=True),
        sa.Column("report_data", sa.Text(), nullable=False),
        sa.Column("documents_analyzed", sa.Integer(), default=0),
        sa.Column("total_issues", sa.Integer(), default=0),
        sa.Column("critical_issues", sa.Integer(), default=0),
        sa.Column("risk_level", sa.String(), nullable=True),
        sa.Column("cache_key", sa.String(), nullable=True),
        sa.Column("analyzed_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_analysis_reports_cache_key", "analysis_reports", ["cache_key"])

    op.create_table(
        "chat_sessions",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("user_id", sa.String(), sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
        sa.Column("title", sa.String(), nullable=True),
        sa.Column("doc_ids", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_chat_sessions_user_id", "chat_sessions", ["user_id"])
    op.create_index("ix_chat_sessions_updated_at", "chat_sessions", ["updated_at"])

    op.create_table(
        "chat_messages",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("session_id", sa.String(), sa.ForeignKey("chat_sessions.id", ondelete="CASCADE"), nullable=False),
        sa.Column("role", sa.String(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_chat_messages_session_id", "chat_messages", ["session_id"])

    op.create_table(
        "drawings",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("filename", sa.String(), nullable=False),
        sa.Column("filepath", sa.String(), nullable=False),
        sa.Column("title", sa.String(), nullable=True),
        sa.Column("drawing_number", sa.String(), nullable=True),
        sa.Column("equipment_tags", sa.Text(), nullable=True),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("page_count", sa.Integer(), default=1),
        sa.Column("text_content", sa.Text(), nullable=True),
        sa.Column("uploaded_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_drawings_uploaded_at", "drawings", ["uploaded_at"])

    op.create_table(
        "isolation_packages",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("cert_number", sa.String(), nullable=False, unique=True),
        sa.Column("equipment_tag", sa.String(), nullable=False),
        sa.Column("work_description", sa.String(), nullable=False),
        sa.Column("work_type", sa.String(), nullable=False),
        sa.Column("fluid_service", sa.String(), nullable=True),
        sa.Column("facility", sa.String(), nullable=True),
        sa.Column("regime", sa.String(), nullable=True),
        sa.Column("special_requirements", sa.Text(), nullable=True),
        sa.Column("drawing_ids", sa.Text(), nullable=False),
        sa.Column("package_data", sa.Text(), nullable=False),
        sa.Column("hazard_classification", sa.String(), nullable=True),
        sa.Column("valve_count", sa.Integer(), default=0),
        sa.Column("blind_count", sa.Integer(), default=0),
        sa.Column("step_count", sa.Integer(), default=0),
        sa.Column("energy_source_count", sa.Integer(), default=0),
        sa.Column("status", sa.String(), default="draft"),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_isolation_packages_status", "isolation_packages", ["status"])
    op.create_index("ix_isolation_packages_created_at", "isolation_packages", ["created_at"])

    op.create_table(
        "update_sessions",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("doc_id", sa.String(), sa.ForeignKey("documents.id", ondelete="CASCADE"), nullable=False),
        sa.Column("user_id", sa.String(), sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
        sa.Column("title", sa.String(), nullable=True),
        sa.Column("regulation_query", sa.Text(), nullable=True),
        sa.Column("regulation_results", sa.Text(), nullable=True),
        sa.Column("updates_json", sa.Text(), nullable=True),
        sa.Column("accepted_ids", sa.Text(), nullable=True),
        sa.Column("status", sa.String(), default="draft"),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_update_sessions_doc_id", "update_sessions", ["doc_id"])
    op.create_index("ix_update_sessions_user_id", "update_sessions", ["user_id"])

    op.create_table(
        "agent_cache",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("user_id", sa.String(), sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
        sa.Column("cache_key", sa.String(), nullable=False),
        sa.Column("agent_type", sa.String(), nullable=False),
        sa.Column("model_used", sa.String(), nullable=False),
        sa.Column("result_data", sa.Text(), nullable=False),
        sa.Column("doc_ids", sa.Text(), nullable=False),
        sa.Column("params_summary", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("expires_at", sa.DateTime(), nullable=True),
    )
    op.create_index("ix_agent_cache_cache_key", "agent_cache", ["cache_key"])
    op.create_index("ix_agent_cache_user_id", "agent_cache", ["user_id"])
    op.create_index("ix_agent_cache_expires_at", "agent_cache", ["expires_at"])

    op.create_table(
        "code_sessions",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("user_id", sa.String(), sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
        sa.Column("title", sa.String(), nullable=True),
        sa.Column("doc_ids", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_code_sessions_user_id", "code_sessions", ["user_id"])
    op.create_index("ix_code_sessions_updated_at", "code_sessions", ["updated_at"])

    op.create_table(
        "code_messages",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("session_id", sa.String(), sa.ForeignKey("code_sessions.id", ondelete="CASCADE"), nullable=False),
        sa.Column("role", sa.String(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_code_messages_session_id", "code_messages", ["session_id"])

    op.create_table(
        "posters",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("user_id", sa.String(), sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
        sa.Column("title", sa.String(), nullable=False),
        sa.Column("prompt_history", sa.Text(), nullable=False),
        sa.Column("html_content", sa.Text(), nullable=False),
        sa.Column("thumbnail", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_posters_user_id", "posters", ["user_id"])


def downgrade() -> None:
    for t in ["code_messages", "code_sessions", "agent_cache", "update_sessions",
              "isolation_packages", "drawings", "chat_messages", "chat_sessions",
              "analysis_reports", "search_results", "documents", "posters", "users"]:
        op.drop_table(t)
