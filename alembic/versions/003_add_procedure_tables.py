"""Add procedure builder tables.

Creates: procedure_sessions, procedure_messages

Revision ID: 003
Revises: 002
"""
from alembic import op
import sqlalchemy as sa

revision = "003"
down_revision = "002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "procedure_sessions",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("user_id", sa.String(), sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
        sa.Column("title", sa.String(), nullable=True),
        sa.Column("source_doc_id", sa.String(), sa.ForeignKey("documents.id", ondelete="SET NULL"), nullable=True),
        sa.Column("status", sa.String(), nullable=False, server_default="gathering"),
        sa.Column("gathered_data", sa.Text(), nullable=True),
        sa.Column("style_config", sa.Text(), nullable=True),
        sa.Column("template_config", sa.Text(), nullable=True),
        sa.Column("output_content", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_procedure_sessions_user_id", "procedure_sessions", ["user_id"])
    op.create_index("ix_procedure_sessions_updated_at", "procedure_sessions", ["updated_at"])

    op.create_table(
        "procedure_messages",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("session_id", sa.String(), sa.ForeignKey("procedure_sessions.id", ondelete="CASCADE"), nullable=False),
        sa.Column("role", sa.String(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_procedure_messages_session_id", "procedure_messages", ["session_id"])


def downgrade() -> None:
    op.drop_table("procedure_messages")
    op.drop_table("procedure_sessions")
