"""Add is_scta and proc_type columns to procedure_sessions.

Revision ID: 005
Revises: 004
"""
from alembic import op
import sqlalchemy as sa

revision = "005"
down_revision = "004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("procedure_sessions", sa.Column("is_scta", sa.String(), nullable=True))
    op.add_column("procedure_sessions", sa.Column("proc_type", sa.String(), nullable=True))


def downgrade() -> None:
    op.drop_column("procedure_sessions", "proc_type")
    op.drop_column("procedure_sessions", "is_scta")
