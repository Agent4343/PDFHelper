"""Add facility/category to procedure sessions and app_settings table.

Revision ID: 004
Revises: 003
"""
from alembic import op
import sqlalchemy as sa

revision = "004"
down_revision = "003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("procedure_sessions", sa.Column("facility", sa.String(), nullable=True))
    op.add_column("procedure_sessions", sa.Column("category", sa.String(), nullable=True))

    op.create_table(
        "app_settings",
        sa.Column("key", sa.String(), primary_key=True),
        sa.Column("value", sa.Text(), nullable=True),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("app_settings")
    op.drop_column("procedure_sessions", "category")
    op.drop_column("procedure_sessions", "facility")
