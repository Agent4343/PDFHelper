"""Add junction tables for document references and migrate JSON doc_ids.

Creates: chat_session_documents, code_session_documents,
         analysis_report_documents, agent_cache_documents

Migrates existing JSON doc_ids data into junction rows, then drops
the old doc_ids columns.

Revision ID: 002
Revises: 001
"""
import json

from alembic import op
import sqlalchemy as sa

revision = "002"
down_revision = "001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "chat_session_documents",
        sa.Column("session_id", sa.String(), sa.ForeignKey("chat_sessions.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("document_id", sa.String(), sa.ForeignKey("documents.id", ondelete="CASCADE"), primary_key=True),
    )
    op.create_table(
        "code_session_documents",
        sa.Column("session_id", sa.String(), sa.ForeignKey("code_sessions.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("document_id", sa.String(), sa.ForeignKey("documents.id", ondelete="CASCADE"), primary_key=True),
    )
    op.create_table(
        "analysis_report_documents",
        sa.Column("report_id", sa.String(), sa.ForeignKey("analysis_reports.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("document_id", sa.String(), sa.ForeignKey("documents.id", ondelete="CASCADE"), primary_key=True),
    )
    op.create_table(
        "agent_cache_documents",
        sa.Column("cache_id", sa.String(), sa.ForeignKey("agent_cache.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("document_id", sa.String(), sa.ForeignKey("documents.id", ondelete="CASCADE"), primary_key=True),
    )

    bind = op.get_bind()

    # Migrate chat_sessions.doc_ids
    rows = bind.execute(sa.text("SELECT id, doc_ids FROM chat_sessions WHERE doc_ids IS NOT NULL")).fetchall()
    for row in rows:
        try:
            ids = json.loads(row[1])
        except (json.JSONDecodeError, TypeError):
            continue
        for did in ids:
            if did:
                bind.execute(sa.text(
                    "INSERT OR IGNORE INTO chat_session_documents (session_id, document_id) VALUES (:sid, :did)"
                    if "sqlite" in bind.dialect.name else
                    "INSERT INTO chat_session_documents (session_id, document_id) VALUES (:sid, :did) ON CONFLICT DO NOTHING"
                ), {"sid": row[0], "did": did})

    # Migrate code_sessions.doc_ids
    rows = bind.execute(sa.text("SELECT id, doc_ids FROM code_sessions WHERE doc_ids IS NOT NULL")).fetchall()
    for row in rows:
        try:
            ids = json.loads(row[1])
        except (json.JSONDecodeError, TypeError):
            continue
        for did in ids:
            if did:
                bind.execute(sa.text(
                    "INSERT OR IGNORE INTO code_session_documents (session_id, document_id) VALUES (:sid, :did)"
                    if "sqlite" in bind.dialect.name else
                    "INSERT INTO code_session_documents (session_id, document_id) VALUES (:sid, :did) ON CONFLICT DO NOTHING"
                ), {"sid": row[0], "did": did})

    # Migrate analysis_reports.doc_ids
    rows = bind.execute(sa.text("SELECT id, doc_ids FROM analysis_reports WHERE doc_ids IS NOT NULL")).fetchall()
    for row in rows:
        try:
            ids = json.loads(row[1])
        except (json.JSONDecodeError, TypeError):
            continue
        for did in ids:
            if did:
                bind.execute(sa.text(
                    "INSERT OR IGNORE INTO analysis_report_documents (report_id, document_id) VALUES (:rid, :did)"
                    if "sqlite" in bind.dialect.name else
                    "INSERT INTO analysis_report_documents (report_id, document_id) VALUES (:rid, :did) ON CONFLICT DO NOTHING"
                ), {"rid": row[0], "did": did})

    # Migrate agent_cache.doc_ids
    rows = bind.execute(sa.text("SELECT id, doc_ids FROM agent_cache WHERE doc_ids IS NOT NULL")).fetchall()
    for row in rows:
        try:
            ids = json.loads(row[1])
        except (json.JSONDecodeError, TypeError):
            continue
        for did in ids:
            if did:
                bind.execute(sa.text(
                    "INSERT OR IGNORE INTO agent_cache_documents (cache_id, document_id) VALUES (:cid, :did)"
                    if "sqlite" in bind.dialect.name else
                    "INSERT INTO agent_cache_documents (cache_id, document_id) VALUES (:cid, :did) ON CONFLICT DO NOTHING"
                ), {"cid": row[0], "did": did})

    # Drop old doc_ids columns
    # SQLite doesn't support DROP COLUMN before 3.35; use batch mode
    with op.batch_alter_table("chat_sessions") as batch_op:
        batch_op.drop_column("doc_ids")
    with op.batch_alter_table("code_sessions") as batch_op:
        batch_op.drop_column("doc_ids")
    with op.batch_alter_table("analysis_reports") as batch_op:
        batch_op.drop_column("doc_ids")
    with op.batch_alter_table("agent_cache") as batch_op:
        batch_op.drop_column("doc_ids")


def downgrade() -> None:
    with op.batch_alter_table("chat_sessions") as batch_op:
        batch_op.add_column(sa.Column("doc_ids", sa.Text(), nullable=True))
    with op.batch_alter_table("code_sessions") as batch_op:
        batch_op.add_column(sa.Column("doc_ids", sa.Text(), nullable=True))
    with op.batch_alter_table("analysis_reports") as batch_op:
        batch_op.add_column(sa.Column("doc_ids", sa.Text(), nullable=True))
    with op.batch_alter_table("agent_cache") as batch_op:
        batch_op.add_column(sa.Column("doc_ids", sa.Text(), nullable=True))

    bind = op.get_bind()

    # Restore JSON doc_ids from junction tables
    for table, fk_col, src_table in [
        ("chat_sessions", "session_id", "chat_session_documents"),
        ("code_sessions", "session_id", "code_session_documents"),
        ("analysis_reports", "report_id", "analysis_report_documents"),
        ("agent_cache", "cache_id", "agent_cache_documents"),
    ]:
        rows = bind.execute(sa.text(f"SELECT DISTINCT {fk_col} FROM {src_table}")).fetchall()
        for (parent_id,) in rows:
            doc_rows = bind.execute(sa.text(
                f"SELECT document_id FROM {src_table} WHERE {fk_col} = :pid"
            ), {"pid": parent_id}).fetchall()
            doc_ids_json = json.dumps([r[0] for r in doc_rows])
            bind.execute(sa.text(
                f"UPDATE {table} SET doc_ids = :val WHERE id = :pid"
            ), {"val": doc_ids_json, "pid": parent_id})

    op.drop_table("agent_cache_documents")
    op.drop_table("analysis_report_documents")
    op.drop_table("code_session_documents")
    op.drop_table("chat_session_documents")
