"""
Audit logging for PDFHelper.

Logs all sensitive operations (uploads, searches, deletes, auth failures)
to a structured log that can be reviewed later.
"""

import logging
import os
import sys
from datetime import datetime, timezone


def setup_audit_logger() -> logging.Logger:
    """Create and return the audit logger."""
    logger = logging.getLogger("pdfhelper.audit")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s | AUDIT | %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%SZ",
        ))
        logger.addHandler(handler)

        # Also log to file if configured
        log_file = os.getenv("AUDIT_LOG_FILE")
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(logging.Formatter(
                "%(asctime)s | AUDIT | %(message)s",
                datefmt="%Y-%m-%dT%H:%M:%SZ",
            ))
            logger.addHandler(file_handler)

    return logger


audit_log = setup_audit_logger()


def log_upload(client_ip: str, filename: str, doc_id: str, pages: int):
    audit_log.info(
        "UPLOAD | ip=%s | file=%s | doc_id=%s | pages=%d",
        client_ip, filename, doc_id, pages,
    )


def log_search(client_ip: str, search_id: str, terms: list | None, ai_query: str | None,
               docs_searched: int, results: int, flagged: int):
    audit_log.info(
        "SEARCH | ip=%s | search_id=%s | terms=%s | ai_query=%s | docs=%d | results=%d | flagged=%d",
        client_ip, search_id, terms, ai_query, docs_searched, results, flagged,
    )


def log_delete(client_ip: str, doc_id: str, filename: str):
    audit_log.info(
        "DELETE | ip=%s | doc_id=%s | file=%s",
        client_ip, doc_id, filename,
    )


def log_auth_failure(client_ip: str, path: str):
    audit_log.warning(
        "AUTH_FAILURE | ip=%s | path=%s",
        client_ip, path,
    )


def log_access(client_ip: str, method: str, path: str, status: int):
    audit_log.info(
        "ACCESS | ip=%s | %s %s | status=%d",
        client_ip, method, path, status,
    )
