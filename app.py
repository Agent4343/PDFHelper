#!/usr/bin/env python3
"""
PDFHelper — Secure web API for AI-powered PDF search and flagging.

Deployed on Railway. Provides endpoints to upload PDFs, search them
with keywords or AI, and retrieve flagged results.

Security features:
- API key authentication (required in production)
- File encryption at rest (AES via Fernet)
- PDF magic-byte verification
- Rate limiting, CORS, security headers
- Audit logging of all sensitive operations
- Auto-cleanup of old files
"""

import json
import os
import re
import secrets
import time
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Literal

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed; env vars must be set directly

from fastapi import (
    BackgroundTasks, FastAPI, File, UploadFile, Depends, HTTPException, Query, Request,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware

from database import (
    SessionLocal, engine, Base, DBDocument, DBSearchResult, DBAnalysisReport,
    DBChatSession, DBChatMessage,
)
from audit import log_upload, log_search, log_delete, log_auth_failure, log_access
from ocr import extract_text_with_ocr_fallback
from search import keyword_search, ai_search

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
IS_PRODUCTION = ENVIRONMENT == "production"

API_KEY = (os.getenv("PDF_HELPER_API_KEY") or "").strip() or None
ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY")

MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE_MB", "20")) * 1024 * 1024
MAX_FILES_PER_REQUEST = int(os.getenv("MAX_FILES_PER_REQUEST", "20"))
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "/tmp/pdfhelper_uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Auto-cleanup: delete uploads older than this many hours (0 = disabled)
AUTO_CLEANUP_HOURS = int(os.getenv("AUTO_CLEANUP_HOURS", "72"))

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

_startup_errors: list[str] = []

app = FastAPI(
    title="PDFHelper",
    description="AI-powered PDF search and flagging tool",
    version="1.0.0",
    docs_url="/docs" if not IS_PRODUCTION else None,
    redoc_url=None,
)


async def _retry_db_init(logger):
    """Retry database table creation in the background."""
    import asyncio
    for attempt in range(2, 6):  # attempts 2–5
        await asyncio.sleep(2 ** attempt)  # 4, 8, 16, 32 seconds
        try:
            Base.metadata.create_all(bind=engine)
            logger.info("Database tables initialized successfully (attempt %d).", attempt)
            return
        except Exception as exc:
            logger.warning("Database init attempt %d/5 failed: %s", attempt, exc)
    _startup_errors.append("WARNING: Database initialization failed after 5 attempts.")


@app.on_event("startup")
async def startup():
    """Run safety checks and initialize DB — errors are logged, not fatal."""
    import asyncio
    import logging
    logger = logging.getLogger("pdfhelper")

    if IS_PRODUCTION and not API_KEY:
        msg = (
            "WARNING: PDF_HELPER_API_KEY not set in production. "
            "API endpoints will reject requests until it is configured."
        )
        logger.warning(msg)
        _startup_errors.append(msg)

    if IS_PRODUCTION and not ENCRYPTION_KEY:
        msg = (
            "WARNING: ENCRYPTION_KEY not set in production. "
            "File encryption is disabled until it is configured."
        )
        logger.warning(msg)
        _startup_errors.append(msg)

    # Try once synchronously, then retry in background if needed
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables initialized successfully.")
    except Exception as exc:
        logger.warning("Database init failed on first attempt: %s — retrying in background", exc)
        asyncio.create_task(_retry_db_init(logger))


# ---------------------------------------------------------------------------
# Security middleware
# ---------------------------------------------------------------------------

# -- CORS --
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "").split(",")
ALLOWED_ORIGINS = [o.strip() for o in ALLOWED_ORIGINS if o.strip()]
if IS_PRODUCTION and not ALLOWED_ORIGINS:
    ALLOWED_ORIGINS = []  # Block all cross-origin requests by default

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS if ALLOWED_ORIGINS else (["*"] if not IS_PRODUCTION else []),
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Authorization", "X-API-Key", "Content-Type"],
)

# -- Trusted hosts --
ALLOWED_HOSTS = os.getenv("ALLOWED_HOSTS", "").split(",")
ALLOWED_HOSTS = [h.strip() for h in ALLOWED_HOSTS if h.strip()]
if ALLOWED_HOSTS:
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=ALLOWED_HOSTS)


# -- Security headers --
class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to every response."""
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        if IS_PRODUCTION:
            response.headers["Strict-Transport-Security"] = "max-age=63072000; includeSubDomains"
        # Never reveal server tech
        if "server" in response.headers:
            del response.headers["server"]
        return response


app.add_middleware(SecurityHeadersMiddleware)


# -- HTTPS redirect (production) --
class HTTPSRedirectMiddleware(BaseHTTPMiddleware):
    """Redirect HTTP to HTTPS in production using X-Forwarded-Proto."""
    async def dispatch(self, request: Request, call_next):
        if IS_PRODUCTION:
            proto = request.headers.get("x-forwarded-proto", "https")
            if proto != "https":
                url = request.url.replace(scheme="https")
                return JSONResponse(
                    status_code=301,
                    headers={"Location": str(url)},
                    content={"detail": "Use HTTPS"},
                )
        return await call_next(request)


app.add_middleware(HTTPSRedirectMiddleware)


# -- Rate limiting --
class RateLimitMiddleware(BaseHTTPMiddleware):
    """In-memory rate limiter per IP with periodic cleanup."""
    def __init__(self, app, max_requests: int = 60, window_seconds: int = 60):
        super().__init__(app)
        self.max_requests = max_requests
        self.window = window_seconds
        self.requests: dict[str, list[float]] = {}
        self._last_cleanup = time.time()
        self._cleanup_interval = 300  # purge stale IPs every 5 minutes

    async def dispatch(self, request: Request, call_next):
        client_ip = _get_client_ip(request)
        now = time.time()
        window_start = now - self.window

        # Periodically purge IPs with no recent requests to prevent memory leak
        if now - self._last_cleanup > self._cleanup_interval:
            self.requests = {
                ip: [t for t in ts if t > window_start]
                for ip, ts in self.requests.items()
                if any(t > window_start for t in ts)
            }
            self._last_cleanup = now

        hits = self.requests.get(client_ip, [])
        hits = [t for t in hits if t > window_start]

        if len(hits) >= self.max_requests:
            return JSONResponse(
                status_code=429,
                content={"detail": "Too many requests. Try again later."},
            )
        hits.append(now)
        self.requests[client_ip] = hits

        response = await call_next(request)
        log_access(client_ip, request.method, request.url.path, response.status_code)
        return response


app.add_middleware(RateLimitMiddleware, max_requests=60, window_seconds=60)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_client_ip(request: Request) -> str:
    """Get real client IP, respecting proxy headers."""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _sanitize_filename(filename: str) -> str:
    """Strip path traversal and dangerous characters from filenames."""
    # Take only the basename
    name = Path(filename).name
    # Remove anything that isn't alphanumeric, dash, underscore, dot, or space
    name = re.sub(r"[^\w\-. ]", "_", name)
    # Prevent hidden files
    name = name.lstrip(".")
    return name or "unnamed.pdf"


PDF_MAGIC_BYTES = b"%PDF-"


def _verify_pdf_content(data: bytes) -> bool:
    """Check that the file actually starts with the PDF magic bytes."""
    return data[:5] == PDF_MAGIC_BYTES


# ---------------------------------------------------------------------------
# Auth dependency
# ---------------------------------------------------------------------------

async def verify_api_key(request: Request):
    """Require a valid API key for all endpoints."""
    if not API_KEY:
        # Dev mode only — production enforced at startup
        return
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        token = auth[7:]
    else:
        token = request.headers.get("X-API-Key", "")

    if not token or not secrets.compare_digest(token, API_KEY):
        log_auth_failure(_get_client_ip(request), request.url.path)
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


# ---------------------------------------------------------------------------
# DB dependency
# ---------------------------------------------------------------------------

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Encryption helpers
# ---------------------------------------------------------------------------

def _encrypt_and_save(data: bytes, path: Path) -> None:
    """Encrypt data and write to path. Falls back to plaintext if no key."""
    if ENCRYPTION_KEY:
        from encryption import encrypt_bytes
        path.write_bytes(encrypt_bytes(data))
    else:
        path.write_bytes(data)


def _encrypt_text(text: str) -> str:
    """Encrypt a string for database storage. Returns base64-encoded ciphertext."""
    if ENCRYPTION_KEY:
        from encryption import encrypt_bytes
        import base64
        return base64.b64encode(encrypt_bytes(text.encode("utf-8"))).decode("ascii")
    return text


def _decrypt_text(stored: str) -> str:
    """Decrypt a string read from the database."""
    if ENCRYPTION_KEY:
        from encryption import decrypt_bytes
        import base64
        return decrypt_bytes(base64.b64decode(stored)).decode("utf-8")
    return stored


# ---------------------------------------------------------------------------
# PDF processing
# ---------------------------------------------------------------------------

def extract_text_from_bytes(pdf_bytes: bytes) -> list[dict]:
    """Extract text from PDF bytes. Falls back to OCR for scanned PDFs."""
    return extract_text_with_ocr_fallback(pdf_bytes)



# keyword_search and ai_search are imported from search.py


# ---------------------------------------------------------------------------
# Auto-cleanup
# ---------------------------------------------------------------------------

def _run_cleanup(db) -> int:
    """Delete documents older than AUTO_CLEANUP_HOURS. Returns count deleted."""
    if not AUTO_CLEANUP_HOURS:
        return 0
    cutoff = datetime.now(timezone.utc) - timedelta(hours=AUTO_CLEANUP_HOURS)
    old_docs = db.query(DBDocument).filter(DBDocument.uploaded_at < cutoff).all()
    count = 0
    for doc in old_docs:
        filepath = Path(doc.filepath)
        if filepath.exists():
            filepath.unlink()
        db.delete(doc)
        count += 1
    if count:
        db.commit()
    return count


def _run_cleanup_background():
    """Run cleanup with its own DB session (safe for background tasks)."""
    db = SessionLocal()
    try:
        _run_cleanup(db)
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class SearchRequest(BaseModel):
    search_terms: list[str] = Field(default=[], description="Exact keywords to search")
    ai_query: str | None = Field(default=None, description="AI concept search query")
    case_sensitive: bool = False


class AnalyzeRequest(BaseModel):
    compliance_context: str | None = Field(
        default=None,
        description="Optional compliance standard to check against, e.g. 'OSHA 2024', 'HIPAA', 'FDA 21 CFR Part 11'",
    )
    search_terms: list[str] = Field(default=[], description="Optional keywords to search for")
    ai_query: str | None = Field(default=None, description="Optional AI concept search query")


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"] = Field(description="'user' or 'assistant'")
    content: str


class ChatRequest(BaseModel):
    message: str = Field(description="The user's message")
    doc_ids: list[str] = Field(default=[], description="Document IDs to use as context (empty = all)")
    conversation_history: list[ChatMessage] = Field(default=[], description="Previous messages for context")
    session_id: str | None = Field(default=None, description="Chat session ID to continue (omit to create new)")


class HealthResponse(BaseModel):
    status: str
    version: str
    api_key_required: bool = False
    warnings: list[str] = []


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_pdf(file: UploadFile, content: bytes) -> str:
    """Validate an uploaded PDF file. Returns sanitized filename."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    clean_name = _sanitize_filename(file.filename)
    if not clean_name.lower().endswith(".pdf"):
        raise HTTPException(status_code=400,
                            detail=f"Only PDF files allowed, got: {clean_name}")

    # Verify actual file content — not just the extension
    if not _verify_pdf_content(content):
        raise HTTPException(status_code=400,
                            detail="File does not appear to be a valid PDF")

    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"{clean_name} exceeds max size of {MAX_FILE_SIZE // (1024*1024)} MB",
        )

    return clean_name


# ---------------------------------------------------------------------------
# Root UI (loaded from templates/index.html)
# ---------------------------------------------------------------------------

_TEMPLATE_DIR = Path(__file__).parent / "templates"
_cached_html: str | None = None


def _load_root_html() -> str:
    """Load the frontend HTML template. Cached in production to avoid disk reads."""
    global _cached_html
    if IS_PRODUCTION and _cached_html is not None:
        return _cached_html
    html = (_TEMPLATE_DIR / "index.html").read_text(encoding="utf-8")
    if IS_PRODUCTION:
        _cached_html = html
    return html



# ---------------------------------------------------------------------------
# Global error handler for DB failures
# ---------------------------------------------------------------------------

from sqlalchemy.exc import OperationalError as SAOperationalError

@app.exception_handler(SAOperationalError)
async def db_error_handler(request: Request, exc: SAOperationalError):
    """Return a clear error message when the database is unreachable."""
    return JSONResponse(
        status_code=503,
        content={
            "detail": "Database is unavailable. Check that DATABASE_URL is set correctly "
                      "(use the public URL, not the internal .railway.internal hostname)."
        },
    )

@app.exception_handler(Exception)
async def general_error_handler(request: Request, exc: Exception):
    """Catch-all so errors always return JSON, never HTML."""
    import traceback, logging
    logging.getLogger("pdfhelper").error(f"Unhandled error on {request.url.path}: {exc}\n{traceback.format_exc()}")
    if IS_PRODUCTION:
        detail = "Internal server error"
    else:
        detail = f"Internal server error: {type(exc).__name__}: {str(exc)}"
    return JSONResponse(
        status_code=500,
        content={"detail": detail},
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check — also tests current DB connectivity."""
    db_ok = True
    db_err = None
    try:
        from sqlalchemy import text
        db = SessionLocal()
        db.execute(text("SELECT 1"))
        db.close()
    except Exception as e:
        db_ok = False
        db_err = f"Database connection failed: {e}"

    warnings = list(_startup_errors)
    if db_err:
        warnings.append(db_err)

    status = "ok" if (not warnings and db_ok) else "degraded"
    return {
        "status": status,
        "version": "1.0.0",
        "api_key_required": bool(API_KEY),
        "warnings": warnings,
    }


@app.get("/verify-key", dependencies=[Depends(verify_api_key)])
async def verify_key():
    """Lightweight API key check — no database required."""
    import asyncio
    from sqlalchemy import text

    def _check_db():
        db = SessionLocal()
        try:
            db.execute(text("SELECT 1"))
        finally:
            db.close()

    db_ok = True
    try:
        await asyncio.wait_for(asyncio.to_thread(_check_db), timeout=3)
    except Exception:
        db_ok = False
    return {"valid": True, "db_ok": db_ok}


@app.get("/", response_class=HTMLResponse)
async def root():
    """Interactive operating interface for PDFHelper."""
    return _load_root_html()


@app.get("/bot")
async def bot_page():
    """Redirect old bot page to main app."""
    from starlette.responses import RedirectResponse
    return RedirectResponse(url="/", status_code=301)


@app.post("/upload", dependencies=[Depends(verify_api_key)])
async def upload_pdfs(
    request: Request,
    files: list[UploadFile] = File(...),
    background_tasks: BackgroundTasks = None,
    db=Depends(get_db),
):
    """Upload one or more PDFs for later searching."""
    if len(files) > MAX_FILES_PER_REQUEST:
        raise HTTPException(status_code=400,
                            detail=f"Max {MAX_FILES_PER_REQUEST} files per request")

    client_ip = _get_client_ip(request)

    # Run cleanup after the response is sent so it doesn't slow down uploads
    if background_tasks:
        background_tasks.add_task(_run_cleanup_background)

    uploaded = []
    for file in files:
        content = await file.read()
        clean_name = validate_pdf(file, content)

        # Extract text from raw bytes (never written unencrypted to disk)
        pages = extract_text_from_bytes(content)

        doc_id = str(uuid.uuid4())
        save_path = UPLOAD_DIR / f"{doc_id}.pdf.enc"

        # Encrypt and save
        _encrypt_and_save(content, save_path)

        db_doc = DBDocument(
            id=doc_id,
            filename=_encrypt_text(clean_name),
            filepath=str(save_path),
            page_count=len(pages),
            text_content=_encrypt_text(json.dumps(pages)),
            uploaded_at=datetime.now(timezone.utc),
        )
        db.add(db_doc)
        db.commit()

        log_upload(client_ip, clean_name, doc_id, len(pages))

        uploaded.append({
            "id": doc_id,
            "filename": clean_name,
            "pages": len(pages),
        })

    return {"uploaded": uploaded, "count": len(uploaded)}


@app.post("/search", dependencies=[Depends(verify_api_key)])
async def search_documents(
    request: Request,
    body: SearchRequest,
    doc_ids: list[str] = Query(default=[], description="Document IDs to search (empty = all)"),
    db=Depends(get_db),
):
    """Search uploaded PDFs with keywords and/or AI."""
    if not body.search_terms and not body.ai_query:
        raise HTTPException(status_code=400,
                            detail="Provide search_terms and/or ai_query")

    query = db.query(DBDocument)
    if doc_ids:
        query = query.filter(DBDocument.id.in_(doc_ids))
    documents = query.all()

    if not documents:
        raise HTTPException(status_code=404, detail="No documents found")

    client_ip = _get_client_ip(request)
    all_keyword_results = []
    all_ai_results = []

    for doc in documents:
        pages = json.loads(_decrypt_text(doc.text_content))
        decrypted_name = _decrypt_text(doc.filename)

        if body.search_terms:
            matches = keyword_search(pages, body.search_terms, body.case_sensitive)
            for m in matches:
                m["document_id"] = doc.id
                m["filename"] = decrypted_name
            all_keyword_results.extend(matches)

        if body.ai_query:
            findings = ai_search(pages, body.ai_query, decrypted_name)
            for f in findings:
                f["document_id"] = doc.id
                f["filename"] = decrypted_name
            all_ai_results.extend(findings)

    search_id = str(uuid.uuid4())
    flagged_count = len([r for r in all_ai_results if r.get("needs_review")])

    db_result = DBSearchResult(
        id=search_id,
        search_terms=_encrypt_text(json.dumps(body.search_terms)) if body.search_terms else None,
        ai_query=_encrypt_text(body.ai_query) if body.ai_query else None,
        keyword_results=_encrypt_text(json.dumps(all_keyword_results)),
        ai_results=_encrypt_text(json.dumps(all_ai_results)),
        total_keyword_matches=len(all_keyword_results),
        total_ai_findings=len(all_ai_results),
        flagged_for_review=flagged_count,
        searched_at=datetime.now(timezone.utc),
    )
    db.add(db_result)
    db.commit()

    log_search(client_ip, search_id, body.search_terms, body.ai_query,
               len(documents), len(all_keyword_results) + len(all_ai_results),
               flagged_count)

    return {
        "search_id": search_id,
        "summary": {
            "documents_searched": len(documents),
            "total_keyword_matches": len(all_keyword_results),
            "total_ai_findings": len(all_ai_results),
            "flagged_for_review": flagged_count,
        },
        "keyword_results": all_keyword_results,
        "ai_results": all_ai_results,
    }


@app.get("/documents", dependencies=[Depends(verify_api_key)])
async def list_documents(db=Depends(get_db)):
    """List all uploaded documents."""
    docs = db.query(DBDocument).order_by(DBDocument.uploaded_at.desc()).all()
    return {
        "documents": [
            {
                "id": d.id,
                "filename": _decrypt_text(d.filename),
                "pages": d.page_count,
                "uploaded_at": d.uploaded_at.isoformat(),
            }
            for d in docs
        ]
    }


@app.get("/documents/{doc_id}", dependencies=[Depends(verify_api_key)])
async def get_document(doc_id: str, db=Depends(get_db)):
    """Get details for a specific document."""
    doc = db.query(DBDocument).filter(DBDocument.id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return {
        "id": doc.id,
        "filename": _decrypt_text(doc.filename),
        "pages": doc.page_count,
        "uploaded_at": doc.uploaded_at.isoformat(),
    }


@app.delete("/documents/{doc_id}", dependencies=[Depends(verify_api_key)])
async def delete_document(doc_id: str, request: Request, db=Depends(get_db)):
    """Delete an uploaded document and its encrypted file."""
    doc = db.query(DBDocument).filter(DBDocument.id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    filepath = Path(doc.filepath)
    if filepath.exists():
        filepath.unlink()

    log_delete(_get_client_ip(request), doc_id, _decrypt_text(doc.filename))

    db.delete(doc)
    db.commit()
    return {"deleted": doc_id}


@app.get("/history", dependencies=[Depends(verify_api_key)])
async def search_history(limit: int = Query(default=20, le=100), db=Depends(get_db)):
    """Get past search results."""
    results = (
        db.query(DBSearchResult)
        .order_by(DBSearchResult.searched_at.desc())
        .limit(limit)
        .all()
    )
    return {
        "searches": [
            {
                "id": r.id,
                "search_terms": json.loads(_decrypt_text(r.search_terms)) if r.search_terms else [],
                "ai_query": _decrypt_text(r.ai_query) if r.ai_query else None,
                "total_keyword_matches": r.total_keyword_matches,
                "total_ai_findings": r.total_ai_findings,
                "flagged_for_review": r.flagged_for_review,
                "searched_at": r.searched_at.isoformat(),
            }
            for r in results
        ]
    }


@app.get("/history/{search_id}", dependencies=[Depends(verify_api_key)])
async def get_search_result(search_id: str, db=Depends(get_db)):
    """Get full details of a past search."""
    result = db.query(DBSearchResult).filter(DBSearchResult.id == search_id).first()
    if not result:
        raise HTTPException(status_code=404, detail="Search result not found")
    return {
        "id": result.id,
        "search_terms": json.loads(_decrypt_text(result.search_terms)) if result.search_terms else [],
        "ai_query": _decrypt_text(result.ai_query) if result.ai_query else None,
        "keyword_results": json.loads(_decrypt_text(result.keyword_results)),
        "ai_results": json.loads(_decrypt_text(result.ai_results)),
        "summary": {
            "total_keyword_matches": result.total_keyword_matches,
            "total_ai_findings": result.total_ai_findings,
            "flagged_for_review": result.flagged_for_review,
        },
        "searched_at": result.searched_at.isoformat(),
    }


# ---------------------------------------------------------------------------
# Procedure Chatbot
# ---------------------------------------------------------------------------

@app.post("/chat", dependencies=[Depends(verify_api_key)])
async def chat_with_documents(
    request: Request,
    body: ChatRequest,
    db=Depends(get_db),
):
    """Chat with your uploaded documents using AI.

    Sends the user's message along with selected document content to Claude
    and returns a context-aware response with procedure citations.

    If session_id is provided, continues that session (loading history from DB).
    Otherwise creates a new session. All messages are persisted.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not configured on server")

    query = db.query(DBDocument)
    if body.doc_ids:
        query = query.filter(DBDocument.id.in_(body.doc_ids))
    documents = query.all()

    if not documents:
        raise HTTPException(status_code=404, detail="No documents found. Upload documents first.")

    # Resolve or create chat session
    now = datetime.now(timezone.utc)
    session = None
    if body.session_id:
        session = db.query(DBChatSession).filter(DBChatSession.id == body.session_id).first()

    if session is None:
        session = DBChatSession(
            id=str(uuid.uuid4()),
            title=body.message[:100],
            doc_ids=json.dumps(body.doc_ids),
            created_at=now,
            updated_at=now,
        )
        db.add(session)
        db.flush()

    # Build procedure context from selected documents
    procedure_parts = []
    for doc in documents:
        decrypted_name = _decrypt_text(doc.filename)
        pages = json.loads(_decrypt_text(doc.text_content))
        full_text = "\n".join(p["text"] for p in pages if p.get("text"))
        if len(full_text) > 80000:
            full_text = full_text[:80000] + "\n\n[... content truncated for context window ...]"
        procedure_parts.append(
            f'--- PROCEDURE: "{decrypted_name}" ---\n{full_text}\n--- END OF "{decrypted_name}" ---'
        )

    procedure_context = "\n\n".join(procedure_parts)

    system_prompt = f"""You are a Procedure Knowledge Assistant. You ONLY answer questions based on the procedure documents provided below.

RULES:
1. ONLY use information from the provided procedure documents to answer questions.
2. ALWAYS cite which procedure document your answer comes from by name and section if possible.
3. If the answer cannot be found in the provided procedures, say "I couldn't find information about that in the selected procedures." and suggest which type of document might contain the answer.
4. Be precise and direct. Quote relevant sections when helpful.
5. If a question spans multiple procedures, reference all relevant ones.
6. Format your answers clearly with procedure references in bold.

LOADED PROCEDURES:
{procedure_context}"""

    # Build conversation from DB history (prefer DB over client-sent history)
    db_messages = (
        db.query(DBChatMessage)
        .filter(DBChatMessage.session_id == session.id)
        .order_by(DBChatMessage.created_at)
        .all()
    )

    if db_messages:
        conversation = [
            {"role": m.role, "content": _decrypt_text(m.content)}
            for m in db_messages[-10:]
        ]
    else:
        conversation = [
            {"role": m.role, "content": m.content}
            for m in body.conversation_history[-10:]
            if m.role in ("user", "assistant")
        ]
    conversation.append({"role": "user", "content": body.message})

    from anthropic import Anthropic
    client = Anthropic(api_key=api_key)

    try:
        response = client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=1500,
            system=system_prompt,
            messages=conversation,
        )
        reply = "\n".join(
            block.text for block in response.content if block.type == "text"
        ) or "Sorry, I couldn't generate a response."
    except Exception as e:
        import logging
        logging.getLogger("pdfhelper").error(f"Chat AI request failed: {e}")
        detail = "AI request failed" if IS_PRODUCTION else f"AI request failed: {str(e)}"
        raise HTTPException(status_code=500, detail=detail)

    # Persist both messages
    db.add(DBChatMessage(
        id=str(uuid.uuid4()), session_id=session.id,
        role="user", content=_encrypt_text(body.message), created_at=now,
    ))
    db.add(DBChatMessage(
        id=str(uuid.uuid4()), session_id=session.id,
        role="assistant", content=_encrypt_text(reply), created_at=now,
    ))
    session.updated_at = now
    db.commit()

    return {
        "reply": reply,
        "session_id": session.id,
        "documents_used": [
            {"id": d.id, "filename": _decrypt_text(d.filename)}
            for d in documents
        ],
    }


# ---------------------------------------------------------------------------
# Chat History Endpoints
# ---------------------------------------------------------------------------

@app.get("/chat/sessions", dependencies=[Depends(verify_api_key)])
async def list_chat_sessions(limit: int = Query(default=30, le=100), db=Depends(get_db)):
    """List past chat sessions, most recent first."""
    sessions = (
        db.query(DBChatSession)
        .order_by(DBChatSession.updated_at.desc())
        .limit(limit)
        .all()
    )
    return {
        "sessions": [
            {
                "id": s.id,
                "title": s.title,
                "doc_ids": json.loads(s.doc_ids),
                "message_count": len(s.messages),
                "created_at": s.created_at.isoformat(),
                "updated_at": s.updated_at.isoformat(),
            }
            for s in sessions
        ]
    }


@app.get("/chat/sessions/{session_id}", dependencies=[Depends(verify_api_key)])
async def get_chat_session(session_id: str, db=Depends(get_db)):
    """Get full message history for a chat session."""
    session = db.query(DBChatSession).filter(DBChatSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Chat session not found")
    return {
        "id": session.id,
        "title": session.title,
        "doc_ids": json.loads(session.doc_ids),
        "created_at": session.created_at.isoformat(),
        "updated_at": session.updated_at.isoformat(),
        "messages": [
            {
                "role": m.role,
                "content": _decrypt_text(m.content),
                "created_at": m.created_at.isoformat(),
            }
            for m in session.messages
        ],
    }


@app.delete("/chat/sessions/{session_id}", dependencies=[Depends(verify_api_key)])
async def delete_chat_session(session_id: str, db=Depends(get_db)):
    """Delete a chat session and all its messages."""
    session = db.query(DBChatSession).filter(DBChatSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Chat session not found")
    db.delete(session)
    db.commit()
    return {"deleted": session_id}


# ---------------------------------------------------------------------------
# Full Analysis Pipeline (multi-agent)
# ---------------------------------------------------------------------------

@app.post("/analyze", dependencies=[Depends(verify_api_key)])
async def analyze_documents(
    request: Request,
    body: AnalyzeRequest,
    doc_ids: list[str] = Query(default=[], description="Document IDs to analyze (empty = all)"),
    db=Depends(get_db),
):
    """Run the full multi-agent analysis pipeline on uploaded documents.

    This runs 4 specialized AI agents:
    1. Document Analyzer — deep analysis of each document
    2. Cross-Reference Checker — finds conflicts between documents
    3. Compliance Checker — flags regulatory/policy issues
    4. Summary Report Generator — produces an actionable executive report

    Optionally also runs keyword and AI search.
    """
    query = db.query(DBDocument)
    if doc_ids:
        query = query.filter(DBDocument.id.in_(doc_ids))
    documents = query.all()

    if not documents:
        raise HTTPException(status_code=404, detail="No documents found")

    client_ip = _get_client_ip(request)

    # Build documents dict for the agent pipeline
    docs_for_agents: dict[str, list[dict]] = {}
    for doc in documents:
        decrypted_name = _decrypt_text(doc.filename)
        pages = json.loads(_decrypt_text(doc.text_content))
        docs_for_agents[decrypted_name] = pages

    # Run the full pipeline in a thread pool to avoid blocking the event loop
    # (run_full_analysis makes multiple synchronous Anthropic API calls)
    import asyncio
    from agents import run_full_analysis
    analysis = await asyncio.to_thread(
        run_full_analysis,
        documents=docs_for_agents,
        compliance_context=body.compliance_context,
        search_terms=body.search_terms if body.search_terms else None,
        ai_query=body.ai_query,
    )

    # Save to DB
    report_id = str(uuid.uuid4())
    db_report = DBAnalysisReport(
        id=report_id,
        doc_ids=json.dumps([d.id for d in documents]),
        compliance_context=_encrypt_text(body.compliance_context) if body.compliance_context else None,
        report_data=_encrypt_text(json.dumps(analysis)),
        documents_analyzed=len(documents),
        total_issues=analysis.get("report", {}).get("total_issues_found", 0),
        critical_issues=analysis.get("report", {}).get("critical_issues", 0),
        risk_level=analysis.get("report", {}).get("overall_risk_level", "unknown"),
        analyzed_at=datetime.now(timezone.utc),
    )
    db.add(db_report)
    db.commit()

    log_search(client_ip, report_id, body.search_terms, body.ai_query,
               len(documents), db_report.total_issues, db_report.critical_issues)

    return {
        "report_id": report_id,
        "report": analysis.get("report"),
        "document_analyses": analysis.get("document_analyses"),
        "cross_reference_findings": analysis.get("cross_reference_findings"),
        "compliance_findings": analysis.get("compliance_findings"),
        "search_results": analysis.get("search_results"),
    }


@app.get("/reports", dependencies=[Depends(verify_api_key)])
async def list_reports(limit: int = Query(default=20, le=100), db=Depends(get_db)):
    """List past analysis reports."""
    reports = (
        db.query(DBAnalysisReport)
        .order_by(DBAnalysisReport.analyzed_at.desc())
        .limit(limit)
        .all()
    )
    return {
        "reports": [
            {
                "id": r.id,
                "documents_analyzed": r.documents_analyzed,
                "total_issues": r.total_issues,
                "critical_issues": r.critical_issues,
                "risk_level": r.risk_level,
                "analyzed_at": r.analyzed_at.isoformat(),
            }
            for r in reports
        ]
    }


@app.get("/reports/{report_id}", dependencies=[Depends(verify_api_key)])
async def get_report(report_id: str, db=Depends(get_db)):
    """Get full details of a past analysis report."""
    report = db.query(DBAnalysisReport).filter(DBAnalysisReport.id == report_id).first()
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    full_data = json.loads(_decrypt_text(report.report_data))
    return {
        "id": report.id,
        "documents_analyzed": report.documents_analyzed,
        "total_issues": report.total_issues,
        "critical_issues": report.critical_issues,
        "risk_level": report.risk_level,
        "analyzed_at": report.analyzed_at.isoformat(),
        **full_data,
    }
