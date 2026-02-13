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

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed; env vars must be set directly

import fitz  # PyMuPDF
from fastapi import (
    FastAPI, File, UploadFile, Depends, HTTPException, Query, Request,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware

from database import SessionLocal, engine, Base, DBDocument, DBSearchResult, DBAnalysisReport
from audit import log_upload, log_search, log_delete, log_auth_failure, log_access
from ocr import extract_text_with_ocr_fallback

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
    """In-memory rate limiter per IP."""
    def __init__(self, app, max_requests: int = 60, window_seconds: int = 60):
        super().__init__(app)
        self.max_requests = max_requests
        self.window = window_seconds
        self.requests: dict[str, list[float]] = {}

    async def dispatch(self, request: Request, call_next):
        client_ip = _get_client_ip(request)
        now = time.time()
        window_start = now - self.window

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


def _read_and_decrypt(path: Path) -> bytes:
    """Read file and decrypt. Falls back to plain read if no key."""
    if ENCRYPTION_KEY:
        from encryption import decrypt_file
        return decrypt_file(str(path))
    return path.read_bytes()


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


def keyword_search(pages: list[dict], search_terms: list[str],
                   case_sensitive: bool = False) -> list[dict]:
    results = []
    flags = 0 if case_sensitive else re.IGNORECASE
    for page_info in pages:
        text = page_info["text"]
        for term in search_terms:
            pattern = re.compile(re.escape(term), flags)
            for match in pattern.finditer(text):
                start = max(0, match.start() - 80)
                end = min(len(text), match.end() + 80)
                context = text[start:end].replace("\n", " ").strip()
                results.append({
                    "page": page_info["page"],
                    "term": term,
                    "matched_text": match.group(),
                    "context": f"...{context}...",
                })
    return results


def ai_search(pages: list[dict], query: str, filename: str) -> list[dict]:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500,
                            detail="ANTHROPIC_API_KEY not configured on server")

    from anthropic import Anthropic
    client = Anthropic(api_key=api_key)
    results = []

    batch_size = 5
    for i in range(0, len(pages), batch_size):
        batch = pages[i:i + batch_size]
        page_texts = ""
        for p in batch:
            page_texts += f"\n--- PAGE {p['page']} ---\n{p['text']}\n"
        if not page_texts.strip():
            continue

        prompt = f"""You are a document reviewer. Analyze the following PDF pages from "{filename}" and search for content related to this query:

QUERY: {query}

DOCUMENT CONTENT:
{page_texts}

Instructions:
- Find any text that is relevant to the query — not just exact matches, but related concepts, synonyms, and ideas.
- For each finding, determine if it might need to be reviewed or changed.
- Respond ONLY with a JSON array of findings. Each finding should have:
  - "page": the page number
  - "matched_text": the specific text that matched (quote directly)
  - "reason": why this is relevant to the query
  - "needs_review": true/false — whether this likely needs changes
  - "suggestion": if needs_review is true, what change might be needed

If nothing relevant is found, respond with an empty array: []
Respond with ONLY valid JSON, no other text."""

        try:
            response = client.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
            )
            response_text = response.content[0].text.strip()
            if response_text.startswith("```"):
                response_text = re.sub(r"^```(?:json)?\n?", "", response_text)
                response_text = re.sub(r"\n?```$", "", response_text)

            findings = json.loads(response_text)
            for finding in findings:
                results.append({
                    "page": finding.get("page", "?"),
                    "matched_text": finding.get("matched_text", ""),
                    "reason": finding.get("reason", ""),
                    "needs_review": finding.get("needs_review", False),
                    "suggestion": finding.get("suggestion", ""),
                })
        except (json.JSONDecodeError, Exception):
            continue

    return results


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
    role: str = Field(description="'user' or 'assistant'")
    content: str


class ChatRequest(BaseModel):
    message: str = Field(description="The user's message")
    doc_ids: list[str] = Field(default=[], description="Document IDs to use as context (empty = all)")
    conversation_history: list[ChatMessage] = Field(default=[], description="Previous messages for context")


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
# Root UI (interactive single-page interface)
# ---------------------------------------------------------------------------

_ROOT_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>PDFHelper</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
:root{--bg:#0f172a;--card:#1e293b;--border:#334155;--text:#e2e8f0;--muted:#94a3b8;
--accent:#3b82f6;--green:#22c55e;--red:#ef4444;--orange:#f59e0b;--code:#0d1117}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
background:var(--bg);color:var(--text);line-height:1.6;margin:0}
.app{display:flex;height:100vh;height:100dvh}
/* Sidebar */
.sidebar{width:220px;background:var(--card);border-right:1px solid var(--border);
display:flex;flex-direction:column;flex-shrink:0}
.sidebar .logo{padding:1.25rem 1rem;font-weight:700;font-size:1.1rem;
border-bottom:1px solid var(--border)}
.sidebar .logo span{color:var(--accent)}
.sidebar nav{flex:1;padding:0.5rem 0;display:flex;flex-direction:column}
.sidebar nav button{display:flex;align-items:center;gap:0.6rem;width:100%;
padding:0.6rem 1rem;background:none;border:none;color:var(--muted);cursor:pointer;
font-size:0.88rem;text-align:left;transition:all .15s}
.sidebar nav button:hover{background:#ffffff08;color:var(--text)}
.sidebar nav button.active{background:#3b82f618;color:var(--accent);
border-right:2px solid var(--accent)}
.sidebar .status{padding:1rem;border-top:1px solid var(--border);font-size:0.78rem}
.status-dot{display:inline-block;width:7px;height:7px;border-radius:50%;margin-right:4px}
.status-dot.ok{background:var(--green)}.status-dot.bad{background:var(--red)}
/* Main */
.main{flex:1;overflow-y:auto;padding:2rem 2.5rem}
.main h2{font-size:1.4rem;margin-bottom:1.25rem}
/* Cards */
.card{background:var(--card);border:1px solid var(--border);border-radius:10px;
padding:1.25rem 1.5rem;margin-bottom:1rem}
.card h3{font-size:0.95rem;margin-bottom:0.75rem;color:var(--accent)}
/* Forms */
label{display:block;font-size:0.82rem;color:var(--muted);margin-bottom:0.3rem;font-weight:500}
input[type=text],input[type=password],textarea{width:100%;padding:0.55rem 0.75rem;
background:var(--code);border:1px solid var(--border);border-radius:6px;color:var(--text);
font-size:0.88rem;font-family:inherit;outline:none;transition:border .15s}
input:focus,textarea:focus{border-color:var(--accent)}
textarea{resize:vertical;min-height:60px}
.btn{display:inline-flex;align-items:center;gap:0.4rem;padding:0.5rem 1.1rem;
border-radius:6px;border:none;cursor:pointer;font-size:0.85rem;font-weight:600;
transition:all .15s}
.btn-primary{background:var(--accent);color:#fff}
.btn-primary:hover{background:#2563eb}
.btn-danger{background:var(--red);color:#fff}
.btn-danger:hover{background:#dc2626}
.btn-secondary{background:var(--border);color:var(--text)}
.btn-secondary:hover{background:#475569}
.btn:disabled{opacity:0.5;cursor:not-allowed}
/* File upload zone */
.drop-zone{border:2px dashed var(--border);border-radius:10px;padding:2rem;
text-align:center;cursor:pointer;transition:all .2s;margin-bottom:0.75rem}
.drop-zone:hover,.drop-zone.dragover{border-color:var(--accent);background:#3b82f608}
.drop-zone input{display:none}
.drop-zone .icon{font-size:2rem;margin-bottom:0.5rem}
.drop-zone p{color:var(--muted);font-size:0.88rem}
.drop-zone .selected{color:var(--green);font-size:0.85rem;margin-top:0.5rem}
/* API key bar */
.api-bar{display:flex;gap:0.75rem;align-items:end;margin-bottom:1.5rem}
.api-bar .field{flex:1}
.api-bar .btn{margin-bottom:1px}
/* Document list */
.doc-row{display:flex;align-items:center;gap:1rem;padding:0.65rem 0;
border-bottom:1px solid var(--border);font-size:0.88rem}
.doc-row:last-child{border-bottom:none}
.doc-icon{font-size:1.3rem}
.doc-info{flex:1}
.doc-info .name{font-weight:600}
.doc-info .meta{font-size:0.78rem;color:var(--muted)}
.doc-check{width:16px;height:16px;accent-color:var(--accent)}
/* Results */
.result-card{background:var(--code);border:1px solid var(--border);border-radius:8px;
padding:0.85rem 1rem;margin-bottom:0.6rem;font-size:0.85rem}
.result-card .label{font-size:0.75rem;color:var(--muted);text-transform:uppercase;
letter-spacing:0.5px;margin-bottom:0.25rem}
.result-card .text{color:var(--text)}
.flag{display:inline-block;font-size:0.72rem;padding:2px 7px;border-radius:3px;font-weight:600}
.flag-review{background:#f59e0b22;color:var(--orange);border:1px solid #f59e0b44}
.flag-ok{background:#22c55e22;color:var(--green);border:1px solid #22c55e44}
.flag-critical{background:#ef444422;color:var(--red);border:1px solid #ef444455}
/* Summary stats */
.stats{display:flex;gap:1rem;margin-bottom:1rem;flex-wrap:wrap}
.stat{background:var(--code);border:1px solid var(--border);border-radius:8px;
padding:0.75rem 1rem;flex:1;min-width:120px;text-align:center}
.stat .num{font-size:1.5rem;font-weight:700}
.stat .lbl{font-size:0.75rem;color:var(--muted)}
/* Tabs */
.tabs{display:flex;gap:0;margin-bottom:1rem;border-bottom:1px solid var(--border)}
.tab{padding:0.5rem 1rem;background:none;border:none;color:var(--muted);cursor:pointer;
font-size:0.85rem;border-bottom:2px solid transparent;transition:all .15s}
.tab:hover{color:var(--text)}.tab.active{color:var(--accent);border-bottom-color:var(--accent)}
/* Loading */
.spinner{display:inline-block;width:16px;height:16px;border:2px solid var(--border);
border-top-color:var(--accent);border-radius:50%;animation:spin .6s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
.loading-overlay{display:flex;flex-direction:column;align-items:center;gap:0.75rem;
padding:2rem;color:var(--muted)}
/* Toast */
.toast{position:fixed;top:1.5rem;left:50%;transform:translateX(-50%);padding:0.85rem 1.5rem;border-radius:8px;
font-size:0.9rem;font-weight:600;z-index:999;opacity:0;transition:opacity .3s;pointer-events:none;
max-width:90%;text-align:center;box-shadow:0 4px 20px rgba(0,0,0,0.4)}
.toast.show{opacity:1}.toast.success{background:#16a34a;color:#fff}
.toast.error{background:var(--red);color:#fff}
/* Connection status */
.api-status{font-size:0.8rem;margin-top:0.3rem;font-weight:600}
.api-status.connected{color:var(--green)}.api-status.disconnected{color:var(--red)}
.api-status.warning{color:var(--orange)}
.api-status.unchecked{color:var(--muted)}
/* Inline error */
.inline-error{background:#ef444422;border:1px solid #ef444455;border-radius:8px;
padding:0.75rem 1rem;margin-top:0.75rem;color:var(--red);font-size:0.88rem;font-weight:500}
/* Hide all pages, show active */
.page{display:none}.page.active{display:block}
/* Responsive */
@media(max-width:768px){
.sidebar{width:56px}
.sidebar .logo span,.sidebar nav button span,.sidebar .status span{display:none}
.sidebar nav button{justify-content:center;padding:0.75rem}
.main{padding:1rem 0.75rem;height:100%}
.api-bar{flex-direction:column;align-items:stretch}
.api-bar .field{width:100%}
.api-bar .btn{width:100%;justify-content:center}
input[type=text],input[type=password],textarea{font-size:16px}
.btn{padding:0.65rem 1rem;font-size:0.9rem}
.drop-zone{padding:1.25rem}
.stat{min-width:80px}
}
</style>
</head>
<body>
<div class="app">
  <!-- Sidebar -->
  <aside class="sidebar">
    <div class="logo"><span>PDF</span>Helper</div>
    <nav>
      <button class="active" onclick="showPage('upload')" id="nav-upload">
        <svg width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
        <span>Upload</span>
      </button>
      <button onclick="showPage('documents')" id="nav-documents">
        <svg width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
        <span>Documents</span>
      </button>
      <button onclick="showPage('search')" id="nav-search">
        <svg width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
        <span>Search</span>
      </button>
      <button onclick="showPage('analyze')" id="nav-analyze">
        <svg width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M12 20V10"/><path d="M18 20V4"/><path d="M6 20v-4"/></svg>
        <span>Analyze</span>
      </button>
      <button onclick="showPage('history')" id="nav-history">
        <svg width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>
        <span>History</span>
      </button>
      <button onclick="window.location.href='/bot'" id="nav-bot" style="margin-top:auto;border-top:1px solid var(--border)">
        <svg width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><rect x="3" y="11" width="18" height="10" rx="2"/><circle cx="12" cy="5" r="2"/><path d="M12 7v4"/><line x1="8" y1="16" x2="8" y2="16"/><line x1="16" y1="16" x2="16" y2="16"/></svg>
        <span>Bot</span>
      </button>
    </nav>
    <div class="status" id="health-status"></div>
  </aside>

  <!-- Main content -->
  <main class="main">

    <!-- API Key bar (persistent) -->
    <div class="api-bar">
      <div class="field">
        <label for="apikey">API Key</label>
        <input type="password" id="apikey" placeholder="Enter your PDF_HELPER_API_KEY" autocomplete="off">
        <div class="api-status unchecked" id="api-status">Not connected — enter your API key and click Connect</div>
      </div>
      <button class="btn btn-secondary" onclick="toggleKeyVisibility(event)">Show</button>
      <button class="btn btn-primary" onclick="testConnection()">Connect</button>
    </div>

    <!-- Upload page -->
    <div class="page active" id="page-upload">
      <h2>Upload PDFs</h2>
      <div class="card">
        <div class="drop-zone" id="drop-zone" onclick="document.getElementById('file-input').click()">
          <input type="file" id="file-input" accept=".pdf" multiple>
          <div class="icon">&#128196;</div>
          <p>Click or drag PDF files here</p>
          <p style="font-size:0.78rem;color:var(--muted)">Max 20 files, 20 MB each</p>
          <div class="selected" id="file-list"></div>
        </div>
        <button class="btn btn-primary" id="upload-btn" onclick="uploadFiles()">Upload</button>
      </div>
      <div id="upload-result"></div>
    </div>

    <!-- Documents page -->
    <div class="page" id="page-documents">
      <h2>Documents</h2>
      <div class="card">
        <button class="btn btn-secondary" onclick="loadDocuments()" style="margin-bottom:0.75rem">Refresh</button>
        <div id="doc-list"><p style="color:var(--muted);font-size:0.88rem">Click Refresh to load documents.</p></div>
      </div>
    </div>

    <!-- Search page -->
    <div class="page" id="page-search">
      <h2>Search Documents</h2>
      <div class="card">
        <h3>Keyword Search</h3>
        <label for="search-terms">Keywords (comma-separated)</label>
        <input type="text" id="search-terms" placeholder='e.g. safety, hazard, compliance'>
        <div style="margin-top:0.75rem">
          <label for="ai-query">AI Semantic Search</label>
          <textarea id="ai-query" placeholder="Describe what you're looking for in plain language, e.g. 'references to outdated regulations' or 'sections about employee training requirements'"></textarea>
        </div>
        <div style="margin-top:0.75rem">
          <label>Filter by documents (optional)</label>
          <div id="search-doc-select" style="font-size:0.85rem;color:var(--muted)">Load documents first from the Documents tab</div>
        </div>
        <div style="margin-top:1rem">
          <button class="btn btn-primary" id="search-btn" onclick="runSearch()">Search</button>
        </div>
      </div>
      <div id="search-result"></div>
    </div>

    <!-- Analyze page -->
    <div class="page" id="page-analyze">
      <h2>Multi-Agent Analysis</h2>
      <div class="card">
        <p style="color:var(--muted);font-size:0.85rem;margin-bottom:1rem">
          Runs 4 AI agents: Document Analyzer, Cross-Reference Checker, Compliance Checker, and Summary Report Generator.
        </p>
        <label for="compliance-ctx">Compliance Standard (optional)</label>
        <input type="text" id="compliance-ctx" placeholder="e.g. OSHA 2024, HIPAA, FDA 21 CFR Part 11">
        <div style="margin-top:0.75rem">
          <label for="analyze-terms">Keywords (optional, comma-separated)</label>
          <input type="text" id="analyze-terms" placeholder="e.g. PPE, training">
        </div>
        <div style="margin-top:0.75rem">
          <label for="analyze-query">AI Query (optional)</label>
          <textarea id="analyze-query" placeholder="e.g. outdated safety procedures"></textarea>
        </div>
        <div style="margin-top:0.75rem">
          <label>Filter by documents (optional)</label>
          <div id="analyze-doc-select" style="font-size:0.85rem;color:var(--muted)">Load documents first from the Documents tab</div>
        </div>
        <div style="margin-top:1rem">
          <button class="btn btn-primary" id="analyze-btn" onclick="runAnalysis()">Run Analysis</button>
        </div>
      </div>
      <div id="analyze-result"></div>
    </div>

    <!-- History page -->
    <div class="page" id="page-history">
      <h2>History &amp; Reports</h2>
      <div class="tabs">
        <button class="tab active" onclick="showHistoryTab('searches',this)">Searches</button>
        <button class="tab" onclick="showHistoryTab('reports',this)">Analysis Reports</button>
      </div>
      <div id="history-searches">
        <button class="btn btn-secondary" onclick="loadHistory()" style="margin-bottom:0.75rem">Load Searches</button>
        <div id="history-list"></div>
      </div>
      <div id="history-reports" style="display:none">
        <button class="btn btn-secondary" onclick="loadReports()" style="margin-bottom:0.75rem">Load Reports</button>
        <div id="reports-list"></div>
      </div>
    </div>

  </main>
</div>

<div class="toast" id="toast"></div>

<script>
const API = window.location.origin;
let apiConnected = false;

function getKey(){ return document.getElementById('apikey').value.trim(); }
function saveKey(){ const k=getKey(); if(k) localStorage.setItem('pdfhelper_apikey',k); }
async function loadKey(){
  // Check if the server even requires an API key
  try{
    const r=await fetch(API+'/health');
    if(r.ok){
      const h=await r.json();
      if(!h.api_key_required){
        apiConnected=true;
        document.querySelector('.api-bar').style.display='none';
        setApiStatus('Connected (no API key required)','connected');
        return;
      }
    }
  }catch(e){}
  const k=localStorage.getItem('pdfhelper_apikey');
  if(k){ document.getElementById('apikey').value=k; testConnection(); }
}
function headers(json){
  const h = {};
  const k = getKey();
  if(k) h['X-API-Key']=k;
  if(json) h['Content-Type']='application/json';
  return h;
}
function toast(msg,type='success'){
  const t=document.getElementById('toast');
  t.textContent=msg; t.className='toast '+type+' show';
  setTimeout(()=>t.classList.remove('show'),4000);
}
function setApiStatus(msg, cls){
  const el=document.getElementById('api-status');
  el.textContent=msg; el.className='api-status '+cls;
}
function showInlineError(containerId, msg){
  const el=document.getElementById(containerId);
  el.innerHTML='<div class="inline-error">'+esc(msg)+'</div>';
}
async function testConnection(){
  const key=getKey();
  if(!key){ setApiStatus('Enter your API key above','disconnected'); return; }
  setApiStatus('Connecting...','unchecked');
  try{
    const r=await fetch(API+'/verify-key',{headers:{'X-API-Key':key}});
    if(r.ok){
      const d=await r.json().catch(()=>({}));
      apiConnected=true; saveKey();
      if(d.db_ok===false){
        setApiStatus('API key valid — but database is unavailable (uploads will fail until DB is fixed)','warning');
        toast('API key accepted, but database is down — uploads will not work yet','error');
      } else {
        setApiStatus('Connected','connected');
        toast('Connected successfully');
      }
    } else {
      apiConnected=false;
      if(r.status===401){
        setApiStatus('Key does not match the server\\'s PDF_HELPER_API_KEY','disconnected');
        toast('Key mismatch — enter the exact PDF_HELPER_API_KEY value from your Railway environment variables','error');
      } else {
        const d=await r.json().catch(()=>({}));
        setApiStatus('Connection failed: '+(d.detail||r.status),'disconnected');
        toast(d.detail||'Connection failed','error');
      }
    }
  }catch(e){
    apiConnected=false;
    setApiStatus('Cannot reach server: '+e.message,'disconnected');
    toast('Cannot reach server: '+e.message,'error');
  }
}
function requireKey(errorContainerId){
  if(apiConnected) return true;
  if(!getKey()){
    toast('Enter your API key and click Connect first','error');
    setApiStatus('API key required','disconnected');
    if(errorContainerId) showInlineError(errorContainerId,'Enter your API key at the top and click Connect before continuing.');
  } else {
    toast('Click Connect to verify your API key first','error');
    setApiStatus('Not connected — click Connect','disconnected');
  }
  return false;
}
function showPage(name){
  document.querySelectorAll('.page').forEach(p=>p.classList.remove('active'));
  document.getElementById('page-'+name).classList.add('active');
  document.querySelectorAll('.sidebar nav button').forEach(b=>b.classList.remove('active'));
  document.getElementById('nav-'+name).classList.add('active');
  if(name==='documents') loadDocuments();
  if(name==='history') loadHistory();
}
function toggleKeyVisibility(e){
  const inp=document.getElementById('apikey');
  const btn=e?e.target:document.querySelector('.api-bar .btn-secondary');
  if(inp.type==='password'){inp.type='text';btn.textContent='Hide';}
  else{inp.type='password';btn.textContent='Show';}
}
function loading(el){el.innerHTML='<div class="loading-overlay"><div class="spinner"></div><span>Working...</span></div>';}

/* ---- File upload ---- */
const dz=document.getElementById('drop-zone');
const fi=document.getElementById('file-input');
let selectedFiles=[];
fi.addEventListener('change',()=>{selectedFiles=[...fi.files];showFileNames();});
dz.addEventListener('dragover',e=>{e.preventDefault();dz.classList.add('dragover');});
dz.addEventListener('dragleave',()=>dz.classList.remove('dragover'));
dz.addEventListener('drop',e=>{
  e.preventDefault();dz.classList.remove('dragover');
  selectedFiles=[...e.dataTransfer.files].filter(f=>f.name.toLowerCase().endsWith('.pdf'));
  showFileNames();
});
function showFileNames(){
  const el=document.getElementById('file-list');
  if(!selectedFiles.length){el.textContent='';return;}
  el.textContent=selectedFiles.map(f=>f.name).join(', ');
}
async function uploadFiles(){
  const res=document.getElementById('upload-result');
  if(!requireKey('upload-result')) return;
  if(!selectedFiles.length){
    toast('Select PDF files first','error');
    showInlineError('upload-result','Click the drop zone above to select PDF files first.');
    return;
  }
  res.innerHTML='';
  const btn=document.getElementById('upload-btn');
  btn.disabled=true;btn.innerHTML='<span class="spinner"></span> Uploading...';
  const fd=new FormData();
  selectedFiles.forEach(f=>fd.append('files',f));
  try{
    const r=await fetch(API+'/upload',{method:'POST',headers:headers(false),body:fd});
    let d;
    try{ d=await r.json(); }catch(jsonErr){ throw new Error('Server returned invalid response (status '+r.status+'). The database may be down — check DATABASE_URL in your Railway variables.'); }
    if(r.status===401) throw new Error('Invalid API key. Enter your key above and click Connect first.');
    if(r.status===503) throw new Error('Database is unavailable. Check that DATABASE_URL is set in your Railway service variables.');
    if(!r.ok) throw new Error(d.detail||'Upload failed (status '+r.status+')');
    res.innerHTML='<div class="card"><h3>Uploaded '+d.count+' file(s)</h3>'+
      d.uploaded.map(u=>'<div class="doc-row"><span class="doc-icon">&#128196;</span><div class="doc-info"><div class="name">'+
      esc(u.filename)+'</div><div class="meta">'+u.pages+' pages &middot; ID: '+u.id.slice(0,8)+'...</div></div></div>').join('')+'</div>';
    toast('Uploaded '+d.count+' file(s)');
    selectedFiles=[];fi.value='';document.getElementById('file-list').textContent='';
  }catch(e){
    toast(e.message,'error');
    showInlineError('upload-result', e.message);
  }
  btn.disabled=false;btn.textContent='Upload';
}

/* ---- Documents ---- */
let allDocs=[];
async function loadDocuments(){
  if(!requireKey('doc-list')) return;
  const el=document.getElementById('doc-list');
  loading(el);
  try{
    const r=await fetch(API+'/documents',{headers:headers()});
    const d=await r.json();
    if(!r.ok) throw new Error(d.detail||'Failed');
    allDocs=d.documents||[];
    if(!allDocs.length){el.innerHTML='<p style="color:var(--muted)">No documents uploaded yet.</p>';updateDocSelectors();return;}
    el.innerHTML=allDocs.map(doc=>
      '<div class="doc-row"><span class="doc-icon">&#128196;</span><div class="doc-info"><div class="name">'+esc(doc.filename)+
      '</div><div class="meta">'+doc.pages+' pages &middot; '+new Date(doc.uploaded_at).toLocaleString()+
      ' &middot; <code style="font-size:0.75rem">'+doc.id.slice(0,8)+'...</code></div></div>'+
      '<button class="btn btn-danger" style="font-size:0.75rem;padding:0.3rem 0.6rem" onclick="deleteDoc(&apos;'+doc.id+'&apos;)">Delete</button></div>'
    ).join('');
    updateDocSelectors();
  }catch(e){el.innerHTML='<p style="color:var(--red)">'+esc(e.message)+'</p>';}
}
function updateDocSelectors(){
  ['search-doc-select','analyze-doc-select'].forEach(id=>{
    const el=document.getElementById(id);
    if(!allDocs.length){el.innerHTML='<span style="color:var(--muted)">No documents available</span>';return;}
    el.innerHTML=allDocs.map(d=>
      '<label style="display:flex;align-items:center;gap:0.5rem;margin-bottom:0.3rem;cursor:pointer;font-size:0.85rem">'+
      '<input type="checkbox" class="doc-check" value="'+d.id+'"> '+esc(d.filename)+
      ' <span style="color:var(--muted);font-size:0.75rem">('+d.pages+' pg)</span></label>'
    ).join('');
  });
}
async function deleteDoc(id){
  if(!confirm('Delete this document permanently?'))return;
  try{
    const r=await fetch(API+'/documents/'+id,{method:'DELETE',headers:headers()});
    if(!r.ok){const d=await r.json();throw new Error(d.detail||'Failed');}
    toast('Document deleted');loadDocuments();
  }catch(e){toast(e.message,'error');}
}

/* ---- Search ---- */
async function runSearch(){
  if(!requireKey('search-result')) return;
  const terms=document.getElementById('search-terms').value.split(',').map(s=>s.trim()).filter(Boolean);
  const aiQ=document.getElementById('ai-query').value.trim();
  if(!terms.length&&!aiQ){toast('Enter keywords or an AI query','error');return;}
  const selDocs=[...document.querySelectorAll('#search-doc-select .doc-check:checked')].map(c=>c.value);
  const btn=document.getElementById('search-btn');
  btn.disabled=true;btn.innerHTML='<span class="spinner"></span> Searching...';
  const res=document.getElementById('search-result');
  loading(res);
  try{
    let url=API+'/search';
    if(selDocs.length) url+='?'+selDocs.map(id=>'doc_ids='+id).join('&');
    const body={};
    if(terms.length) body.search_terms=terms;
    if(aiQ) body.ai_query=aiQ;
    const r=await fetch(url,{method:'POST',headers:headers(true),body:JSON.stringify(body)});
    const d=await r.json();
    if(!r.ok) throw new Error(d.detail||'Search failed');
    renderSearchResults(d,res);
    toast('Search complete');
  }catch(e){res.innerHTML='<div class="card" style="color:var(--red)">'+esc(e.message)+'</div>';}
  btn.disabled=false;btn.textContent='Search';
}
function renderSearchResults(d,el){
  const s=d.summary||{};
  let html='<div class="stats">';
  html+='<div class="stat"><div class="num">'+s.documents_searched+'</div><div class="lbl">Docs Searched</div></div>';
  html+='<div class="stat"><div class="num">'+s.total_keyword_matches+'</div><div class="lbl">Keyword Matches</div></div>';
  html+='<div class="stat"><div class="num">'+s.total_ai_findings+'</div><div class="lbl">AI Findings</div></div>';
  html+='<div class="stat"><div class="num" style="color:var(--orange)">'+s.flagged_for_review+'</div><div class="lbl">Flagged for Review</div></div>';
  html+='</div>';
  if(d.keyword_results&&d.keyword_results.length){
    html+='<div class="card"><h3>Keyword Matches</h3>';
    d.keyword_results.forEach(r=>{
      html+='<div class="result-card"><div class="label">'+esc(r.filename)+' &middot; Page '+r.page+' &middot; Term: <strong>'+esc(r.term)+'</strong></div>';
      html+='<div class="text">'+esc(r.context)+'</div></div>';
    });
    html+='</div>';
  }
  if(d.ai_results&&d.ai_results.length){
    html+='<div class="card"><h3>AI Findings</h3>';
    d.ai_results.forEach(r=>{
      const flag=r.needs_review?'<span class="flag flag-review">Needs Review</span>':'<span class="flag flag-ok">OK</span>';
      html+='<div class="result-card"><div class="label">'+esc(r.filename)+' &middot; Page '+r.page+' '+flag+'</div>';
      html+='<div class="text"><strong>Found:</strong> '+esc(r.matched_text)+'</div>';
      html+='<div class="text" style="margin-top:0.25rem"><strong>Reason:</strong> '+esc(r.reason)+'</div>';
      if(r.suggestion) html+='<div class="text" style="margin-top:0.25rem;color:var(--orange)"><strong>Suggestion:</strong> '+esc(r.suggestion)+'</div>';
      html+='</div>';
    });
    html+='</div>';
  }
  if(!d.keyword_results?.length&&!d.ai_results?.length) html+='<div class="card"><p style="color:var(--muted)">No results found.</p></div>';
  el.innerHTML=html;
}

/* ---- Analyze ---- */
async function runAnalysis(){
  if(!requireKey('analyze-result')) return;
  const btn=document.getElementById('analyze-btn');
  btn.disabled=true;btn.innerHTML='<span class="spinner"></span> Analyzing (this may take a minute)...';
  const res=document.getElementById('analyze-result');
  loading(res);
  const selDocs=[...document.querySelectorAll('#analyze-doc-select .doc-check:checked')].map(c=>c.value);
  const body={};
  const ctx=document.getElementById('compliance-ctx').value.trim();
  if(ctx) body.compliance_context=ctx;
  const terms=document.getElementById('analyze-terms').value.split(',').map(s=>s.trim()).filter(Boolean);
  if(terms.length) body.search_terms=terms;
  const q=document.getElementById('analyze-query').value.trim();
  if(q) body.ai_query=q;
  try{
    let url=API+'/analyze';
    if(selDocs.length) url+='?'+selDocs.map(id=>'doc_ids='+id).join('&');
    const r=await fetch(url,{method:'POST',headers:headers(true),body:JSON.stringify(body)});
    const d=await r.json();
    if(!r.ok) throw new Error(d.detail||'Analysis failed');
    renderAnalysis(d,res);
    toast('Analysis complete');
  }catch(e){res.innerHTML='<div class="card" style="color:var(--red)">'+esc(e.message)+'</div>';}
  btn.disabled=false;btn.textContent='Run Analysis';
}
function renderAnalysis(d,el){
  const rpt=d.report||{};
  const risk=rpt.overall_risk_level||'unknown';
  const riskColor=risk==='high'?'var(--red)':risk==='medium'?'var(--orange)':'var(--green)';
  let html='<div class="stats">';
  html+='<div class="stat"><div class="num">'+rpt.documents_reviewed+'</div><div class="lbl">Docs Reviewed</div></div>';
  html+='<div class="stat"><div class="num">'+rpt.total_issues_found+'</div><div class="lbl">Issues Found</div></div>';
  html+='<div class="stat"><div class="num" style="color:var(--red)">'+rpt.critical_issues+'</div><div class="lbl">Critical</div></div>';
  html+='<div class="stat"><div class="num" style="color:'+riskColor+'">'+risk.toUpperCase()+'</div><div class="lbl">Risk Level</div></div>';
  html+='</div>';
  if(rpt.executive_summary) html+='<div class="card"><h3>Executive Summary</h3><p style="font-size:0.88rem">'+esc(rpt.executive_summary)+'</p></div>';
  if(rpt.recommendation) html+='<div class="card"><h3>Recommendation</h3><p style="font-size:0.88rem">'+esc(rpt.recommendation)+'</p></div>';
  if(rpt.action_items&&rpt.action_items.length){
    html+='<div class="card"><h3>Action Items</h3>';
    rpt.action_items.forEach((a,i)=>{
      const txt=typeof a==='string'?a:(a.description||a.action||JSON.stringify(a));
      const prio=typeof a==='object'&&a.priority?' <span class="flag '+(a.priority==='critical'?'flag-critical':'flag-review')+'">'+a.priority+'</span>':'';
      html+='<div class="result-card"><div class="text">'+(i+1)+'. '+esc(txt)+prio+'</div></div>';
    });
    html+='</div>';
  }
  if(d.cross_reference_findings&&d.cross_reference_findings.length){
    html+='<div class="card"><h3>Cross-Reference Findings</h3>';
    d.cross_reference_findings.forEach(f=>{
      const txt=typeof f==='string'?f:(f.description||f.issue||JSON.stringify(f));
      html+='<div class="result-card"><div class="text">'+esc(txt)+'</div></div>';
    });
    html+='</div>';
  }
  if(d.compliance_findings){
    const cf=d.compliance_findings;
    const items=Array.isArray(cf)?cf:Object.values(cf).flat();
    if(items.length){
      html+='<div class="card"><h3>Compliance Findings</h3>';
      items.forEach(f=>{
        const txt=typeof f==='string'?f:(f.finding||f.description||f.issue||JSON.stringify(f));
        html+='<div class="result-card"><div class="text">'+esc(txt)+'</div></div>';
      });
      html+='</div>';
    }
  }
  el.innerHTML=html;
}

/* ---- History ---- */
function showHistoryTab(tab,btn){
  document.getElementById('history-searches').style.display=tab==='searches'?'':'none';
  document.getElementById('history-reports').style.display=tab==='reports'?'':'none';
  document.querySelectorAll('#page-history .tab').forEach(t=>t.classList.remove('active'));
  btn.classList.add('active');
  if(tab==='reports') loadReports();
}
async function loadHistory(){
  if(!requireKey('history-list')) return;
  const el=document.getElementById('history-list');loading(el);
  try{
    const r=await fetch(API+'/history?limit=50',{headers:headers()});
    const d=await r.json();
    if(!r.ok) throw new Error(d.detail||'Failed');
    const items=d.searches||[];
    if(!items.length){el.innerHTML='<p style="color:var(--muted)">No search history.</p>';return;}
    el.innerHTML=items.map(s=>
      '<div class="result-card" style="cursor:pointer" onclick="viewSearch(&apos;'+s.id+'&apos;)">'+
      '<div class="label">'+new Date(s.searched_at).toLocaleString()+'</div>'+
      '<div class="text">'+
        (s.search_terms?.length?'Keywords: <strong>'+s.search_terms.map(esc).join(', ')+'</strong> &middot; ':'')+
        (s.ai_query?'AI: <em>'+esc(s.ai_query)+'</em> &middot; ':'')+
        s.total_keyword_matches+' keyword / '+s.total_ai_findings+' AI findings'+
        (s.flagged_for_review?' &middot; <span class="flag flag-review">'+s.flagged_for_review+' flagged</span>':'')+
      '</div></div>'
    ).join('');
  }catch(e){el.innerHTML='<p style="color:var(--red)">'+esc(e.message)+'</p>';}
}
async function viewSearch(id){
  if(!getKey()) return;
  const el=document.getElementById('history-list');loading(el);
  try{
    const r=await fetch(API+'/history/'+id,{headers:headers()});
    const d=await r.json();
    if(!r.ok) throw new Error(d.detail||'Failed');
    let html='<button class="btn btn-secondary" onclick="loadHistory()" style="margin-bottom:0.75rem">&larr; Back</button>';
    const wrap=document.createElement('div');wrap.id='history-list';
    el.parentNode.replaceChild(wrap,el);
    wrap.innerHTML=html;
    const resDiv=document.createElement('div');
    wrap.appendChild(resDiv);
    renderSearchResults(d,resDiv);
  }catch(e){el.innerHTML='<p style="color:var(--red)">'+esc(e.message)+'</p>';}
}
async function loadReports(){
  if(!requireKey('reports-list')) return;
  const el=document.getElementById('reports-list');loading(el);
  try{
    const r=await fetch(API+'/reports?limit=50',{headers:headers()});
    const d=await r.json();
    if(!r.ok) throw new Error(d.detail||'Failed');
    const items=d.reports||[];
    if(!items.length){el.innerHTML='<p style="color:var(--muted)">No reports yet.</p>';return;}
    el.innerHTML=items.map(rp=>{
      const rc=rp.risk_level==='high'?'var(--red)':rp.risk_level==='medium'?'var(--orange)':'var(--green)';
      return '<div class="result-card" style="cursor:pointer" onclick="viewReport(&apos;'+rp.id+'&apos;)">'+
        '<div class="label">'+new Date(rp.analyzed_at).toLocaleString()+'</div>'+
        '<div class="text">'+rp.documents_analyzed+' docs &middot; '+rp.total_issues+' issues &middot; '+
        rp.critical_issues+' critical &middot; <span style="color:'+rc+';font-weight:700">'+rp.risk_level.toUpperCase()+' RISK</span></div></div>';
    }).join('');
  }catch(e){el.innerHTML='<p style="color:var(--red)">'+esc(e.message)+'</p>';}
}
async function viewReport(id){
  if(!getKey()) return;
  const el=document.getElementById('reports-list');loading(el);
  try{
    const r=await fetch(API+'/reports/'+id,{headers:headers()});
    const d=await r.json();
    if(!r.ok) throw new Error(d.detail||'Failed');
    let html='<button class="btn btn-secondary" onclick="loadReports()" style="margin-bottom:0.75rem">&larr; Back</button>';
    const wrap=document.createElement('div');wrap.id='reports-list';
    el.parentNode.replaceChild(wrap,el);
    wrap.innerHTML=html;
    const resDiv=document.createElement('div');
    wrap.appendChild(resDiv);
    renderAnalysis(d,resDiv);
  }catch(e){el.innerHTML='<p style="color:var(--red)">'+esc(e.message)+'</p>';}
}

/* ---- Health check ---- */
async function checkHealth(){
  try{
    const r=await fetch(API+'/health');const d=await r.json();
    const ok=d.status==='ok';
    document.getElementById('health-status').innerHTML=
      '<span class="status-dot '+(ok?'ok':'bad')+'"></span><span>'+d.version+' &middot; '+(ok?'Healthy':'Degraded')+'</span>';
  }catch(e){
    document.getElementById('health-status').innerHTML='<span class="status-dot bad"></span><span>Offline</span>';
  }
}
checkHealth();setInterval(checkHealth,30000);

function esc(s){if(!s)return'';const d=document.createElement('div');d.textContent=String(s);return d.innerHTML;}

/* ---- Load saved API key on startup ---- */
loadKey();
</script>
</body>
</html>"""

_BOT_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>PDFHelper — Procedure Bot</title>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
*{margin:0;padding:0;box-sizing:border-box}
:root{
  --bg:#0a0e17;--surface:#111827;--surface-light:#1a2236;
  --border:#1e293b;--border-active:#3b82f6;
  --primary:#3b82f6;--primary-dark:#2563eb;--primary-glow:rgba(59,130,246,0.15);
  --accent:#10b981;--accent-glow:rgba(16,185,129,0.15);
  --warning:#f59e0b;--text:#e2e8f0;--text-muted:#94a3b8;--text-dim:#64748b;
  --white:#ffffff;--danger:#ef4444;
}
body{font-family:'IBM Plex Sans','Segoe UI',system-ui,sans-serif;
  background:var(--bg);color:var(--text);overflow:hidden;margin:0}

.app{display:flex;height:100vh;height:100dvh;width:100vw}

/* ---- Sidebar ---- */
.sidebar{width:320px;min-width:320px;background:var(--surface);
  border-right:1px solid var(--border);display:flex;flex-direction:column;
  transition:all .3s ease;overflow:hidden}
.sidebar.collapsed{width:0;min-width:0}

.sidebar-header{padding:20px;border-bottom:1px solid var(--border)}
.sidebar-header .label{display:flex;align-items:center;gap:10px;margin-bottom:4px}
.sidebar-header .dot{width:8px;height:8px;border-radius:50%;background:var(--text-dim);transition:all .3s}
.sidebar-header .dot.active{background:var(--accent);box-shadow:0 0 8px var(--accent)}
.sidebar-header .title{font-size:13px;font-weight:600;text-transform:uppercase;
  letter-spacing:0.08em;color:var(--text-muted)}
.sidebar-header .meta{font-size:12px;color:var(--text-dim);margin-top:6px;
  font-family:'IBM Plex Mono',monospace}

/* API Key Section */
.api-section{padding:12px 20px;border-bottom:1px solid var(--border)}
.api-section label{font-size:11px;color:var(--text-dim);margin-bottom:4px;display:block}
.api-section .key-row{display:flex;gap:6px}
.api-section input{flex:1;padding:6px 10px;background:var(--bg);border:1px solid var(--border);
  border-radius:6px;color:var(--text);font-size:12px;outline:none;font-family:inherit}
.api-section input:focus{border-color:var(--primary)}
.api-section .connect-btn{padding:6px 12px;background:var(--primary);color:var(--white);
  border:none;border-radius:6px;font-size:11px;font-weight:600;cursor:pointer}
.api-section .connect-btn:hover{background:var(--primary-dark)}
.api-section .status{font-size:11px;margin-top:4px}
.api-section .status.ok{color:var(--accent)}
.api-section .status.bad{color:var(--danger)}
.api-section .status.muted{color:var(--text-dim)}

/* Doc List */
.doc-list{flex:1;overflow-y:auto;padding:0 12px 12px}
.doc-empty{text-align:center;padding:30px 20px;color:var(--text-dim);font-size:13px}
.doc-item{display:flex;align-items:flex-start;gap:10px;padding:12px;margin-bottom:4px;
  border-radius:8px;cursor:pointer;transition:all .2s;
  border:1px solid transparent}
.doc-item.selected{background:var(--primary-glow);border-color:rgba(59,130,246,0.25)}
.doc-item .checkbox{width:18px;height:18px;border-radius:4px;
  border:2px solid var(--text-dim);display:flex;align-items:center;
  justify-content:center;flex-shrink:0;margin-top:1px;transition:all .15s}
.doc-item.selected .checkbox{border-color:var(--primary);background:var(--primary)}
.doc-item .info{flex:1;min-width:0}
.doc-item .name{font-size:13px;font-weight:500;color:var(--text-muted);
  overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.doc-item.selected .name{color:var(--white)}
.doc-item .pages{font-size:11px;color:var(--text-dim);margin-top:2px;
  font-family:'IBM Plex Mono',monospace}
.doc-actions{padding:12px 20px;border-top:1px solid var(--border);display:flex;gap:8px}
.doc-actions button{flex:1;padding:6px;font-size:11px;font-weight:500;
  background:transparent;border:1px solid var(--border);color:var(--text-muted);
  border-radius:6px;cursor:pointer}
.doc-actions button:hover{border-color:var(--primary);color:var(--white)}

/* Back link */
.back-link{padding:12px 20px;border-top:1px solid var(--border)}
.back-link a{font-size:12px;color:var(--text-dim);text-decoration:none;
  display:flex;align-items:center;gap:6px}
.back-link a:hover{color:var(--primary)}

/* ---- Main chat ---- */
.main{flex:1;display:flex;flex-direction:column;min-width:0;min-height:0}

.topbar{padding:14px 24px;border-bottom:1px solid var(--border);
  display:flex;align-items:center;gap:14px;background:var(--surface)}
.toggle-btn{background:none;border:1px solid var(--border);color:var(--text-muted);
  cursor:pointer;padding:6px 10px;border-radius:6px;font-size:16px;line-height:1;
  display:flex;align-items:center}
.topbar .bot-info .title{font-size:15px;font-weight:600;color:var(--white)}
.topbar .bot-info .subtitle{font-size:11px;color:var(--text-dim)}
.clear-btn{margin-left:auto;background:none;border:1px solid var(--border);
  color:var(--text-dim);cursor:pointer;padding:6px 12px;border-radius:6px;font-size:12px}
.clear-btn:hover{border-color:var(--primary);color:var(--white)}

/* Messages */
.messages{flex:1;overflow-y:auto;padding:24px;min-height:0}
.empty-state{display:flex;flex-direction:column;align-items:center;
  justify-content:center;height:100%;gap:16px;opacity:0.7}
.empty-state .icon-box{width:64px;height:64px;border-radius:16px;
  background:var(--primary-glow);display:flex;align-items:center;justify-content:center}
.empty-state h3{font-size:18px;font-weight:600;color:var(--white);margin-bottom:6px}
.empty-state p{font-size:13px;color:var(--text-dim);max-width:400px;line-height:1.6;text-align:center}
.suggestions{display:flex;gap:8px;margin-top:8px;flex-wrap:wrap;justify-content:center}
.suggestion{background:var(--surface-light);border:1px solid var(--border);
  color:var(--text-muted);padding:8px 14px;border-radius:20px;font-size:12px;
  cursor:pointer;transition:all .15s}
.suggestion:hover{border-color:var(--primary);color:var(--white)}

.message{display:flex;margin-bottom:16px}
.message.user{justify-content:flex-end}
.message.assistant{justify-content:flex-start}
.bubble{max-width:75%;padding:12px 16px;font-size:14px;line-height:1.6}
.message.user .bubble{background:var(--primary);color:var(--white);
  border-radius:16px 16px 4px 16px}
.message.assistant .bubble{background:var(--surface-light);color:var(--text);
  border:1px solid var(--border);border-radius:16px 16px 16px 4px}
.bubble strong{color:var(--white)}

.typing{display:flex;justify-content:flex-start;margin-bottom:16px}
.typing-bubble{padding:12px 20px;border-radius:16px 16px 16px 4px;
  background:var(--surface-light);border:1px solid var(--border);
  display:flex;gap:6px;align-items:center}
.typing-dot{width:8px;height:8px;border-radius:50%;background:var(--primary);
  animation:pulse 1.2s ease-in-out infinite}
.typing-dot:nth-child(2){animation-delay:0.2s}
.typing-dot:nth-child(3){animation-delay:0.4s}
@keyframes pulse{0%,100%{opacity:.3;transform:scale(.8)}50%{opacity:1;transform:scale(1)}}

/* Input */
.input-area{padding:16px 24px;border-top:1px solid var(--border);background:var(--surface)}
.input-row{display:flex;gap:10px;align-items:flex-end;background:var(--surface-light);
  border-radius:14px;padding:6px 6px 6px 16px;border:1px solid var(--border)}
.input-row textarea{flex:1;background:none;border:none;color:var(--text);font-size:14px;
  font-family:inherit;resize:none;outline:none;padding:8px 0;line-height:1.5;
  max-height:120px;overflow-y:auto}
.input-row textarea::placeholder{color:var(--text-dim)}
.send-btn{width:40px;height:40px;border-radius:10px;border:none;
  color:var(--white);cursor:pointer;display:flex;align-items:center;
  justify-content:center;flex-shrink:0;transition:all .15s}
.send-btn.active{background:var(--primary)}
.send-btn.inactive{background:var(--border);cursor:default}
.input-footer{font-size:11px;color:var(--text-dim);margin-top:8px;text-align:center}

/* Scrollbar */
::-webkit-scrollbar{width:6px}
::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{background:var(--border);border-radius:3px}
::-webkit-scrollbar-thumb:hover{background:var(--text-dim)}

/* ---- Mobile ---- */
@media(max-width:768px){
  html,body{height:100%;overflow:hidden;margin:0;padding:0}

  /* position:fixed + inset:0 = always matches visible viewport
     on ALL mobile browsers, regardless of address bar state.
     flex-direction:column so .main fills remaining space. */
  .app{
    position:fixed;top:0;left:0;right:0;bottom:0;
    display:flex;flex-direction:column;
    width:auto;height:auto;overflow:hidden
  }

  /* Sidebar: fixed overlay drawer, hidden by default on mobile.
     Uses .sidebar-open to show — no JS needed for initial hide. */
  .sidebar{
    position:fixed;top:0;left:0;bottom:0;
    width:280px;min-width:280px;z-index:100;
    transform:translateX(-100%);transition:transform .25s ease;
    overflow-y:auto;box-shadow:none;pointer-events:none
  }
  .sidebar.sidebar-open{
    transform:translateX(0);
    box-shadow:4px 0 24px rgba(0,0,0,0.5);
    pointer-events:auto
  }
  .sidebar-backdrop{
    display:none;position:fixed;inset:0;z-index:99;
    background:rgba(0,0,0,0.5);-webkit-tap-highlight-color:transparent
  }
  .sidebar-backdrop.visible{display:block}

  /* Main chat area — only in-flow child of .app, takes all space */
  .main{
    flex:1;display:flex;flex-direction:column;
    min-height:0;min-width:0;width:100%;overflow:hidden
  }

  /* Topbar: fixed height, never shrinks */
  .topbar{flex-shrink:0;padding:10px 14px;gap:10px}
  .topbar .bot-info .title{font-size:14px}

  /* Messages: fills remaining space, scrolls independently */
  .messages{
    flex:1;min-height:0;overflow-y:auto;
    padding:16px 12px;-webkit-overflow-scrolling:touch
  }

  /* Empty state: auto height so it never clips, centered via padding */
  .empty-state{
    height:auto;min-height:100%;padding:20px 16px;
    justify-content:center
  }
  .empty-state p{font-size:12px;max-width:300px}
  .suggestions{flex-direction:column;align-items:center}
  .suggestion{width:100%;max-width:260px;text-align:center}

  /* Input area: pinned to bottom, never shrinks */
  .input-area{
    flex-shrink:0;padding:10px 12px;
    padding-bottom:calc(10px + env(safe-area-inset-bottom,0px))
  }
  .input-row{border-radius:12px;padding:4px 4px 4px 12px}
  .input-row textarea{font-size:16px}
  .send-btn{width:36px;height:36px;border-radius:8px}

  /* Chat bubbles */
  .bubble{max-width:88%}

  /* Sidebar touch targets */
  .doc-item{padding:14px 12px}
  .doc-actions{padding:10px 12px}
  .doc-actions button{padding:10px;font-size:12px}
  .back-link{padding:10px 12px}
}

@media(max-width:380px){
  .sidebar{width:100vw;min-width:100vw}
}
</style>
</head>
<body>
<div class="app">
  <!-- Sidebar -->
  <div class="sidebar" id="sidebar">
    <div class="sidebar-header">
      <div class="label">
        <div class="dot" id="status-dot"></div>
        <span class="title">Procedure Library</span>
      </div>
      <div class="meta" id="doc-stats">0 loaded &middot; 0 active</div>
    </div>

    <!-- API Key (hidden by default, shown only if server requires it) -->
    <div class="api-section" id="api-section" style="display:none">
      <label>API Key</label>
      <div class="key-row">
        <input type="password" id="apikey" placeholder="PDF_HELPER_API_KEY" autocomplete="off">
        <button class="connect-btn" onclick="connectKey()">Connect</button>
      </div>
      <div class="status muted" id="key-status">Enter key to connect</div>
    </div>

    <!-- Document list -->
    <div class="doc-list" id="doc-list">
      <div class="doc-empty" id="doc-empty">Connecting to server...</div>
    </div>

    <!-- Select all / none / refresh -->
    <div class="doc-actions" id="doc-actions" style="display:none">
      <button onclick="selectAll()">Select All</button>
      <button onclick="deselectAll()">Deselect All</button>
    </div>
    <div class="doc-actions">
      <button onclick="loadDocs()" style="width:100%">&#8635; Refresh Documents</button>
    </div>

    <!-- Back link -->
    <div class="back-link">
      <a href="/">&larr; Back to PDFHelper</a>
    </div>
  </div>
  <div class="sidebar-backdrop" id="sidebar-backdrop" onclick="closeSidebar()"></div>

  <!-- Main chat -->
  <div class="main">
    <div class="topbar">
      <button class="toggle-btn" onclick="toggleSidebar()">
        <span id="toggle-icon">&#9665;</span>
      </button>
      <div style="display:flex;align-items:center;gap:10px">
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#3b82f6" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <rect x="3" y="11" width="18" height="10" rx="2"/><circle cx="12" cy="5" r="2"/>
          <path d="M12 7v4"/><line x1="8" y1="16" x2="8" y2="16"/><line x1="16" y1="16" x2="16" y2="16"/>
        </svg>
        <div class="bot-info">
          <div class="title">Procedure Assistant</div>
          <div class="subtitle" id="bot-subtitle">No documents selected</div>
        </div>
      </div>
      <button class="clear-btn" id="clear-btn" style="display:none" onclick="clearChat()">Clear Chat</button>
    </div>

    <div class="messages" id="messages">
      <div class="empty-state" id="empty-state">
        <div id="doc-status-banner"></div>
        <div class="icon-box">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#3b82f6" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <rect x="3" y="11" width="18" height="10" rx="2"/><circle cx="12" cy="5" r="2"/>
            <path d="M12 7v4"/><line x1="8" y1="16" x2="8" y2="16"/><line x1="16" y1="16" x2="16" y2="16"/>
          </svg>
        </div>
        <div>
          <h3>Procedure Knowledge Bot</h3>
          <p>Upload your procedure documents via the main app, select which ones to include, then ask questions. Answers come strictly from your data with source citations.</p>
        </div>
        <div class="suggestions">
          <button class="suggestion" onclick="useSuggestion(this)">What are the lockout steps?</button>
          <button class="suggestion" onclick="useSuggestion(this)">Who approves SIMOPS?</button>
          <button class="suggestion" onclick="useSuggestion(this)">What PPE is required?</button>
        </div>
      </div>
    </div>

    <div class="input-area">
      <div class="input-row">
        <textarea id="chat-input" rows="1"
          placeholder="Upload and select procedures first..."
          onkeydown="handleKey(event)"
          oninput="autoResize(this);updateSendBtn()"></textarea>
        <button class="send-btn inactive" id="send-btn" onclick="sendMessage()">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/>
          </svg>
        </button>
      </div>
      <div class="input-footer">Answers sourced strictly from your selected procedures &middot; AI may occasionally misinterpret content</div>
    </div>
  </div>
</div>

<script>
const API = window.location.origin;
let connected = false;
let allDocs = [];
let selectedIds = new Set();
let messages = [];
let sending = false;
let docsLoading = false;

function isMobile(){ return window.innerWidth<=768; }

/* ---- API Key ---- */
function getKey(){ return document.getElementById('apikey').value.trim(); }

function connectKey(){
  const key = getKey();
  if(!key){ setKeyStatus('Enter your API key','bad'); return; }
  setKeyStatus('Connecting...','muted');
  fetch(API+'/verify-key',{headers:{'X-API-Key':key}})
    .then(async r=>{
      if(r.ok){
        connected=true;
        localStorage.setItem('pdfhelper_apikey',key);
        setKeyStatus('Connected','ok');
        loadDocs();
      } else if(r.status===401){
        setKeyStatus('Invalid API key','bad');
      } else {
        setKeyStatus('Connection failed','bad');
      }
    })
    .catch(e=>setKeyStatus('Cannot reach server','bad'));
}

function setKeyStatus(msg,cls){
  const el=document.getElementById('key-status');
  el.textContent=msg; el.className='status '+cls;
}

function hdrs(json){
  const h={};
  const k=getKey();
  if(k) h['X-API-Key']=k;
  if(json) h['Content-Type']='application/json';
  return h;
}

/* Auto-load: check if API key is needed */
(function(){
  fetch(API+'/health').then(function(r){ return r.json(); }).then(function(d){
    if(!d.api_key_required){
      // No key needed — connect directly
      connected=true;
      loadDocs();
      return;
    }
    // Key IS required — show the API key section
    document.getElementById('api-section').style.display='';
    var k=localStorage.getItem('pdfhelper_apikey');
    if(k){document.getElementById('apikey').value=k;connectKey();}
    else{
      document.getElementById('doc-list').innerHTML=
        '<div class="doc-empty">Enter your API key above to load documents.</div>';
    }
  }).catch(function(){
    // Can't reach server — try again after 3s, then show error with retry
    setTimeout(function(){
      fetch(API+'/health').then(function(r){ return r.json(); }).then(function(d){
        if(!d.api_key_required){
          connected=true;
          loadDocs();
        } else {
          document.getElementById('api-section').style.display='';
          var k=localStorage.getItem('pdfhelper_apikey');
          if(k){document.getElementById('apikey').value=k;connectKey();}
          else{
            document.getElementById('doc-list').innerHTML=
              '<div class="doc-empty">Enter your API key above to load documents.</div>';
          }
        }
      }).catch(function(){
        document.getElementById('api-section').style.display='';
        document.getElementById('doc-list').innerHTML=
          '<div class="doc-empty">Cannot reach server.<br>'+
          '<button onclick="location.reload()" style="margin-top:8px;padding:6px 14px;background:var(--primary);color:var(--white);border:none;border-radius:6px;font-size:12px;cursor:pointer">Reload Page</button></div>';
        updateEmptyState();
      });
    },3000);
  });
})();

/* ---- Documents ---- */
var loadDocsRetries=0;
function loadDocs(){
  if(!connected) return;
  docsLoading=true;
  document.getElementById('doc-list').innerHTML='<div class="doc-empty">Loading documents...</div>';
  updateEmptyState();
  fetch(API+'/documents',{headers:hdrs()})
    .then(function(r){
      if(!r.ok) throw new Error('Failed to load ('+r.status+')');
      return r.json();
    })
    .then(function(d){
      allDocs=d.documents||[];
      selectedIds=new Set(allDocs.map(function(doc){return doc.id;}));
      docsLoading=false;
      loadDocsRetries=0;
      renderDocs();
    })
    .catch(function(e){
      docsLoading=false;
      loadDocsRetries++;
      if(loadDocsRetries<3){
        // Auto-retry with backoff
        var delay=loadDocsRetries*2000;
        document.getElementById('doc-list').innerHTML=
          '<div class="doc-empty">Retrying in '+Math.round(delay/1000)+'s...</div>';
        setTimeout(loadDocs,delay);
      } else {
        loadDocsRetries=0;
        document.getElementById('doc-list').innerHTML=
          '<div class="doc-empty">Failed to load documents: '+esc(e.message)+
          '<br><button onclick="loadDocs()" style="margin-top:8px;padding:6px 14px;background:var(--primary);color:var(--white);border:none;border-radius:6px;font-size:12px;cursor:pointer">Retry</button></div>';
      }
      updateEmptyState();
    });
}

/* Auto-reload documents when user switches back to this tab */
document.addEventListener('visibilitychange',function(){
  if(!document.hidden && connected){ loadDocs(); }
});

function renderDocs(){
  const el=document.getElementById('doc-list');
  const actions=document.getElementById('doc-actions');

  if(!allDocs.length){
    el.innerHTML='<div class="doc-empty">No documents uploaded yet.<br>Upload via the <a href="/" style="color:var(--primary)">main app</a> first.</div>';
    actions.style.display='none';
    updateStatus();
    return;
  }

  actions.style.display=allDocs.length>1?'flex':'none';
  el.innerHTML=allDocs.map(doc=>{
    const sel=selectedIds.has(doc.id);
    return '<div class="doc-item'+(sel?' selected':'')+'" onclick="toggleDoc(\''+doc.id+'\')">'+
      '<div class="checkbox">'+(sel?'<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="3"><polyline points="20 6 9 17 4 12"/></svg>':'')+'</div>'+
      '<div class="info"><div class="name">'+esc(doc.filename)+'</div>'+
      '<div class="pages">'+doc.pages+' pages</div></div></div>';
  }).join('');
  updateStatus();
}

function toggleDoc(id){
  if(selectedIds.has(id)) selectedIds.delete(id);
  else selectedIds.add(id);
  renderDocs();
}
function selectAll(){ selectedIds=new Set(allDocs.map(d=>d.id)); renderDocs(); }
function deselectAll(){ selectedIds.clear(); renderDocs(); }

function updateStatus(){
  const dot=document.getElementById('status-dot');
  const stats=document.getElementById('doc-stats');
  const subtitle=document.getElementById('bot-subtitle');
  const input=document.getElementById('chat-input');
  const sel=selectedIds.size;

  dot.className='dot'+(sel>0?' active':'');
  stats.textContent=allDocs.length+' loaded '+String.fromCharCode(183)+' '+sel+' active';
  subtitle.textContent=sel>0?'Answering from '+sel+' procedure'+(sel>1?'s':''):'No documents selected';
  input.placeholder=sel>0?'Ask a question about your procedures...':'Select procedures from the sidebar first...';
  updateEmptyState();
}

function updateEmptyState(){
  var es=document.getElementById('empty-state');
  if(!es) return;
  var sel=selectedIds.size;
  var statusHtml='';
  if(docsLoading){
    statusHtml='<div style="background:rgba(59,130,246,0.15);border:1px solid rgba(59,130,246,0.3);border-radius:8px;padding:10px 16px;font-size:13px;color:#94a3b8;margin-bottom:8px">'+
      'Loading documents...</div>';
  } else if(sel>0){
    statusHtml='<div style="background:rgba(16,185,129,0.15);border:1px solid rgba(16,185,129,0.3);border-radius:8px;padding:10px 16px;font-size:13px;color:#10b981;margin-bottom:8px">'+
      '<strong>'+sel+' document'+(sel>1?'s':'')+' loaded and ready</strong></div>';
  } else if(allDocs.length>0){
    statusHtml='<div style="background:rgba(245,158,11,0.15);border:1px solid rgba(245,158,11,0.3);border-radius:8px;padding:10px 16px;font-size:13px;color:#f59e0b;margin-bottom:8px">'+
      'Documents loaded but none selected. <a href="#" onclick="event.preventDefault();toggleSidebar()" style="color:#3b82f6;text-decoration:underline">Open sidebar</a> to select.</div>';
  } else if(!connected){
    statusHtml='<div style="background:rgba(239,68,68,0.15);border:1px solid rgba(239,68,68,0.3);border-radius:8px;padding:10px 16px;font-size:13px;color:#ef4444;margin-bottom:8px">'+
      'Not connected to server. <button onclick="location.reload()" style="color:#3b82f6;background:none;border:none;text-decoration:underline;cursor:pointer;font-size:13px">Reload page</button></div>';
  } else {
    statusHtml='<div style="background:rgba(59,130,246,0.15);border:1px solid rgba(59,130,246,0.3);border-radius:8px;padding:10px 16px;font-size:13px;color:#94a3b8;margin-bottom:8px">'+
      'No documents uploaded yet. <a href="/" style="color:#3b82f6;text-decoration:underline">Go to main app</a> to upload PDFs first.</div>';
  }
  var statusEl=document.getElementById('doc-status-banner');
  if(statusEl){ statusEl.innerHTML=statusHtml; }
}

/* ---- Sidebar toggle ---- */
function toggleSidebar(){
  var sb=document.getElementById('sidebar');
  var ic=document.getElementById('toggle-icon');
  var bd=document.getElementById('sidebar-backdrop');
  if(isMobile()){
    sb.classList.toggle('sidebar-open');
    var open=sb.classList.contains('sidebar-open');
    ic.innerHTML=open?'&#9665;':'&#9776;';
    bd.className='sidebar-backdrop'+(open?' visible':'');
  } else {
    sb.classList.toggle('collapsed');
    var closed=sb.classList.contains('collapsed');
    ic.innerHTML=closed?'&#9776;':'&#9665;';
  }
}

function closeSidebar(){
  var sb=document.getElementById('sidebar');
  if(isMobile()){
    sb.classList.remove('sidebar-open');
  } else {
    sb.classList.add('collapsed');
  }
  document.getElementById('toggle-icon').innerHTML='&#9776;';
  document.getElementById('sidebar-backdrop').className='sidebar-backdrop';
}

/* On mobile, sidebar starts hidden via CSS (no JS needed).
   On desktop, set the toggle icon to the correct default. */
if(isMobile()){
  document.getElementById('toggle-icon').innerHTML='&#9776;';
}

/* ---- Chat ---- */
function clearChat(){
  messages=[];
  renderMessages();
}

function useSuggestion(btn){
  document.getElementById('chat-input').value=btn.textContent;
  autoResize(document.getElementById('chat-input'));
  updateSendBtn();
  document.getElementById('chat-input').focus();
}

function handleKey(e){
  if(e.key==='Enter'&&!e.shiftKey){ e.preventDefault(); sendMessage(); }
}

function autoResize(el){
  el.style.height='auto';
  el.style.height=Math.min(el.scrollHeight,120)+'px';
}

function updateSendBtn(){
  const btn=document.getElementById('send-btn');
  const val=document.getElementById('chat-input').value.trim();
  btn.className='send-btn '+(val&&!sending?'active':'inactive');
}

function sendMessage(){
  const input=document.getElementById('chat-input');
  const text=input.value.trim();
  if(!text||sending) return;

  if(!connected){
    messages.push({role:'user',content:text});
    messages.push({role:'assistant',content:'Please connect your API key first using the sidebar.'});
    input.value='';
    autoResize(input);
    renderMessages();
    return;
  }

  if(selectedIds.size===0){
    messages.push({role:'user',content:text});
    messages.push({role:'assistant',content:'No procedures are selected. Please select at least one procedure document from the sidebar before asking questions.'});
    input.value='';
    autoResize(input);
    renderMessages();
    return;
  }

  messages.push({role:'user',content:text});
  input.value='';
  autoResize(input);
  sending=true;
  updateSendBtn();
  renderMessages();

  const body={
    message:text,
    doc_ids:[...selectedIds],
    conversation_history:messages.slice(0,-1).slice(-10)
  };

  fetch(API+'/chat',{method:'POST',headers:hdrs(true),body:JSON.stringify(body)})
    .then(async r=>{
      const d=await r.json();
      if(!r.ok) throw new Error(d.detail||'Request failed');
      messages.push({role:'assistant',content:d.reply});
    })
    .catch(e=>{
      messages.push({role:'assistant',content:'Error: '+e.message});
    })
    .finally(()=>{
      sending=false;
      updateSendBtn();
      renderMessages();
    });
}

function renderMessages(){
  const el=document.getElementById('messages');
  const clearBtn=document.getElementById('clear-btn');

  if(!messages.length){
    el.innerHTML='';
    el.appendChild(createEmptyState());
    clearBtn.style.display='none';
    return;
  }

  clearBtn.style.display='';
  let html='';
  messages.forEach(m=>{
    html+='<div class="message '+m.role+'"><div class="bubble">'+formatContent(m.content)+'</div></div>';
  });
  if(sending){
    html+='<div class="typing"><div class="typing-bubble">'+
      '<div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div>'+
      '</div></div>';
  }
  html+='<div id="chat-end"></div>';
  el.innerHTML=html;
  document.getElementById('chat-end')?.scrollIntoView({behavior:'smooth'});
}

function createEmptyState(){
  const div=document.createElement('div');
  div.className='empty-state';
  div.id='empty-state';
  div.innerHTML=
    '<div id="doc-status-banner"></div>'+
    '<div class="icon-box"><svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#3b82f6" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="11" width="18" height="10" rx="2"/><circle cx="12" cy="5" r="2"/><path d="M12 7v4"/><line x1="8" y1="16" x2="8" y2="16"/><line x1="16" y1="16" x2="16" y2="16"/></svg></div>'+
    '<div><h3>Procedure Knowledge Bot</h3><p>Upload your procedure documents via the main app, select which ones to include, then ask questions. Answers come strictly from your data with source citations.</p></div>'+
    '<div class="suggestions">'+
    '<button class="suggestion" onclick="useSuggestion(this)">What are the lockout steps?</button>'+
    '<button class="suggestion" onclick="useSuggestion(this)">Who approves SIMOPS?</button>'+
    '<button class="suggestion" onclick="useSuggestion(this)">What PPE is required?</button></div>';
  updateEmptyState();
  return div;
}

function formatContent(text){
  var safe=esc(text);
  // Bold **text**
  safe=safe.replace(/[*][*](.+?)[*][*]/g,'<strong style="color:var(--white)">$1</strong>');
  // Bullet lists
  safe=safe.replace(/^[-] (.+)/gm,'<li style="margin-left:16px;list-style:disc">$1</li>');
  // Numbered lists
  safe=safe.replace(/^(\\d+)[.] (.+)/gm,'<li style="margin-left:16px;list-style:decimal">$2</li>');
  // Paragraphs (split on double newline)
  var nl=String.fromCharCode(10);
  var parts=safe.split(nl+nl);
  if(parts.length>1){
    return parts.map(function(p){return '<p style="margin:0 0 8px 0">'+p.split(nl).join('<br>')+'</p>';}).join('');
  }
  return '<p style="margin:0 0 8px 0">'+safe.split(nl).join('<br>')+'</p>';
}

function esc(s){if(!s)return '';const d=document.createElement('div');d.textContent=String(s);return d.innerHTML;}

/* ---- Mobile keyboard handling ---- */
if(isMobile()){
  document.getElementById('chat-input').addEventListener('focus',function(){
    setTimeout(function(){
      var end=document.getElementById('chat-end');
      if(end) end.scrollIntoView({behavior:'smooth'});
    },350);
  });
}
</script>
</body>
</html>"""


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
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {type(exc).__name__}: {str(exc)}"},
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check — also tests current DB connectivity."""
    db_ok = True
    try:
        from sqlalchemy import text
        db = SessionLocal()
        db.execute(text("SELECT 1"))
        db.close()
    except Exception as e:
        db_ok = False
        db_err = f"Database connection failed: {e}"
        if db_err not in _startup_errors:
            _startup_errors.append(db_err)

    status = "ok" if (not _startup_errors and db_ok) else "degraded"
    return {
        "status": status,
        "version": "1.0.0",
        "api_key_required": bool(API_KEY),
        "warnings": _startup_errors,
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
    return _ROOT_HTML


@app.get("/bot", response_class=HTMLResponse)
async def bot_page():
    """Procedure Knowledge Bot — chat with your uploaded documents."""
    return _BOT_HTML


@app.post("/upload", dependencies=[Depends(verify_api_key)])
async def upload_pdfs(
    request: Request,
    files: list[UploadFile] = File(...),
    db=Depends(get_db),
):
    """Upload one or more PDFs for later searching."""
    if len(files) > MAX_FILES_PER_REQUEST:
        raise HTTPException(status_code=400,
                            detail=f"Max {MAX_FILES_PER_REQUEST} files per request")

    client_ip = _get_client_ip(request)

    # Run cleanup on upload to keep storage in check
    _run_cleanup(db)

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

    # Build procedure context from selected documents
    procedure_parts = []
    for doc in documents:
        decrypted_name = _decrypt_text(doc.filename)
        pages = json.loads(_decrypt_text(doc.text_content))
        full_text = "\n".join(p["text"] for p in pages if p.get("text"))
        # Truncate very long documents to stay within context limits
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

    # Build conversation messages (last 10 for context)
    conversation = [
        {"role": m.role, "content": m.content}
        for m in body.conversation_history[-10:]
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
        raise HTTPException(status_code=500, detail=f"AI request failed: {str(e)}")

    return {
        "reply": reply,
        "documents_used": [
            {"id": d.id, "filename": _decrypt_text(d.filename)}
            for d in documents
        ],
    }


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

    # Run the full pipeline
    from agents import run_full_analysis
    analysis = run_full_analysis(
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
