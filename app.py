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

import fitz  # PyMuPDF
from fastapi import (
    FastAPI, File, UploadFile, Depends, HTTPException, Query, Request,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
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

API_KEY = os.getenv("PDF_HELPER_API_KEY")
ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY")

MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE_MB", "20")) * 1024 * 1024
MAX_FILES_PER_REQUEST = int(os.getenv("MAX_FILES_PER_REQUEST", "20"))
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "/tmp/pdfhelper_uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Auto-cleanup: delete uploads older than this many hours (0 = disabled)
AUTO_CLEANUP_HOURS = int(os.getenv("AUTO_CLEANUP_HOURS", "72"))

# ---------------------------------------------------------------------------
# Startup safety checks
# ---------------------------------------------------------------------------

if IS_PRODUCTION and not API_KEY:
    raise RuntimeError(
        "FATAL: PDF_HELPER_API_KEY must be set in production. "
        "Generate one with: python -c \"import secrets; print(secrets.token_urlsafe(32))\""
    )

if IS_PRODUCTION and not ENCRYPTION_KEY:
    raise RuntimeError(
        "FATAL: ENCRYPTION_KEY must be set in production for file encryption. "
        "Generate one with: python -c \"from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())\""
    )

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="PDFHelper",
    description="AI-powered PDF search and flagging tool",
    version="1.0.0",
    docs_url="/docs" if not IS_PRODUCTION else None,
    redoc_url=None,
)

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
        response.headers.pop("server", None)
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


class HealthResponse(BaseModel):
    status: str
    version: str


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
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return {"status": "ok", "version": "1.0.0"}


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
