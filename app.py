#!/usr/bin/env python3
"""
PDFHelper — Web API for AI-powered PDF search and flagging.

Deployed on Railway. Provides endpoints to upload PDFs, search them
with keywords or AI, and retrieve flagged results.
"""

import json
import os
import re
import secrets
import tempfile
import uuid
from datetime import datetime, timezone
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

from database import SessionLocal, engine, Base, DBDocument, DBSearchResult

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="PDFHelper",
    description="AI-powered PDF search and flagging tool",
    version="1.0.0",
    docs_url="/docs" if os.getenv("ENVIRONMENT") != "production" else None,
    redoc_url=None,
)

# ---------------------------------------------------------------------------
# Security middleware
# ---------------------------------------------------------------------------

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "").split(",")
ALLOWED_ORIGINS = [o.strip() for o in ALLOWED_ORIGINS if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS or ["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)

ALLOWED_HOSTS = os.getenv("ALLOWED_HOSTS", "").split(",")
ALLOWED_HOSTS = [h.strip() for h in ALLOWED_HOSTS if h.strip()]
if ALLOWED_HOSTS:
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=ALLOWED_HOSTS)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple in-memory rate limiter per IP."""
    def __init__(self, app, max_requests: int = 60, window_seconds: int = 60):
        super().__init__(app)
        self.max_requests = max_requests
        self.window = window_seconds
        self.requests: dict[str, list[float]] = {}

    async def dispatch(self, request: Request, call_next):
        import time
        client_ip = request.client.host if request.client else "unknown"
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
        return await call_next(request)


app.add_middleware(RateLimitMiddleware, max_requests=60, window_seconds=60)


# ---------------------------------------------------------------------------
# Auth dependency
# ---------------------------------------------------------------------------

API_KEY = os.getenv("PDF_HELPER_API_KEY")


async def verify_api_key(request: Request):
    """Require a valid API key for all endpoints."""
    if not API_KEY:
        return  # No key configured = auth disabled (dev mode)
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        token = auth[7:]
    else:
        token = request.headers.get("X-API-Key", "")
    if not secrets.compare_digest(token, API_KEY):
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
# Config
# ---------------------------------------------------------------------------

MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE_MB", "20")) * 1024 * 1024  # default 20 MB
MAX_FILES_PER_REQUEST = int(os.getenv("MAX_FILES_PER_REQUEST", "20"))
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "/tmp/pdfhelper_uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# PDF processing (reused from pdf_helper.py core logic)
# ---------------------------------------------------------------------------

def extract_text_from_pdf(pdf_path: str) -> list[dict]:
    pages = []
    doc = fitz.open(pdf_path)
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        pages.append({"page": page_num + 1, "text": text})
    doc.close()
    return pages


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
# Request / response models
# ---------------------------------------------------------------------------

class SearchRequest(BaseModel):
    search_terms: list[str] = Field(default=[], description="Exact keywords to search")
    ai_query: str | None = Field(default=None, description="AI concept search query")
    case_sensitive: bool = False


class HealthResponse(BaseModel):
    status: str
    version: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def validate_pdf(file: UploadFile) -> None:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail=f"Only PDF files allowed, got: {file.filename}")
    if file.content_type and file.content_type != "application/pdf":
        raise HTTPException(status_code=400,
                            detail=f"Invalid content type: {file.content_type}")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return {"status": "ok", "version": "1.0.0"}


@app.post("/upload", dependencies=[Depends(verify_api_key)])
async def upload_pdfs(
    files: list[UploadFile] = File(...),
    db=Depends(get_db),
):
    """Upload one or more PDFs for later searching."""
    if len(files) > MAX_FILES_PER_REQUEST:
        raise HTTPException(status_code=400,
                            detail=f"Max {MAX_FILES_PER_REQUEST} files per request")

    uploaded = []
    for file in files:
        validate_pdf(file)

        content = await file.read()
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"{file.filename} exceeds max size of "
                       f"{MAX_FILE_SIZE // (1024*1024)} MB",
            )

        doc_id = str(uuid.uuid4())
        save_path = UPLOAD_DIR / f"{doc_id}.pdf"
        save_path.write_bytes(content)

        # Extract text
        pages = extract_text_from_pdf(str(save_path))

        db_doc = DBDocument(
            id=doc_id,
            filename=file.filename,
            filepath=str(save_path),
            page_count=len(pages),
            text_content=json.dumps(pages),
            uploaded_at=datetime.now(timezone.utc),
        )
        db.add(db_doc)
        db.commit()

        uploaded.append({
            "id": doc_id,
            "filename": file.filename,
            "pages": len(pages),
        })

    return {"uploaded": uploaded, "count": len(uploaded)}


@app.post("/search", dependencies=[Depends(verify_api_key)])
async def search_documents(
    request: SearchRequest,
    doc_ids: list[str] = Query(default=[], description="Document IDs to search (empty = all)"),
    db=Depends(get_db),
):
    """Search uploaded PDFs with keywords and/or AI."""
    if not request.search_terms and not request.ai_query:
        raise HTTPException(status_code=400,
                            detail="Provide search_terms and/or ai_query")

    # Get documents
    query = db.query(DBDocument)
    if doc_ids:
        query = query.filter(DBDocument.id.in_(doc_ids))
    documents = query.all()

    if not documents:
        raise HTTPException(status_code=404, detail="No documents found")

    all_keyword_results = []
    all_ai_results = []

    for doc in documents:
        pages = json.loads(doc.text_content)

        if request.search_terms:
            matches = keyword_search(pages, request.search_terms, request.case_sensitive)
            for m in matches:
                m["document_id"] = doc.id
                m["filename"] = doc.filename
            all_keyword_results.extend(matches)

        if request.ai_query:
            findings = ai_search(pages, request.ai_query, doc.filename)
            for f in findings:
                f["document_id"] = doc.id
                f["filename"] = doc.filename
            all_ai_results.extend(findings)

    # Save results to DB
    search_id = str(uuid.uuid4())
    flagged_count = len([r for r in all_ai_results if r.get("needs_review")])

    db_result = DBSearchResult(
        id=search_id,
        search_terms=json.dumps(request.search_terms) if request.search_terms else None,
        ai_query=request.ai_query,
        keyword_results=json.dumps(all_keyword_results),
        ai_results=json.dumps(all_ai_results),
        total_keyword_matches=len(all_keyword_results),
        total_ai_findings=len(all_ai_results),
        flagged_for_review=flagged_count,
        searched_at=datetime.now(timezone.utc),
    )
    db.add(db_result)
    db.commit()

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
                "filename": d.filename,
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
        "filename": doc.filename,
        "pages": doc.page_count,
        "uploaded_at": doc.uploaded_at.isoformat(),
    }


@app.delete("/documents/{doc_id}", dependencies=[Depends(verify_api_key)])
async def delete_document(doc_id: str, db=Depends(get_db)):
    """Delete an uploaded document."""
    doc = db.query(DBDocument).filter(DBDocument.id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    # Remove file from disk
    filepath = Path(doc.filepath)
    if filepath.exists():
        filepath.unlink()

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
                "search_terms": json.loads(r.search_terms) if r.search_terms else [],
                "ai_query": r.ai_query,
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
        "search_terms": json.loads(result.search_terms) if result.search_terms else [],
        "ai_query": result.ai_query,
        "keyword_results": json.loads(result.keyword_results),
        "ai_results": json.loads(result.ai_results),
        "summary": {
            "total_keyword_matches": result.total_keyword_matches,
            "total_ai_findings": result.total_ai_findings,
            "flagged_for_review": result.flagged_for_review,
        },
        "searched_at": result.searched_at.isoformat(),
    }
