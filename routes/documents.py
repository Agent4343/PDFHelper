"""Document management routes — upload, search, list, download, merge, split, annotate."""

import asyncio
import hashlib
import json
import secrets
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
)
from fastapi.responses import Response
from pydantic import BaseModel, Field

from auth import verify_auth, verify_api_key, get_db, _get_client_ip, _decode_jwt
from config import UPLOAD_DIR, MAX_FILES_PER_REQUEST, ENCRYPTION_KEY, CHAT_MODEL, CHAT_MAX_TOKENS, IS_PRODUCTION, API_KEY
from database import (
    SessionLocal,
    DBDocument,
    DBSearchResult,
    DBChatSession,
    DBCodeSession,
    DBUpdateSession,
    DBAnalysisReport,
    DBAgentCache,
)
from models import SearchRequest, AnalyzeRequest, MergeRequest, SplitRequest
from helpers import (
    validate_upload,
    _encrypt_and_save,
    _encrypt_text,
    _decrypt_text,
    _safe_decrypt,
    extract_text_from_bytes,
    _load_pdf_bytes,
    _load_stored_text,
    _stored_text_to_structured,
    _safe_unlink,
    _run_cleanup_background,
    _sanitize_filename,
    _is_image_file,
    _is_spreadsheet_file,
    _is_word_file,
    _is_text_file,
    _extract_text_file,
    _extract_word_text,
    _extract_spreadsheet_text,
    _decrypt_and_load,
    _extract_image_base64,
    _detect_image_media_type,
)
from ocr import extract_structured_text
from search import keyword_search, ai_search
from audit import log_upload, log_search, log_delete

router = APIRouter()


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------


@router.post("/upload", dependencies=[Depends(verify_api_key)])
async def upload_pdfs(
    request: Request,
    files: list[UploadFile] = File(...),
    background_tasks: BackgroundTasks = None,
    db=Depends(get_db),
):
    """Upload files for later searching.

    Supported formats: PDF, DOCX, JS, HTML, CSS, MD, TXT, JSON, XML, YAML, PY,
    JPG, PNG, TIFF, BMP, WebP, XLSX, XLS, CSV (and more code/text formats).
    """
    if len(files) > MAX_FILES_PER_REQUEST:
        raise HTTPException(status_code=400,
                            detail=f"Max {MAX_FILES_PER_REQUEST} files per request")

    client_ip = _get_client_ip(request)

    if background_tasks:
        background_tasks.add_task(_run_cleanup_background)

    uploaded = []
    for file in files:
        raw_content = await file.read()
        clean_name, file_bytes, file_type = validate_upload(file, raw_content)

        if file_type == "spreadsheet":
            pages = _extract_spreadsheet_text(raw_content, clean_name)
        elif file_type == "word":
            pages = _extract_word_text(raw_content)
        elif file_type == "text":
            pages = _extract_text_file(raw_content, clean_name)
        elif file_type == "image":
            pages = extract_text_from_bytes(file_bytes)
        else:
            pages = extract_structured_text(file_bytes)

        doc_id = str(uuid.uuid4())
        save_path = UPLOAD_DIR / f"{doc_id}.pdf.enc"

        _encrypt_and_save(raw_content, save_path)

        if file_type == "image":
            img_save_path = UPLOAD_DIR / f"{doc_id}.img.enc"
            _encrypt_and_save(raw_content, img_save_path)

        content_hash = hashlib.sha256(raw_content).hexdigest()

        db_doc = DBDocument(
            id=doc_id,
            filename=_encrypt_text(clean_name),
            filepath=str(save_path),
            page_count=len(pages),
            text_content=_encrypt_text(json.dumps(pages)),
            content_hash=content_hash,
            uploaded_at=datetime.now(timezone.utc),
        )
        db.add(db_doc)
        db.commit()

        log_upload(client_ip, clean_name, doc_id, len(pages))

        uploaded.append({
            "id": doc_id,
            "filename": clean_name,
            "pages": len(pages),
            "type": file_type,
        })

    return {"uploaded": uploaded, "count": len(uploaded)}


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


def _search_sync(documents, search_terms, ai_query, case_sensitive):
    """Sync helper for heavy decrypt+parse work — run via asyncio.to_thread."""
    all_keyword_results = []
    all_ai_results = []
    for doc in documents:
        pages = json.loads(_safe_decrypt(doc.text_content, "[]"))
        decrypted_name = _safe_decrypt(doc.filename, f"Document {doc.id[:8]}")
        if search_terms:
            matches = keyword_search(pages, search_terms, case_sensitive)
            for m in matches:
                m["document_id"] = doc.id
                m["filename"] = decrypted_name
            all_keyword_results.extend(matches)
        if ai_query:
            findings = ai_search(pages, ai_query, decrypted_name)
            for f in findings:
                f["document_id"] = doc.id
                f["filename"] = decrypted_name
            all_ai_results.extend(findings)
    return all_keyword_results, all_ai_results


@router.post("/search", dependencies=[Depends(verify_api_key)])
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

    all_keyword_results, all_ai_results = await asyncio.to_thread(
        _search_sync, documents, body.search_terms, body.ai_query, body.case_sensitive
    )

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


# ---------------------------------------------------------------------------
# List / Get / Delete documents
# ---------------------------------------------------------------------------


def _list_documents_sync(docs):
    """Sync helper — decrypt document metadata off the event loop."""
    doc_list = []
    for d in docs:
        try:
            fname = _decrypt_text(d.filename)
        except Exception:
            fname = f"Document {d.id[:8]}"
        doc_list.append({
            "id": d.id,
            "filename": fname,
            "pages": d.page_count,
            "uploaded_at": d.uploaded_at.isoformat(),
        })
    return doc_list


@router.get("/documents", dependencies=[Depends(verify_api_key)])
async def list_documents(
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=200),
    db=Depends(get_db),
):
    """List all uploaded documents."""
    total = db.query(DBDocument).count()
    docs = (
        db.query(DBDocument)
        .order_by(DBDocument.uploaded_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )
    doc_list = await asyncio.to_thread(_list_documents_sync, docs)
    return {"documents": doc_list, "total": total}


@router.get("/documents/{doc_id}", dependencies=[Depends(verify_api_key)])
async def get_document(doc_id: str, db=Depends(get_db)):
    """Get details for a specific document."""
    doc = db.query(DBDocument).filter(DBDocument.id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return {
        "id": doc.id,
        "filename": _safe_decrypt(doc.filename, f"Document {doc.id[:8]}"),
        "pages": doc.page_count,
        "uploaded_at": doc.uploaded_at.isoformat(),
    }


@router.delete("/documents/{doc_id}", dependencies=[Depends(verify_api_key)])
async def delete_document(doc_id: str, request: Request, db=Depends(get_db)):
    """Delete an uploaded document, its files, and all related sessions."""
    doc = db.query(DBDocument).filter(DBDocument.id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    # Delete encrypted files from disk
    _safe_unlink(Path(doc.filepath))
    img_path = Path(doc.filepath.replace(".pdf.enc", ".img.enc"))
    _safe_unlink(img_path)

    try:
        log_delete(_get_client_ip(request), doc_id, _decrypt_text(doc.filename))
    except Exception:
        log_delete(_get_client_ip(request), doc_id, f"Document {doc_id[:8]}")

    # Cascade: delete sessions referencing this document (SQL filter, not .all())
    doc_id_pattern = f'"{doc_id}"'
    db.query(DBChatSession).filter(DBChatSession.doc_ids.contains(doc_id_pattern)).delete(synchronize_session="fetch")
    db.query(DBCodeSession).filter(DBCodeSession.doc_ids.contains(doc_id_pattern)).delete(synchronize_session="fetch")
    db.query(DBUpdateSession).filter(DBUpdateSession.doc_id == doc_id).delete(synchronize_session="fetch")
    db.query(DBAnalysisReport).filter(DBAnalysisReport.doc_ids.contains(doc_id_pattern)).delete(synchronize_session="fetch")
    db.query(DBAgentCache).filter(DBAgentCache.doc_ids.contains(doc_id_pattern)).delete(synchronize_session="fetch")

    db.delete(doc)
    db.commit()
    return {"deleted": doc_id, "cascade": True}


# ---------------------------------------------------------------------------
# Search history
# ---------------------------------------------------------------------------


@router.get("/history", dependencies=[Depends(verify_api_key)])
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
                "search_terms": json.loads(_safe_decrypt(r.search_terms, "[]")) if r.search_terms else [],
                "ai_query": _safe_decrypt(r.ai_query) if r.ai_query else None,
                "total_keyword_matches": r.total_keyword_matches,
                "total_ai_findings": r.total_ai_findings,
                "flagged_for_review": r.flagged_for_review,
                "searched_at": r.searched_at.isoformat(),
            }
            for r in results
        ]
    }


@router.get("/history/{search_id}", dependencies=[Depends(verify_api_key)])
async def get_search_result(search_id: str, db=Depends(get_db)):
    """Get full details of a past search."""
    result = db.query(DBSearchResult).filter(DBSearchResult.id == search_id).first()
    if not result:
        raise HTTPException(status_code=404, detail="Search result not found")
    return {
        "id": result.id,
        "search_terms": json.loads(_safe_decrypt(result.search_terms, "[]")) if result.search_terms else [],
        "ai_query": _safe_decrypt(result.ai_query) if result.ai_query else None,
        "keyword_results": json.loads(_safe_decrypt(result.keyword_results, "[]")),
        "ai_results": json.loads(_safe_decrypt(result.ai_results, "[]")),
        "summary": {
            "total_keyword_matches": result.total_keyword_matches,
            "total_ai_findings": result.total_ai_findings,
            "flagged_for_review": result.flagged_for_review,
        },
        "searched_at": result.searched_at.isoformat(),
    }


# ---------------------------------------------------------------------------
# Download / View
# ---------------------------------------------------------------------------


@router.get("/documents/{doc_id}/download", dependencies=[Depends(verify_api_key)])
async def download_document(doc_id: str, db=Depends(get_db)):
    """Download the original PDF file."""
    doc = db.query(DBDocument).filter(DBDocument.id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    pdf_bytes = _load_pdf_bytes(doc)
    filename = _safe_decrypt(doc.filename, f"Document {doc.id[:8]}.pdf")
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/documents/{doc_id}/view")
async def view_document_pdf(
    doc_id: str,
    request: Request,
    token: str = Query(default=""),
    key: str = Query(default=""),
    db=Depends(get_db),
):
    """Serve decrypted PDF inline for in-browser viewing.

    Accepts auth via query params (token= or key=) for iframe embedding,
    in addition to the standard Authorization header.
    """
    # Try standard header auth first
    authed = False
    try:
        await verify_auth(request)
        authed = True
    except HTTPException:
        pass

    # Fall back to query-param auth for iframe usage
    if not authed and token:
        payload = _decode_jwt(token)
        if payload:
            authed = True
    if not authed and key and API_KEY:
        if secrets.compare_digest(key, API_KEY):
            authed = True

    if not authed:
        raise HTTPException(status_code=401, detail="Invalid or missing credentials")

    doc = db.query(DBDocument).filter(DBDocument.id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    pdf_bytes = _load_pdf_bytes(doc)
    filename = _safe_decrypt(doc.filename, f"Document {doc.id[:8]}.pdf")
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'inline; filename="{filename}"'},
    )


# ---------------------------------------------------------------------------
# Dashboard stats
# ---------------------------------------------------------------------------


@router.get("/dashboard/stats", dependencies=[Depends(verify_api_key)])
async def dashboard_stats(db=Depends(get_db)):
    """Return summary stats for the dashboard."""
    doc_count = db.query(DBDocument).count()
    chat_count = db.query(DBChatSession).count()
    cache_count = db.query(DBAgentCache).count()
    search_count = db.query(DBSearchResult).count()
    analysis_count = db.query(DBAnalysisReport).count()

    recent_docs = db.query(DBDocument).order_by(DBDocument.uploaded_at.desc()).limit(5).all()
    recent_chats = db.query(DBChatSession).order_by(DBChatSession.updated_at.desc()).limit(5).all()
    recent_cache = db.query(DBAgentCache).order_by(DBAgentCache.created_at.desc()).limit(5).all()

    return {
        "documents": doc_count,
        "chat_sessions": chat_count,
        "agent_cache": cache_count,
        "searches": search_count,
        "analyses": analysis_count,
        "recent_documents": [
            {"id": d.id, "filename": _safe_decrypt(d.filename, f"Document {d.id[:8]}"),
             "page_count": d.page_count, "uploaded_at": d.uploaded_at.isoformat() if d.uploaded_at else None}
            for d in recent_docs
        ],
        "recent_chats": [
            {"id": s.id, "title": s.title, "updated_at": s.updated_at.isoformat() if s.updated_at else None}
            for s in recent_chats
        ],
        "recent_agents": [
            {"id": c.id, "agent_type": c.agent_type, "model_used": c.model_used,
             "params_summary": c.params_summary, "created_at": c.created_at.isoformat() if c.created_at else None}
            for c in recent_cache
        ],
    }


# ---------------------------------------------------------------------------
# Merge / Split / Annotate
# ---------------------------------------------------------------------------


@router.post("/documents/merge", dependencies=[Depends(verify_api_key)])
async def merge_documents(body: MergeRequest, request: Request, db=Depends(get_db)):
    """Merge multiple PDFs into a single new document."""
    import fitz  # PyMuPDF

    docs_db = []
    for did in body.doc_ids:
        doc = db.query(DBDocument).filter(DBDocument.id == did).first()
        if not doc:
            raise HTTPException(status_code=404, detail=f"Document {did} not found")
        docs_db.append(doc)

    merged = fitz.open()
    try:
        for doc in docs_db:
            pdf_bytes = _load_pdf_bytes(doc)
            src = fitz.open(stream=pdf_bytes, filetype="pdf")
            merged.insert_pdf(src)
            src.close()

        merged_bytes = merged.tobytes()
    finally:
        merged.close()

    # Save the merged PDF as a new document
    clean_name = _sanitize_filename(body.output_filename)
    if not clean_name.lower().endswith(".pdf"):
        clean_name += ".pdf"

    doc_id = str(uuid.uuid4())
    save_path = UPLOAD_DIR / f"{doc_id}.pdf.enc"
    _encrypt_and_save(merged_bytes, save_path)

    pages = extract_structured_text(merged_bytes)
    content_hash = hashlib.sha256(merged_bytes).hexdigest()

    db_doc = DBDocument(
        id=doc_id,
        filename=_encrypt_text(clean_name),
        filepath=str(save_path),
        page_count=len(pages),
        text_content=_encrypt_text(json.dumps(pages)),
        content_hash=content_hash,
        uploaded_at=datetime.now(timezone.utc),
    )
    db.add(db_doc)
    db.commit()

    log_upload(_get_client_ip(request), clean_name, doc_id, len(pages))

    return {
        "id": doc_id,
        "filename": clean_name,
        "pages": len(pages),
        "merged_from": body.doc_ids,
    }


@router.post("/documents/{doc_id}/split", dependencies=[Depends(verify_api_key)])
async def split_document(doc_id: str, body: SplitRequest, request: Request, db=Depends(get_db)):
    """Extract specific pages from a PDF into a new document."""
    import fitz

    doc = db.query(DBDocument).filter(DBDocument.id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    pdf_bytes = _load_pdf_bytes(doc)
    src = fitz.open(stream=pdf_bytes, filetype="pdf")

    # Validate page numbers (convert 1-based to 0-based)
    total_pages = src.page_count
    zero_based = []
    for p in body.pages:
        if p < 1 or p > total_pages:
            src.close()
            raise HTTPException(
                status_code=400,
                detail=f"Page {p} out of range (document has {total_pages} pages)",
            )
        zero_based.append(p - 1)

    new_pdf = fitz.open()
    try:
        for pg in zero_based:
            new_pdf.insert_pdf(src, from_page=pg, to_page=pg)
        split_bytes = new_pdf.tobytes()
    finally:
        new_pdf.close()
        src.close()

    clean_name = _sanitize_filename(body.output_filename)
    if not clean_name.lower().endswith(".pdf"):
        clean_name += ".pdf"

    new_id = str(uuid.uuid4())
    save_path = UPLOAD_DIR / f"{new_id}.pdf.enc"
    _encrypt_and_save(split_bytes, save_path)

    pages = extract_structured_text(split_bytes)
    content_hash = hashlib.sha256(split_bytes).hexdigest()

    db_doc = DBDocument(
        id=new_id,
        filename=_encrypt_text(clean_name),
        filepath=str(save_path),
        page_count=len(pages),
        text_content=_encrypt_text(json.dumps(pages)),
        content_hash=content_hash,
        uploaded_at=datetime.now(timezone.utc),
    )
    db.add(db_doc)
    db.commit()

    log_upload(_get_client_ip(request), clean_name, new_id, len(pages))

    return {
        "id": new_id,
        "filename": clean_name,
        "pages": len(pages),
        "extracted_from": doc_id,
        "page_numbers": body.pages,
    }


@router.post("/documents/{doc_id}/annotate", dependencies=[Depends(verify_api_key)])
async def annotate_document(
    doc_id: str,
    request: Request,
    db=Depends(get_db),
    text: str = Form(..., description="Text to add"),
    page: int = Form(default=1, description="Page number (1-based)"),
    x: float = Form(default=72, description="X position in points from left"),
    y: float = Form(default=72, description="Y position in points from top"),
    font_size: float = Form(default=12, description="Font size"),
    color: str = Form(default="0,0,0", description="RGB color as 'r,g,b' (0-1 range)"),
    save_as_new: bool = Form(default=True, description="Save as new document instead of overwriting"),
    output_filename: str = Form(default="", description="Output filename (only if save_as_new)"),
):
    """Add text annotation/watermark to a PDF page."""
    import fitz

    doc = db.query(DBDocument).filter(DBDocument.id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    pdf_bytes = _load_pdf_bytes(doc)
    pdf = fitz.open(stream=pdf_bytes, filetype="pdf")

    if page < 1 or page > pdf.page_count:
        pdf.close()
        raise HTTPException(
            status_code=400,
            detail=f"Page {page} out of range (document has {pdf.page_count} pages)",
        )

    # Parse and validate color
    try:
        rgb = tuple(float(c.strip()) for c in color.split(","))
        if len(rgb) != 3 or not all(0.0 <= v <= 1.0 for v in rgb):
            raise ValueError
    except (ValueError, TypeError):
        pdf.close()
        raise HTTPException(status_code=400, detail="Color must be three comma-separated floats between 0.0 and 1.0, e.g. '0,0,0' for black")

    pg = pdf[page - 1]
    pg.insert_text(
        fitz.Point(x, y),
        text,
        fontsize=font_size,
        color=rgb,
    )

    annotated_bytes = pdf.tobytes()
    pdf.close()

    client_ip = _get_client_ip(request)
    original_name = _safe_decrypt(doc.filename, f"Document {doc.id[:8]}.pdf")

    if save_as_new:
        clean_name = _sanitize_filename(output_filename or f"annotated_{original_name}")
        if not clean_name.lower().endswith(".pdf"):
            clean_name += ".pdf"
        new_id = str(uuid.uuid4())
        save_path = UPLOAD_DIR / f"{new_id}.pdf.enc"
        _encrypt_and_save(annotated_bytes, save_path)

        pages_data = extract_structured_text(annotated_bytes)
        content_hash = hashlib.sha256(annotated_bytes).hexdigest()

        db_doc = DBDocument(
            id=new_id,
            filename=_encrypt_text(clean_name),
            filepath=str(save_path),
            page_count=len(pages_data),
            text_content=_encrypt_text(json.dumps(pages_data)),
            content_hash=content_hash,
            uploaded_at=datetime.now(timezone.utc),
        )
        db.add(db_doc)
        db.commit()
        log_upload(client_ip, clean_name, new_id, len(pages_data))
        return {"id": new_id, "filename": clean_name, "pages": len(pages_data)}
    else:
        # Overwrite existing document
        save_path = Path(doc.filepath)
        _encrypt_and_save(annotated_bytes, save_path)
        pages_data = extract_text_from_bytes(annotated_bytes)
        doc.page_count = len(pages_data)
        doc.text_content = _encrypt_text(json.dumps(pages_data))
        doc.content_hash = hashlib.sha256(annotated_bytes).hexdigest()
        db.commit()
        return {"id": doc_id, "filename": original_name, "pages": len(pages_data), "updated": True}
