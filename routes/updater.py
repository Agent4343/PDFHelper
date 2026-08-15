"""Doc Updater routes — Structure, Regulation Search, Updates, Review, Sessions."""

import io
import json
import logging
import os
import re
import uuid
import zipfile
from datetime import datetime, timezone

from anthropic import Anthropic
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, Response, StreamingResponse

from auth import verify_api_key, get_db
from config import CHAT_MODEL, CHAT_MAX_TOKENS, AGENT_MAX_TOKENS, CHAT_WEB_SEARCH, IS_PRODUCTION
from database import DBDocument, DBUpdateSession
from helpers import _encrypt_text, _safe_decrypt, _load_pdf_bytes, _markdown_to_docx, _build_vba_module
from models import (
    RegulationSearchRequest,
    GenerateUpdatesRequest,
    ReviewSectionRequest,
    ApplyUpdatesRequest,
    SaveSessionRequest,
)
from ocr import extract_structured_text

router = APIRouter()

logger = logging.getLogger("pdfhelper")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/documents/{doc_id}/structure", dependencies=[Depends(verify_api_key)])
async def get_document_structure(doc_id: str, db=Depends(get_db)):
    """Return structured content extraction for a document (headings, paragraphs, lists, tables)."""
    doc = db.query(DBDocument).filter(DBDocument.id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    pdf_bytes = _load_pdf_bytes(doc)
    structured = extract_structured_text(pdf_bytes)
    return {"doc_id": doc_id, "filename": _safe_decrypt(doc.filename, f"Document {doc.id[:8]}"), "pages": structured}


@router.get("/documents/{doc_id}/html", dependencies=[Depends(verify_api_key)])
async def get_document_html(doc_id: str, db=Depends(get_db)):
    """Return an HTML rendering of a document's structured content for the in-browser viewer."""
    doc = db.query(DBDocument).filter(DBDocument.id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    pdf_bytes = _load_pdf_bytes(doc)
    structured = extract_structured_text(pdf_bytes)

    html_parts = []
    block_idx = 0
    for page_data in structured:
        page_num = page_data["page"]
        html_parts.append(f'<div class="doc-page" data-page="{page_num}">')
        html_parts.append(f'<div class="page-header">Page {page_num}</div>')
        for block in page_data.get("blocks", []):
            btype = block["type"]
            text = block["text"].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            bid = f"blk-{block_idx}"
            block_idx += 1
            if btype == "heading":
                html_parts.append(f'<h3 class="doc-heading doc-block" id="{bid}" data-type="heading" data-idx="{block_idx}">{text}</h3>')
            elif btype == "list_item":
                html_parts.append(f'<p class="doc-list-item doc-block" id="{bid}" data-type="list_item" data-idx="{block_idx}">{text}</p>')
            elif btype == "table":
                rows = block.get("rows", [])
                if rows:
                    html_parts.append(f'<table class="doc-table doc-block" id="{bid}" data-type="table" data-idx="{block_idx}"><tbody>')
                    for ri, row in enumerate(rows):
                        tag = "th" if ri == 0 else "td"
                        cells = "".join(
                            f"<{tag}>{str(c).replace('&','&amp;').replace('<','&lt;').replace('>','&gt;')}</{tag}>"
                            for c in row
                        )
                        html_parts.append(f"<tr>{cells}</tr>")
                    html_parts.append("</tbody></table>")
                else:
                    html_parts.append(f'<pre class="doc-table-text doc-block" id="{bid}" data-type="table" data-idx="{block_idx}">{text}</pre>')
            else:
                html_parts.append(f'<p class="doc-para doc-block" id="{bid}" data-type="paragraph" data-idx="{block_idx}">{text}</p>')
        html_parts.append("</div>")

    return HTMLResponse("\n".join(html_parts))


@router.get("/documents/{doc_id}/detect-regulations", dependencies=[Depends(verify_api_key)])
async def detect_regulations(doc_id: str, db=Depends(get_db)):
    """Scan a document and return detected regulation/standard references."""
    doc = db.query(DBDocument).filter(DBDocument.id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    pages = json.loads(_safe_decrypt(doc.text_content, "[]"))
    full_text = "\n".join(p["text"] for p in pages if p.get("text"))

    patterns = [
        r'(?:OSHA|29\s*CFR)\s*[\d.]+(?:\([a-z]\))?',
        r'(?:API|ASME|ANSI|NFPA|ISO|IEC|IEEE|ASTM|CSA|CGA|DOT|EPA|MSHA)\s*[\d][\w.\-]*',
        r'(?:AS|BS|EN|DIN|JIS|NF|GB)\s*\d[\w.\-]*',
        r'(?:NEC|NESC|CFR|USC|FR)\s*[\d.]+',
        r'(?:Part|Section|Subpart)\s+\d[\w.\-]*',
        r'(?:29|30|33|40|46|49)\s*CFR\s*[\d.]+',
    ]
    found = set()
    for pattern in patterns:
        for match in re.finditer(pattern, full_text, re.IGNORECASE):
            ref = match.group(0).strip()
            if len(ref) > 3:
                found.add(ref)

    refs = sorted(found)
    suggested_query = ""
    if refs:
        top_refs = refs[:10]
        suggested_query = "Current requirements for: " + ", ".join(top_refs)

    return {
        "doc_id": doc_id,
        "regulations_found": refs,
        "count": len(refs),
        "suggested_query": suggested_query,
    }


@router.post("/regulations/search", dependencies=[Depends(verify_api_key)])
async def search_regulations(body: RegulationSearchRequest, db=Depends(get_db)):
    """Search the web for current regulations relevant to a query or document."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not configured on server")

    doc_context = ""
    if body.doc_id:
        doc = db.query(DBDocument).filter(DBDocument.id == body.doc_id).first()
        if doc:
            pages = json.loads(_safe_decrypt(doc.text_content, "[]"))
            full_text = "\n".join(p["text"] for p in pages if p.get("text"))
            if len(full_text) > 30000:
                full_text = full_text[:30000] + "\n[... truncated ...]"
            doc_context = f"\n\nDOCUMENT CONTENT TO CHECK AGAINST:\n{full_text}"

    client = Anthropic(api_key=api_key)

    system = f"""You are a regulatory compliance researcher. Search the web for current regulations, standards, and requirements related to the user's query.{doc_context}

Return your findings as a structured analysis with these sections:
1. **Regulations Found** — list each regulation/standard with its current version and source
2. **Key Requirements** — summarize the main requirements from each regulation
3. **Relevance to Document** — if a document was provided, explain how each regulation applies
4. **Recommended Updates** — specific changes the document should make for compliance

Use markdown formatting. Cite sources with URLs when available."""

    create_kwargs = dict(
        model=CHAT_MODEL,
        max_tokens=CHAT_MAX_TOKENS,
        system=system,
        messages=[{"role": "user", "content": body.query + (f"\n\nAdditional context: {body.context}" if body.context else "")}],
        tools=[{"type": "web_search_20250305", "name": "web_search"}],
    )

    async def stream_search():
        full_reply = ""
        try:
            with client.messages.stream(**create_kwargs) as stream:
                for event in stream:
                    if hasattr(event, 'type'):
                        if event.type == 'content_block_start':
                            if hasattr(event.content_block, 'type') and event.content_block.type == 'server_tool_use':
                                yield f"data: {json.dumps({'type': 'status', 'message': 'Searching the web for regulations...'})}\n\n"
                        elif event.type == 'content_block_delta':
                            if hasattr(event.delta, 'text'):
                                full_reply += event.delta.text
                                yield f"data: {json.dumps({'type': 'chunk', 'text': event.delta.text})}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'full_text': full_reply})}\n\n"
        except Exception as e:
            logger.error("Regulation search failed: %s", e)
            err_msg = "Search failed" if IS_PRODUCTION else f"Search failed: {str(e)}"
            yield f"data: {json.dumps({'type': 'error', 'detail': err_msg})}\n\n"

    return StreamingResponse(stream_search(), media_type="text/event-stream")


@router.post("/documents/{doc_id}/generate-updates", dependencies=[Depends(verify_api_key)])
async def generate_updates(doc_id: str, body: GenerateUpdatesRequest, db=Depends(get_db)):
    """Generate proposed updates for a document based on regulation findings."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not configured on server")

    doc = db.query(DBDocument).filter(DBDocument.id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    pdf_bytes = _load_pdf_bytes(doc)
    structured = extract_structured_text(pdf_bytes)
    doc_content = ""
    for page_data in structured:
        doc_content += f"\n--- Page {page_data['page']} ---\n"
        for block in page_data.get("blocks", []):
            prefix = f"[{block['type'].upper()}] " if block['type'] != 'paragraph' else ""
            doc_content += f"{prefix}{block['text']}\n"
    if len(doc_content) > 80000:
        doc_content = doc_content[:80000] + "\n[... truncated ...]"

    client = Anthropic(api_key=api_key)

    system = f"""You are a document update specialist. You have the original document structure and regulation findings. Generate specific, localized updates.

ORIGINAL DOCUMENT:
{doc_content}

REGULATION FINDINGS:
{body.regulation_text[:50000]}

{('ADDITIONAL INSTRUCTIONS: ' + body.additional_instructions) if body.additional_instructions else ''}

You MUST respond with a JSON array of update objects. Each object has these fields:
- "id": a unique short identifier like "upd-1", "upd-2", etc.
- "section": the section name or heading this update applies to
- "change_type": one of "replace", "insert", or "delete"
- "original_text": the exact original text being changed (quote it precisely)
- "proposed_text": the new text to replace it with (empty string for deletions)
- "rationale": why this change is needed, citing the specific regulation

Respond ONLY with a valid JSON array. No markdown, no explanation outside the JSON. Example:
[
  {{"id": "upd-1", "section": "PPE Requirements", "change_type": "replace", "original_text": "Hard hats required", "proposed_text": "Hard hats and safety glasses required per OSHA 1926.100", "rationale": "OSHA 1926.100 requires eye protection in addition to head protection"}}
]"""

    create_kwargs = dict(
        model=CHAT_MODEL,
        max_tokens=AGENT_MAX_TOKENS,
        system=system,
        messages=[{"role": "user", "content": "Analyze the document and generate all necessary updates as a JSON array."}],
    )

    response = client.messages.create(**create_kwargs)
    full_text = ""
    for block in response.content:
        if hasattr(block, "text"):
            full_text += block.text

    # Try to parse as JSON array
    updates = []
    try:
        # Strip markdown code fences if present
        cleaned = full_text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r'^```\w*\n?', '', cleaned)
            cleaned = re.sub(r'\n?```$', '', cleaned)
        updates = json.loads(cleaned)
        if not isinstance(updates, list):
            updates = [updates]
    except json.JSONDecodeError:
        # Fallback: return as single raw text block
        updates = [{"id": "upd-1", "section": "Full Document", "change_type": "replace",
                     "original_text": "", "proposed_text": full_text, "rationale": "AI-generated update (could not parse structured blocks)"}]

    return {"updates": updates, "raw_text": full_text}


@router.post("/documents/{doc_id}/review-section", dependencies=[Depends(verify_api_key)])
async def review_section(doc_id: str, body: ReviewSectionRequest, db=Depends(get_db)):
    """AI-review a highlighted section of a document for compliance and improvements."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not configured on server")

    doc = db.query(DBDocument).filter(DBDocument.id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    pages = json.loads(_safe_decrypt(doc.text_content, "[]"))
    full_text = "\n".join(p["text"] for p in pages if p.get("text"))
    if len(full_text) > 40000:
        full_text = full_text[:40000] + "\n[... truncated ...]"

    client = Anthropic(api_key=api_key)

    focus_areas = [f.strip() for f in body.focus.split(",") if f.strip()]
    focus_str = ", ".join(focus_areas) if focus_areas else "compliance, clarity, completeness"

    system = f"""You are a document review specialist. The user has highlighted a section of a procedure document for review.

FULL DOCUMENT CONTEXT:
{full_text}

Review the highlighted section focusing on: {focus_str}

You MUST respond with a JSON object containing:
- "issues": array of strings describing problems found
- "suggested_replacement": the rewritten section text
- "rationale": explanation of why changes were made
- "regulation_refs": array of relevant regulation references

Respond ONLY with valid JSON. No markdown outside the JSON."""

    create_kwargs = dict(
        model=CHAT_MODEL,
        max_tokens=CHAT_MAX_TOKENS,
        system=system,
        messages=[{"role": "user", "content": f"Review this highlighted section:\n\n{body.highlighted_text}" + (f"\n\nAdditional context: {body.context}" if body.context else "")}],
    )
    if CHAT_WEB_SEARCH:
        create_kwargs["tools"] = [{"type": "web_search_20250305", "name": "web_search"}]

    async def stream_review():
        full_reply = ""
        try:
            with client.messages.stream(**create_kwargs) as stream:
                for event in stream:
                    if hasattr(event, 'type'):
                        if event.type == 'content_block_start':
                            if hasattr(event.content_block, 'type') and event.content_block.type == 'server_tool_use':
                                yield f"data: {json.dumps({'type': 'status', 'message': 'Searching regulations...'})}\n\n"
                        elif event.type == 'content_block_delta':
                            if hasattr(event.delta, 'text'):
                                full_reply += event.delta.text
                                yield f"data: {json.dumps({'type': 'chunk', 'text': event.delta.text})}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'full_text': full_reply})}\n\n"
        except Exception as e:
            logger.error("Section review failed: %s", e)
            err_msg = "Review failed" if IS_PRODUCTION else f"Review failed: {str(e)}"
            yield f"data: {json.dumps({'type': 'error', 'detail': err_msg})}\n\n"

    return StreamingResponse(stream_review(), media_type="text/event-stream")


@router.post("/documents/{doc_id}/apply-updates", dependencies=[Depends(verify_api_key)])
async def apply_updates_to_document(doc_id: str, body: ApplyUpdatesRequest, db=Depends(get_db)):
    """Generate a Word document with the proposed updates applied."""
    doc = db.query(DBDocument).filter(DBDocument.id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not configured on server")

    pages = json.loads(_safe_decrypt(doc.text_content, "[]"))
    full_text = "\n".join(p["text"] for p in pages if p.get("text"))
    if len(full_text) > 80000:
        full_text = full_text[:80000] + "\n[... truncated ...]"

    client = Anthropic(api_key=api_key)

    system = f"""You are a professional document writer. You have the original document and a set of proposed updates. Write the COMPLETE updated document incorporating all the accepted changes.

ORIGINAL DOCUMENT:
{full_text}

PROPOSED UPDATES:
{body.updates_markdown[:80000]}

Write the complete updated document using markdown formatting:
- Use # for main title, ## for sections, ### for subsections
- Use **bold** for emphasis
- Use numbered lists (1. 2. 3.) and bullet lists (- item)
- Preserve the original document structure and tone
- Incorporate all the proposed changes
- The output should be the FULL document, not just the changed sections"""

    response = client.messages.create(
        model=CHAT_MODEL,
        max_tokens=AGENT_MAX_TOKENS,
        system=system,
        messages=[{"role": "user", "content": f"Please write the complete updated document titled '{body.title}'."}],
    )
    full_doc_text = ""
    for block in response.content:
        if hasattr(block, "text"):
            full_doc_text += block.text

    if not full_doc_text.strip():
        raise HTTPException(status_code=500, detail="AI failed to generate document")

    docx_bytes = _markdown_to_docx(full_doc_text, body.title)
    safe_title = re.sub(r'[^\w\s-]', '', body.title)[:50].strip() or "updated-document"

    if body.include_vba:
        zip_buf = io.BytesIO()
        vba_code = _build_vba_module(body.title)
        with zipfile.ZipFile(zip_buf, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(f"{safe_title}.docx", docx_bytes)
            zf.writestr(f"{safe_title}_macros.bas", vba_code)
            zf.writestr("README.txt",
                f"UPDATED DOCUMENT PACKAGE\n"
                f"========================\n\n"
                f"1. {safe_title}.docx - The updated document\n"
                f"2. {safe_title}_macros.bas - VBA macros for formatting\n\n"
                f"To use macros: Open .docx in Word, Alt+F11, File > Import File\n"
            )
        return Response(
            content=zip_buf.getvalue(),
            media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="{safe_title}_package.zip"'},
        )

    return Response(
        content=docx_bytes,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": f'attachment; filename="{safe_title}.docx"'},
    )


# -- Update Sessions: save / load / list --

@router.post("/updater/sessions", dependencies=[Depends(verify_api_key)])
async def save_update_session(body: SaveSessionRequest, request: Request, db=Depends(get_db)):
    """Save a Doc Updater session so the user can resume later."""
    now = datetime.now(timezone.utc)
    current_user_id = getattr(request.state, "user_id", None)
    session = DBUpdateSession(
        id=str(uuid.uuid4()),
        doc_id=body.doc_id,
        user_id=current_user_id,
        title=body.title or f"Session {now.strftime('%b %d %H:%M')}",
        regulation_query=body.regulation_query,
        regulation_results=_encrypt_text(body.regulation_results) if body.regulation_results else None,
        updates_json=_encrypt_text(body.updates_json) if body.updates_json else None,
        accepted_ids=json.dumps(body.accepted_ids),
        status="draft",
        created_at=now,
        updated_at=now,
    )
    db.add(session)
    db.commit()
    return {"id": session.id, "title": session.title, "created_at": now.isoformat()}


@router.get("/updater/sessions", dependencies=[Depends(verify_api_key)])
async def list_update_sessions(request: Request, db=Depends(get_db)):
    """List saved Doc Updater sessions."""
    current_user_id = getattr(request.state, "user_id", None)
    query = db.query(DBUpdateSession).order_by(DBUpdateSession.updated_at.desc())
    if current_user_id:
        query = query.filter(DBUpdateSession.user_id == current_user_id)
    sessions = query.limit(50).all()
    return {"sessions": [{
        "id": s.id, "doc_id": s.doc_id, "title": s.title,
        "status": s.status, "regulation_query": s.regulation_query,
        "created_at": s.created_at.isoformat(), "updated_at": s.updated_at.isoformat(),
    } for s in sessions]}


@router.get("/updater/sessions/{session_id}", dependencies=[Depends(verify_api_key)])
async def get_update_session(session_id: str, request: Request, db=Depends(get_db)):
    """Load a saved Doc Updater session."""
    session = db.query(DBUpdateSession).filter(DBUpdateSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    current_user_id = getattr(request.state, "user_id", None)
    if current_user_id and session.user_id and session.user_id != current_user_id:
        raise HTTPException(status_code=403, detail="You do not own this session")
    return {
        "id": session.id, "doc_id": session.doc_id, "title": session.title,
        "regulation_query": session.regulation_query,
        "regulation_results": _safe_decrypt(session.regulation_results) if session.regulation_results else "",
        "updates_json": _safe_decrypt(session.updates_json, "[]") if session.updates_json else "[]",
        "accepted_ids": json.loads(session.accepted_ids) if session.accepted_ids else [],
        "status": session.status,
    }


@router.delete("/updater/sessions/{session_id}", dependencies=[Depends(verify_api_key)])
async def delete_update_session(session_id: str, request: Request, db=Depends(get_db)):
    """Delete a saved Doc Updater session."""
    session = db.query(DBUpdateSession).filter(DBUpdateSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    current_user_id = getattr(request.state, "user_id", None)
    if current_user_id and session.user_id and session.user_id != current_user_id:
        raise HTTPException(status_code=403, detail="You do not own this session")
    db.delete(session)
    db.commit()
    return {"deleted": True}
