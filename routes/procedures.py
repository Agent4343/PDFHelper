"""
Procedure Builder routes — AI-guided procedure writing with business templates.

Covers: /procedures, /procedures/{id}, /procedures/{id}/chat,
        /procedures/{id}/generate, /procedures/{id}/download
"""

import json
import uuid
from datetime import datetime, timezone

from anthropic import Anthropic
from fastapi import APIRouter, Depends, HTTPException, Query, Request, UploadFile, File
from fastapi.responses import Response, StreamingResponse

from auth import verify_auth, get_db
from config import CHAT_MODEL, CHAT_MAX_TOKENS
from database import (
    DBDocument,
    DBProcedureSession,
    DBProcedureMessage,
    SessionLocal,
)
from helpers import _encrypt_text, _decrypt_text, _safe_decrypt, _load_stored_text

router = APIRouter(dependencies=[Depends(verify_auth)])

PROCEDURE_SYSTEM_PROMPT = """You are an expert technical procedure writer specializing in human performance principles. Your job is to help create clear, precise, and safe work procedures.

When gathering information, ask focused questions one at a time about:
- Procedure title and number
- Purpose and scope
- Required personnel and qualifications
- Tools, equipment, and materials needed
- Precautions and safety considerations
- Prerequisites and initial conditions
- Step-by-step instructions (action-first, one action per step)
- Verification and sign-off requirements
- References and related documents

Writing style principles:
- Action-first steps (start with a verb)
- One action per step
- Specific and measurable language
- Clear hold points and decision points
- Proper use of notes, cautions, and warnings (placed BEFORE the step they apply to)
- No ambiguous words (ensure, appropriate, proper — replace with specifics)

{style_config}

{template_config}

If the user provides an existing procedure to update, identify what needs changing and ask targeted questions about the updates needed.

Always be thorough — a missing step or unclear instruction in a procedure can lead to safety incidents."""


@router.get("/procedures")
async def list_procedures(
    request: Request,
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    db=Depends(get_db),
):
    user_id = getattr(request.state, "user_id", None)
    query = db.query(DBProcedureSession)
    if user_id:
        query = query.filter(DBProcedureSession.user_id == user_id)
    total = query.count()
    sessions = query.order_by(DBProcedureSession.updated_at.desc()).offset(skip).limit(limit).all()
    return {
        "sessions": [
            {
                "id": s.id,
                "title": _safe_decrypt(s.title) or "Untitled Procedure",
                "status": s.status,
                "created_at": s.created_at.isoformat(),
                "updated_at": s.updated_at.isoformat(),
            }
            for s in sessions
        ],
        "total": total,
    }


@router.post("/procedures")
async def create_procedure(request: Request, db=Depends(get_db)):
    body = await request.json()
    title = body.get("title", "").strip() or "New Procedure"
    source_doc_id = body.get("source_doc_id")

    if source_doc_id:
        doc = db.query(DBDocument).filter(DBDocument.id == source_doc_id).first()
        if not doc:
            raise HTTPException(status_code=404, detail="Source document not found")

    now = datetime.now(timezone.utc)
    session = DBProcedureSession(
        id=str(uuid.uuid4()),
        user_id=getattr(request.state, "user_id", None),
        title=_encrypt_text(title),
        source_doc_id=source_doc_id,
        status="gathering",
        created_at=now,
        updated_at=now,
    )
    db.add(session)
    db.commit()

    initial_msg = "I'm ready to help you write a procedure"
    if source_doc_id:
        initial_msg = "I can see you've attached an existing procedure to update. Let me review it and ask you about the changes needed."

    assistant_msg = DBProcedureMessage(
        id=str(uuid.uuid4()),
        session_id=session.id,
        role="assistant",
        content=_encrypt_text(initial_msg),
        created_at=now,
    )
    db.add(assistant_msg)
    db.commit()

    return {
        "id": session.id,
        "title": title,
        "status": session.status,
        "messages": [{"role": "assistant", "content": initial_msg}],
    }


@router.get("/procedures/{session_id}")
async def get_procedure(session_id: str, db=Depends(get_db)):
    session = db.query(DBProcedureSession).filter(DBProcedureSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Procedure session not found")

    messages = [
        {"role": m.role, "content": _safe_decrypt(m.content) or m.content}
        for m in session.messages
    ]

    return {
        "id": session.id,
        "title": _safe_decrypt(session.title) or "Untitled Procedure",
        "status": session.status,
        "source_doc_id": session.source_doc_id,
        "messages": messages,
        "created_at": session.created_at.isoformat(),
        "updated_at": session.updated_at.isoformat(),
    }


@router.delete("/procedures/{session_id}")
async def delete_procedure(session_id: str, db=Depends(get_db)):
    session = db.query(DBProcedureSession).filter(DBProcedureSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Procedure session not found")
    db.delete(session)
    db.commit()
    return {"detail": "Procedure session deleted"}


@router.post("/procedures/{session_id}/chat")
async def procedure_chat(session_id: str, request: Request, db=Depends(get_db)):
    session = db.query(DBProcedureSession).filter(DBProcedureSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Procedure session not found")

    body = await request.json()
    user_message = body.get("message", "").strip()
    if not user_message:
        raise HTTPException(status_code=400, detail="Message is required")

    now = datetime.now(timezone.utc)
    user_msg = DBProcedureMessage(
        id=str(uuid.uuid4()),
        session_id=session.id,
        role="user",
        content=_encrypt_text(user_message),
        created_at=now,
    )
    db.add(user_msg)
    session.updated_at = now
    db.commit()

    style_config = ""
    if session.style_config:
        style_config = f"Additional style rules:\n{_safe_decrypt(session.style_config) or ''}"

    template_config = ""
    if session.template_config:
        template_config = f"Template structure:\n{_safe_decrypt(session.template_config) or ''}"

    source_context = ""
    if session.source_doc_id:
        doc = db.query(DBDocument).filter(DBDocument.id == session.source_doc_id).first()
        if doc:
            text = _load_stored_text(doc)
            if text:
                source_context = f"\n\n--- EXISTING PROCEDURE (to update) ---\n{text[:8000]}\n--- END ---"

    system_prompt = PROCEDURE_SYSTEM_PROMPT.format(
        style_config=style_config,
        template_config=template_config,
    )
    if source_context:
        system_prompt += source_context

    history = []
    for m in session.messages:
        content = _safe_decrypt(m.content) or m.content
        if m.role == "user":
            history.append({"role": "user", "content": content})
        else:
            history.append({"role": "assistant", "content": content})

    history.append({"role": "user", "content": user_message})

    client = Anthropic()

    def generate():
        full_response = []
        with client.messages.stream(
            model=CHAT_MODEL,
            max_tokens=CHAT_MAX_TOKENS,
            system=system_prompt,
            messages=history,
        ) as stream:
            for text in stream.text_stream:
                full_response.append(text)
                yield f"data: {json.dumps({'type': 'text', 'content': text})}\n\n"

        assistant_content = "".join(full_response)
        save_db = SessionLocal()
        try:
            assistant_msg = DBProcedureMessage(
                id=str(uuid.uuid4()),
                session_id=session.id,
                role="assistant",
                content=_encrypt_text(assistant_content),
                created_at=datetime.now(timezone.utc),
            )
            save_db.add(assistant_msg)
            save_db.commit()
        finally:
            save_db.close()

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@router.post("/procedures/{session_id}/generate")
async def generate_procedure(session_id: str, request: Request, db=Depends(get_db)):
    """Generate the final Word document from gathered information."""
    session = db.query(DBProcedureSession).filter(DBProcedureSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Procedure session not found")

    style_config = ""
    if session.style_config:
        style_config = f"Additional style rules:\n{_safe_decrypt(session.style_config) or ''}"

    template_config = ""
    if session.template_config:
        template_config = f"Template structure:\n{_safe_decrypt(session.template_config) or ''}"

    history = []
    for m in session.messages:
        content = _safe_decrypt(m.content) or m.content
        history.append({"role": m.role, "content": content})

    generation_prompt = """Based on all the information gathered in this conversation, generate the complete procedure document now.

Format it with clear sections, numbered steps, and proper use of Notes/Cautions/Warnings.
Output the procedure in a structured format that can be converted to a Word document.
Use the template structure and style rules provided."""

    history.append({"role": "user", "content": generation_prompt})

    client = Anthropic()

    def generate():
        full_response = []
        with client.messages.stream(
            model=CHAT_MODEL,
            max_tokens=CHAT_MAX_TOKENS,
            system=PROCEDURE_SYSTEM_PROMPT.format(
                style_config=style_config,
                template_config=template_config,
            ),
            messages=history,
        ) as stream:
            for text in stream.text_stream:
                full_response.append(text)
                yield f"data: {json.dumps({'type': 'text', 'content': text})}\n\n"

        content = "".join(full_response)
        save_db = SessionLocal()
        try:
            proc = save_db.query(DBProcedureSession).filter(DBProcedureSession.id == session_id).first()
            if proc:
                proc.output_content = _encrypt_text(content)
                proc.status = "complete"
                proc.updated_at = datetime.now(timezone.utc)
                save_db.commit()
        finally:
            save_db.close()

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@router.get("/procedures/{session_id}/download")
async def download_procedure(session_id: str, db=Depends(get_db)):
    """Download the generated procedure as a Word document."""
    session = db.query(DBProcedureSession).filter(DBProcedureSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Procedure session not found")
    if not session.output_content:
        raise HTTPException(status_code=400, detail="Procedure not yet generated")

    from docx import Document
    from docx.shared import Pt, Inches
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    import io

    content = _safe_decrypt(session.output_content) or ""
    title = _safe_decrypt(session.title) or "Procedure"

    doc = Document()
    style = doc.styles["Normal"]
    style.font.name = "Arial"
    style.font.size = Pt(11)

    doc.add_heading(title, level=0)

    for line in content.split("\n"):
        line = line.rstrip()
        if not line:
            doc.add_paragraph("")
        elif line.startswith("# "):
            doc.add_heading(line[2:], level=1)
        elif line.startswith("## "):
            doc.add_heading(line[3:], level=2)
        elif line.startswith("### "):
            doc.add_heading(line[4:], level=3)
        elif line.startswith("**") and line.endswith("**"):
            p = doc.add_paragraph()
            run = p.add_run(line.strip("*"))
            run.bold = True
        elif line.startswith("- ") or line.startswith("• "):
            doc.add_paragraph(line[2:], style="List Bullet")
        elif line[0:1].isdigit() and ". " in line[:5]:
            idx = line.index(". ")
            doc.add_paragraph(line[idx + 2:], style="List Number")
        else:
            doc.add_paragraph(line)

    buf = io.BytesIO()
    doc.save(buf)
    buf.seek(0)

    safe_title = "".join(c for c in title if c.isalnum() or c in " -_")[:50].strip() or "procedure"
    return Response(
        content=buf.getvalue(),
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": f'attachment; filename="{safe_title}.docx"'},
    )
