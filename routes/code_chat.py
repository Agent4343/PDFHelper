import asyncio
import json
import logging
import os
import uuid
from datetime import datetime, timezone

from anthropic import Anthropic
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from starlette.responses import StreamingResponse

from auth import verify_auth, verify_api_key, get_db
from config import CHAT_MODEL, CHAT_MAX_TOKENS, AGENT_MODELS, IS_PRODUCTION
from database import DBDocument, DBCodeSession, DBCodeMessage, SessionLocal, code_session_documents
from models import CodeChatRequest
from helpers import _encrypt_text, _decrypt_text, _safe_decrypt, _load_stored_text, _stored_text_to_structured

router = APIRouter()


# ---------------------------------------------------------------------------
# Code Chat — Iterative Code Generation
# ---------------------------------------------------------------------------

@router.post("/code-chat", dependencies=[Depends(verify_api_key)])
async def code_chat(
    request: Request,
    body: CodeChatRequest,
    db=Depends(get_db),
):
    """Chat-style iterative code generation using document data."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not configured on server")

    query = db.query(DBDocument)
    if body.doc_ids:
        query = query.filter(DBDocument.id.in_(body.doc_ids))
    documents = query.all()

    if not documents:
        raise HTTPException(status_code=404, detail="No documents found. Upload documents first.")

    now = datetime.now(timezone.utc)
    current_user_id = getattr(request.state, "user_id", None)
    session = None
    if body.session_id:
        session = db.query(DBCodeSession).filter(DBCodeSession.id == body.session_id).first()
        if session and current_user_id and session.user_id and session.user_id != current_user_id:
            raise HTTPException(status_code=403, detail="You do not own this code session")

    if session is None:
        docs = db.query(DBDocument).filter(DBDocument.id.in_(body.doc_ids)).all() if body.doc_ids else []
        session = DBCodeSession(
            id=str(uuid.uuid4()),
            user_id=current_user_id,
            title=body.message[:100],
            documents=docs,
            created_at=now,
            updated_at=now,
        )
        db.add(session)
        db.flush()

    procedure_parts = []
    for doc in documents:
        try:
            decrypted_name = _decrypt_text(doc.filename)
        except Exception:
            decrypted_name = f"Document {doc.id[:8]}"
        try:
            pages = _load_stored_text(doc)
            full_text = _stored_text_to_structured(pages)
        except Exception:
            full_text = f"(Could not load content for {decrypted_name})"
        per_doc_limit = max(200000, 2500000 // max(len(documents), 1))
        if len(full_text) > per_doc_limit:
            full_text = full_text[:per_doc_limit] + "\n\n[... content truncated ...]"
        procedure_parts.append(
            f'--- PROCEDURE: "{decrypted_name}" ---\n{full_text}\n--- END OF "{decrypted_name}" ---'
        )

    doc_index_lines = ["DOCUMENT INDEX:"]
    for i, doc in enumerate(documents, 1):
        try:
            dn = _decrypt_text(doc.filename)
        except Exception:
            dn = f"Document {doc.id[:8]}"
        doc_index_lines.append(f"  {i}. \"{dn}\" — {doc.page_count} pages")
    doc_index = "\n".join(doc_index_lines)
    procedure_context = doc_index + "\n\n" + "\n\n".join(procedure_parts)

    db_messages = (
        db.query(DBCodeMessage)
        .filter(DBCodeMessage.session_id == session.id)
        .order_by(DBCodeMessage.created_at)
        .all()
    )

    if db_messages:
        conversation = []
        for m in db_messages[-20:]:
            try:
                conversation.append({"role": m.role, "content": _decrypt_text(m.content)})
            except Exception:
                conversation.append({"role": m.role, "content": "(message could not be decrypted)"})
    else:
        conversation = [
            {"role": m.role, "content": m.content}
            for m in body.conversation_history[-20:]
            if m.role in ("user", "assistant")
        ]
    while conversation and conversation[0]["role"] != "user":
        conversation.pop(0)
    conversation.append({"role": "user", "content": body.message})

    MAX_TOTAL_CHARS = 3200000
    def _msg_text_len(m):
        c = m["content"]
        return len(c) if isinstance(c, str) else 0

    conv_chars = sum(_msg_text_len(m) for m in conversation)
    while conv_chars > 200000 and len(conversation) > 1:
        removed = conversation.pop(0)
        conv_chars -= _msg_text_len(removed)

    budget_for_procedures = MAX_TOTAL_CHARS - conv_chars
    if budget_for_procedures < 10000:
        budget_for_procedures = 10000
    if len(procedure_context) > budget_for_procedures:
        procedure_context = procedure_context[:budget_for_procedures] + "\n\n[... procedures truncated ...]"

    system_prompt = """You are a Code Generation Assistant. You generate complete, working code using data from the documents loaded below. You support ANY programming language or format the user requests, including but not limited to:

- HTML/CSS/JavaScript (web apps, dashboards, reports)
- VBA (Excel macros, Outlook automation, Word macros)
- Python, SQL, PowerShell, Bash
- JSON, XML, YAML configuration files
- Any other language or format

RULES:
1. ALL data in the generated code MUST come from the loaded documents. Never invent content.
2. Always generate COMPLETE files — never partial snippets. When updating, output the ENTIRE file.
3. For HTML: use modern HTML5, CSS3, and vanilla JavaScript. Make it responsive and professional.
4. For VBA: include proper Sub/Function declarations, error handling, and clear comments explaining each section.
5. If data from the documents includes tables, lists, or structured content, preserve that structure.
6. Cite which document and page the data came from (in comments appropriate to the language).
7. When asked to fix or change something, keep all existing functionality and only modify what was requested.

RESPONSE FORMAT:
- Start with a brief description of what you built/changed (1-2 sentences).
- Then provide the complete code in a single code block with the appropriate language tag (```html, ```vba, ```python, etc.)
- After the code block, list what data you used and from which documents/pages.
- If the user's request is unclear, ask clarifying questions before generating code."""

    system_blocks = [
        {
            "type": "text",
            "text": system_prompt,
            "cache_control": {"type": "ephemeral"},
        },
        {
            "type": "text",
            "text": f"LOADED PROCEDURES:\n{procedure_context}",
            "cache_control": {"type": "ephemeral"},
        },
    ]

    client = Anthropic(api_key=api_key)

    db.add(DBCodeMessage(
        id=str(uuid.uuid4()), session_id=session.id,
        role="user", content=_encrypt_text(body.message), created_at=now,
    ))
    db.commit()

    session_id = session.id
    doc_info = []
    for d in documents:
        try:
            doc_info.append({"id": d.id, "filename": _decrypt_text(d.filename)})
        except Exception:
            doc_info.append({"id": d.id, "filename": f"Document {d.id[:8]}"})

    async def stream_code_chat():
        full_reply = ""
        try:
            yield f"data: {json.dumps({'type': 'meta', 'session_id': session_id, 'documents_used': doc_info})}\n\n"

            chat_model = AGENT_MODELS.get(body.model, CHAT_MODEL) if body.model else CHAT_MODEL
            create_kwargs = dict(
                model=chat_model,
                max_tokens=CHAT_MAX_TOKENS,
                system=system_blocks,
                messages=conversation,
            )

            max_retries = 3
            for attempt in range(max_retries):
                try:
                    with client.messages.stream(**create_kwargs) as stream:
                        for event in stream:
                            if hasattr(event, 'type'):
                                if event.type == 'content_block_start':
                                    if hasattr(event.content_block, 'type') and event.content_block.type == 'thinking':
                                        yield f"data: {json.dumps({'type': 'status', 'message': 'Thinking...'})}\n\n"
                                elif event.type == 'content_block_delta':
                                    if hasattr(event.delta, 'text'):
                                        full_reply += event.delta.text
                                        yield f"data: {json.dumps({'type': 'chunk', 'text': event.delta.text})}\n\n"
                    break
                except Exception as retry_err:
                    err_str = str(retry_err)
                    is_retryable = "overloaded" in err_str.lower() or "529" in err_str or "rate" in err_str.lower()
                    if is_retryable and attempt < max_retries - 1:
                        wait_time = 2 ** (attempt + 1)
                        yield f"data: {json.dumps({'type': 'status', 'message': f'Server busy, retrying in {wait_time}s...'})}\n\n"
                        await asyncio.sleep(wait_time)
                        full_reply = ""
                        continue
                    raise

            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        except Exception as e:
            logging.getLogger("pdfhelper").error("Code chat stream failed: %s", e)
            err_msg = "AI request failed — please try again" if IS_PRODUCTION else f"AI request failed: {str(e)}"
            yield f"data: {json.dumps({'type': 'error', 'detail': err_msg})}\n\n"
            full_reply = full_reply or err_msg
        finally:
            save_db = SessionLocal()
            try:
                save_db.add(DBCodeMessage(
                    id=str(uuid.uuid4()), session_id=session_id,
                    role="assistant",
                    content=_encrypt_text(full_reply or "Sorry, I couldn't generate a response."),
                    created_at=datetime.now(timezone.utc),
                ))
                sess = save_db.query(DBCodeSession).filter(DBCodeSession.id == session_id).first()
                if sess:
                    sess.updated_at = datetime.now(timezone.utc)
                save_db.commit()
            except Exception:
                save_db.rollback()
            finally:
                save_db.close()

    return StreamingResponse(stream_code_chat(), media_type="text/event-stream")


@router.get("/code-chat/sessions", dependencies=[Depends(verify_api_key)])
async def list_code_sessions(request: Request, limit: int = Query(default=30, le=100), db=Depends(get_db)):
    """List code chat sessions, most recent first."""
    current_user_id = getattr(request.state, "user_id", None)
    query = db.query(DBCodeSession)
    if current_user_id:
        query = query.filter(
            (DBCodeSession.user_id == current_user_id) | (DBCodeSession.user_id.is_(None))
        )
    sessions = query.order_by(DBCodeSession.updated_at.desc()).limit(limit).all()
    return {
        "sessions": [
            {
                "id": s.id,
                "title": s.title,
                "doc_ids": [d.id for d in s.documents],
                "message_count": len(s.messages),
                "created_at": s.created_at.isoformat(),
                "updated_at": s.updated_at.isoformat(),
            }
            for s in sessions
        ]
    }


@router.get("/code-chat/sessions/{session_id}", dependencies=[Depends(verify_api_key)])
async def get_code_session(session_id: str, request: Request, db=Depends(get_db)):
    """Get full message history for a code session."""
    session = db.query(DBCodeSession).filter(DBCodeSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Code session not found")
    current_user_id = getattr(request.state, "user_id", None)
    if current_user_id and session.user_id and session.user_id != current_user_id:
        raise HTTPException(status_code=403, detail="You do not own this code session")
    messages = []
    for m in session.messages:
        try:
            content = _decrypt_text(m.content)
        except Exception:
            content = "(message could not be decrypted)"
        messages.append({
            "role": m.role,
            "content": content,
            "created_at": m.created_at.isoformat(),
        })
    return {
        "id": session.id,
        "title": session.title,
        "doc_ids": [d.id for d in session.documents],
        "created_at": session.created_at.isoformat(),
        "updated_at": session.updated_at.isoformat(),
        "messages": messages,
    }


@router.delete("/code-chat/sessions/{session_id}", dependencies=[Depends(verify_api_key)])
async def delete_code_session(session_id: str, request: Request, db=Depends(get_db)):
    """Delete a code session and all its messages."""
    session = db.query(DBCodeSession).filter(DBCodeSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Code session not found")
    current_user_id = getattr(request.state, "user_id", None)
    if current_user_id and session.user_id and session.user_id != current_user_id:
        raise HTTPException(status_code=403, detail="You do not own this code session")
    db.delete(session)
    db.commit()
    return {"deleted": session_id}
