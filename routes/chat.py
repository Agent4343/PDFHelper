"""
Chat router – extracted from app.py.

Covers: /chat, /chat/sessions, /chat/export, /chat/markdown-to-docx,
        /chat/generate-doc, /chat/improve-procedure
"""

import io
import json
import logging
import os
import re
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from anthropic import Anthropic
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import Response, StreamingResponse

from auth import verify_api_key, get_db
from config import (
    CHAT_MODEL,
    CHAT_MAX_TOKENS,
    CHAT_WEB_SEARCH,
    AGENT_MODELS,
    AGENT_MAX_TOKENS,
    IS_PRODUCTION,
)
from database import DBDocument, DBChatSession, DBChatMessage, SessionLocal
from helpers import (
    _encrypt_text,
    _decrypt_text,
    _safe_decrypt,
    _load_stored_text,
    _stored_text_to_structured,
    _load_pdf_bytes,
    _decrypt_and_load,
    _extract_image_base64,
    _detect_image_media_type,
    _markdown_to_docx,
    _build_vba_module,
)
from models import (
    ChatRequest,
    ExportChatRequest,
    GenerateDocRequest,
    MarkdownToDocxRequest,
    ImproveProcedureRequest,
)

router = APIRouter()


# ---------------------------------------------------------------------------
# Procedure Chatbot
# ---------------------------------------------------------------------------

@router.post("/chat", dependencies=[Depends(verify_api_key)])
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
    current_user_id = getattr(request.state, "user_id", None)
    session = None
    if body.session_id:
        session = db.query(DBChatSession).filter(DBChatSession.id == body.session_id).first()
        # Enforce session ownership: user can only access their own sessions
        if session and current_user_id and session.user_id and session.user_id != current_user_id:
            raise HTTPException(status_code=403, detail="You do not own this chat session")

    if session is None:
        session = DBChatSession(
            id=str(uuid.uuid4()),
            user_id=current_user_id,
            title=body.message[:100],
            doc_ids=json.dumps(body.doc_ids),
            created_at=now,
            updated_at=now,
        )
        db.add(session)
        db.flush()

    # Build procedure context from stored document text (extracted once at upload)
    procedure_parts = []
    image_content_blocks = []  # Claude vision blocks for uploaded images
    for doc in documents:
        try:
            decrypted_name = _decrypt_text(doc.filename)
        except Exception:
            decrypted_name = f"Document {doc.id[:8]}"
        try:
            pages = _load_stored_text(doc)
            full_text = _stored_text_to_structured(pages)
        except Exception as exc:
            logging.getLogger("pdfhelper").warning("text_content failed for %s: %s", doc.id, exc)
            full_text = f"(Could not load content for {decrypted_name})"
        per_doc_limit = max(200000, 2500000 // max(len(documents), 1))
        if len(full_text) > per_doc_limit:
            full_text = full_text[:per_doc_limit] + "\n\n[... content truncated for context window ...]"
        procedure_parts.append(
            f'--- PROCEDURE: "{decrypted_name}" ---\n{full_text}\n--- END OF "{decrypted_name}" ---'
        )

        # Check if this document has an associated image file for vision
        img_path = Path(doc.filepath.replace(".pdf.enc", ".img.enc"))
        if img_path.exists() and len(image_content_blocks) < 10:  # max 10 images
            try:
                img_bytes = _decrypt_and_load(img_path)
                media_type = _detect_image_media_type(img_bytes)
                b64 = _extract_image_base64(img_bytes)
                image_content_blocks.append({
                    "type": "text",
                    "text": f"[Image: {decrypted_name}]"
                })
                image_content_blocks.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": b64,
                    }
                })
            except Exception:
                pass  # skip if image can't be loaded

    doc_index_lines = ["DOCUMENT INDEX:"]
    for i, doc in enumerate(documents, 1):
        try:
            dn = _decrypt_text(doc.filename)
        except Exception:
            dn = f"Document {doc.id[:8]}"
        doc_index_lines.append(f"  {i}. \"{dn}\" — {doc.page_count} pages")
    doc_index = "\n".join(doc_index_lines)
    procedure_context = doc_index + "\n\n" + "\n\n".join(procedure_parts)

    # Build conversation from DB history (prefer DB over client-sent history)
    db_messages = (
        db.query(DBChatMessage)
        .filter(DBChatMessage.session_id == session.id)
        .order_by(DBChatMessage.created_at)
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
    # Build the user message -- include images via Claude vision only on the
    # first message of a session so they aren't re-sent on every turn.
    include_images = image_content_blocks and not db_messages
    if include_images:
        user_content = list(image_content_blocks)  # copy
        user_content.append({"type": "text", "text": body.message})
        conversation.append({"role": "user", "content": user_content})
    else:
        conversation.append({"role": "user", "content": body.message})

    # Budget the total context to stay within the model's context window.
    # Reserve chars for the system prompt template, response tokens, and safety margin.
    # Approximate: 1 token ~ 4 chars.  Model context ~ 200K tokens ~ 800K chars.
    # Each image ~ 1600 tokens, so subtract from budget when included.
    MAX_TOTAL_CHARS = 3200000  # ~800K tokens; uses most of the 1M-token context window
    if include_images:
        image_char_budget = len(image_content_blocks) // 2 * 6400  # ~1600 tokens * 4 chars per image
        MAX_TOTAL_CHARS -= image_char_budget

    def _msg_text_len(m):
        c = m["content"]
        if isinstance(c, str):
            return len(c)
        if isinstance(c, list):
            return sum(len(b.get("text", "")) for b in c if isinstance(b, dict) and b.get("type") == "text")
        return 0

    conv_chars = sum(_msg_text_len(m) for m in conversation)

    # If conversation history alone is too large, trim older messages (keep latest)
    while conv_chars > 200000 and len(conversation) > 1:
        removed = conversation.pop(0)
        conv_chars -= _msg_text_len(removed)

    budget_for_procedures = MAX_TOTAL_CHARS - conv_chars
    if budget_for_procedures < 10000:
        budget_for_procedures = 10000  # always keep at least some procedure context

    # Truncate procedure context if it exceeds budget
    if len(procedure_context) > budget_for_procedures:
        procedure_context = procedure_context[:budget_for_procedures] + "\n\n[... procedures truncated to fit context window ...]"

    system_prompt = """You are a Procedure Knowledge Assistant. Answer questions ONLY from the procedure documents loaded below.

RULES — READ THESE FIRST:
1. EVERY answer MUST come from the loaded documents. If you cannot find it, say "I could not find this in the loaded procedures." Do NOT guess or use general knowledge.
2. ALWAYS quote the relevant text from the document (in a blockquote), then explain it.
3. ALWAYS cite the document name AND page number (e.g. **"WMS Manual 4.0.1" — Page 12**).
4. If you're unsure, say so. Never fabricate procedure content.
5. If multiple documents cover the same topic, cite all of them and flag any differences.
6. If the user asks you to build or generate code (HTML, apps, etc.), you may do so, but ALL data in the code MUST come from the loaded documents — never invent content.

RESPONSE FORMAT:
- Simple questions: 1-3 sentences with a quote and citation
- Procedure steps: numbered list preserving the exact steps from the source
- Comparisons: markdown table with citations per cell
- Use **bold** for procedure names and key terms
- Preserve table structure from the source when a [TABLE] block is referenced"""

    # Use structured system prompt with cache_control for Anthropic prompt caching.
    # The rules block is cached (stable across messages), and the procedures block
    # is cached separately (stable within a session). This means follow-up messages
    # in the same session reuse cached tokens instead of re-processing everything,
    # cutting input costs by up to 90%.
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

    # Save user message now (assistant message saved after stream completes)
    db.add(DBChatMessage(
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

    # Configure tools -- optionally include web search
    chat_tools = []
    if CHAT_WEB_SEARCH:
        chat_tools.append({"type": "web_search_20250305", "name": "web_search"})

    async def stream_chat():
        """Stream the AI response as Server-Sent Events."""
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
            if chat_tools:
                create_kwargs["tools"] = chat_tools

            max_retries = 3
            for attempt in range(max_retries):
                try:
                    with client.messages.stream(**create_kwargs) as stream:
                        for event in stream:
                            if hasattr(event, 'type'):
                                if event.type == 'content_block_start':
                                    if hasattr(event.content_block, 'type'):
                                        if event.content_block.type == 'server_tool_use':
                                            yield f"data: {json.dumps({'type': 'status', 'message': 'Searching the web...'})}\n\n"
                                        elif event.content_block.type == 'thinking':
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
                        import asyncio
                        wait_time = 2 ** (attempt + 1)
                        yield f"data: {json.dumps({'type': 'status', 'message': f'Server busy, retrying in {wait_time}s...'})}\n\n"
                        await asyncio.sleep(wait_time)
                        full_reply = ""
                        continue
                    raise

            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        except Exception as e:
            logging.getLogger("pdfhelper").error("Chat AI stream failed: %s", e)
            err_msg = "AI request failed — please try again" if IS_PRODUCTION else f"AI request failed: {str(e)}"
            yield f"data: {json.dumps({'type': 'error', 'detail': err_msg})}\n\n"
            full_reply = full_reply or err_msg
        finally:
            # Persist assistant reply after stream completes
            save_db = SessionLocal()
            try:
                save_db.add(DBChatMessage(
                    id=str(uuid.uuid4()), session_id=session_id,
                    role="assistant",
                    content=_encrypt_text(full_reply or "Sorry, I couldn't generate a response."),
                    created_at=datetime.now(timezone.utc),
                ))
                sess = save_db.query(DBChatSession).filter(DBChatSession.id == session_id).first()
                if sess:
                    sess.updated_at = datetime.now(timezone.utc)
                save_db.commit()
            except Exception:
                save_db.rollback()
            finally:
                save_db.close()

    return StreamingResponse(stream_chat(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Chat History Endpoints
# ---------------------------------------------------------------------------

@router.get("/chat/sessions", dependencies=[Depends(verify_api_key)])
async def list_chat_sessions(request: Request, limit: int = Query(default=30, le=100), db=Depends(get_db)):
    """List past chat sessions, most recent first. Filtered to current user."""
    current_user_id = getattr(request.state, "user_id", None)
    query = db.query(DBChatSession)
    if current_user_id:
        query = query.filter(
            (DBChatSession.user_id == current_user_id) | (DBChatSession.user_id.is_(None))
        )
    sessions = query.order_by(DBChatSession.updated_at.desc()).limit(limit).all()
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


@router.get("/chat/sessions/{session_id}", dependencies=[Depends(verify_api_key)])
async def get_chat_session(session_id: str, request: Request, db=Depends(get_db)):
    """Get full message history for a chat session."""
    session = db.query(DBChatSession).filter(DBChatSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Chat session not found")
    current_user_id = getattr(request.state, "user_id", None)
    if current_user_id and session.user_id and session.user_id != current_user_id:
        raise HTTPException(status_code=403, detail="You do not own this chat session")
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
        "doc_ids": json.loads(session.doc_ids),
        "created_at": session.created_at.isoformat(),
        "updated_at": session.updated_at.isoformat(),
        "messages": messages,
    }


@router.delete("/chat/sessions/{session_id}", dependencies=[Depends(verify_api_key)])
async def delete_chat_session(session_id: str, request: Request, db=Depends(get_db)):
    """Delete a chat session and all its messages."""
    session = db.query(DBChatSession).filter(DBChatSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Chat session not found")
    current_user_id = getattr(request.state, "user_id", None)
    if current_user_id and session.user_id and session.user_id != current_user_id:
        raise HTTPException(status_code=403, detail="You do not own this chat session")
    db.delete(session)
    db.commit()
    return {"deleted": session_id}


# ---------------------------------------------------------------------------
# Chat -> Word Document Export
# ---------------------------------------------------------------------------

@router.post("/chat/export", dependencies=[Depends(verify_api_key)])
async def export_chat_to_docx(body: ExportChatRequest, request: Request, db=Depends(get_db)):
    """Export a chat session's AI responses as a Word document.

    Collects all assistant messages from the session and formats them
    into a downloadable .docx file.
    """
    session = db.query(DBChatSession).filter(DBChatSession.id == body.session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Chat session not found")
    current_user_id = getattr(request.state, "user_id", None)
    if current_user_id and session.user_id and session.user_id != current_user_id:
        raise HTTPException(status_code=403, detail="You do not own this chat session")

    messages = (
        db.query(DBChatMessage)
        .filter(DBChatMessage.session_id == session.id)
        .order_by(DBChatMessage.created_at)
        .all()
    )

    # Build document content from the conversation
    parts = []
    for m in messages:
        content = _safe_decrypt(m.content)
        if m.role == "user":
            parts.append(f"**Question:** {content}")
        else:
            parts.append(content)
        parts.append("")  # blank line separator

    full_text = "\n".join(parts)
    title = session.title or "Chat Export"
    docx_bytes = _markdown_to_docx(full_text, title)

    safe_title = re.sub(r'[^\w\s-]', '', title)[:50].strip() or "chat-export"
    filename = f"{safe_title}.docx"

    return Response(
        content=docx_bytes,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.post("/chat/markdown-to-docx", dependencies=[Depends(verify_api_key)])
async def markdown_to_docx(body: MarkdownToDocxRequest):
    """Convert markdown text to a downloadable Word document."""
    docx_bytes = _markdown_to_docx(body.markdown, body.title)
    safe_title = re.sub(r'[^\w\s-]', '', body.title)[:50].strip() or "document"
    filename = f"{safe_title}.docx"
    return Response(
        content=docx_bytes,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.post("/chat/generate-doc", dependencies=[Depends(verify_api_key)])
async def generate_document_from_chat(body: GenerateDocRequest, request: Request, db=Depends(get_db)):
    """Use AI to generate a Word document based on chat context and instructions.

    The AI writes a complete document (procedure, report, summary, etc.)
    using the uploaded procedures and conversation history as context,
    then returns it as a downloadable .docx file.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not configured on server")

    # Load session history for context
    session = db.query(DBChatSession).filter(DBChatSession.id == body.session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Chat session not found")
    current_user_id = getattr(request.state, "user_id", None)
    if current_user_id and session.user_id and session.user_id != current_user_id:
        raise HTTPException(status_code=403, detail="You do not own this chat session")

    # Get conversation context
    db_messages = (
        db.query(DBChatMessage)
        .filter(DBChatMessage.session_id == session.id)
        .order_by(DBChatMessage.created_at)
        .all()
    )
    chat_context = "\n\n".join(
        f"{'User' if m.role == 'user' else 'Assistant'}: {_safe_decrypt(m.content)}"
        for m in db_messages[-10:]
    )

    # Get procedure context from selected documents
    query = db.query(DBDocument)
    if body.doc_ids:
        query = query.filter(DBDocument.id.in_(body.doc_ids))
    documents = query.all()

    procedure_parts = []
    for doc in documents:
        decrypted_name = _safe_decrypt(doc.filename, f"Document {doc.id[:8]}")
        pages = json.loads(_safe_decrypt(doc.text_content, "[]"))
        full_text = "\n".join(p["text"] for p in pages if p.get("text"))
        if len(full_text) > 40000:
            full_text = full_text[:40000] + "\n[... truncated ...]"
        procedure_parts.append(f'--- "{decrypted_name}" ---\n{full_text}')
    procedure_context = "\n\n".join(procedure_parts) if procedure_parts else "(No procedures loaded)"

    client = Anthropic(api_key=api_key)

    system = f"""You are a professional document writer. You create well-structured, detailed documents based on the user's instructions and the reference materials provided.

REFERENCE PROCEDURES:
{procedure_context[:200000]}

RECENT CHAT CONTEXT:
{chat_context[:50000]}

INSTRUCTIONS:
- Write the document in clean, professional language
- Use markdown headings (#, ##, ###), bold (**text**), bullet points (- item), and numbered lists (1. item)
- Include all relevant details from the reference procedures
- Structure the document logically with clear sections
- The document should be complete and ready to use — not a draft or outline"""

    create_kwargs = dict(
        model=CHAT_MODEL,
        max_tokens=AGENT_MAX_TOKENS,
        system=system,
        messages=[{"role": "user", "content": body.instructions}],
    )
    if CHAT_WEB_SEARCH:
        create_kwargs["tools"] = [{"type": "web_search_20250305", "name": "web_search"}]

    response = client.messages.create(**create_kwargs)
    full_text = ""
    for block in response.content:
        if hasattr(block, "text"):
            full_text += block.text

    if not full_text.strip():
        raise HTTPException(status_code=500, detail="AI failed to generate document content")

    docx_bytes = _markdown_to_docx(full_text, body.title)

    safe_title = re.sub(r'[^\w\s-]', '', body.title)[:50].strip() or "generated-document"

    if body.include_vba:
        zip_buf = io.BytesIO()
        vba_code = _build_vba_module(body.title)
        with zipfile.ZipFile(zip_buf, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(f"{safe_title}.docx", docx_bytes)
            zf.writestr(f"{safe_title}_macros.bas", vba_code)
            zf.writestr("README.txt",
                f"DOCUMENT PACKAGE\n"
                f"================\n\n"
                f"1. {safe_title}.docx — The generated document\n"
                f"2. {safe_title}_macros.bas — VBA macros for Word formatting\n\n"
                f"To use macros: Open .docx in Word, press Alt+F11,\n"
                f"File > Import File, select the .bas, then Alt+F8 > FormatProcedure > Run\n"
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


@router.post("/chat/improve-procedure", dependencies=[Depends(verify_api_key)])
async def improve_procedure(body: ImproveProcedureRequest, request: Request, db=Depends(get_db)):
    """Cross-reference uploaded documents to produce an improved procedure.

    Takes one procedure document as the base, compares it against reference
    documents (standards, regulations, other procedures), and generates an
    improved version as a Word document. Optionally includes VBA macro code
    for automated formatting in Word.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not configured on server")

    # Load the procedure to improve
    proc_doc = db.query(DBDocument).filter(DBDocument.id == body.procedure_doc_id).first()
    if not proc_doc:
        raise HTTPException(status_code=404, detail="Procedure document not found")
    proc_name = _safe_decrypt(proc_doc.filename, f"Document {proc_doc.id[:8]}")
    proc_pages = json.loads(_safe_decrypt(proc_doc.text_content, "[]"))
    proc_text = "\n".join(p["text"] for p in proc_pages if p.get("text"))
    if len(proc_text) > 100000:
        proc_text = proc_text[:100000] + "\n[... truncated ...]"

    # Load reference documents
    ref_parts = []
    if body.reference_doc_ids:
        ref_docs = db.query(DBDocument).filter(DBDocument.id.in_(body.reference_doc_ids)).all()
        for doc in ref_docs:
            name = _safe_decrypt(doc.filename, f"Document {doc.id[:8]}")
            pages = json.loads(_safe_decrypt(doc.text_content, "[]"))
            text = "\n".join(p["text"] for p in pages if p.get("text"))
            if len(text) > 60000:
                text = text[:60000] + "\n[... truncated ...]"
            ref_parts.append(f'--- REFERENCE: "{name}" ---\n{text}\n--- END ---')
    ref_context = "\n\n".join(ref_parts) if ref_parts else "(No additional reference documents)"

    # Load chat context
    session = db.query(DBChatSession).filter(DBChatSession.id == body.session_id).first()
    chat_context = ""
    if session:
        current_user_id = getattr(request.state, "user_id", None)
        if current_user_id and session.user_id and session.user_id != current_user_id:
            raise HTTPException(status_code=403, detail="You do not own this chat session")
        db_messages = (
            db.query(DBChatMessage)
            .filter(DBChatMessage.session_id == session.id)
            .order_by(DBChatMessage.created_at)
            .all()
        )
        chat_context = "\n\n".join(
            f"{'User' if m.role == 'user' else 'Assistant'}: {_safe_decrypt(m.content)}"
            for m in db_messages[-10:]
        )

    focus = body.focus_areas or "clarity, completeness, safety, step-by-step structure, compliance"

    client = Anthropic(api_key=api_key)

    system = f"""You are an expert technical procedure writer. Your task is to IMPROVE an existing procedure document by cross-referencing it against other reference documents.

=== ORIGINAL PROCEDURE TO IMPROVE ===
Document: "{proc_name}"
{proc_text}

=== REFERENCE DOCUMENTS ===
{ref_context[:200000]}

=== RECENT DISCUSSION ===
{chat_context[:30000]}

=== YOUR TASK ===
Produce an IMPROVED version of the original procedure. Focus on: {focus}

RULES:
1. Cross-reference the original procedure against ALL reference documents
2. Incorporate missing steps, safety requirements, compliance items found in references
3. Keep the improved procedure clear, concise, and actionable
4. Use proper procedure structure:
   - Title and document number
   - Purpose / Scope
   - Definitions / Abbreviations
   - Responsibilities
   - Required Tools / Materials / PPE
   - Precautions (DANGER / WARNING / CAUTION / NOTE)
   - Step-by-step instructions (numbered, with verification checkboxes)
   - Acceptance criteria
   - References
   - Revision history placeholder
5. Highlight what changed from the original with [ADDED], [MODIFIED], or [IMPROVED] tags
6. Use markdown: # headings, **bold**, - bullets, 1. numbered lists
7. Be specific — include actual values, part numbers, limits from the reference documents
8. Every safety-critical step should have a verification checkbox [ ]"""

    create_kwargs = dict(
        model=CHAT_MODEL,
        max_tokens=AGENT_MAX_TOKENS,
        system=system,
        messages=[{"role": "user", "content": f"Improve this procedure. Focus areas: {focus}\n\nProduce the complete improved procedure now."}],
    )
    if CHAT_WEB_SEARCH:
        create_kwargs["tools"] = [{"type": "web_search_20250305"}]

    response = client.messages.create(**create_kwargs)
    full_text = ""
    for block in response.content:
        if hasattr(block, "text"):
            full_text += block.text

    if not full_text.strip():
        raise HTTPException(status_code=500, detail="AI failed to generate the improved procedure")

    docx_bytes = _markdown_to_docx(full_text, body.title)

    # Build response files
    safe_title = re.sub(r'[^\w\s-]', '', body.title)[:50].strip() or "improved-procedure"

    if body.include_vba:
        # Return a ZIP containing the .docx and the .bas VBA module
        zip_buf = io.BytesIO()
        vba_code = _build_vba_module(body.title)
        with zipfile.ZipFile(zip_buf, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(f"{safe_title}.docx", docx_bytes)
            zf.writestr(f"{safe_title}_macros.bas", vba_code)
            zf.writestr("README.txt",
                f"IMPROVED PROCEDURE PACKAGE\n"
                f"=========================\n\n"
                f"This package contains:\n\n"
                f"1. {safe_title}.docx\n"
                f"   The improved procedure document. Open in Word to review and edit.\n\n"
                f"2. {safe_title}_macros.bas\n"
                f"   VBA macro module for Word. To use:\n"
                f"   a) Open the .docx in Word\n"
                f"   b) Press Alt+F11 to open the VBA editor\n"
                f"   c) Go to File > Import File, select the .bas file\n"
                f"   d) Close VBA editor\n"
                f"   e) Press Alt+F8, select 'FormatProcedure', click Run\n\n"
                f"   Available macros:\n"
                f"   - FormatProcedure: Applies all formatting at once\n"
                f"   - InsertRevisionTable: Adds a revision history table\n"
                f"   - InsertSignOffBlock: Adds a signature/approval block\n\n"
                f"3. Save as .docm (macro-enabled) to keep the VBA macros.\n"
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
