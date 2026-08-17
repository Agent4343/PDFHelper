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

PROCEDURE_SYSTEM_PROMPT = """You are an expert technical procedure writer for upstream oil and gas operations, trained on:
- PPA AP-907-005 Procedure Writer's Manual (Rev. 3)
- Upstream Best Practices: Procedure Writing Rules
- Operations Integrity Protocol 6.1 (OIMS Element 6)
- Upstream Procedure Tools: Task Analysis
- Safety Critical Task Analysis (SCTA) programme requirements

Your job is to help create clear, precise, and safe work procedures that meet OIMS 6.1 requirements and human performance principles.

=== GATHERING INFORMATION ===
Ask focused questions ONE AT A TIME in this order:
1. Procedure title and designation number (use logical numbering: Unit-Type-System-Sequence)
2. Purpose — what, when, and why (do NOT simply repeat the title)
3. Scope — activities covered, boundaries, applicable personnel and equipment
4. References and commitments — regulatory docs, operating experience, P&IDs, vendor manuals
5. Definitions — terms unique to this procedure (alphabetical, do not define self-explanatory terms)
6. Responsibilities — who does what (high-level summary, not a repeat of steps)
7. Precautions — equipment/personnel/public protection measures (state effect AND cause)
8. Limitations — specific regulatory or administrative limits with values
9. Prerequisites — conditions that must exist before starting
10. Step-by-step instructions — walk through each action
11. Acceptance criteria — quantitative/qualitative pass/fail criteria
12. Attachments needed — data sheets, checklists, figures, P&ID excerpts
13. Is this an SCTA/safeguard critical task? If so, identify safety critical steps and hold points.
14. Who needs to perform the task? Use standard job roles (e.g., CRO, OPER, INST, MECH, BCO, SUP)

=== PROCEDURE STRUCTURE (Table 1 — PPA AP-907-005) ===
Required sections for technical procedures in this order:
1. Cover Page (title, number, revision, level of use, effective date, approver)
2. Table of Contents
3. Purpose (R)
4. Scope (R)
5. References and Commitments (R)
6. Definitions (O)
7. Responsibilities (O)
8. Precautions and Limitations (R)
9. Prerequisites (R)
10. Instructions (R)
11. Acceptance Criteria (R for testing, O for maintenance)
12. Summary of Alterations (O)
13. Attachments (O)

=== WRITING RULES ===

ACTION STEPS:
- Every action step starts with an ACTION VERB in UPPERCASE BOLD (e.g., OPEN, CLOSE, VERIFY, CHECK, RECORD, PERFORM)
- One action per step — never combine two actions
- Active voice only — the step directs the user to act
- Include WHO performs the action and a checkbox

Step format (upstream table style):
No. | Action | Who | Check
1.  | OPEN inlet valve XX-XXX-001 to Amine Circulating Pump XX-XXXX | Ops | [ ]

EMPHASIS TECHNIQUES:
- Action verbs: UPPERCASE BOLD (e.g., OPEN, CLOSE, VERIFY)
- Conditional/logic terms: UPPERCASE, UNDERLINED, BOLD (IF, THEN, WHEN, AND, OR, NOT, WHILE)
- Component positions: UPPERCASE (OPEN, CLOSED, ON, OFF, AUTO)
- Component noun names: Title Case (Heater Pump, Amine Discharge Valve)
- Locations: inside parentheses — (inside the MMC)
- Condition and action on SEPARATE lines

CONDITIONAL STEPS:
- IF introduces a condition that may or may not be true
- WHEN introduces a condition expected to occur
- THEN goes between condition and action (never between actions)
- IF AT ANY TIME introduces a condition that may occur during procedure execution
- WHILE introduces a continuous action
- Do NOT use AND/OR construction
- State conditions positively (IF valve is OPEN, not IF valve is NOT CLOSED)
- For 3+ conditions, use a decision table

IF/THEN table format:
No. | Action | Who | Check
1.  | IF Temperature exceeds 100C | |
    | THEN OPEN XX-XXX-0011 inlet to XXX exchanger | Ops | [ ]

NOTES, CAUTIONS, AND WARNINGS:
- Place BEFORE the step they apply to (never after)
- Sequence: Note first, then Caution, then Warning (most important closest to step)
- Must appear on the same page as the impacted step
- Written in passive voice, short and concise
- Must NOT contain action steps or implied instructions
- If removed, procedure performance would not be affected
- WARNING: personnel injury, loss of life, health hazards — format: ! WARNING: [text]
- CAUTION: equipment damage, process risk — format: CAUTION: [text]
- NOTE: supplemental/explanatory information — format: Note: [text]

HOLD POINTS:
- A pre-selected step beyond which work may NOT proceed until required action is performed
- Identified by SCTA for HC scenarios or by SME
- Format: "Hold Point" label before the action step
- Require explicit authorization to proceed past

INDEPENDENT VERIFICATION:
- Used for safeguard critical steps
- Two individuals working independently to confirm component condition
- Format includes Name/Sign fields after the step

SAFEGUARD CRITICAL STEPS:
- Only used when identified by SCTA
- Preceded by a WARNING box: "WARNING - SAFEGUARD CRITICAL STEP"
- State: independent verification required, what happens if step is done wrong (clear hazard/consequence)
- Easy to understand and execute (limit potential for error)
- Consider hold points for steps requiring additional review

STEP NUMBERING:
- Up to 4 levels: 1. / a. / (1) / (a)
- Alphanumeric steps performed in written order unless stated otherwise
- Bulleted steps within a single alphanumeric step may be performed in any order
- Sections: 1.0 TITLE / 1.1 Subtitle / 1.1.1 Subtitle

VOCABULARY:
- No ambiguous words: replace "ensure," "appropriate," "proper" with specific measurable language
- SHALL = requirement, SHOULD = recommendation, MAY = permission
- Use action verbs from PPA Attachment 1 (ADJUST, ALIGN, CHECK, CLOSE, CONNECT, DE-ENERGIZE, DRAIN, ENERGIZE, FILL, FLUSH, INSTALL, ISOLATE, MARK, MEASURE, MONITOR, NOTIFY, OBSERVE, OPEN, OPERATE, PLACE, POSITION, PRESS, PULL, PUSH, RAISE, RECORD, RELEASE, REMOVE, REPLACE, RESET, RESTORE, ROTATE, SELECT, SET, START, STOP, TAG, TEST, TORQUE, TURN, VERIFY, etc.)

ABBREVIATIONS:
- Spell out on first reference in each major section, followed by acronym in parentheses
- Example: Competency Assurance Standard (CAS)
- Plurals: add lowercase s (BUs not BU's)
- Use standard industry abbreviations: JSA, PSV, FPSO, P&ID, PFD, HAZOP

FORMAT:
- Font: Arial 11 or 12 point
- Paper: Portrait, 8.5 x 11
- Margins: 0.8 inch left/right, 0.5 inch top/bottom
- Left justify all text
- Single line spacing with white space between steps
- Page header on every page: procedure title, number, revision, page X of Y
- Keep steps unbroken on same page
- Use continuation headings: "4.2 (cont.)" when content spans pages

TASK ANALYSIS (when building from scratch):
- Gather P&IDs, PFDs, vendor info, system descriptions
- Identify operating system using engineering system numbering
- List major equipment from P&IDs
- Consolidate common equipment (no duplicates)
- Populate tasks per equipment, then device actions within each task

=== OIMS 6.1 COMPLIANCE ===
Procedures must address:
- Level of Operations Integrity risk (determines detail and verification needed)
- Operating envelopes with consequence of deviation and response
- Transient operations as applicable
- Regulatory requirements
- Human Factors including capabilities and limitations
- Simplifying processes or tasks to reduce potential for error

{style_config}

{template_config}

If the user provides an existing procedure to update, review it against these standards, identify deficiencies, and ask targeted questions about the changes needed.

Always be thorough — a missing step or unclear instruction in a procedure can lead to safety incidents. Every safeguard critical step must be clearly identified with proper warnings and verification requirements."""


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

OUTPUT FORMAT RULES:
- Use # for main sections (e.g., # 1.0 PURPOSE)
- Use ## for subsections (e.g., ## 6.1 Equipment Lineup)
- Use ### for sub-subsections
- Action steps as numbered list: "1. OPEN valve XX-XXX-001"
- Action verbs in UPPERCASE: OPEN, CLOSE, VERIFY, CHECK, RECORD, PERFORM, etc.
- Conditional terms in UPPERCASE: IF, THEN, WHEN, AND, OR, NOT, WHILE
- Component positions in UPPERCASE: OPEN, CLOSED, ON, OFF, AUTO
- Component names in Title Case: Amine Discharge Valve
- Warnings as: ! WARNING: [text]
- Cautions as: CAUTION: [text]
- Notes as: NOTE: [text]
- Place warnings/cautions/notes BEFORE the step they apply to
- Tables as markdown tables with | pipes |
- Use the action step table format where applicable:
| No. | Action | Who | Check |
| --- | --- | --- | --- |
| 1. | OPEN inlet valve XX-XXX-001 | Ops | [ ] |

Include ALL required sections per PPA AP-907-005:
1. Purpose
2. Scope
3. References and Commitments
4. Definitions (if needed)
5. Responsibilities
6. Precautions and Limitations
7. Prerequisites
8. Instructions (step-by-step with proper formatting)
9. Acceptance Criteria (if applicable)
10. Attachments (if applicable)

Use the template structure and style rules provided. Every action step must start with an uppercase bold action verb and contain only one action."""

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
    from docx.shared import Pt, Emu, RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.enum.table import WD_TABLE_ALIGNMENT
    from docx.oxml.ns import qn
    import io
    import re

    NAVY = RGBColor(0x0B, 0x25, 0x45)
    GREY = RGBColor(0x5B, 0x64, 0x72)

    content = _safe_decrypt(session.output_content) or ""
    title = _safe_decrypt(session.title) or "Procedure"

    doc = Document()

    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(10.5)
    style.paragraph_format.space_after = Pt(6)

    for hs in ["Heading 1", "Heading 2", "Heading 3"]:
        if hs in doc.styles:
            h_style = doc.styles[hs]
            h_style.font.name = "Calibri"
            h_style.font.color.rgb = NAVY
            h_style.font.bold = True
    if "Heading 1" in doc.styles:
        doc.styles["Heading 1"].font.size = Pt(13.5)
    if "Heading 2" in doc.styles:
        doc.styles["Heading 2"].font.size = Pt(12)
    if "Heading 3" in doc.styles:
        doc.styles["Heading 3"].font.size = Pt(11)

    for section in doc.sections:
        section.top_margin = Emu(635000)
        section.bottom_margin = Emu(635000)
        section.left_margin = Emu(698500)
        section.right_margin = Emu(698500)

        header = section.header
        header.is_linked_to_previous = False
        hp = header.paragraphs[0] if header.paragraphs else header.add_paragraph()
        hp.text = ""
        hp.alignment = WD_ALIGN_PARAGRAPH.LEFT
        run = hp.add_run(title)
        run.font.name = "Calibri"
        run.font.size = Pt(8)
        run.font.color.rgb = GREY

        footer = section.footer
        footer.is_linked_to_previous = False
        fp = footer.paragraphs[0] if footer.paragraphs else footer.add_paragraph()
        fp.text = ""
        fp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        frun = fp.add_run("Page ")
        frun.font.name = "Calibri"
        frun.font.size = Pt(8)
        frun.font.color.rgb = GREY
        fld_xml = (
            '<w:fldSimple xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'
            ' w:instr=" PAGE "><w:r><w:t>1</w:t></w:r></w:fldSimple>'
        )
        from lxml import etree
        fp._element.append(etree.fromstring(fld_xml))
        frun2 = fp.add_run(" of ")
        frun2.font.name = "Calibri"
        frun2.font.size = Pt(8)
        frun2.font.color.rgb = GREY
        fld_xml2 = (
            '<w:fldSimple xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'
            ' w:instr=" NUMPAGES "><w:r><w:t>1</w:t></w:r></w:fldSimple>'
        )
        fp._element.append(etree.fromstring(fld_xml2))

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    run = p.add_run(title)
    run.bold = True
    run.font.size = Pt(16)
    run.font.name = "Calibri"
    run.font.color.rgb = NAVY
    doc.add_paragraph()

    def _add_warning_box(doc, text):
        tbl = doc.add_table(rows=1, cols=1)
        tbl.alignment = WD_TABLE_ALIGNMENT.LEFT
        cell = tbl.cell(0, 0)
        shading = cell._element.get_or_add_tcPr()
        shading_elm = shading.makeelement(qn("w:shd"), {
            qn("w:fill"): "FFE0E0", qn("w:val"): "clear",
        })
        shading.append(shading_elm)
        p = cell.paragraphs[0]
        run = p.add_run("! WARNING: ")
        run.bold = True
        run.font.name = "Calibri"
        run.font.color.rgb = RGBColor(0xCC, 0x00, 0x00)
        run.font.size = Pt(10.5)
        run2 = p.add_run(text)
        run2.font.name = "Calibri"
        run2.font.size = Pt(10.5)

    def _add_caution_box(doc, text):
        tbl = doc.add_table(rows=1, cols=1)
        tbl.alignment = WD_TABLE_ALIGNMENT.LEFT
        cell = tbl.cell(0, 0)
        shading = cell._element.get_or_add_tcPr()
        shading_elm = shading.makeelement(qn("w:shd"), {
            qn("w:fill"): "FFF3CD", qn("w:val"): "clear",
        })
        shading.append(shading_elm)
        p = cell.paragraphs[0]
        run = p.add_run("CAUTION: ")
        run.bold = True
        run.font.name = "Calibri"
        run.font.color.rgb = RGBColor(0xCC, 0x88, 0x00)
        run.font.size = Pt(10.5)
        run2 = p.add_run(text)
        run2.font.name = "Calibri"
        run2.font.size = Pt(10.5)

    def _add_note_box(doc, text):
        tbl = doc.add_table(rows=1, cols=1)
        tbl.alignment = WD_TABLE_ALIGNMENT.LEFT
        cell = tbl.cell(0, 0)
        shading = cell._element.get_or_add_tcPr()
        shading_elm = shading.makeelement(qn("w:shd"), {
            qn("w:fill"): "E8F0FE", qn("w:val"): "clear",
        })
        shading.append(shading_elm)
        p = cell.paragraphs[0]
        run = p.add_run("NOTE: ")
        run.bold = True
        run.font.name = "Calibri"
        run.font.color.rgb = RGBColor(0x00, 0x55, 0xCC)
        run.font.size = Pt(10.5)
        run2 = p.add_run(text)
        run2.font.name = "Calibri"
        run2.font.size = Pt(10.5)

    for line in content.split("\n"):
        line = line.rstrip()
        if not line:
            doc.add_paragraph("")
            continue

        line_upper = line.strip().upper()
        if line_upper.startswith("! WARNING:") or line_upper.startswith("WARNING:"):
            text = re.sub(r'^!?\s*WARNING:\s*', '', line.strip(), flags=re.IGNORECASE)
            _add_warning_box(doc, text)
        elif line_upper.startswith("CAUTION:"):
            text = re.sub(r'^CAUTION:\s*', '', line.strip(), flags=re.IGNORECASE)
            _add_caution_box(doc, text)
        elif line_upper.startswith("NOTE:"):
            text = re.sub(r'^NOTE:\s*', '', line.strip(), flags=re.IGNORECASE)
            _add_note_box(doc, text)
        elif line.startswith("# "):
            doc.add_heading(line[2:], level=1)
        elif line.startswith("## "):
            doc.add_heading(line[3:], level=2)
        elif line.startswith("### "):
            doc.add_heading(line[4:], level=3)
        elif line.strip().startswith("**") and line.strip().endswith("**"):
            p = doc.add_paragraph()
            run = p.add_run(line.strip().strip("*"))
            run.bold = True
            run.font.name = "Calibri"
            run.font.color.rgb = NAVY
            run.font.size = Pt(12)
        elif line.strip().startswith("- ") or line.strip().startswith("• "):
            text = line.strip()[2:]
            doc.add_paragraph(text, style="List Bullet")
        elif re.match(r'^\d+\.\s', line.strip()):
            text = re.sub(r'^\d+\.\s+', '', line.strip())
            doc.add_paragraph(text, style="List Number")
        elif line.strip().startswith("|") and line.strip().endswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if all(set(c) <= set("- :") for c in cells):
                continue
            if not hasattr(doc, '_proc_table_active'):
                doc._proc_table_active = False
            if not doc._proc_table_active:
                tbl = doc.add_table(rows=1, cols=len(cells))
                tbl.style = "Table Grid"
                tbl.alignment = WD_TABLE_ALIGNMENT.LEFT
                for i, val in enumerate(cells):
                    cell = tbl.cell(0, i)
                    cell.text = val
                    for p in cell.paragraphs:
                        for run in p.runs:
                            run.bold = True
                            run.font.size = Pt(9.5)
                            run.font.name = "Calibri"
                doc._proc_table_active = True
                doc._proc_table_ref = tbl
            else:
                row = doc._proc_table_ref.add_row()
                for i, val in enumerate(cells):
                    if i < len(row.cells):
                        row.cells[i].text = val
                        for p in row.cells[i].paragraphs:
                            for run in p.runs:
                                run.font.size = Pt(9.5)
                                run.font.name = "Calibri"
        else:
            if hasattr(doc, '_proc_table_active') and doc._proc_table_active:
                doc._proc_table_active = False
            doc.add_paragraph(line)

    if hasattr(doc, '_proc_table_active'):
        del doc._proc_table_active
    if hasattr(doc, '_proc_table_ref'):
        del doc._proc_table_ref

    buf = io.BytesIO()
    doc.save(buf)
    buf.seek(0)

    safe_title = "".join(c for c in title if c.isalnum() or c in " -_")[:50].strip() or "procedure"
    return Response(
        content=buf.getvalue(),
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": f'attachment; filename="{safe_title}.docx"'},
    )
