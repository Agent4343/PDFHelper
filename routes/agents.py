"""Agent endpoints extracted from app.py — analysis pipeline, bulk audit,
compliance audit, document comparison, procedure writer, code builder,
and agent cache management."""

import asyncio
import hashlib
import json
import logging
import os
import uuid
from datetime import datetime, timezone

from anthropic import Anthropic
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

from auth import verify_api_key, get_db
from config import CHAT_MAX_TOKENS, AGENT_MAX_TOKENS, IS_PRODUCTION, CHAT_WEB_SEARCH
from database import (
    DBDocument,
    DBAnalysisReport,
    DBAgentCache,
    SessionLocal,
)
from models import (
    AnalyzeRequest,
    BulkAuditRequest,
    ComplianceAuditRequest,
    CompareDocsRequest,
    ProcedureWriterRequest,
    CodeBuilderRequest,
)
from helpers import (
    _encrypt_text,
    _safe_decrypt,
    _load_pdf_bytes,
    _agent_step,
    _agent_chunk,
    _agent_done,
    _agent_error,
    _resolve_agent_model,
    _call_claude,
    _call_claude_bg,
    _stream_claude,
    _agent_cache_key,
    _get_doc_hash,
    _check_agent_cache,
    _save_agent_cache,
)
from search import keyword_search, ai_search
from audit import log_search as log_search_audit
from auth import _get_client_ip
from ocr import extract_structured_text

router = APIRouter()

# ---------------------------------------------------------------------------
# Full Analysis Pipeline (multi-agent)
# ---------------------------------------------------------------------------


@router.post("/analyze", dependencies=[Depends(verify_api_key)])
async def analyze_documents(
    request: Request,
    body: AnalyzeRequest,
    doc_ids: list[str] = Query(default=[], description="Document IDs to analyze (empty = all)"),
    db=Depends(get_db),
):
    """Run the full multi-agent analysis pipeline on uploaded documents.

    This runs 4 specialized AI agents:
    1. Document Analyzer -- deep analysis of each document
    2. Cross-Reference Checker -- finds conflicts between documents
    3. Compliance Checker -- flags regulatory/policy issues
    4. Summary Report Generator -- produces an actionable executive report

    Optionally also runs keyword and AI search.
    """
    query = db.query(DBDocument)
    if doc_ids:
        query = query.filter(DBDocument.id.in_(doc_ids))
    documents = query.all()

    if not documents:
        raise HTTPException(status_code=404, detail="No documents found")

    client_ip = _get_client_ip(request)

    # Build a cache key from document content hashes + analysis parameters
    # This lets us skip re-analysis when the same documents are analyzed
    # with the same compliance context (search is fast enough to always re-run)
    doc_hashes = sorted(d.content_hash or d.id for d in documents)
    cache_key_input = json.dumps({
        "hashes": doc_hashes,
        "compliance_context": body.compliance_context,
    }, sort_keys=True)
    cache_key = hashlib.sha256(cache_key_input.encode()).hexdigest()

    # Check for a cached analysis with the same content + parameters
    cached_report = (
        db.query(DBAnalysisReport)
        .filter(DBAnalysisReport.cache_key == cache_key)
        .order_by(DBAnalysisReport.analyzed_at.desc())
        .first()
    )
    if cached_report:
        cached_analysis = json.loads(_safe_decrypt(cached_report.report_data, "{}"))
        # Re-run search if requested (cheap), but reuse the cached analysis
        if body.search_terms or body.ai_query:
            docs_for_agents: dict[str, list[dict]] = {}
            for doc in documents:
                decrypted_name = _safe_decrypt(doc.filename, f"Document {doc.id[:8]}")
                pages = json.loads(_safe_decrypt(doc.text_content, "[]"))
                docs_for_agents[decrypted_name] = pages
            search_results = {"keyword_results": [], "ai_results": []}
            for filename, pages in docs_for_agents.items():
                if body.search_terms:
                    kw_matches = keyword_search(pages, body.search_terms)
                    for m in kw_matches:
                        m["filename"] = filename
                    search_results["keyword_results"].extend(kw_matches)
                if body.ai_query:
                    ai_matches = await asyncio.to_thread(ai_search, pages, body.ai_query, filename)
                    for m in ai_matches:
                        m["filename"] = filename
                    search_results["ai_results"].extend(ai_matches)
            cached_analysis["search_results"] = search_results

        return {
            "report_id": cached_report.id,
            "cached": True,
            "report": cached_analysis.get("report"),
            "document_analyses": cached_analysis.get("document_analyses"),
            "cross_reference_findings": cached_analysis.get("cross_reference_findings"),
            "compliance_findings": cached_analysis.get("compliance_findings"),
            "search_results": cached_analysis.get("search_results"),
        }

    # Build documents dict for the agent pipeline
    docs_for_agents: dict[str, list[dict]] = {}
    for doc in documents:
        decrypted_name = _safe_decrypt(doc.filename, f"Document {doc.id[:8]}")
        pages = json.loads(_safe_decrypt(doc.text_content, "[]"))
        docs_for_agents[decrypted_name] = pages

    # Run the full pipeline in a thread pool to avoid blocking the event loop
    # (run_full_analysis makes multiple synchronous Anthropic API calls)
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
        cache_key=cache_key,
        analyzed_at=datetime.now(timezone.utc),
    )
    db.add(db_report)
    db.commit()

    log_search_audit(client_ip, report_id, body.search_terms, body.ai_query,
                     len(documents), db_report.total_issues, db_report.critical_issues)

    response = {
        "report_id": report_id,
        "cached": False,
        "report": analysis.get("report"),
        "document_analyses": analysis.get("document_analyses"),
        "cross_reference_findings": analysis.get("cross_reference_findings"),
        "compliance_findings": analysis.get("compliance_findings"),
        "search_results": analysis.get("search_results"),
    }
    if analysis.get("warnings"):
        response["warnings"] = analysis["warnings"]
    return response


@router.get("/reports", dependencies=[Depends(verify_api_key)])
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


@router.get("/reports/{report_id}", dependencies=[Depends(verify_api_key)])
async def get_report(report_id: str, db=Depends(get_db)):
    """Get full details of a past analysis report."""
    report = db.query(DBAnalysisReport).filter(DBAnalysisReport.id == report_id).first()
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    full_data = json.loads(_safe_decrypt(report.report_data, "{}"))
    return {
        "id": report.id,
        "documents_analyzed": report.documents_analyzed,
        "total_issues": report.total_issues,
        "critical_issues": report.critical_issues,
        "risk_level": report.risk_level,
        "analyzed_at": report.analyzed_at.isoformat(),
        **full_data,
    }


# ---------------------------------------------------------------------------
# Bulk Audit Agent
# ---------------------------------------------------------------------------


@router.post("/agents/bulk-audit", dependencies=[Depends(verify_api_key)])
async def agent_bulk_audit(body: BulkAuditRequest, request: Request, db=Depends(get_db)):
    """Run compliance audit on multiple documents, streaming progress."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not configured on server")

    current_user_id = getattr(request.state, "user_id", None)

    if body.doc_ids:
        docs = db.query(DBDocument).filter(DBDocument.id.in_(body.doc_ids)).all()
    else:
        docs = db.query(DBDocument).all()

    if not docs:
        raise HTTPException(status_code=404, detail="No documents found")

    model = _resolve_agent_model(body.model)
    focus = body.focus_areas

    client = Anthropic(api_key=api_key)
    web_tools = [{"type": "web_search_20250305", "name": "web_search"}] if CHAT_WEB_SEARCH else None

    async def run_bulk():
        total = len(docs)
        yield f"data: {json.dumps({'type': 'bulk_start', 'total': total})}\n\n"

        for idx, doc in enumerate(docs):
            doc_name = _safe_decrypt(doc.filename, f"Document {doc.id[:8]}")
            yield f"data: {json.dumps({'type': 'bulk_progress', 'current': idx + 1, 'total': total, 'doc_name': doc_name, 'status': 'running'})}\n\n"

            doc_hash = _get_doc_hash(doc)
            cache_key = _agent_cache_key("audit", model, [doc_hash], focus)
            save_db = SessionLocal()
            try:
                cached = _check_agent_cache(save_db, cache_key, current_user_id)
            finally:
                save_db.close()

            if cached:
                yield f"data: {json.dumps({'type': 'bulk_result', 'current': idx + 1, 'doc_name': doc_name, 'doc_id': doc.id, 'cached': True, 'summary': cached[:500]})}\n\n"
                continue

            try:
                pdf_bytes = _load_pdf_bytes(doc)
                structured = extract_structured_text(pdf_bytes)
                doc_content = ""
                for page_data in structured:
                    doc_content += f"\n--- Page {page_data['page']} ---\n"
                    for block in page_data.get("blocks", []):
                        prefix = f"[{block['type'].upper()}] " if block['type'] != 'paragraph' else ""
                        doc_content += f"{prefix}{block['text']}\n"
                if len(doc_content) > 50000:
                    doc_content = doc_content[:50000] + "\n[... truncated ...]"

                report = _call_claude(client,
                    f"""You are a compliance auditor. Analyze this document against current regulations.

DOCUMENT: {doc_name}
{doc_content}

Provide a concise compliance audit with:
1. Overall compliance rating (percentage)
2. Risk level (HIGH/MEDIUM/LOW)
3. Key findings (max 5 bullet points)
4. Critical issues requiring immediate attention

Use markdown formatting.""",
                    "Audit this document for compliance.", tools=web_tools,
                    max_tokens=AGENT_MAX_TOKENS, model=model)

                save_db = SessionLocal()
                try:
                    _save_agent_cache(save_db, cache_key, "audit", model,
                                      report, [doc.id], f"bulk|focus: {focus}" if focus else "bulk",
                                      user_id=current_user_id)
                finally:
                    save_db.close()

                yield f"data: {json.dumps({'type': 'bulk_result', 'current': idx + 1, 'doc_name': doc_name, 'doc_id': doc.id, 'cached': False, 'summary': report[:500]})}\n\n"

            except Exception as e:
                logging.getLogger("pdfhelper").error("Bulk audit failed for %s: %s", doc_name, e)
                err = "Audit failed" if IS_PRODUCTION else str(e)
                yield f"data: {json.dumps({'type': 'bulk_result', 'current': idx + 1, 'doc_name': doc_name, 'doc_id': doc.id, 'error': err})}\n\n"

        yield _agent_done()

    return StreamingResponse(run_bulk(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Compliance Audit Agent
# ---------------------------------------------------------------------------


@router.post("/agents/compliance-audit", dependencies=[Depends(verify_api_key)])
async def agent_compliance_audit(body: ComplianceAuditRequest, request: Request, db=Depends(get_db)):
    """Multi-step compliance audit agent."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not configured on server")

    doc = db.query(DBDocument).filter(DBDocument.id == body.doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    current_user_id = getattr(request.state, "user_id", None)
    model = _resolve_agent_model(body.model)
    doc_hash = _get_doc_hash(doc)
    cache_key = _agent_cache_key("audit", model, [doc_hash], body.focus_areas)

    cached = _check_agent_cache(db, cache_key, current_user_id)
    if cached:
        async def return_cached():
            yield f"data: {json.dumps({'type': 'cached', 'message': 'Returning cached result'})}\n\n"
            yield _agent_chunk(cached)
            yield _agent_done()
        return StreamingResponse(return_cached(), media_type="text/event-stream")

    pdf_bytes = _load_pdf_bytes(doc)
    structured = extract_structured_text(pdf_bytes)
    doc_name = _safe_decrypt(doc.filename, f"Document {doc.id[:8]}")
    focus = body.focus_areas

    doc_content = ""
    for page_data in structured:
        doc_content += f"\n--- Page {page_data['page']} ---\n"
        for block in page_data.get("blocks", []):
            prefix = f"[{block['type'].upper()}] " if block['type'] != 'paragraph' else ""
            doc_content += f"{prefix}{block['text']}\n"
    if len(doc_content) > 80000:
        doc_content = doc_content[:80000] + "\n[... truncated ...]"

    client = Anthropic(api_key=api_key)
    web_tools = [{"type": "web_search_20250305", "name": "web_search"}] if CHAT_WEB_SEARCH else None

    async def run_audit():
        full_report = ""
        try:
            yield _agent_step(1, 4, "Analyzing document structure")
            task = asyncio.create_task(_call_claude_bg(client,
                "You are a document analyst. Identify the key sections, scope, and purpose of this procedure document. List each section with a one-line summary.",
                f"Analyze this document:\n\nFILENAME: {doc_name}\n\n{doc_content}", model=model))
            while not task.done():
                yield ":\n\n"
                await asyncio.sleep(3)
            analysis = task.result()
            yield _agent_step(1, 4, "Analyzing document structure", "done")

            yield _agent_step(2, 4, "Searching current regulations")
            reg_query = f"Current regulations and standards applicable to: {doc_name}."
            if focus:
                reg_query += f" Focus areas: {focus}."
            reg_query += f"\n\nDocument sections found:\n{analysis[:3000]}"
            task = asyncio.create_task(_call_claude_bg(client,
                "You are a regulatory researcher. Search the web for current, applicable regulations, standards, and industry requirements. List each regulation with its full title, version/year, and key requirements.",
                reg_query, tools=web_tools, model=model))
            while not task.done():
                yield ":\n\n"
                await asyncio.sleep(3)
            regulations = task.result()
            yield _agent_step(2, 4, "Searching current regulations", "done")

            yield _agent_step(3, 4, "Cross-referencing sections against regulations")
            task = asyncio.create_task(_call_claude_bg(client,
                f"""You are a compliance auditor. You have:

DOCUMENT: {doc_name}
{doc_content[:40000]}

APPLICABLE REGULATIONS:
{regulations[:20000]}

For EACH section of the document, determine:
- PASS: Section meets regulatory requirements
- FAIL: Section is missing required content or contradicts regulations
- WARNING: Section is partially compliant or could be improved

Be specific about what's missing or wrong. Cite the exact regulation.""",
                "Perform the cross-reference audit for every section.", model=model))
            while not task.done():
                yield ":\n\n"
                await asyncio.sleep(3)
            cross_ref = task.result()
            yield _agent_step(3, 4, "Cross-referencing sections against regulations", "done")

            yield _agent_step(4, 4, "Generating audit report")
            for chunk in _stream_claude(client,
                f"""You are a compliance audit report writer. Using the analysis below, write a complete, professional audit report.

DOCUMENT ANALYZED: {doc_name}
{('FOCUS AREAS: ' + focus) if focus else ''}

DOCUMENT STRUCTURE ANALYSIS:
{analysis[:5000]}

APPLICABLE REGULATIONS:
{regulations[:10000]}

CROSS-REFERENCE FINDINGS:
{cross_ref[:20000]}

Format the report with these sections:
# Compliance Audit Report: [Document Name]

## Executive Summary
Brief overview with overall compliance rating (percentage) and risk level.

## Regulations Reviewed
Table of all regulations checked.

## Section-by-Section Findings
For each section: status (PASS/FAIL/WARNING), finding detail, regulation reference, recommended action.

## Critical Issues
List any FAIL items that need immediate attention.

## Recommendations
Prioritized list of changes to achieve full compliance.

Use markdown formatting with tables where appropriate.""",
                "Write the complete audit report.", max_tokens=CHAT_MAX_TOKENS * 3, model=model):
                full_report += chunk
                yield _agent_chunk(chunk)

            yield _agent_step(4, 4, "Generating audit report", "done")

            save_db = SessionLocal()
            try:
                _save_agent_cache(save_db, cache_key, "audit", model,
                                  full_report, [body.doc_id], f"focus: {focus}" if focus else "",
                                  user_id=current_user_id)
            finally:
                save_db.close()

            yield _agent_done()

        except Exception as e:
            logging.getLogger("pdfhelper").error("Compliance audit agent failed: %s", e)
            err_msg = "Audit failed" if IS_PRODUCTION else f"Audit failed: {str(e)}"
            yield _agent_error(err_msg)

    return StreamingResponse(run_audit(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Compare Documents Agent
# ---------------------------------------------------------------------------


@router.post("/agents/compare-docs", dependencies=[Depends(verify_api_key)])
async def agent_compare_docs(body: CompareDocsRequest, request: Request, db=Depends(get_db)):
    """Multi-step document comparison agent."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not configured on server")

    doc1 = db.query(DBDocument).filter(DBDocument.id == body.doc_id_1).first()
    doc2 = db.query(DBDocument).filter(DBDocument.id == body.doc_id_2).first()
    if not doc1 or not doc2:
        raise HTTPException(status_code=404, detail="One or both documents not found")

    current_user_id = getattr(request.state, "user_id", None)
    model = _resolve_agent_model(body.model)
    h1, h2 = _get_doc_hash(doc1), _get_doc_hash(doc2)
    cache_key = _agent_cache_key("compare", model, [h1, h2], body.focus_areas)

    cached = _check_agent_cache(db, cache_key, current_user_id)
    if cached:
        async def return_cached():
            yield f"data: {json.dumps({'type': 'cached', 'message': 'Returning cached result'})}\n\n"
            yield _agent_chunk(cached)
            yield _agent_done()
        return StreamingResponse(return_cached(), media_type="text/event-stream")

    def _get_content(doc):
        pdf_bytes = _load_pdf_bytes(doc)
        structured = extract_structured_text(pdf_bytes)
        text = ""
        for page_data in structured:
            text += f"\n--- Page {page_data['page']} ---\n"
            for block in page_data.get("blocks", []):
                prefix = f"[{block['type'].upper()}] " if block['type'] != 'paragraph' else ""
                text += f"{prefix}{block['text']}\n"
        return text[:60000] if len(text) > 60000 else text

    name1 = _safe_decrypt(doc1.filename, f"Document {doc1.id[:8]}")
    name2 = _safe_decrypt(doc2.filename, f"Document {doc2.id[:8]}")
    content1 = _get_content(doc1)
    content2 = _get_content(doc2)
    focus = body.focus_areas

    client = Anthropic(api_key=api_key)

    async def run_compare():
        full_report = ""
        try:
            yield _agent_step(1, 3, "Analyzing document structures")
            task = asyncio.create_task(_call_claude_bg(client,
                "You are a document analyst. Compare the structure (sections, headings, organization) of these two documents. List the sections in each and note which sections exist in one but not the other.",
                f"DOCUMENT A: {name1}\n{content1[:30000]}\n\nDOCUMENT B: {name2}\n{content2[:30000]}", model=model))
            while not task.done():
                yield ":\n\n"
                await asyncio.sleep(3)
            structures = task.result()
            yield _agent_step(1, 3, "Analyzing document structures", "done")

            yield _agent_step(2, 3, "Comparing content section by section")
            task = asyncio.create_task(_call_claude_bg(client,
                f"""You are a document comparison specialist. Compare these two documents in detail.

DOCUMENT A: {name1}
{content1[:40000]}

DOCUMENT B: {name2}
{content2[:40000]}

STRUCTURE ANALYSIS:
{structures[:5000]}

For each shared section, identify:
- Content that is IDENTICAL or equivalent
- Content that DIFFERS (quote both versions)
- Content that is MISSING from one document
- Content that CONFLICTS between documents

{('FOCUS AREAS: ' + focus) if focus else ''}""",
                "Perform the detailed comparison.", model=model))
            while not task.done():
                yield ":\n\n"
                await asyncio.sleep(3)
            differences = task.result()
            yield _agent_step(2, 3, "Comparing content section by section", "done")

            yield _agent_step(3, 3, "Generating comparison report")
            for chunk in _stream_claude(client,
                f"""You are a document comparison report writer. Write a professional comparison report using the analysis below.

DOCUMENT A: {name1}
DOCUMENT B: {name2}
{('FOCUS AREAS: ' + focus) if focus else ''}

STRUCTURE ANALYSIS:
{structures[:5000]}

DETAILED DIFFERENCES:
{differences[:20000]}

Format the report as:
# Document Comparison: {name1} vs {name2}

## Summary
Overall similarity rating, key differences count, which document is more comprehensive.

## Structure Comparison
Table showing sections side-by-side.

## Key Differences
Each significant difference with quotes from both documents.

## Conflicts Found
Any contradictions between the documents (these are critical).

## Gaps
Content present in one but missing from the other.

## Recommendation
Which document is more complete and what each needs to match the other.

Use markdown with tables.""",
                "Write the complete comparison report.", max_tokens=CHAT_MAX_TOKENS * 3, model=model):
                full_report += chunk
                yield _agent_chunk(chunk)

            yield _agent_step(3, 3, "Generating comparison report", "done")

            save_db = SessionLocal()
            try:
                _save_agent_cache(save_db, cache_key, "compare", model,
                                  full_report, [body.doc_id_1, body.doc_id_2],
                                  f"focus: {focus}" if focus else "",
                                  user_id=current_user_id)
            finally:
                save_db.close()

            yield _agent_done()

        except Exception as e:
            logging.getLogger("pdfhelper").error("Compare docs agent failed: %s", e)
            err_msg = "Comparison failed" if IS_PRODUCTION else f"Comparison failed: {str(e)}"
            yield _agent_error(err_msg)

    return StreamingResponse(run_compare(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Procedure Writer Agent
# ---------------------------------------------------------------------------


@router.post("/agents/procedure-writer", dependencies=[Depends(verify_api_key)])
async def agent_procedure_writer(body: ProcedureWriterRequest, request: Request, db=Depends(get_db)):
    """Multi-step procedure writing agent."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not configured on server")

    current_user_id = getattr(request.state, "user_id", None)
    model = _resolve_agent_model(body.model)

    # Extract source document content (the document to base the procedure on)
    source_content = ""
    source_name = ""
    doc_hashes = []
    if body.source_doc_id:
        source_doc = db.query(DBDocument).filter(DBDocument.id == body.source_doc_id).first()
        if source_doc:
            source_name = _safe_decrypt(source_doc.filename, f"Document {source_doc.id[:8]}")
            doc_hashes.append(_get_doc_hash(source_doc))
            pdf_bytes = _load_pdf_bytes(source_doc)
            structured = extract_structured_text(pdf_bytes)
            for page_data in structured:
                source_content += f"\n--- Page {page_data['page']} ---\n"
                for block in page_data.get("blocks", []):
                    prefix = f"[{block['type'].upper()}] " if block['type'] != 'paragraph' else ""
                    source_content += f"{prefix}{block['text']}\n"
            if len(source_content) > 60000:
                source_content = source_content[:60000] + "\n[... truncated ...]"

    # Extract reference documents (for style matching)
    ref_content = ""
    ref_names = []
    if body.reference_doc_ids:
        ref_ids = [rid for rid in body.reference_doc_ids if rid != body.source_doc_id]
        if ref_ids:
            docs = db.query(DBDocument).filter(DBDocument.id.in_(ref_ids)).all()
            for doc in docs[:3]:
                name = _safe_decrypt(doc.filename, f"Document {doc.id[:8]}")
                ref_names.append(name)
                doc_hashes.append(_get_doc_hash(doc))
                pdf_bytes = _load_pdf_bytes(doc)
                structured = extract_structured_text(pdf_bytes)
                text = ""
                for page_data in structured:
                    for block in page_data.get("blocks", []):
                        prefix = f"[{block['type'].upper()}] " if block['type'] != 'paragraph' else ""
                        text += f"{prefix}{block['text']}\n"
                if len(text) > 40000:
                    text = text[:40000] + "\n[... truncated ...]"
                ref_content += f'\n--- REFERENCE: "{name}" ---\n{text}\n'

    cache_params = f"{body.description}|src={body.source_doc_id}|regs={body.include_regulations}"
    cache_key = _agent_cache_key("writer", model, doc_hashes or ["no-refs"], cache_params)

    cached = _check_agent_cache(db, cache_key, current_user_id)
    if cached:
        async def return_cached():
            yield f"data: {json.dumps({'type': 'cached', 'message': 'Returning cached result'})}\n\n"
            yield _agent_chunk(cached)
            yield _agent_done()
        return StreamingResponse(return_cached(), media_type="text/event-stream")

    client = Anthropic(api_key=api_key)
    web_tools = [{"type": "web_search_20250305", "name": "web_search"}] if CHAT_WEB_SEARCH else None
    steps = 4 if body.include_regulations else 3

    async def run_writer():
        full_doc = ""
        try:
            # Step 1: Research and outline
            yield _agent_step(1, steps, "Analyzing document and creating outline")
            outline_prompt = f"Create a detailed outline for this procedure:\n\nDESCRIPTION: {body.description}"
            if source_content:
                outline_prompt += f'\n\nSOURCE DOCUMENT ("{source_name}") — base the procedure on this content:\n{source_content[:30000]}'
            if ref_content:
                outline_prompt += f"\n\nREFERENCE PROCEDURES (match their style and structure):\n{ref_content[:15000]}"
            outline_system = "You are a technical procedure writer. Create a detailed section-by-section outline for the requested procedure."
            if source_content:
                outline_system += " The user has provided a SOURCE DOCUMENT — your outline MUST be based on the actual content, topics, processes, and specifics from that document. Extract the real procedures, steps, equipment, roles, and safety information from it. Do NOT invent generic content — use what the document actually says."
            outline_system += " Include all standard sections (Purpose, Scope, Definitions, Responsibilities, Procedure Steps, Safety, Emergency, References). Note what content goes in each section."
            task = asyncio.create_task(_call_claude_bg(client, outline_system, outline_prompt, model=model))
            while not task.done():
                yield ":\n\n"
                await asyncio.sleep(3)
            outline = task.result()
            yield _agent_step(1, steps, "Analyzing document and creating outline", "done")

            regulations = ""
            step_num = 2
            if body.include_regulations:
                yield _agent_step(2, steps, "Searching applicable regulations")
                task = asyncio.create_task(_call_claude_bg(client,
                    "You are a regulatory researcher. Search the web for all regulations, standards, and industry best practices that apply to this procedure. List each with its key requirements.",
                    f"Find applicable regulations for: {body.description}", tools=web_tools, model=model))
                while not task.done():
                    yield ":\n\n"
                    await asyncio.sleep(3)
                regulations = task.result()
                yield _agent_step(2, steps, "Searching applicable regulations", "done")
                step_num = 3

            yield _agent_step(step_num, steps, "Writing procedure content")
            write_system = f"""You are an expert technical procedure writer. Write a complete, professional procedure document.

DESCRIPTION: {body.description}

OUTLINE:
{outline[:10000]}

{('SOURCE DOCUMENT ("' + source_name + '") — base the procedure on this content:' + chr(10) + source_content[:40000]) if source_content else ''}

{('APPLICABLE REGULATIONS:' + chr(10) + regulations[:10000]) if regulations else ''}

{('REFERENCE PROCEDURES (match their style):' + chr(10) + ref_content[:15000]) if ref_content else ''}

Write a complete procedure document with:
- Professional formatting using markdown headings, numbered lists, and tables
- Clear, actionable steps with responsible parties
- Safety warnings and precautions in bold
- Regulatory references where applicable
- Standard sections: Purpose, Scope, Definitions, Responsibilities, Procedure, Safety Requirements, Emergency Procedures, References
- Specific details (not generic placeholders)"""
            if source_content:
                write_system += f"""

CRITICAL: The SOURCE DOCUMENT contains the actual content you must base this procedure on. Use the real processes, equipment names, roles, locations, safety requirements, and specific details from that document. Do NOT generate generic or hypothetical content — extract and organize what the source document actually describes into a well-structured procedure format."""

            for chunk in _stream_claude(client, write_system,
                "Write the complete procedure document now.", max_tokens=CHAT_MAX_TOKENS * 3, model=model):
                full_doc += chunk
                yield _agent_chunk(chunk)

            yield _agent_step(step_num, steps, "Writing procedure content", "done")

            yield _agent_step(steps, steps, "Running quality review")
            task = asyncio.create_task(_call_claude_bg(client,
                f"""You are a procedure quality reviewer. Review this draft procedure for:
1. Completeness — are any standard sections missing?
2. Clarity — are steps clear and unambiguous?
3. Safety — are all hazards addressed?
4. Compliance — does it meet the regulations found?
5. Consistency — any contradictions?

DRAFT:
{full_doc[:40000]}

{('REGULATIONS:' + chr(10) + regulations[:5000]) if regulations else ''}

If issues are found, list them briefly. If the document is good, say so.""",
                "Review the draft and list any issues.", model=model))
            while not task.done():
                yield ":\n\n"
                await asyncio.sleep(3)
            review = task.result()
            yield _agent_step(steps, steps, "Running quality review", "done")

            if "issue" in review.lower() or "missing" in review.lower() or "should" in review.lower():
                full_doc += "\n\n---\n\n## Quality Review Notes\n\n" + review
                yield _agent_chunk("\n\n---\n\n## Quality Review Notes\n\n" + review)

            save_db = SessionLocal()
            try:
                _save_agent_cache(save_db, cache_key, "writer", model,
                                  full_doc, body.reference_doc_ids,
                                  body.description[:200],
                                  user_id=current_user_id)
            finally:
                save_db.close()

            yield _agent_done()

        except Exception as e:
            logging.getLogger("pdfhelper").error("Procedure writer agent failed: %s", e)
            err_msg = "Writing failed" if IS_PRODUCTION else f"Writing failed: {str(e)}"
            yield _agent_error(err_msg)

    return StreamingResponse(run_writer(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Code Builder Agent
# ---------------------------------------------------------------------------


@router.post("/agents/code-builder", dependencies=[Depends(verify_api_key)])
async def agent_code_builder(body: CodeBuilderRequest, request: Request, db=Depends(get_db)):
    """Multi-step code builder agent that generates complete HTML applications from document data."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not configured on server")

    current_user_id = getattr(request.state, "user_id", None)
    model = _resolve_agent_model(body.model)

    doc_content = ""
    doc_names = []
    doc_hashes = []
    selected_docs = db.query(DBDocument).filter(DBDocument.id.in_(body.doc_ids)).all() if body.doc_ids else []
    if not selected_docs:
        selected_docs = db.query(DBDocument).filter(
            DBDocument.user_id == current_user_id
        ).all() if current_user_id else []

    for doc in selected_docs[:5]:
        name = _safe_decrypt(doc.filename, f"Document {doc.id[:8]}")
        doc_names.append(name)
        doc_hashes.append(_get_doc_hash(doc))
        pdf_bytes = _load_pdf_bytes(doc)
        structured = extract_structured_text(pdf_bytes)
        text = ""
        for page_data in structured:
            text += f"\n--- Page {page_data['page']} ---\n"
            for block in page_data.get("blocks", []):
                prefix = f"[{block['type'].upper()}] " if block['type'] != 'paragraph' else ""
                text += f"{prefix}{block['text']}\n"
        if len(text) > 80000:
            text = text[:80000] + "\n[... truncated ...]"
        doc_content += f'\n===== DOCUMENT: "{name}" =====\n{text}\n'

    if len(doc_content) > 300000:
        doc_content = doc_content[:300000] + "\n[... truncated ...]"

    cache_params = f"{body.description}|type={body.app_type}"
    cache_key = _agent_cache_key("code-builder", model, doc_hashes or ["no-docs"], cache_params)

    cached = _check_agent_cache(db, cache_key, current_user_id)
    if cached:
        async def return_cached():
            yield f"data: {json.dumps({'type': 'cached', 'message': 'Returning cached result'})}\n\n"
            yield _agent_chunk(cached)
            yield _agent_done()
        return StreamingResponse(return_cached(), media_type="text/event-stream")

    client = Anthropic(api_key=api_key)

    async def run_builder():
        full_code = ""
        try:
            # Step 1: Extract and structure the data
            yield _agent_step(1, 3, "Extracting data from documents")
            extract_system = """You are a data extraction specialist. Extract ALL relevant data from the provided documents and organize it into structured JSON-like format.

Extract:
- All tables (preserve rows and columns)
- All lists and checklists
- All named items, categories, and their properties
- All numerical data, dates, statuses
- All personnel roles, responsibilities
- All procedures, steps, requirements
- All section headings and their content hierarchy

Output the extracted data in a clear, organized format that a code generator can use. Include EVERY piece of data — do not summarize or skip anything."""

            task = asyncio.create_task(_call_claude_bg(client, extract_system,
                f"Extract all data from these documents:\n\n{doc_content[:150000]}",
                max_tokens=CHAT_MAX_TOKENS, model=model))
            while not task.done():
                yield ":\n\n"
                await asyncio.sleep(3)
            extracted_data = task.result()
            yield _agent_step(1, 3, "Extracting data from documents", "done")

            # Step 2: Plan the application
            yield _agent_step(2, 3, "Planning application structure")
            plan_system = f"""You are a senior web application architect. Plan a complete single-file HTML5 application.

APPLICATION TYPE: {body.app_type}
USER REQUEST: {body.description}
DOCUMENTS USED: {', '.join(doc_names)}

Plan the application with:
1. Component layout (what sections, panels, and UI elements)
2. Data model (how the extracted data maps to the UI)
3. Interactivity (filters, search, sorting, tabs, modals)
4. Color scheme and visual design approach
5. Responsive layout strategy

Keep the plan focused and specific to the actual data provided."""

            task = asyncio.create_task(_call_claude_bg(client, plan_system,
                f"Plan the application using this extracted data:\n\n{extracted_data[:50000]}",
                model=model))
            while not task.done():
                yield ":\n\n"
                await asyncio.sleep(3)
            plan = task.result()
            yield _agent_step(2, 3, "Planning application structure", "done")

            # Step 3: Generate the complete code
            yield _agent_step(3, 3, "Generating complete HTML application")
            code_system = f"""You are an expert front-end developer. Generate a COMPLETE, PRODUCTION-READY, single-file HTML5 application.

APPLICATION TYPE: {body.app_type}
USER REQUEST: {body.description}
DOCUMENTS USED: {', '.join(doc_names)}

APPLICATION PLAN:
{plan[:15000]}

ABSOLUTE REQUIREMENTS — YOUR CODE WILL BE REJECTED IF ANY ARE VIOLATED:
1. Output ONLY the HTML code — start with <!DOCTYPE html> and end with </html>. No explanations, no markdown fences, no commentary before or after the code.
2. SINGLE FILE — all CSS in <style> tags, all JavaScript in <script> tags. ZERO external dependencies (no CDN links, no imports, no external fonts).
3. ALL DATA EMBEDDED — every piece of data from the documents must be hardcoded as JavaScript arrays/objects inside the file. Never use placeholder data like "Item 1", "Lorem ipsum", or "TODO".
4. FULLY FUNCTIONAL — every button, filter, search box, tab, and interactive element must work. Test mentally: click each button, type in each input — does it do something?
5. RESPONSIVE — must work on both desktop (1200px+) and mobile (375px). Use CSS Grid/Flexbox, relative units, and media queries.
6. PROFESSIONAL DESIGN — clean modern UI with a cohesive color scheme, proper spacing, shadows, rounded corners, hover states on interactive elements.
7. COMPLETE — do not truncate, abbreviate, or skip any section. The output must be the entire working application.
8. INCLUDE: search/filter functionality, sorting where applicable, print-friendly styles (@media print), and a professional header with the application title.

Remember: Output ONLY the raw HTML code. No markdown, no explanations."""

            for chunk in _stream_claude(client, code_system,
                f"Generate the complete HTML application using this data:\n\n{extracted_data[:80000]}",
                max_tokens=CHAT_MAX_TOKENS * 3, model=model):
                full_code += chunk
                yield _agent_chunk(chunk)

            yield _agent_step(3, 3, "Generating complete HTML application", "done")

            save_db = SessionLocal()
            try:
                _save_agent_cache(save_db, cache_key, "code-builder", model,
                                  full_code, body.doc_ids,
                                  body.description[:200],
                                  user_id=current_user_id)
            finally:
                save_db.close()

            yield _agent_done()

        except Exception as e:
            logging.getLogger("pdfhelper").error("Code builder agent failed: %s", e)
            err_msg = "Code generation failed" if IS_PRODUCTION else f"Code generation failed: {str(e)}"
            yield _agent_error(err_msg)

    return StreamingResponse(run_builder(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Agent Cache Management
# ---------------------------------------------------------------------------


@router.get("/agents/cache", dependencies=[Depends(verify_api_key)])
async def list_agent_cache(request: Request, db=Depends(get_db)):
    """List cached agent results for the current user (metadata only)."""
    current_user_id = getattr(request.state, "user_id", None)
    q = db.query(DBAgentCache)
    if current_user_id:
        q = q.filter(DBAgentCache.user_id == current_user_id)
    else:
        q = q.filter(DBAgentCache.user_id.is_(None))
    entries = q.order_by(DBAgentCache.created_at.desc()).limit(50).all()
    return {
        "cache_entries": [
            {
                "id": e.id,
                "agent_type": e.agent_type,
                "model_used": e.model_used,
                "doc_ids": json.loads(e.doc_ids),
                "params_summary": e.params_summary,
                "created_at": e.created_at.isoformat() if e.created_at else None,
                "expires_at": e.expires_at.isoformat() if e.expires_at else None,
            }
            for e in entries
        ],
        "total": len(entries),
    }


@router.delete("/agents/cache", dependencies=[Depends(verify_api_key)])
async def clear_all_agent_cache(request: Request, db=Depends(get_db)):
    """Clear all cached agent results for the current user only."""
    current_user_id = getattr(request.state, "user_id", None)
    q = db.query(DBAgentCache)
    if current_user_id:
        q = q.filter(DBAgentCache.user_id == current_user_id)
    else:
        q = q.filter(DBAgentCache.user_id.is_(None))
    count = q.delete()
    db.commit()
    return {"deleted": count}


@router.delete("/agents/cache/{cache_id}", dependencies=[Depends(verify_api_key)])
async def delete_agent_cache_entry(cache_id: str, request: Request, db=Depends(get_db)):
    """Delete a specific cached agent result, enforcing ownership."""
    current_user_id = getattr(request.state, "user_id", None)
    entry = db.query(DBAgentCache).filter(DBAgentCache.id == cache_id).first()
    if not entry:
        raise HTTPException(status_code=404, detail="Cache entry not found")
    if current_user_id and entry.user_id and entry.user_id != current_user_id:
        raise HTTPException(status_code=403, detail="You do not own this cache entry")
    db.delete(entry)
    db.commit()
    return {"deleted": True}
