"""
PDFHelper AI Agent System.

Specialized agents that each handle a different aspect of document analysis.
The orchestrator coordinates them and merges their results into a single report.
"""

import json
import os
import re

from anthropic import Anthropic


def _get_client() -> Anthropic:
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY not configured")
    return Anthropic(api_key=key)


MODEL = "claude-sonnet-4-5-20250929"


def _call_ai(system_prompt: str, user_prompt: str, max_tokens: int = 4096) -> str:
    """Make a single AI call with a system prompt and user prompt."""
    client = _get_client()
    response = client.messages.create(
        model=MODEL,
        max_tokens=max_tokens,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return response.content[0].text.strip()


def _parse_json_response(text: str) -> list | dict:
    """Extract JSON from an AI response, handling markdown code blocks."""
    cleaned = text
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned)
    return json.loads(cleaned)


# ---------------------------------------------------------------------------
# Agent: Document Analyzer
# ---------------------------------------------------------------------------

def document_analysis_agent(filename: str, pages: list[dict]) -> dict:
    """Deeply analyze a single document — structure, topics, key findings.

    Returns a structured summary of what the document contains.
    """
    # Build a condensed view (first 8000 chars per page batch to stay in limits)
    page_text = ""
    for p in pages:
        page_text += f"\n--- PAGE {p['page']} ---\n{p['text'][:2000]}\n"

    system = """You are a document analysis specialist. Your job is to deeply
analyze documents and produce structured summaries. Be thorough and precise.
Respond with ONLY valid JSON, no other text."""

    prompt = f"""Analyze this document: "{filename}"

CONTENT:
{page_text}

Produce a JSON object with:
{{
  "title": "document title if identifiable",
  "type": "procedure/policy/manual/report/form/other",
  "topics": ["list of main topics covered"],
  "key_dates": ["any dates mentioned (revision dates, effective dates, deadlines)"],
  "key_references": ["any regulations, standards, or other documents referenced"],
  "sections": ["list of major sections or headings"],
  "summary": "2-3 sentence summary of the document's purpose and content"
}}"""

    try:
        result = _parse_json_response(_call_ai(system, prompt))
        return result if isinstance(result, dict) else {}
    except (json.JSONDecodeError, Exception):
        return {"error": f"Could not analyze {filename}"}


# ---------------------------------------------------------------------------
# Agent: Cross-Reference Checker
# ---------------------------------------------------------------------------

def cross_reference_agent(documents: list[dict]) -> list[dict]:
    """Compare content across multiple documents to find inconsistencies.

    Each item in documents: {"filename": str, "summary": str, "topics": list, "key_references": list}

    Returns a list of inconsistencies/conflicts found.
    """
    if len(documents) < 2:
        return []

    system = """You are a cross-reference specialist. Your job is to compare
multiple documents and find inconsistencies, conflicts, contradictions, or
outdated cross-references between them. Be specific about what conflicts and
where. Respond with ONLY valid JSON, no other text."""

    doc_summaries = ""
    for i, doc in enumerate(documents, 1):
        doc_summaries += f"""
--- DOCUMENT {i}: {doc['filename']} ---
Summary: {doc.get('summary', 'N/A')}
Topics: {', '.join(doc.get('topics', []))}
Key References: {', '.join(doc.get('key_references', []))}
Key Dates: {', '.join(doc.get('key_dates', []))}
Sections: {', '.join(doc.get('sections', []))}
"""

    prompt = f"""Compare these {len(documents)} documents and find any inconsistencies,
conflicts, or contradictions between them:

{doc_summaries}

Return a JSON array of findings. Each finding:
{{
  "type": "conflict/inconsistency/outdated_reference/gap",
  "documents_involved": ["doc1.pdf", "doc2.pdf"],
  "description": "what the issue is",
  "severity": "high/medium/low",
  "recommendation": "what should be done to resolve it"
}}

If no issues found, return an empty array: []"""

    try:
        result = _parse_json_response(_call_ai(system, prompt, max_tokens=4096))
        return result if isinstance(result, list) else []
    except (json.JSONDecodeError, Exception):
        return []


# ---------------------------------------------------------------------------
# Agent: Compliance Checker
# ---------------------------------------------------------------------------

def compliance_agent(filename: str, pages: list[dict],
                     compliance_context: str | None = None) -> list[dict]:
    """Check a document for compliance issues — outdated regulations,
    missing required language, policy gaps.

    compliance_context: optional user-provided context like
    "Check against OSHA 2024 standards" or "FDA 21 CFR Part 11"
    """
    page_text = ""
    for p in pages:
        page_text += f"\n--- PAGE {p['page']} ---\n{p['text'][:2000]}\n"

    context_instruction = ""
    if compliance_context:
        context_instruction = f"\nThe user specifically wants you to check against: {compliance_context}\n"

    system = """You are a compliance and regulatory specialist. Your job is to
review documents for compliance issues: outdated regulatory references, missing
required language, policy gaps, and procedural deficiencies. Be specific about
page numbers and exact text. Respond with ONLY valid JSON, no other text."""

    prompt = f"""Review this document for compliance issues: "{filename}"
{context_instruction}
CONTENT:
{page_text}

Return a JSON array of compliance findings. Each finding:
{{
  "page": page_number,
  "issue_type": "outdated_reference/missing_language/policy_gap/procedural_deficiency/formatting_issue",
  "found_text": "the specific text that has the issue (quote directly)",
  "issue": "description of the compliance problem",
  "severity": "critical/high/medium/low",
  "recommendation": "specific suggestion to fix the issue"
}}

If the document appears compliant with no issues, return an empty array: []"""

    try:
        result = _parse_json_response(_call_ai(system, prompt, max_tokens=4096))
        return result if isinstance(result, list) else []
    except (json.JSONDecodeError, Exception):
        return []


# ---------------------------------------------------------------------------
# Agent: Summary Report Generator
# ---------------------------------------------------------------------------

def summary_report_agent(
    doc_analyses: list[dict],
    cross_ref_findings: list[dict],
    compliance_findings: dict[str, list[dict]],
    search_results: dict | None = None,
) -> dict:
    """Take all agent outputs and produce a clean, actionable summary report.

    Returns a structured report with priorities and action items.
    """
    system = """You are a report generation specialist. Your job is to take raw
analysis data and produce a clear, actionable executive summary. Prioritize
findings by severity. Be concise but thorough. Respond with ONLY valid JSON,
no other text."""

    # Build the input data
    input_data = {
        "document_analyses": doc_analyses,
        "cross_reference_issues": cross_ref_findings,
        "compliance_findings_by_document": compliance_findings,
    }
    if search_results:
        input_data["search_results"] = search_results

    prompt = f"""Generate an executive summary report from this analysis data:

{json.dumps(input_data, indent=2)[:12000]}

Return a JSON object:
{{
  "executive_summary": "2-4 sentence overview of all findings",
  "documents_reviewed": number,
  "total_issues_found": number,
  "critical_issues": number,
  "action_items": [
    {{
      "priority": "critical/high/medium/low",
      "document": "filename",
      "action": "what needs to be done",
      "details": "more context about why"
    }}
  ],
  "cross_reference_summary": "summary of cross-document issues if any",
  "compliance_summary": "summary of compliance issues if any",
  "overall_risk_level": "high/medium/low",
  "recommendation": "top-level recommendation for next steps"
}}"""

    try:
        result = _parse_json_response(_call_ai(system, prompt, max_tokens=4096))
        return result if isinstance(result, dict) else {}
    except (json.JSONDecodeError, Exception):
        return {"error": "Could not generate summary report"}


# ---------------------------------------------------------------------------
# Orchestrator — runs the full pipeline
# ---------------------------------------------------------------------------

def run_full_analysis(
    documents: dict[str, list[dict]],
    compliance_context: str | None = None,
    search_terms: list[str] | None = None,
    ai_query: str | None = None,
) -> dict:
    """Run the full multi-agent analysis pipeline.

    documents: {"filename": [{"page": 1, "text": "..."}, ...], ...}

    Returns the complete analysis with all agent outputs merged.
    """
    # Step 1: Analyze each document individually
    doc_analyses = []
    for filename, pages in documents.items():
        analysis = document_analysis_agent(filename, pages)
        analysis["filename"] = filename
        doc_analyses.append(analysis)

    # Step 2: Cross-reference check (only if multiple documents)
    cross_ref_findings = []
    if len(documents) > 1:
        cross_ref_findings = cross_reference_agent(doc_analyses)

    # Step 3: Compliance check on each document
    compliance_findings = {}
    for filename, pages in documents.items():
        findings = compliance_agent(filename, pages, compliance_context)
        if findings:
            compliance_findings[filename] = findings

    # Step 4: Run keyword/AI search if requested
    search_results = None
    if search_terms or ai_query:
        from app import keyword_search, ai_search
        search_results = {"keyword_results": [], "ai_results": []}
        for filename, pages in documents.items():
            if search_terms:
                kw_matches = keyword_search(pages, search_terms)
                for m in kw_matches:
                    m["filename"] = filename
                search_results["keyword_results"].extend(kw_matches)
            if ai_query:
                ai_matches = ai_search(pages, ai_query, filename)
                for m in ai_matches:
                    m["filename"] = filename
                search_results["ai_results"].extend(ai_matches)

    # Step 5: Generate summary report
    report = summary_report_agent(
        doc_analyses, cross_ref_findings, compliance_findings, search_results,
    )

    return {
        "report": report,
        "document_analyses": doc_analyses,
        "cross_reference_findings": cross_ref_findings,
        "compliance_findings": compliance_findings,
        "search_results": search_results,
    }
