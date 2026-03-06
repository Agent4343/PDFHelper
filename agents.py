"""
PDFHelper AI Agent System.

Specialized agents that each handle a different aspect of document analysis.
The orchestrator coordinates them and merges their results into a single report.
"""

import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from anthropic import Anthropic

logger = logging.getLogger(__name__)


_client: Anthropic | None = None


def _get_client() -> Anthropic:
    """Return a shared Anthropic client instance (created once, reused)."""
    global _client
    if _client is None:
        key = os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY not configured")
        _client = Anthropic(api_key=key)
    return _client


MODEL = "claude-sonnet-4-5-20250929"


MAX_RETRIES = 3
RETRY_BACKOFF = 2  # seconds, doubled each retry


def _call_ai(system_prompt: str, user_prompt: str, max_tokens: int = 4096) -> str:
    """Make a single AI call with a system prompt and user prompt.

    Retries on transient errors (rate limits, server errors) with exponential backoff.
    """
    for attempt in range(MAX_RETRIES):
        try:
            response = _get_client().messages.create(
                model=MODEL,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            return response.content[0].text.strip()
        except Exception as exc:
            is_retryable = _is_retryable_error(exc)
            if is_retryable and attempt < MAX_RETRIES - 1:
                wait = RETRY_BACKOFF * (2 ** attempt)
                logger.warning("AI call failed (attempt %d/%d), retrying in %ds: %s",
                               attempt + 1, MAX_RETRIES, wait, exc)
                time.sleep(wait)
            else:
                raise


def _is_retryable_error(exc: Exception) -> bool:
    """Check if an API error is transient and worth retrying."""
    exc_str = str(exc).lower()
    retryable_indicators = ["rate_limit", "overloaded", "529", "500", "502", "503", "timeout"]
    return any(indicator in exc_str for indicator in retryable_indicators)


def _parse_json_response(text: str) -> list | dict:
    """Extract JSON from an AI response, handling markdown code blocks."""
    cleaned = text
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned)
    return json.loads(cleaned)


# ---------------------------------------------------------------------------
# Text budget helpers
# ---------------------------------------------------------------------------

# Total character budget for text sent to each agent call.
# Claude Sonnet has ~680K char context; we stay well under with room for
# the system prompt, JSON schema, and response.
_AGENT_TEXT_BUDGET = 180_000


def _build_page_text(pages: list[dict], budget: int = _AGENT_TEXT_BUDGET) -> str:
    """Build concatenated page text that fits within a character budget.

    Distributes the budget evenly across pages, but allows shorter pages to
    donate their unused budget to longer ones (two-pass).  This avoids the
    old fixed 2000-char-per-page limit that truncated dense safety tables.
    """
    if not pages:
        return ""

    # First pass: find pages shorter than their equal share
    equal_share = budget // len(pages)
    short_pages = {}  # page index -> actual length
    surplus = 0
    for i, p in enumerate(pages):
        length = len(p["text"])
        if length <= equal_share:
            short_pages[i] = length
            surplus += equal_share - length

    # Second pass: long pages split the surplus
    long_count = len(pages) - len(short_pages)
    long_limit = equal_share + (surplus // max(long_count, 1)) if long_count else equal_share

    parts = []
    for i, p in enumerate(pages):
        limit = len(p["text"]) if i in short_pages else long_limit
        parts.append(f"\n--- PAGE {p['page']} ---\n{p['text'][:limit]}\n")
    return "".join(parts)


# ---------------------------------------------------------------------------
# Agent: Document Analyzer
# ---------------------------------------------------------------------------

def document_analysis_agent(filename: str, pages: list[dict]) -> dict:
    """Deeply analyze a single document — structure, topics, key findings.

    Returns a structured summary of what the document contains.
    """
    page_text = _build_page_text(pages)

    system = """You are a document analysis specialist. Your job is to deeply
analyze documents and produce structured summaries. Pay special attention to:
- Revision history and document version information
- Regulatory references (specific regulation sections, standards codes)
- Safety-critical procedures, roles, and responsibilities
- Tables containing specifications, approved lists, or reference values
Be thorough and precise. Respond with ONLY valid JSON, no other text."""

    prompt = f"""Analyze this document: "{filename}"

CONTENT:
{page_text}

Produce a JSON object with:
{{
  "title": "document title if identifiable",
  "type": "procedure/policy/manual/report/form/safety_document/regulatory/other",
  "topics": ["list of main topics covered"],
  "key_dates": ["any dates mentioned (revision dates, effective dates, deadlines)"],
  "current_revision": "revision identifier if found (e.g. D14, Rev 3)",
  "key_references": ["any regulations, standards, or other documents referenced — include section numbers"],
  "regulatory_references": ["specific regulatory citations e.g. 'OHS Regulations Section 144(3)', 'CSA 117.2'"],
  "sections": ["list of major sections or headings"],
  "roles_identified": ["any roles/positions mentioned (e.g. PIC, Permit Holder, Area Authority)"],
  "summary": "2-3 sentence summary of the document's purpose and content"
}}"""

    try:
        result = _parse_json_response(_call_ai(system, prompt))
        return result if isinstance(result, dict) else {}
    except json.JSONDecodeError:
        logger.warning("Document analysis returned invalid JSON for %s", filename)
        return {"error": f"Could not analyze {filename}"}
    except Exception:
        logger.error("Document analysis failed for %s", filename, exc_info=True)
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
Revision: {doc.get('current_revision', 'N/A')}
Topics: {', '.join(doc.get('topics', []))}
Key References: {', '.join(doc.get('key_references', []))}
Regulatory References: {', '.join(doc.get('regulatory_references', []))}
Key Dates: {', '.join(doc.get('key_dates', []))}
Sections: {', '.join(doc.get('sections', []))}
Roles: {', '.join(doc.get('roles_identified', []))}
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
    except json.JSONDecodeError:
        logger.warning("Cross-reference agent returned invalid JSON")
        return []
    except Exception:
        logger.error("Cross-reference agent failed", exc_info=True)
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
    page_text = _build_page_text(pages)

    context_instruction = ""
    if compliance_context:
        context_instruction = f"\nThe user specifically wants you to check against: {compliance_context}\n"

    system = """You are a compliance and regulatory specialist with expertise in
industrial safety, offshore operations, and occupational health regulations.
Your job is to review documents for compliance issues including:
- Outdated regulatory references or standards citations
- Missing required language (e.g. required permit conditions, safety statements)
- Policy gaps where procedures don't cover required scenarios
- Inconsistencies between referenced regulations and stated procedures
- Missing or outdated revision dates
- Incomplete role/responsibility definitions
Be specific about page numbers and quote exact text. Respond with ONLY valid JSON,
no other text."""

    prompt = f"""Review this document for compliance issues: "{filename}"
{context_instruction}
CONTENT:
{page_text}

Return a JSON array of compliance findings. Each finding:
{{
  "page": page_number,
  "issue_type": "outdated_reference/missing_language/policy_gap/procedural_deficiency/formatting_issue/role_gap/isolation_concern",
  "found_text": "the specific text that has the issue (quote directly)",
  "issue": "description of the compliance problem",
  "severity": "critical/high/medium/low",
  "regulation": "the specific regulation or standard this relates to, if applicable",
  "recommendation": "specific suggestion to fix the issue"
}}

If the document appears compliant with no issues, return an empty array: []"""

    try:
        result = _parse_json_response(_call_ai(system, prompt, max_tokens=4096))
        return result if isinstance(result, list) else []
    except json.JSONDecodeError:
        logger.warning("Compliance agent returned invalid JSON for %s", filename)
        return []
    except Exception:
        logger.error("Compliance agent failed for %s", filename, exc_info=True)
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

{json.dumps(input_data, indent=2)[:30000]}

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
        result = _parse_json_response(_call_ai(system, prompt, max_tokens=8192))
        return result if isinstance(result, dict) else {}
    except json.JSONDecodeError:
        logger.warning("Summary report agent returned invalid JSON")
        return {"error": "Could not generate summary report"}
    except Exception:
        logger.error("Summary report agent failed", exc_info=True)
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
    # Steps 1 & 3: Run document analysis and compliance checks in parallel
    # (each document is independent, so all calls can run concurrently)
    doc_analyses = []
    compliance_findings = {}

    with ThreadPoolExecutor(max_workers=min(4, len(documents) * 2)) as pool:
        # Submit all document analysis tasks
        analysis_futures = {
            pool.submit(document_analysis_agent, filename, pages): filename
            for filename, pages in documents.items()
        }
        # Submit all compliance tasks at the same time
        compliance_futures = {
            pool.submit(compliance_agent, filename, pages, compliance_context): filename
            for filename, pages in documents.items()
        }

        # Collect document analysis results
        for future in as_completed(analysis_futures):
            filename = analysis_futures[future]
            analysis = future.result()
            analysis["filename"] = filename
            doc_analyses.append(analysis)

        # Collect compliance results
        for future in as_completed(compliance_futures):
            filename = compliance_futures[future]
            findings = future.result()
            if findings:
                compliance_findings[filename] = findings

    # Step 2 & 3: Cross-reference check and search run in parallel
    # (cross-ref needs doc_analyses from step 1, search is independent)
    cross_ref_findings = []
    search_results = None

    def _run_cross_ref():
        if len(documents) > 1:
            return cross_reference_agent(doc_analyses)
        return []

    def _run_search():
        if not search_terms and not ai_query:
            return None
        from search import keyword_search, ai_search
        results = {"keyword_results": [], "ai_results": []}
        for filename, pages in documents.items():
            if search_terms:
                kw_matches = keyword_search(pages, search_terms)
                for m in kw_matches:
                    m["filename"] = filename
                results["keyword_results"].extend(kw_matches)
            if ai_query:
                ai_matches = ai_search(pages, ai_query, filename)
                for m in ai_matches:
                    m["filename"] = filename
                results["ai_results"].extend(ai_matches)
        return results

    with ThreadPoolExecutor(max_workers=2) as pool:
        cross_ref_future = pool.submit(_run_cross_ref)
        search_future = pool.submit(_run_search)
        cross_ref_findings = cross_ref_future.result()
        search_results = search_future.result()

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
