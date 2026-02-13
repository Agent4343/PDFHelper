"""
Shared search functions for PDFHelper.

Used by both the web API (app.py) and the agent pipeline (agents.py).
"""

import json
import os
import re


def keyword_search(pages: list[dict], search_terms: list[str],
                   case_sensitive: bool = False) -> list[dict]:
    """Search pages for exact keyword/phrase matches.

    Returns a list of match dicts with page, term, matched_text, and context.
    """
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
    """Use Claude AI to semantically search pages for concepts.

    Returns a list of findings with page, matched_text, reason, needs_review,
    and suggestion fields.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not configured on server")

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
