"""
Shared search functions for PDFHelper.

Used by both the web API (app.py) and the agent pipeline (agents.py).
"""

import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils import is_retryable_error as _is_retryable_error, parse_json_response as _parse_json_response

logger = logging.getLogger(__name__)

MODEL = os.getenv("SEARCH_MODEL", "claude-sonnet-4-5-20250929")

_client = None


def _get_client():
    """Return a shared Anthropic client instance (created once, reused)."""
    global _client
    if _client is None:
        key = os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY not configured on server")
        from anthropic import Anthropic
        _client = Anthropic(api_key=key)
    return _client


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


def _build_batch_prompt(filename: str, query: str, batch: list[dict]) -> str:
    """Build the AI search prompt for a batch of pages."""
    page_texts = ""
    for p in batch:
        page_texts += f"\n--- PAGE {p['page']} ---\n{p['text']}\n"

    return f"""You are a document reviewer. Analyze the following PDF pages from "{filename}" and search for content related to this query:

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


MAX_RETRIES = 3
RETRY_BACKOFF = 2


def _search_batch(client, prompt: str) -> list[dict]:
    """Send a single batch to the AI and parse the response. Thread-safe.

    Retries on transient API errors with exponential backoff.
    """
    response_text = ""
    for attempt in range(MAX_RETRIES):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
            )
            response_text = response.content[0].text.strip()
            findings = _parse_json_response(response_text)
            return [
                {
                    "page": f.get("page", "?"),
                    "matched_text": f.get("matched_text", ""),
                    "reason": f.get("reason", ""),
                    "needs_review": f.get("needs_review", False),
                    "suggestion": f.get("suggestion", ""),
                }
                for f in findings
            ]
        except json.JSONDecodeError:
            logger.warning("AI search returned invalid JSON: %.200s", response_text)
            return []
        except Exception as exc:
            if _is_retryable_error(exc) and attempt < MAX_RETRIES - 1:
                wait = RETRY_BACKOFF * (2 ** attempt)
                logger.warning("AI search batch failed (attempt %d/%d), retrying in %ds: %s",
                               attempt + 1, MAX_RETRIES, wait, exc)
                time.sleep(wait)
            else:
                logger.error("AI search batch failed", exc_info=True)
                return []
    return []


def ai_search(pages: list[dict], query: str, filename: str) -> list[dict]:
    """Use Claude AI to semantically search pages for concepts.

    Sends page batches in parallel for faster results on large documents.

    Returns a list of findings with page, matched_text, reason, needs_review,
    and suggestion fields.
    """
    client = _get_client()

    # Build batches
    batch_size = 5
    batches = []
    for i in range(0, len(pages), batch_size):
        batch = pages[i:i + batch_size]
        page_texts = "".join(p["text"] for p in batch)
        if not page_texts.strip():
            continue
        prompt = _build_batch_prompt(filename, query, batch)
        batches.append(prompt)

    if not batches:
        return []

    # Single batch — no need for threading overhead
    if len(batches) == 1:
        return _search_batch(client, batches[0])

    # Multiple batches — run in parallel (up to 3 concurrent API calls)
    results = []
    max_workers = min(3, len(batches))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_search_batch, client, prompt) for prompt in batches]
        for future in as_completed(futures):
            results.extend(future.result())

    return results
