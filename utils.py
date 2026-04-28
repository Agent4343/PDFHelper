"""
Shared utilities for PDFHelper.

Centralises helpers that are used by multiple modules (search, agents, etc.)
to avoid duplication.
"""

import json
import re


def is_retryable_error(exc: Exception) -> bool:
    """Check if an API error is transient and worth retrying."""
    exc_str = str(exc).lower()
    retryable_indicators = ["rate_limit", "overloaded", "529", "500", "502", "503", "timeout"]
    return any(indicator in exc_str for indicator in retryable_indicators)


def parse_json_response(text: str) -> list | dict:
    """Extract JSON from an AI response, handling markdown code blocks."""
    cleaned = text
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned)
    return json.loads(cleaned)
