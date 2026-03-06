#!/usr/bin/env python3
"""
PDFHelper - AI-powered PDF search and flagging tool.

Searches through one or more PDFs for specific words, phrases, or concepts.
Uses both exact keyword matching and AI-powered semantic analysis to find
relevant content and flag items that may need changes.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

try:
    import fitz  # PyMuPDF
except ImportError:
    sys.exit("Error: PyMuPDF is required. Install with: pip install PyMuPDF")

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

console = Console()


# ---------------------------------------------------------------------------
# PDF text extraction
# ---------------------------------------------------------------------------

def _extract_page_text(page) -> str:
    """Extract text from a single page, using table-aware extraction when available.

    PyMuPDF 1.23+ supports find_tables() which preserves tabular structure
    that get_text() garbles (e.g. valve isolation tables, specification charts).
    Falls back to plain get_text() on older versions.
    """
    if hasattr(page, "find_tables"):
        try:
            tables = page.find_tables()
            if tables.tables:
                parts = [page.get_text()]
                for table in tables:
                    try:
                        df = table.to_pandas()
                        parts.append("\n[TABLE]\n" + df.to_string(index=False) + "\n[/TABLE]\n")
                    except Exception:
                        rows = table.extract()
                        if rows:
                            table_lines = []
                            for row in rows:
                                cells = [str(c) if c is not None else "" for c in row]
                                table_lines.append(" | ".join(cells))
                            parts.append("\n[TABLE]\n" + "\n".join(table_lines) + "\n[/TABLE]\n")
                return "\n".join(parts)
        except Exception:
            pass
    return page.get_text()


def extract_text_from_pdf(pdf_path: str) -> list[dict]:
    """Extract text from each page of a PDF.

    Uses table-aware extraction when PyMuPDF supports it, preserving the
    structure of specification tables and similar tabular data.

    Returns a list of dicts: [{"page": 1, "text": "..."}, ...]
    """
    pages = []
    try:
        doc = fitz.open(pdf_path)
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = _extract_page_text(page)
            pages.append({"page": page_num + 1, "text": text})
        doc.close()
    except Exception as e:
        console.print(f"[red]Error reading {pdf_path}: {e}[/red]")
    return pages


def load_pdfs(paths: list[str]) -> dict[str, list[dict]]:
    """Load multiple PDFs and return their extracted text.

    Returns: {"filename.pdf": [{"page": 1, "text": "..."}, ...], ...}
    """
    all_docs = {}
    for path in paths:
        p = Path(path)
        if not p.exists():
            console.print(f"[yellow]Warning: {path} not found, skipping.[/yellow]")
            continue
        if p.is_dir():
            pdf_files = sorted(p.glob("*.pdf"))
            if not pdf_files:
                console.print(f"[yellow]No PDFs found in {path}[/yellow]")
            for pdf_file in pdf_files:
                console.print(f"  Loading [cyan]{pdf_file.name}[/cyan]...")
                all_docs[str(pdf_file)] = extract_text_from_pdf(str(pdf_file))
        else:
            console.print(f"  Loading [cyan]{p.name}[/cyan]...")
            all_docs[str(p)] = extract_text_from_pdf(str(p))
    return all_docs


# ---------------------------------------------------------------------------
# Search functions — delegate to the shared search module
# ---------------------------------------------------------------------------

from search import keyword_search as _kw_search, ai_search as _ai_search


def keyword_search(docs: dict, search_terms: list[str], case_sensitive: bool = False) -> list[dict]:
    """Search all loaded PDFs for exact keyword/phrase matches."""
    results = []
    for filepath, pages in docs.items():
        filename = Path(filepath).name
        matches = _kw_search(pages, search_terms, case_sensitive)
        for m in matches:
            m["file"] = filename
            m["filepath"] = filepath
        results.extend(matches)
    return results


def ai_search(docs: dict, query: str, api_key: str | None = None) -> list[dict]:
    """Use Claude AI to semantically search PDFs for concepts and ideas."""
    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        console.print("[red]Error: ANTHROPIC_API_KEY environment variable not set. "
                       "Set it or pass --api-key to use AI search.[/red]")
        return []
    # Temporarily set the key so search.ai_search can find it
    old_key = os.environ.get("ANTHROPIC_API_KEY")
    os.environ["ANTHROPIC_API_KEY"] = key

    results = []
    try:
        for filepath, pages in docs.items():
            filename = Path(filepath).name
            findings = _ai_search(pages, query, filename)
            for f in findings:
                f["file"] = filename
                f["filepath"] = filepath
            results.extend(findings)
    except Exception as e:
        console.print(f"[red]AI search error: {e}[/red]")
    finally:
        if old_key is not None:
            os.environ["ANTHROPIC_API_KEY"] = old_key
        elif "ANTHROPIC_API_KEY" in os.environ:
            del os.environ["ANTHROPIC_API_KEY"]

    return results


# ---------------------------------------------------------------------------
# Display results
# ---------------------------------------------------------------------------

def display_keyword_results(results: list[dict]) -> None:
    """Pretty-print keyword search results."""
    if not results:
        console.print(Panel("[green]No keyword matches found.[/green]"))
        return

    table = Table(title="Keyword Search Results", show_lines=True)
    table.add_column("File", style="cyan", max_width=25)
    table.add_column("Page", style="magenta", justify="center", width=6)
    table.add_column("Term", style="yellow", max_width=20)
    table.add_column("Context", style="white", max_width=60)

    for r in results:
        table.add_row(r["file"], str(r["page"]), r["matched_text"], r["context"])

    console.print(table)
    console.print(f"\n[bold]Total matches: {len(results)}[/bold]")


def display_ai_results(results: list[dict]) -> None:
    """Pretty-print AI search results."""
    if not results:
        console.print(Panel("[green]No relevant content found by AI search.[/green]"))
        return

    flagged = [r for r in results if r.get("needs_review")]
    clean = [r for r in results if not r.get("needs_review")]

    if flagged:
        table = Table(title="FLAGGED — Items That May Need Changes", show_lines=True,
                       border_style="red")
        table.add_column("File", style="cyan", max_width=20)
        table.add_column("Page", style="magenta", justify="center", width=6)
        table.add_column("Found Text", style="yellow", max_width=30)
        table.add_column("Reason", style="white", max_width=30)
        table.add_column("Suggestion", style="green", max_width=30)

        for r in flagged:
            table.add_row(
                r["file"], str(r["page"]),
                r["matched_text"][:80], r["reason"][:80],
                r.get("suggestion", "")[:80],
            )
        console.print(table)

    if clean:
        table = Table(title="Relevant Content (No Changes Needed)", show_lines=True,
                       border_style="green")
        table.add_column("File", style="cyan", max_width=20)
        table.add_column("Page", style="magenta", justify="center", width=6)
        table.add_column("Found Text", style="yellow", max_width=35)
        table.add_column("Reason", style="white", max_width=40)

        for r in clean:
            table.add_row(r["file"], str(r["page"]),
                          r["matched_text"][:100], r["reason"][:100])
        console.print(table)

    console.print(f"\n[bold]Total findings: {len(results)}  |  "
                  f"[red]Flagged for review: {len(flagged)}[/red]  |  "
                  f"[green]OK: {len(clean)}[/green][/bold]")


def export_results(keyword_results: list[dict], ai_results: list[dict],
                   output_path: str) -> None:
    """Export all results to a JSON file."""
    data = {
        "keyword_matches": keyword_results,
        "ai_findings": ai_results,
        "summary": {
            "total_keyword_matches": len(keyword_results),
            "total_ai_findings": len(ai_results),
            "flagged_for_review": len([r for r in ai_results if r.get("needs_review")]),
        },
    }
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    console.print(f"\n[green]Results exported to {output_path}[/green]")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pdf_helper",
        description="PDFHelper — AI-powered PDF search and flagging tool",
    )
    parser.add_argument(
        "paths", nargs="+",
        help="PDF files or directories containing PDFs to search",
    )
    parser.add_argument(
        "-s", "--search", nargs="+", default=[],
        help="Keywords or phrases to search for (exact match)",
    )
    parser.add_argument(
        "-q", "--query",
        help="AI-powered search query (searches for concepts, not just keywords)",
    )
    parser.add_argument(
        "--search-file",
        help="Path to a text file with one search term per line",
    )
    parser.add_argument(
        "--case-sensitive", action="store_true",
        help="Make keyword searches case-sensitive",
    )
    parser.add_argument(
        "--api-key",
        help="Anthropic API key (or set ANTHROPIC_API_KEY env var)",
    )
    parser.add_argument(
        "--export",
        help="Export results to a JSON file at this path",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not args.search and not args.query and not args.search_file:
        parser.error("Provide at least one of: --search, --query, or --search-file")

    # Gather search terms
    search_terms = list(args.search)
    if args.search_file:
        sf = Path(args.search_file)
        if sf.exists():
            search_terms.extend(
                line.strip() for line in sf.read_text().splitlines() if line.strip()
            )
        else:
            console.print(f"[red]Search file not found: {args.search_file}[/red]")
            sys.exit(1)

    # Load PDFs
    console.print(Panel("[bold]PDFHelper — PDF Search & Flagging Tool[/bold]"))
    console.print("\n[bold]Loading PDFs...[/bold]")
    docs = load_pdfs(args.paths)

    if not docs:
        console.print("[red]No PDFs were loaded. Check your file paths.[/red]")
        sys.exit(1)

    total_pages = sum(len(pages) for pages in docs.values())
    console.print(f"[green]Loaded {len(docs)} PDF(s) with {total_pages} total pages.[/green]\n")

    keyword_results = []
    ai_results = []

    # Keyword search
    if search_terms:
        console.print(f"[bold]Running keyword search for {len(search_terms)} term(s)...[/bold]")
        keyword_results = keyword_search(docs, search_terms, args.case_sensitive)
        display_keyword_results(keyword_results)
        console.print()

    # AI search
    if args.query:
        console.print(f"[bold]Running AI-powered search: \"{args.query}\"[/bold]")
        console.print("[dim]This uses Claude AI to find concepts, not just exact words...[/dim]\n")
        ai_results = ai_search(docs, args.query, args.api_key)
        display_ai_results(ai_results)

    # Export
    if args.export:
        export_results(keyword_results, ai_results, args.export)


if __name__ == "__main__":
    main()
