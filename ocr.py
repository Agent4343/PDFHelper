"""
OCR and structured text extraction for PDFs.

Uses PyMuPDF's built-in text extraction + Tesseract OCR to extract text
from PDFs, preserving document structure (headings, lists, tables).

Performance optimizations:
- Parallel OCR processing using ThreadPoolExecutor
- Per-page OCR decisions so mixed PDFs only OCR the pages that need it
- Configurable DPI (lower = faster, higher = more accurate)
"""

import io
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)

try:
    from PIL import Image
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

OCR_DPI = int(os.getenv("OCR_DPI", "200"))
OCR_WORKERS = int(os.getenv("OCR_WORKERS", "4"))
OCR_CHAR_THRESHOLD = 50


def _page_needs_ocr(text: str) -> bool:
    return len(text.strip()) < OCR_CHAR_THRESHOLD


def _ocr_single_page(page_png_bytes: bytes) -> str:
    try:
        image = Image.open(io.BytesIO(page_png_bytes))
        return pytesseract.image_to_string(image)
    except Exception:
        logger.warning("OCR failed for page image (%d bytes)", len(page_png_bytes), exc_info=True)
        return ""


def _classify_block(block_text: str, font_size: float, is_bold: bool, avg_font_size: float) -> str:
    """Classify a text block as heading, list_item, or paragraph."""
    stripped = block_text.strip()
    if not stripped:
        return "empty"

    # Numbered list items: "1.", "1)", "a.", "a)", "(a)", etc.
    if re.match(r'^(\d+[\.\)]\s|[a-zA-Z][\.\)]\s|\([a-zA-Z]\)\s|\(\d+\)\s)', stripped):
        return "list_item"

    # Bullet points
    if stripped[0] in ('-', '•', '‣', '◦', '⁃', '*') and len(stripped) > 2:
        return "list_item"

    # Headings: larger font size or all-caps short lines
    if font_size > avg_font_size * 1.2 and len(stripped) < 200:
        return "heading"
    if is_bold and len(stripped) < 200 and not stripped.endswith('.'):
        return "heading"
    if stripped.isupper() and len(stripped) > 3 and len(stripped) < 120:
        return "heading"

    # Section number patterns: "1.0", "2.3", "Section 4"
    if re.match(r'^(\d+\.(\d+\.?)*)\s+\S', stripped) and len(stripped) < 200:
        return "heading"

    return "paragraph"


def _extract_structured_page(page) -> dict:
    """Extract text from a page with structural annotations.

    Returns {"page": N, "text": str, "blocks": [{"type": str, "text": str}, ...]}
    where type is heading/paragraph/list_item/table.
    """
    blocks_out = []

    # Gather font size stats for relative heading detection
    text_dict = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
    all_font_sizes = []
    for block in text_dict.get("blocks", []):
        if block.get("type") != 0:  # text blocks only
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                if span.get("text", "").strip():
                    all_font_sizes.append(span["size"])

    avg_font_size = sum(all_font_sizes) / len(all_font_sizes) if all_font_sizes else 11.0

    # Process each text block
    for block in text_dict.get("blocks", []):
        if block.get("type") != 0:
            continue

        block_text_parts = []
        max_font_size = 0
        has_bold = False

        for line in block.get("lines", []):
            line_text = ""
            for span in line.get("spans", []):
                line_text += span.get("text", "")
                max_font_size = max(max_font_size, span.get("size", 0))
                flags = span.get("flags", 0)
                if flags & 2**4:  # bold flag
                    has_bold = True
            block_text_parts.append(line_text)

        block_text = "\n".join(block_text_parts).strip()
        if not block_text:
            continue

        block_type = _classify_block(block_text, max_font_size, has_bold, avg_font_size)
        if block_type != "empty":
            blocks_out.append({"type": block_type, "text": block_text})

    # Extract tables separately
    if hasattr(page, "find_tables"):
        try:
            tables = page.find_tables()
            for table in tables.tables:
                rows = table.extract()
                if rows:
                    table_lines = []
                    for row in rows:
                        cells = [str(c) if c is not None else "" for c in row]
                        table_lines.append(" | ".join(cells))
                    blocks_out.append({
                        "type": "table",
                        "text": "\n".join(table_lines),
                        "rows": [[str(c) if c is not None else "" for c in row] for row in rows],
                    })
        except Exception:
            logger.debug("Table extraction failed for page")

    # Build a plain-text representation with structure markers
    text_parts = []
    for b in blocks_out:
        if b["type"] == "heading":
            text_parts.append(f"\n[HEADING] {b['text']}")
        elif b["type"] == "list_item":
            text_parts.append(f"[LIST] {b['text']}")
        elif b["type"] == "table":
            text_parts.append(f"\n[TABLE]\n{b['text']}\n[/TABLE]")
        else:
            text_parts.append(b["text"])

    plain_text = "\n".join(text_parts)
    return plain_text, blocks_out


def _extract_page_text(page) -> str:
    """Extract text from a page, preserving tables."""
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
                            table_lines = [" | ".join(str(c) if c is not None else "" for c in row) for row in rows]
                            parts.append("\n[TABLE]\n" + "\n".join(table_lines) + "\n[/TABLE]\n")
                return "\n".join(parts)
        except Exception:
            logger.debug("Table extraction failed, falling back to plain text")
    return page.get_text()


def extract_text_with_ocr_fallback(pdf_bytes: bytes) -> list[dict]:
    """Extract text from a PDF, using OCR only on pages that need it.

    Returns list of {"page": int, "text": str}.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page_count = len(doc)

    pages = []
    ocr_needed_indices = []
    for page_num in range(page_count):
        page = doc[page_num]
        text = _extract_page_text(page)
        pages.append({"page": page_num + 1, "text": text})
        if _page_needs_ocr(text):
            ocr_needed_indices.append(page_num)

    if not ocr_needed_indices or not OCR_AVAILABLE:
        doc.close()
        return pages

    page_images = {}
    for page_num in ocr_needed_indices:
        try:
            pix = doc[page_num].get_pixmap(dpi=OCR_DPI)
            page_images[page_num] = pix.tobytes("png")
        except Exception:
            logger.warning("Failed to render page %d for OCR", page_num + 1, exc_info=True)

    doc.close()

    if page_images:
        with ThreadPoolExecutor(max_workers=min(OCR_WORKERS, len(page_images))) as pool:
            futures = {
                page_num: pool.submit(_ocr_single_page, png_bytes)
                for page_num, png_bytes in page_images.items()
            }
            for page_num, future in futures.items():
                ocr_text = future.result()
                if ocr_text.strip():
                    pages[page_num]["text"] = ocr_text

    return pages


def extract_structured_text(pdf_bytes: bytes) -> list[dict]:
    """Extract text from a PDF with structural annotations per block.

    Returns list of {"page": int, "text": str, "blocks": [{"type": str, "text": str}, ...]}.
    Blocks have type: heading, paragraph, list_item, or table.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        plain_text, blocks = _extract_structured_page(page)

        if _page_needs_ocr(plain_text) and OCR_AVAILABLE:
            try:
                pix = page.get_pixmap(dpi=OCR_DPI)
                ocr_text = _ocr_single_page(pix.tobytes("png"))
                if ocr_text.strip():
                    plain_text = ocr_text
                    blocks = [{"type": "paragraph", "text": ocr_text}]
            except Exception:
                pass

        pages.append({"page": page_num + 1, "text": plain_text, "blocks": blocks})

    doc.close()
    return pages
