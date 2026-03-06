"""
OCR support for scanned PDFs.

Uses PyMuPDF's built-in image extraction + Tesseract OCR to extract text
from PDFs that contain scanned images instead of selectable text.

Performance optimizations:
- Parallel OCR processing using ThreadPoolExecutor
- Per-page OCR decisions so mixed PDFs only OCR the pages that need it
- Configurable DPI (lower = faster, higher = more accurate)
"""

import io
import logging
import os
from concurrent.futures import ThreadPoolExecutor

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)

try:
    from PIL import Image
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# Configurable via environment variable. 200 is a good balance of speed vs accuracy.
# 300 is higher quality but ~2x slower. 150 is fast but may miss small text.
OCR_DPI = int(os.getenv("OCR_DPI", "200"))

# Max threads for parallel OCR. Each page is OCR'd independently.
OCR_WORKERS = int(os.getenv("OCR_WORKERS", "4"))

# Pages with fewer than this many characters are considered "needs OCR"
OCR_CHAR_THRESHOLD = 50


def _page_needs_ocr(text: str) -> bool:
    """Check if a single page has too little text and likely needs OCR."""
    return len(text.strip()) < OCR_CHAR_THRESHOLD


def _ocr_single_page(page_png_bytes: bytes) -> str:
    """Run Tesseract OCR on a single page image. Thread-safe."""
    try:
        image = Image.open(io.BytesIO(page_png_bytes))
        return pytesseract.image_to_string(image)
    except Exception:
        logger.warning("OCR failed for page image (%d bytes)", len(page_png_bytes), exc_info=True)
        return ""


def _extract_page_text(page) -> str:
    """Extract text from a single page, using table-aware extraction when available.

    PyMuPDF 1.23+ supports find_tables() which preserves tabular structure
    that get_text() garbles (e.g. valve isolation tables, specification charts).
    Falls back to plain get_text() on older versions.
    """
    # Try table-aware extraction first (PyMuPDF 1.23+)
    if hasattr(page, "find_tables"):
        try:
            tables = page.find_tables()
            if tables.tables:
                # Extract non-table text and tables separately, then combine
                # This preserves both flowing text and table structure
                parts = []
                plain_text = page.get_text()

                # Add the plain text (which includes everything)
                parts.append(plain_text)

                # Append structured table representations
                for table in tables:
                    try:
                        df = table.to_pandas()
                        # Convert table to a readable format with column alignment
                        table_str = "\n[TABLE]\n" + df.to_string(index=False) + "\n[/TABLE]\n"
                        parts.append(table_str)
                    except Exception:
                        # Fallback: extract table as list of rows
                        rows = table.extract()
                        if rows:
                            table_lines = []
                            for row in rows:
                                cells = [str(c) if c is not None else "" for c in row]
                                table_lines.append(" | ".join(cells))
                            parts.append("\n[TABLE]\n" + "\n".join(table_lines) + "\n[/TABLE]\n")

                return "\n".join(parts)
        except Exception:
            logger.debug("Table extraction failed for page, falling back to plain text")

    return page.get_text()


def extract_text_with_ocr_fallback(pdf_bytes: bytes) -> list[dict]:
    """Extract text from a PDF, using OCR only on pages that need it.

    For mixed PDFs (some pages scanned, some with text), this only OCRs the
    pages that have little or no extractable text — much faster than OCR'ing
    the entire document.

    Uses table-aware extraction when PyMuPDF supports it, preserving the
    structure of specification tables, valve lists, and similar tabular data.

    Returns list of {"page": int, "text": str}.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page_count = len(doc)

    # Step 1: Extract text from all pages normally (fast)
    pages = []
    ocr_needed_indices = []
    for page_num in range(page_count):
        page = doc[page_num]
        text = _extract_page_text(page)
        pages.append({"page": page_num + 1, "text": text})
        if _page_needs_ocr(text):
            ocr_needed_indices.append(page_num)

    # Step 2: If no pages need OCR, or OCR isn't available, return early
    if not ocr_needed_indices or not OCR_AVAILABLE:
        doc.close()
        return pages

    # Step 3: Render only the pages that need OCR to PNG images
    # (done in main thread because PyMuPDF isn't thread-safe)
    page_images = {}
    for page_num in ocr_needed_indices:
        try:
            pix = doc[page_num].get_pixmap(dpi=OCR_DPI)
            page_images[page_num] = pix.tobytes("png")
        except Exception:
            logger.warning("Failed to render page %d for OCR", page_num + 1, exc_info=True)

    doc.close()

    # Step 4: Run Tesseract in parallel across the pages that need OCR
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
