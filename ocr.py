"""
OCR support for scanned PDFs.

Uses PyMuPDF's built-in image extraction + Tesseract OCR to extract text
from PDFs that contain scanned images instead of selectable text.
"""

import io
import fitz  # PyMuPDF

try:
    from PIL import Image
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False


def needs_ocr(pages: list[dict]) -> bool:
    """Check if extracted text is mostly empty — indicating a scanned PDF."""
    if not pages:
        return False
    total_chars = sum(len(p["text"].strip()) for p in pages)
    avg_chars_per_page = total_chars / len(pages)
    # If average text per page is very low, it's likely scanned
    return avg_chars_per_page < 50


def ocr_pdf_bytes(pdf_bytes: bytes) -> list[dict]:
    """Run OCR on a PDF's pages by extracting images and running Tesseract.

    Returns list of {"page": int, "text": str} same format as regular extraction.
    """
    if not OCR_AVAILABLE:
        return []

    pages = []
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    for page_num in range(len(doc)):
        page = doc[page_num]
        page_text = ""

        # Render the page as an image and OCR it
        # This handles both image-based and mixed PDFs
        pix = page.get_pixmap(dpi=300)
        img_bytes = pix.tobytes("png")
        image = Image.open(io.BytesIO(img_bytes))
        page_text = pytesseract.image_to_string(image)

        pages.append({"page": page_num + 1, "text": page_text})

    doc.close()
    return pages


def extract_text_with_ocr_fallback(pdf_bytes: bytes) -> list[dict]:
    """Extract text normally, falling back to OCR if the PDF is scanned.

    Returns list of {"page": int, "text": str}.
    """
    # Try normal text extraction first
    pages = []
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        pages.append({"page": page_num + 1, "text": text})
    doc.close()

    # If we got very little text, try OCR
    if needs_ocr(pages) and OCR_AVAILABLE:
        ocr_pages = ocr_pdf_bytes(pdf_bytes)
        if ocr_pages:
            return ocr_pages

    return pages
