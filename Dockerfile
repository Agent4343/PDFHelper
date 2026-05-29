FROM python:3.12-slim

WORKDIR /app

# System deps for Tesseract OCR, PostgreSQL client, and WeasyPrint PDF rendering
# Note: PyMuPDF wheels bundle their own MuPDF — no libmupdf needed
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        tesseract-ocr \
        tesseract-ocr-eng \
        libpq5 \
        libpango-1.0-0 \
        libpangocairo-1.0-0 \
        libgdk-pixbuf2.0-0 \
        libffi-dev \
        libcairo2 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create upload directory
RUN mkdir -p /tmp/pdfhelper_uploads

# Flush Python output immediately so Railway deploy logs show errors
ENV PYTHONUNBUFFERED=1

# Railway sets PORT automatically
CMD ["sh", "-c", "exec python -m uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
