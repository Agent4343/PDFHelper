FROM python:3.12-slim

WORKDIR /app

# System deps for PyMuPDF, Tesseract OCR, and PostgreSQL client
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libmupdf-dev \
        tesseract-ocr \
        tesseract-ocr-eng \
        libpq5 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create upload directory
RUN mkdir -p /tmp/pdfhelper_uploads

# Flush Python output immediately so Railway deploy logs show errors
ENV PYTHONUNBUFFERED=1

# Railway sets PORT automatically; exec replaces shell for proper signal handling
CMD exec uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}
