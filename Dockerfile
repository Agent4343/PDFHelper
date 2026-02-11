FROM python:3.12-slim

WORKDIR /app

# System deps for PyMuPDF
RUN apt-get update && \
    apt-get install -y --no-install-recommends libmupdf-dev && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create upload directory
RUN mkdir -p /tmp/pdfhelper_uploads

# Railway sets PORT automatically
CMD uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}
