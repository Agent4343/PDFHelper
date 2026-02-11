# PDFHelper

AI-powered PDF search, analysis, and flagging tool with multi-agent intelligence. Upload PDFs, run deep analysis with specialized AI agents, and get actionable reports on what needs to change. Deployable on Railway.

## Features

- **Batch PDF processing** — Upload 20+ PDFs at once
- **OCR support** — Automatically detects scanned PDFs and extracts text via Tesseract OCR
- **Keyword search** — Find exact words and phrases across all pages
- **AI-powered search** — Uses Claude AI to find related concepts, synonyms, and ideas
- **Multi-agent analysis pipeline** — 4 specialized AI agents working together:
  - **Document Analyzer** — Deep analysis of each document's structure, topics, and references
  - **Cross-Reference Checker** — Finds conflicts and inconsistencies *between* documents
  - **Compliance Checker** — Flags regulatory issues, outdated references, policy gaps
  - **Summary Report Generator** — Produces an executive report with prioritized action items
- **Flagging** — AI identifies items that may need review and provides suggestions
- **Search & report history** — Database stores all uploads, searches, and analysis reports
- **Fully encrypted** — Files on disk, text in database, filenames, search results — all encrypted at rest
- **Secure** — API key auth, rate limiting, file validation, CORS controls, audit logging

## Deploy to Railway

1. Push this repo to GitHub
2. Create a new project on [Railway](https://railway.app)
3. Connect your GitHub repo
4. Add environment variables in Railway dashboard:

| Variable | Required | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | Yes | Your Anthropic API key for AI search |
| `PDF_HELPER_API_KEY` | Yes | Secret key to protect your API |
| `ENCRYPTION_KEY` | Yes | Fernet key for encrypting data at rest |
| `ENVIRONMENT` | Yes | Set to `production` |
| `DATABASE_URL` | No | Railway auto-sets this if you add a PostgreSQL plugin |
| `ALLOWED_ORIGINS` | No | Comma-separated allowed CORS origins |

5. (Recommended) Add a **PostgreSQL** plugin from the Railway dashboard
6. Deploy — Railway builds using the Dockerfile automatically

## API Endpoints

All endpoints (except `/health`) require the header:
```
Authorization: Bearer YOUR_API_KEY
```

### Health check
```
GET /health
```

### Upload PDFs
```
POST /upload
Content-Type: multipart/form-data
files: (one or more PDF files)
```
Scanned PDFs are automatically OCR'd if no text is detected.

### Search documents
```
POST /search?doc_ids=id1&doc_ids=id2
{
  "search_terms": ["safety", "compliance"],
  "ai_query": "outdated procedures referencing old regulations",
  "case_sensitive": false
}
```

### Full multi-agent analysis (the powerful one)
```
POST /analyze?doc_ids=id1&doc_ids=id2
{
  "compliance_context": "OSHA 2024 standards",
  "search_terms": ["expiration", "review date"],
  "ai_query": "procedures that may be outdated"
}
```
This runs all 4 agents and returns a complete analysis with:
- Per-document deep analysis
- Cross-document conflict detection
- Compliance issue flagging
- Executive summary with prioritized action items

### List documents / reports / history
```
GET /documents
GET /reports
GET /reports/{report_id}
GET /history
GET /history/{search_id}
```

### Delete a document
```
DELETE /documents/{doc_id}
```

## Local Development

```bash
pip install -r requirements.txt
cp .env.example .env   # edit with your keys
uvicorn app:app --reload
```

For OCR support locally, install Tesseract:
```bash
# macOS
brew install tesseract

# Ubuntu/Debian
sudo apt-get install tesseract-ocr
```

Visit `http://localhost:8000/docs` for interactive API docs (Swagger UI).

## CLI Usage (standalone)

The original CLI tool is still available:
```bash
python pdf_helper.py my_procedures/ -s "safety" "compliance" -q "outdated language"
```
