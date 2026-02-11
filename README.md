# PDFHelper

AI-powered PDF search and flagging tool. Upload PDFs, search them with keywords or AI, and get flagged when something may need changes. Deployable on Railway.

## Features

- **Batch PDF processing** — Upload 20+ PDFs at once
- **Keyword search** — Find exact words and phrases across all pages
- **AI-powered search** — Uses Claude AI to find related concepts, synonyms, and ideas
- **Flagging** — AI identifies items that may need review and provides suggestions
- **Search history** — Database stores all uploads and past search results
- **Secure** — API key auth, rate limiting, file validation, CORS controls

## Deploy to Railway

1. Push this repo to GitHub
2. Create a new project on [Railway](https://railway.app)
3. Connect your GitHub repo
4. Add environment variables in Railway dashboard:

| Variable | Required | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | Yes | Your Anthropic API key for AI search |
| `PDF_HELPER_API_KEY` | Yes | Secret key to protect your API |
| `DATABASE_URL` | No | Railway auto-sets this if you add a PostgreSQL plugin |
| `ENVIRONMENT` | No | Set to `production` to disable /docs |
| `ALLOWED_ORIGINS` | No | Comma-separated allowed CORS origins |

5. (Recommended) Add a **PostgreSQL** plugin from the Railway dashboard — the app detects it automatically via `DATABASE_URL`
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

### Search documents
```
POST /search?doc_ids=id1&doc_ids=id2
{
  "search_terms": ["safety", "compliance"],
  "ai_query": "outdated procedures referencing old regulations",
  "case_sensitive": false
}
```
Leave `doc_ids` empty to search all uploaded documents.

### List uploaded documents
```
GET /documents
```

### Delete a document
```
DELETE /documents/{doc_id}
```

### View search history
```
GET /history?limit=20
```

### View a specific search result
```
GET /history/{search_id}
```

## Local Development

```bash
pip install -r requirements.txt
cp .env.example .env   # edit with your keys
uvicorn app:app --reload
```

Visit `http://localhost:8000/docs` for interactive API docs (Swagger UI).

## CLI Usage (standalone)

The original CLI tool is still available:
```bash
python pdf_helper.py my_procedures/ -s "safety" "compliance" -q "outdated language"
```
