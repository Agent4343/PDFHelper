# PDFHelper

AI-powered PDF search and flagging tool. Search through multiple PDFs for specific words, phrases, or concepts — and get flagged when something may need changes.

## Features

- **Batch PDF processing** — Load 20+ PDFs at once (pass individual files or an entire folder)
- **Keyword search** — Find exact words and phrases across all pages
- **AI-powered search** — Uses Claude AI to find related concepts, synonyms, and ideas — not just exact matches
- **Flagging** — AI identifies items that may need review or changes and provides suggestions
- **Export** — Save results to a JSON report

## Setup

```bash
pip install -r requirements.txt
```

For AI-powered search, set your Anthropic API key:
```bash
export ANTHROPIC_API_KEY="your-key-here"
```

## Usage

### Search for exact keywords
```bash
python pdf_helper.py my_procedures/ -s "safety" "compliance" "deadline"
```

### Search using a file of terms (one per line)
```bash
python pdf_helper.py my_procedures/ --search-file terms.txt
```

### AI-powered concept search
```bash
python pdf_helper.py my_procedures/ -q "outdated safety procedures that reference old regulations"
```

### Combine keyword + AI search and export results
```bash
python pdf_helper.py doc1.pdf doc2.pdf -s "OSHA" "EPA" -q "any compliance language that may be outdated" --export report.json
```

### Point at a folder of PDFs
```bash
python pdf_helper.py /path/to/pdf/folder/ -s "review" -q "sections that need updating"
```
