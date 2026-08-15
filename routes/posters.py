import asyncio
import json
import os
import re
import uuid
from datetime import datetime, timezone

import anthropic
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from starlette.responses import StreamingResponse

from auth import verify_api_key, get_db
from database import DBPoster
from helpers import _encrypt_text, _safe_decrypt, _resolve_agent_model, _call_claude_bg
from models import PosterCreateRequest, PosterUpdateRequest, PosterSaveHTMLRequest

router = APIRouter()

# ---------------------------------------------------------------------------
# Poster Generator – constants
# ---------------------------------------------------------------------------

POSTER_SIZES = {
    "letter": {"width": "8.5in", "height": "11in", "px_w": 816, "px_h": 1056},
    "a4": {"width": "210mm", "height": "297mm", "px_w": 794, "px_h": 1123},
    "a3": {"width": "297mm", "height": "420mm", "px_w": 1123, "px_h": 1587},
    "wide": {"width": "11in", "height": "8.5in", "px_w": 1056, "px_h": 816},
    "square": {"width": "10in", "height": "10in", "px_w": 960, "px_h": 960},
    "banner": {"width": "24in", "height": "8in", "px_w": 2304, "px_h": 768},
}

POSTER_STYLES = {
    "bold": "Use strong contrasting colors (black/yellow, red/white), thick borders, large impactful typography, industrial feel.",
    "clean": "Use a clean modern aesthetic with plenty of white space, subtle shadows, thin lines, and a muted professional color palette.",
    "safety": "Use standard safety colors: red for danger/prohibition, yellow for caution, blue for mandatory, green for safe. Include ISO-style safety symbols using Unicode. Add hazard borders with diagonal stripes.",
    "corporate": "Use a polished corporate style with navy/gray tones, structured grid layout, subtle gradients, and professional serif fonts for headings.",
    "vibrant": "Use bright, eye-catching colors with bold gradients, rounded shapes, playful typography, and high visual energy.",
    "retro": "Use a vintage/retro aesthetic with muted earth tones, textured backgrounds via CSS patterns, bold serif fonts, and decorative borders.",
}

POSTER_SYSTEM = """You are a poster design engine. Your ONLY job is to output a complete HTML/CSS poster. You NEVER ask questions, request clarification, provide commentary, or give advice. You ALWAYS respond with raw HTML code and nothing else.

CRITICAL BEHAVIOR:
- NO MATTER WHAT the user writes — whether it's a simple description, a detailed specification, a review request, a consulting prompt, or anything else — you MUST create a poster from it.
- Extract the key content, messages, and themes from the user's prompt and design a poster around them.
- If the user describes an existing document or infographic, create an IMPROVED version as a poster.
- If the user asks for a "review" or "analysis", create a poster that presents those findings visually.
- NEVER respond with text, questions, bullet points, or explanations. ONLY HTML.

DESIGN PRINCIPLES:
- Visual hierarchy: title largest, key message prominent, details smaller
- High contrast for readability — never place light text on light backgrounds
- Balanced whitespace — don't cram everything together
- Consistent alignment and spacing throughout
- Use CSS shapes, gradients, borders, box-shadows for visual interest
- Use Unicode symbols and emoji for icons (e.g. ⚠️ 🔥 ✅ 🏗️ 📋 ☎️ 🚨 ⛑️ 👷 🔒)
- ALL content MUST fit within the poster dimensions — never overflow
- For dense content: use smaller fonts, tighter spacing, multi-column layouts

{style_instruction}

TECHNICAL RULES:
1. Output ONLY the raw HTML — no markdown fences, no explanation, no commentary. NOTHING except HTML.
2. Complete HTML document: <!DOCTYPE html>, <html>, <head> with <title>, <body>.
3. ALL styles in a <style> block. No external stylesheets, images, or JavaScript.
4. Body dimensions: exactly {width} x {height}, margin:0, overflow:hidden.
5. Use web-safe fonts: Arial, Georgia, Impact, Courier New, Trebuchet MS, Verdana.
6. For print quality: use pt/in/cm units for text sizing where appropriate.
7. Add a <title> tag that summarizes the poster content in 3-6 words.
8. CRITICAL — include these print rules in your <style> block so the poster prints cleanly with no browser headers/footers/URLs:
   @page {{ margin: 0; size: {width} {height}; }}
   @media print {{ html, body {{ margin: 0; padding: 0; -webkit-print-color-adjust: exact; print-color-adjust: exact; }} }}

SIZE: {width} x {height}"""

POSTER_UPDATE_SYSTEM = """You are a poster editing engine. Your ONLY job is to output updated HTML/CSS. You NEVER ask questions, provide commentary, or give advice. You ALWAYS respond with raw HTML code and nothing else.

The user will provide the current poster HTML and a description of changes.

RULES:
1. Output ONLY the updated HTML — no markdown fences, no explanation, no commentary. NOTHING except HTML.
2. Keep the same document structure and poster dimensions.
3. Preserve ALL elements the user did NOT ask to change.
4. Apply requested changes precisely — if they say "make title red", only change the title color.
5. Maintain or improve design quality — don't break alignment, spacing, or contrast.
6. All styles stay in <style> block. No external resources, no JavaScript.
7. If the user asks to add content, integrate it naturally into the existing layout.
8. Keep the <title> tag updated if the poster topic changes.
9. ALWAYS keep or add these print rules in the <style> block:
   @page { margin: 0; }
   @media print { html, body { margin: 0; padding: 0; -webkit-print-color-adjust: exact; print-color-adjust: exact; } }"""


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/posters", dependencies=[Depends(verify_api_key)])
async def create_poster(body: PosterCreateRequest, request: Request, db=Depends(get_db)):
    """Generate a new poster from a text prompt using AI."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not configured")

    client = anthropic.Anthropic(api_key=api_key)
    model = _resolve_agent_model(body.model)
    size = POSTER_SIZES.get(body.size, POSTER_SIZES["letter"])

    style_instruction = ""
    if body.style and body.style in POSTER_STYLES:
        style_instruction = f"STYLE: {body.style.upper()}\n{POSTER_STYLES[body.style]}"

    system = POSTER_SYSTEM.format(width=size["width"], height=size["height"], style_instruction=style_instruction)

    async def generate():
        import asyncio
        yield f"data: {json.dumps({'type': 'status', 'message': 'Designing your poster...'})}\n\n"

        task = asyncio.create_task(_call_claude_bg(
            client, system, body.prompt, max_tokens=8000, model=model
        ))
        while not task.done():
            yield ":\n\n"
            await asyncio.sleep(3)
        html_content = task.result()

        html_content = html_content.strip()
        if html_content.startswith("```"):
            html_content = re.sub(r"^```(?:html)?\s*\n?", "", html_content)
            html_content = re.sub(r"\n?```\s*$", "", html_content)

        current_user_id = getattr(request.state, "user_id", None)
        poster_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)

        title_match = re.search(r"<title>(.*?)</title>", html_content, re.IGNORECASE)
        title = title_match.group(1) if title_match else body.prompt[:80]

        poster = DBPoster(
            id=poster_id,
            user_id=current_user_id,
            title=_encrypt_text(title),
            prompt_history=_encrypt_text(json.dumps([body.prompt])),
            html_content=_encrypt_text(html_content),
            created_at=now,
            updated_at=now,
        )
        db.add(poster)
        db.commit()

        yield f"data: {json.dumps({'type': 'result', 'poster_id': poster_id, 'title': title, 'html': html_content, 'size': body.size})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@router.post("/posters/{poster_id}/update", dependencies=[Depends(verify_api_key)])
async def update_poster(poster_id: str, body: PosterUpdateRequest, request: Request, db=Depends(get_db)):
    """Update an existing poster with a new prompt."""
    current_user_id = getattr(request.state, "user_id", None)
    poster = db.query(DBPoster).filter(DBPoster.id == poster_id).first()
    if not poster:
        raise HTTPException(status_code=404, detail="Poster not found")
    if current_user_id and poster.user_id and poster.user_id != current_user_id:
        raise HTTPException(status_code=403, detail="Access denied")

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not configured")

    client = anthropic.Anthropic(api_key=api_key)
    model = _resolve_agent_model(body.model)

    current_html = _safe_decrypt(poster.html_content)
    prompt_history = json.loads(_safe_decrypt(poster.prompt_history, "[]"))

    user_msg = f"CURRENT POSTER HTML:\n{current_html}\n\nREQUESTED CHANGES:\n{body.prompt}"

    async def generate():
        import asyncio
        yield f"data: {json.dumps({'type': 'status', 'message': 'Updating your poster...'})}\n\n"

        task = asyncio.create_task(_call_claude_bg(
            client, POSTER_UPDATE_SYSTEM, user_msg, max_tokens=8000, model=model
        ))
        while not task.done():
            yield ":\n\n"
            await asyncio.sleep(3)
        new_html = task.result()

        new_html = new_html.strip()
        if new_html.startswith("```"):
            new_html = re.sub(r"^```(?:html)?\s*\n?", "", new_html)
            new_html = re.sub(r"\n?```\s*$", "", new_html)

        prompt_history.append(body.prompt)
        title_match = re.search(r"<title>(.*?)</title>", new_html, re.IGNORECASE)
        title = title_match.group(1) if title_match else _safe_decrypt(poster.title, "Untitled Poster")

        poster.title = _encrypt_text(title)
        poster.prompt_history = _encrypt_text(json.dumps(prompt_history))
        poster.html_content = _encrypt_text(new_html)
        poster.updated_at = datetime.now(timezone.utc)
        db.commit()

        yield f"data: {json.dumps({'type': 'result', 'poster_id': poster_id, 'title': title, 'html': new_html})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


def _list_posters_sync(posters):
    """Sync helper — decrypt poster metadata off the event loop."""
    return [
        {
            "id": p.id,
            "title": _safe_decrypt(p.title, "Untitled Poster"),
            "prompt_count": len(json.loads(_safe_decrypt(p.prompt_history, "[]"))),
            "created_at": p.created_at.isoformat() if p.created_at else None,
            "updated_at": p.updated_at.isoformat() if p.updated_at else None,
        }
        for p in posters
    ]


@router.get("/posters", dependencies=[Depends(verify_api_key)])
async def list_posters(
    request: Request,
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=200),
    db=Depends(get_db),
):
    """List all posters for the current user."""
    current_user_id = getattr(request.state, "user_id", None)
    q = db.query(DBPoster)
    if current_user_id:
        q = q.filter(DBPoster.user_id == current_user_id)
    else:
        q = q.filter(DBPoster.user_id.is_(None))
    total = q.count()
    posters = q.order_by(DBPoster.updated_at.desc()).offset(skip).limit(limit).all()
    poster_list = await asyncio.to_thread(_list_posters_sync, posters)
    return {"posters": poster_list, "total": total}


@router.get("/posters/{poster_id}", dependencies=[Depends(verify_api_key)])
async def get_poster(poster_id: str, request: Request, db=Depends(get_db)):
    """Get a specific poster with its full HTML content."""
    current_user_id = getattr(request.state, "user_id", None)
    poster = db.query(DBPoster).filter(DBPoster.id == poster_id).first()
    if not poster:
        raise HTTPException(status_code=404, detail="Poster not found")
    if current_user_id and poster.user_id and poster.user_id != current_user_id:
        raise HTTPException(status_code=403, detail="Access denied")
    return {
        "id": poster.id,
        "title": _safe_decrypt(poster.title, "Untitled Poster"),
        "html": _safe_decrypt(poster.html_content),
        "prompts": json.loads(_safe_decrypt(poster.prompt_history, "[]")),
        "created_at": poster.created_at.isoformat() if poster.created_at else None,
        "updated_at": poster.updated_at.isoformat() if poster.updated_at else None,
    }


@router.delete("/posters/{poster_id}", dependencies=[Depends(verify_api_key)])
async def delete_poster(poster_id: str, request: Request, db=Depends(get_db)):
    """Delete a poster."""
    current_user_id = getattr(request.state, "user_id", None)
    poster = db.query(DBPoster).filter(DBPoster.id == poster_id).first()
    if not poster:
        raise HTTPException(status_code=404, detail="Poster not found")
    if current_user_id and poster.user_id and poster.user_id != current_user_id:
        raise HTTPException(status_code=403, detail="Access denied")
    db.delete(poster)
    db.commit()
    return {"deleted": True}


@router.patch("/posters/{poster_id}", dependencies=[Depends(verify_api_key)])
async def save_poster_html(poster_id: str, body: PosterSaveHTMLRequest, request: Request, db=Depends(get_db)):
    """Save manually edited HTML content for a poster."""
    current_user_id = getattr(request.state, "user_id", None)
    poster = db.query(DBPoster).filter(DBPoster.id == poster_id).first()
    if not poster:
        raise HTTPException(status_code=404, detail="Poster not found")
    if current_user_id and poster.user_id and poster.user_id != current_user_id:
        raise HTTPException(status_code=403, detail="Access denied")

    title_match = re.search(r"<title>(.*?)</title>", body.html, re.IGNORECASE)
    if title_match:
        poster.title = _encrypt_text(title_match.group(1))

    prompt_history = json.loads(_safe_decrypt(poster.prompt_history, "[]"))
    prompt_history.append("[Manual HTML edit]")
    poster.prompt_history = _encrypt_text(json.dumps(prompt_history))
    poster.html_content = _encrypt_text(body.html)
    poster.updated_at = datetime.now(timezone.utc)
    db.commit()
    return {"saved": True, "title": _safe_decrypt(poster.title, "Untitled Poster")}


@router.post("/posters/{poster_id}/duplicate", dependencies=[Depends(verify_api_key)])
async def duplicate_poster(poster_id: str, request: Request, db=Depends(get_db)):
    """Duplicate an existing poster."""
    current_user_id = getattr(request.state, "user_id", None)
    poster = db.query(DBPoster).filter(DBPoster.id == poster_id).first()
    if not poster:
        raise HTTPException(status_code=404, detail="Poster not found")
    if current_user_id and poster.user_id and poster.user_id != current_user_id:
        raise HTTPException(status_code=403, detail="Access denied")

    now = datetime.now(timezone.utc)
    orig_title = _safe_decrypt(poster.title, "Untitled Poster")
    new_poster = DBPoster(
        id=str(uuid.uuid4()),
        user_id=current_user_id,
        title=_encrypt_text(f"{orig_title} (Copy)"),
        prompt_history=poster.prompt_history,
        html_content=poster.html_content,
        created_at=now,
        updated_at=now,
    )
    db.add(new_poster)
    db.commit()
    return {
        "id": new_poster.id,
        "title": f"{orig_title} (Copy)",
    }
