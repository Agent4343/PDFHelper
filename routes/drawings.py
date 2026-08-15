import asyncio
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile
from starlette.responses import StreamingResponse

from auth import verify_api_key, get_db, _get_client_ip
from config import UPLOAD_DIR, MAX_FILE_SIZE, MAX_FILES_PER_REQUEST, ENCRYPTION_KEY, IS_PRODUCTION
from database import DBDrawing, DBIsolationPackage, SessionLocal
from models import IsolationRequest, _VALID_WORK_TYPES
from helpers import (
    _sanitize_filename,
    _encrypt_text,
    _decrypt_text,
    _safe_decrypt,
    _image_to_pdf,
    _validate_filepath,
    _safe_unlink,
    extract_text_from_bytes,
)
from audit import log_upload

router = APIRouter()


# ---------------------------------------------------------------------------
# IsoIntel — P&ID Drawing Management
# ---------------------------------------------------------------------------

@router.post("/drawings/upload", dependencies=[Depends(verify_api_key)])
async def upload_drawing(
    request: Request,
    file: UploadFile = File(...),
    title: str = Query(default="", description="Drawing title"),
    drawing_number: str = Query(default="", description="Drawing number, e.g. HEB-PID-1234"),
    equipment_tags: str = Query(default="", description="Comma-separated equipment tags on this drawing"),
    description: str = Query(default="", description="What system this drawing covers"),
    db=Depends(get_db),
):
    """Upload a P&ID drawing (PDF or image) for use in isolation packages."""
    content = await file.read()

    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail=f"File too large. Max {MAX_FILE_SIZE // (1024*1024)}MB.")

    drawing_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)

    # Determine file type and process
    fname = (file.filename or "drawing.pdf").strip()
    ext = fname.rsplit(".", 1)[-1].lower() if "." in fname else ""

    # Save encrypted file
    safe_name = _sanitize_filename(fname) if fname else "drawing"
    dest = UPLOAD_DIR / f"{drawing_id}_{safe_name}"
    if ENCRYPTION_KEY:
        from encryption import encrypt_bytes
        dest.write_bytes(encrypt_bytes(content))
    else:
        dest.write_bytes(content)

    # Extract text — convert images to PDF first for OCR
    text_content = "[]"
    page_count = 1
    try:
        if ext == "pdf":
            pages = extract_text_from_bytes(content)
        elif ext in ("jpg", "jpeg", "png", "tiff", "tif", "bmp", "webp"):
            pdf_bytes = _image_to_pdf(content)
            pages = extract_text_from_bytes(pdf_bytes)
        else:
            pages = []
        if pages:
            text_content = json.dumps(pages)
            page_count = len(pages)
    except Exception:
        pass

    drawing = DBDrawing(
        id=drawing_id,
        filename=_encrypt_text(fname),
        filepath=_encrypt_text(str(dest)),
        title=_encrypt_text(title) if title else _encrypt_text(fname),
        drawing_number=_encrypt_text(drawing_number) if drawing_number else None,
        equipment_tags=_encrypt_text(equipment_tags) if equipment_tags else None,
        description=_encrypt_text(description) if description else None,
        page_count=page_count,
        text_content=_encrypt_text(text_content),
        uploaded_at=now,
    )
    db.add(drawing)
    db.commit()

    log_upload(_get_client_ip(request), fname, drawing_id, 1)

    return {
        "id": drawing_id,
        "filename": fname,
        "title": title or fname,
        "drawing_number": drawing_number,
        "equipment_tags": equipment_tags,
        "page_count": page_count,
        "uploaded_at": now.isoformat(),
    }


@router.post("/drawings/upload-batch", dependencies=[Depends(verify_api_key)])
async def upload_drawings_batch(
    request: Request,
    files: list[UploadFile] = File(...),
    equipment_tags: str = Query(default="", description="Comma-separated equipment tags shared across all drawings"),
    description: str = Query(default="", description="Shared description for these drawings"),
    db=Depends(get_db),
):
    """Upload multiple P&ID drawings at once. Each file gets its filename as the title and drawing number."""
    if len(files) > MAX_FILES_PER_REQUEST:
        raise HTTPException(status_code=400, detail=f"Max {MAX_FILES_PER_REQUEST} files per request.")

    uploaded = []
    errors = []

    for file in files:
        try:
            content = await file.read()
            if len(content) > MAX_FILE_SIZE:
                errors.append({"filename": file.filename, "error": f"File too large. Max {MAX_FILE_SIZE // (1024*1024)}MB."})
                continue

            drawing_id = str(uuid.uuid4())
            now = datetime.now(timezone.utc)

            fname = (file.filename or "drawing.pdf").strip()
            ext = fname.rsplit(".", 1)[-1].lower() if "." in fname else ""

            # Save encrypted file
            safe_name = _sanitize_filename(fname) if fname else "drawing"
            dest = UPLOAD_DIR / f"{drawing_id}_{safe_name}"
            if ENCRYPTION_KEY:
                from encryption import encrypt_bytes
                dest.write_bytes(encrypt_bytes(content))
            else:
                dest.write_bytes(content)

            # Extract text if PDF
            text_content = "[]"
            page_count = 1
            if ext == "pdf":
                try:
                    pages = extract_text_from_bytes(content)
                    text_content = json.dumps(pages)
                    page_count = len(pages)
                except Exception:
                    pass

            # Use filename (without extension) as default title
            default_title = fname.rsplit(".", 1)[0] if "." in fname else fname

            drawing = DBDrawing(
                id=drawing_id,
                filename=_encrypt_text(fname),
                filepath=_encrypt_text(str(dest)),
                title=_encrypt_text(default_title),
                drawing_number=_encrypt_text(default_title),
                equipment_tags=_encrypt_text(equipment_tags) if equipment_tags else None,
                description=_encrypt_text(description) if description else None,
                page_count=page_count,
                text_content=_encrypt_text(text_content),
                uploaded_at=now,
            )
            db.add(drawing)
            db.commit()

            log_upload(_get_client_ip(request), fname, drawing_id, 1)

            uploaded.append({
                "id": drawing_id,
                "filename": fname,
                "title": default_title,
                "page_count": page_count,
                "uploaded_at": now.isoformat(),
            })
        except Exception as e:
            errors.append({"filename": file.filename or "unknown", "error": str(e)})

    return {
        "uploaded": uploaded,
        "errors": errors,
        "total_uploaded": len(uploaded),
        "total_errors": len(errors),
    }


def _list_drawings_sync(drawings):
    """Sync helper — decrypt drawing metadata off the event loop."""
    return [
        {
            "id": d.id,
            "filename": _safe_decrypt(d.filename, f"Drawing {d.id[:8]}"),
            "title": _safe_decrypt(d.title) if d.title else None,
            "drawing_number": _safe_decrypt(d.drawing_number) if d.drawing_number else None,
            "equipment_tags": _safe_decrypt(d.equipment_tags) if d.equipment_tags else None,
            "description": _safe_decrypt(d.description) if d.description else None,
            "page_count": d.page_count,
            "uploaded_at": d.uploaded_at.isoformat(),
        }
        for d in drawings
    ]


@router.get("/drawings", dependencies=[Depends(verify_api_key)])
async def list_drawings(
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=200),
    db=Depends(get_db),
):
    """List all uploaded P&ID drawings."""
    total = db.query(DBDrawing).count()
    drawings = (
        db.query(DBDrawing)
        .order_by(DBDrawing.uploaded_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )
    drawing_list = await asyncio.to_thread(_list_drawings_sync, drawings)
    return {"drawings": drawing_list, "total": total}


@router.delete("/drawings/{drawing_id}", dependencies=[Depends(verify_api_key)])
async def delete_drawing(drawing_id: str, db=Depends(get_db)):
    """Delete a P&ID drawing."""
    drawing = db.query(DBDrawing).filter(DBDrawing.id == drawing_id).first()
    if not drawing:
        raise HTTPException(status_code=404, detail="Drawing not found")
    # Delete file
    try:
        _safe_unlink(Path(_decrypt_text(drawing.filepath)))
    except Exception:
        pass
    db.delete(drawing)
    db.commit()
    return {"deleted": drawing_id}


# ---------------------------------------------------------------------------
# IsoIntel — Isolation Package Generation
# ---------------------------------------------------------------------------

@router.post("/isolations/generate", dependencies=[Depends(verify_api_key)])
async def generate_isolation(
    request: Request,
    body: IsolationRequest,
    db=Depends(get_db),
):
    """Generate a full isolation package using the two-pass AI pipeline.

    Pass 1 (if no drawing_ids provided): AI selects relevant drawings from library.
    Pass 2: AI reads P&ID images and generates the complete isolation package.
    Response is streamed as Server-Sent Events.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not configured on server")

    from anthropic import Anthropic
    client = Anthropic(api_key=api_key)

    # Load all drawings for metadata
    all_drawings = db.query(DBDrawing).all()
    if not all_drawings:
        raise HTTPException(status_code=404, detail="No P&ID drawings uploaded. Upload drawings first.")

    now = datetime.now(timezone.utc)
    iso_id = str(uuid.uuid4())
    cert_number = f"ISO-{now.strftime('%Y%m%d')}-{iso_id[:8].upper()}"

    job = {
        "equipment_tag": body.equipment_tag,
        "work_description": body.work_description,
        "work_type": body.work_type,
        "fluid_service": body.fluid_service,
        "special_requirements": body.special_requirements,
    }

    # Determine which drawings to use
    selected_ids = list(body.drawing_ids)

    async def stream_isolation():
        """Stream the isolation generation as SSE."""
        import base64
        import logging
        logger = logging.getLogger("pdfhelper")
        from isointel import run_pass1, run_pass2_stream

        nonlocal selected_ids
        full_output = ""

        try:
            # Pass 1: drawing discovery (if not manually specified)
            if not selected_ids:
                drawings_meta = []
                for d in all_drawings:
                    drawings_meta.append({
                        "id": d.id,
                        "title": _decrypt_text(d.title) if d.title else None,
                        "drawingNumber": _decrypt_text(d.drawing_number) if d.drawing_number else None,
                        "equipmentTags": _decrypt_text(d.equipment_tags) if d.equipment_tags else None,
                        "description": _decrypt_text(d.description) if d.description else None,
                    })

                yield f"data: {json.dumps({'type': 'status', 'message': 'Pass 1: Searching drawing library...'})}\n\n"
                selected_ids = run_pass1(client, drawings_meta, job)

                if not selected_ids:
                    yield f"data: {json.dumps({'type': 'error', 'detail': 'Pass 1 could not identify relevant drawings. Try specifying drawing IDs manually.'})}\n\n"
                    return

                selected_titles = []
                for d in all_drawings:
                    if d.id in selected_ids:
                        selected_titles.append(_decrypt_text(d.title) if d.title else d.id)
                yield f"data: {json.dumps({'type': 'pass1_result', 'drawing_ids': selected_ids, 'drawing_titles': selected_titles})}\n\n"

            # Load drawing images for Pass 2
            yield f"data: {json.dumps({'type': 'status', 'message': 'Pass 2: Reading P&ID images and generating isolation package...'})}\n\n"

            drawing_images = []
            for d in all_drawings:
                if d.id not in selected_ids:
                    continue
                try:
                    fpath = _validate_filepath(Path(_decrypt_text(d.filepath)))
                    if ENCRYPTION_KEY:
                        from encryption import decrypt_bytes as dec_bytes
                        raw = dec_bytes(fpath.read_bytes())
                    else:
                        raw = fpath.read_bytes()

                    fname = _decrypt_text(d.filename).lower()
                    if fname.endswith(".pdf"):
                        # Convert first page of PDF to image
                        try:
                            import fitz  # PyMuPDF
                            pdf_doc = fitz.open(stream=raw, filetype="pdf")
                            for page_num in range(min(pdf_doc.page_count, 2)):
                                page = pdf_doc[page_num]
                                pix = page.get_pixmap(dpi=200)
                                img_bytes = pix.tobytes("png")
                                drawing_images.append({
                                    "id": d.id,
                                    "title": _decrypt_text(d.title) if d.title else d.id,
                                    "image_b64": base64.b64encode(img_bytes).decode("ascii"),
                                    "media_type": "image/png",
                                })
                            pdf_doc.close()
                        except ImportError:
                            logger.warning("PyMuPDF not available for PDF-to-image conversion")
                    else:
                        # Image file — use directly
                        media = "image/png"
                        if fname.endswith(".jpg") or fname.endswith(".jpeg"):
                            media = "image/jpeg"
                        elif fname.endswith(".webp"):
                            media = "image/webp"
                        drawing_images.append({
                            "id": d.id,
                            "title": _decrypt_text(d.title) if d.title else d.id,
                            "image_b64": base64.b64encode(raw).decode("ascii"),
                            "media_type": media,
                        })
                except Exception as e:
                    logger.warning("Failed to load drawing %s: %s", d.id, e)

            if not drawing_images:
                yield f"data: {json.dumps({'type': 'error', 'detail': 'Could not load any drawing images. Ensure drawings are uploaded as PDF or image files.'})}\n\n"
                return

            # Send meta
            yield f"data: {json.dumps({'type': 'meta', 'isolation_id': iso_id, 'cert_number': cert_number, 'drawings_used': len(drawing_images)})}\n\n"

            # Pass 2: stream the isolation generation
            for chunk in run_pass2_stream(
                client, job, drawing_images,
                facility=body.facility,
                regime=body.regime,
                cert_number=cert_number,
            ):
                full_output += chunk
                yield f"data: {json.dumps({'type': 'chunk', 'text': chunk})}\n\n"

            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        except Exception as e:
            import logging
            logging.getLogger("pdfhelper").error("Isolation generation failed: %s", e)
            err_msg = "Isolation generation failed" if IS_PRODUCTION else f"Isolation generation failed: {str(e)}"
            yield f"data: {json.dumps({'type': 'error', 'detail': err_msg})}\n\n"
        finally:
            # Save the isolation package to DB
            if full_output:
                save_db = SessionLocal()
                try:
                    # Try to parse stats from the JSON output
                    hazard = "HIGH"
                    v_count = b_count = s_count = e_count = 0
                    try:
                        parsed = json.loads(full_output)
                        stats = parsed.get("stats", {})
                        v_count = stats.get("valveCount", 0)
                        b_count = stats.get("blindCount", 0)
                        s_count = stats.get("stepCount", 0)
                        e_count = stats.get("energySourceCount", 0)
                        hazard = parsed.get("hazardClassification", "HIGH")
                    except (json.JSONDecodeError, AttributeError):
                        pass

                    pkg = DBIsolationPackage(
                        id=iso_id,
                        cert_number=_encrypt_text(cert_number),
                        equipment_tag=_encrypt_text(body.equipment_tag),
                        work_description=_encrypt_text(body.work_description),
                        work_type=_encrypt_text(body.work_type),
                        fluid_service=_encrypt_text(body.fluid_service) if body.fluid_service else None,
                        facility=_encrypt_text(body.facility) if body.facility else None,
                        regime=_encrypt_text(body.regime) if body.regime else None,
                        special_requirements=_encrypt_text(body.special_requirements) if body.special_requirements else None,
                        drawing_ids=json.dumps(selected_ids),
                        package_data=_encrypt_text(full_output),
                        hazard_classification=hazard,
                        valve_count=v_count,
                        blind_count=b_count,
                        step_count=s_count,
                        energy_source_count=e_count,
                        status="draft",
                        created_at=now,
                        updated_at=now,
                    )
                    save_db.add(pkg)
                    save_db.commit()
                except Exception:
                    save_db.rollback()
                finally:
                    save_db.close()

    return StreamingResponse(stream_isolation(), media_type="text/event-stream")


@router.get("/isolations", dependencies=[Depends(verify_api_key)])
async def list_isolations(
    limit: int = Query(default=30, le=100),
    db=Depends(get_db),
):
    """List past isolation packages, most recent first."""
    packages = (
        db.query(DBIsolationPackage)
        .order_by(DBIsolationPackage.created_at.desc())
        .limit(limit)
        .all()
    )
    return {
        "isolations": [
            {
                "id": p.id,
                "cert_number": _safe_decrypt(p.cert_number),
                "equipment_tag": _safe_decrypt(p.equipment_tag),
                "work_description": _safe_decrypt(p.work_description),
                "work_type": _safe_decrypt(p.work_type),
                "hazard_classification": p.hazard_classification,
                "valve_count": p.valve_count,
                "blind_count": p.blind_count,
                "step_count": p.step_count,
                "status": p.status,
                "created_at": p.created_at.isoformat(),
            }
            for p in packages
        ]
    }


@router.get("/isolations/{isolation_id}", dependencies=[Depends(verify_api_key)])
async def get_isolation(isolation_id: str, db=Depends(get_db)):
    """Get the full isolation package by ID."""
    pkg = db.query(DBIsolationPackage).filter(DBIsolationPackage.id == isolation_id).first()
    if not pkg:
        raise HTTPException(status_code=404, detail="Isolation package not found")

    # Try to parse the package data as JSON
    raw_data = _safe_decrypt(pkg.package_data, "{}")
    try:
        package_json = json.loads(raw_data)
    except (json.JSONDecodeError, ValueError):
        package_json = {"raw": raw_data}

    return {
        "id": pkg.id,
        "cert_number": _safe_decrypt(pkg.cert_number),
        "equipment_tag": _safe_decrypt(pkg.equipment_tag),
        "work_description": _safe_decrypt(pkg.work_description),
        "work_type": _safe_decrypt(pkg.work_type),
        "fluid_service": _safe_decrypt(pkg.fluid_service) if pkg.fluid_service else None,
        "facility": _safe_decrypt(pkg.facility) if pkg.facility else None,
        "regime": _safe_decrypt(pkg.regime) if pkg.regime else None,
        "special_requirements": _safe_decrypt(pkg.special_requirements) if pkg.special_requirements else None,
        "drawing_ids": json.loads(pkg.drawing_ids),
        "hazard_classification": pkg.hazard_classification,
        "valve_count": pkg.valve_count,
        "blind_count": pkg.blind_count,
        "step_count": pkg.step_count,
        "energy_source_count": pkg.energy_source_count,
        "status": pkg.status,
        "created_at": pkg.created_at.isoformat(),
        "updated_at": pkg.updated_at.isoformat(),
        "package": package_json,
    }


@router.delete("/isolations/{isolation_id}", dependencies=[Depends(verify_api_key)])
async def delete_isolation(isolation_id: str, db=Depends(get_db)):
    """Delete an isolation package."""
    pkg = db.query(DBIsolationPackage).filter(DBIsolationPackage.id == isolation_id).first()
    if not pkg:
        raise HTTPException(status_code=404, detail="Isolation package not found")
    db.delete(pkg)
    db.commit()
    return {"deleted": isolation_id}
