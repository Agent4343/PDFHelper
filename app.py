#!/usr/bin/env python3
"""
PDFHelper — Secure web API for AI-powered PDF search and flagging.

Deployed on Railway. Provides endpoints to upload PDFs, search them
with keywords or AI, and retrieve flagged results.

Security features:
- API key authentication (required in production)
- File encryption at rest (AES via Fernet)
- PDF magic-byte verification
- Rate limiting, CORS, security headers
- Audit logging of all sensitive operations
- Auto-cleanup of old files
"""

import hashlib
import json
import os
import re
import secrets
import time
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Literal

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed; env vars must be set directly

from fastapi import (
    BackgroundTasks, FastAPI, File, Form, UploadFile, Depends, HTTPException, Query, Request,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, StreamingResponse, Response
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware

from database import (
    SessionLocal, engine, Base, DBUser, DBDocument, DBSearchResult, DBAnalysisReport,
    DBChatSession, DBChatMessage, DBDrawing, DBIsolationPackage,
)
from audit import log_upload, log_search, log_delete, log_auth_failure, log_access
from ocr import extract_text_with_ocr_fallback
from search import keyword_search, ai_search

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
IS_PRODUCTION = ENVIRONMENT == "production"

API_KEY = (os.getenv("PDF_HELPER_API_KEY") or "").strip() or None
ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY")

MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE_MB", "20")) * 1024 * 1024
MAX_FILES_PER_REQUEST = int(os.getenv("MAX_FILES_PER_REQUEST", "20"))
CHAT_MODEL = os.getenv("CHAT_MODEL", "claude-sonnet-4-5-20250929")
CHAT_MAX_TOKENS = int(os.getenv("CHAT_MAX_TOKENS", "8192"))
CHAT_WEB_SEARCH = os.getenv("CHAT_WEB_SEARCH", "true").lower() == "true"
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "/tmp/pdfhelper_uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Auto-cleanup: delete uploads older than this many hours (0 = disabled)
AUTO_CLEANUP_HOURS = int(os.getenv("AUTO_CLEANUP_HOURS", "72"))

# JWT auth config
JWT_SECRET = os.getenv("JWT_SECRET", "").strip() or secrets.token_hex(32)
JWT_ALGORITHM = "HS256"
JWT_EXPIRY_HOURS = int(os.getenv("JWT_EXPIRY_HOURS", "24"))
# Set to "true" to allow new user registration (default: disabled for single-user)
ALLOW_REGISTRATION = os.getenv("ALLOW_REGISTRATION", "false").lower() == "true"
# Auto-create admin account on startup (set both to enable)
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "").strip()
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "").strip()

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

_startup_errors: list[str] = []

app = FastAPI(
    title="PDFHelper",
    description="AI-powered PDF search and flagging tool",
    version="1.0.0",
    docs_url="/docs" if not IS_PRODUCTION else None,
    redoc_url=None,
)


async def _retry_db_init(logger):
    """Retry database table creation in the background."""
    import asyncio
    for attempt in range(2, 6):  # attempts 2–5
        await asyncio.sleep(2 ** attempt)  # 4, 8, 16, 32 seconds
        try:
            Base.metadata.create_all(bind=engine)
            logger.info("Database tables initialized successfully (attempt %d).", attempt)
            return
        except Exception as exc:
            logger.warning("Database init attempt %d/5 failed: %s", attempt, exc)
    _startup_errors.append("WARNING: Database initialization failed after 5 attempts.")


@app.on_event("startup")
async def startup():
    """Run safety checks and initialize DB — errors are logged, not fatal."""
    import asyncio
    import logging
    logger = logging.getLogger("pdfhelper")

    if IS_PRODUCTION and not API_KEY:
        msg = (
            "WARNING: PDF_HELPER_API_KEY not set in production. "
            "API endpoints will reject requests until it is configured."
        )
        logger.warning(msg)
        _startup_errors.append(msg)

    if IS_PRODUCTION and not os.getenv("JWT_SECRET", "").strip():
        msg = (
            "WARNING: JWT_SECRET not set in production. "
            "A random secret was generated — JWTs will not survive restarts."
        )
        logger.warning(msg)
        _startup_errors.append(msg)

    if IS_PRODUCTION and not ENCRYPTION_KEY:
        msg = (
            "WARNING: ENCRYPTION_KEY not set in production. "
            "File encryption is disabled until it is configured."
        )
        logger.warning(msg)
        _startup_errors.append(msg)

    # Try once synchronously, then retry in background if needed
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables initialized successfully.")
    except Exception as exc:
        logger.warning("Database init failed on first attempt: %s — retrying in background", exc)
        asyncio.create_task(_retry_db_init(logger))

    # Auto-create admin if env vars are set and no user with that name exists
    if ADMIN_USERNAME and ADMIN_PASSWORD:
        try:
            db = SessionLocal()
            existing = db.query(DBUser).filter(DBUser.username == ADMIN_USERNAME).first()
            if not existing:
                admin = DBUser(
                    id=str(uuid.uuid4()),
                    username=ADMIN_USERNAME,
                    password_hash=_hash_password(ADMIN_PASSWORD),
                    is_admin=True,
                    created_at=datetime.now(timezone.utc),
                )
                db.add(admin)
                db.commit()
                logger.info("Admin account '%s' created from env vars.", ADMIN_USERNAME)
            db.close()
        except Exception as exc:
            logger.warning("Failed to auto-create admin account: %s", exc)


# ---------------------------------------------------------------------------
# Security middleware
# ---------------------------------------------------------------------------

# -- CORS --
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "").split(",")
ALLOWED_ORIGINS = [o.strip() for o in ALLOWED_ORIGINS if o.strip()]
if IS_PRODUCTION and not ALLOWED_ORIGINS:
    ALLOWED_ORIGINS = []  # Block all cross-origin requests by default

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS if ALLOWED_ORIGINS else (["*"] if not IS_PRODUCTION else []),
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Authorization", "X-API-Key", "Content-Type"],
)

# -- Trusted hosts --
ALLOWED_HOSTS = os.getenv("ALLOWED_HOSTS", "").split(",")
ALLOWED_HOSTS = [h.strip() for h in ALLOWED_HOSTS if h.strip()]
if ALLOWED_HOSTS:
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=ALLOWED_HOSTS)


# -- Security headers --
class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to every response."""
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        if IS_PRODUCTION:
            response.headers["Strict-Transport-Security"] = "max-age=63072000; includeSubDomains"
        # Never reveal server tech
        if "server" in response.headers:
            del response.headers["server"]
        return response


app.add_middleware(SecurityHeadersMiddleware)


# -- HTTPS redirect (production) --
class HTTPSRedirectMiddleware(BaseHTTPMiddleware):
    """Redirect HTTP to HTTPS in production using X-Forwarded-Proto."""
    async def dispatch(self, request: Request, call_next):
        if IS_PRODUCTION:
            proto = request.headers.get("x-forwarded-proto", "https")
            if proto != "https":
                url = request.url.replace(scheme="https")
                return JSONResponse(
                    status_code=301,
                    headers={"Location": str(url)},
                    content={"detail": "Use HTTPS"},
                )
        return await call_next(request)


app.add_middleware(HTTPSRedirectMiddleware)


# -- Rate limiting --
class RateLimitMiddleware(BaseHTTPMiddleware):
    """In-memory rate limiter per IP with periodic cleanup."""
    def __init__(self, app, max_requests: int = 60, window_seconds: int = 60):
        super().__init__(app)
        self.max_requests = max_requests
        self.window = window_seconds
        self.requests: dict[str, list[float]] = {}
        self._last_cleanup = time.time()
        self._cleanup_interval = 300  # purge stale IPs every 5 minutes

    async def dispatch(self, request: Request, call_next):
        client_ip = _get_client_ip(request)
        now = time.time()
        window_start = now - self.window

        # Periodically purge IPs with no recent requests to prevent memory leak
        if now - self._last_cleanup > self._cleanup_interval:
            self.requests = {
                ip: [t for t in ts if t > window_start]
                for ip, ts in self.requests.items()
                if any(t > window_start for t in ts)
            }
            self._last_cleanup = now

        hits = self.requests.get(client_ip, [])
        hits = [t for t in hits if t > window_start]

        if len(hits) >= self.max_requests:
            return JSONResponse(
                status_code=429,
                content={"detail": "Too many requests. Try again later."},
            )
        hits.append(now)
        self.requests[client_ip] = hits

        response = await call_next(request)
        log_access(client_ip, request.method, request.url.path, response.status_code)
        return response


app.add_middleware(RateLimitMiddleware, max_requests=60, window_seconds=60)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_client_ip(request: Request) -> str:
    """Get real client IP, respecting proxy headers."""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _sanitize_filename(filename: str) -> str:
    """Strip path traversal and dangerous characters from filenames."""
    # Take only the basename
    name = Path(filename).name
    # Remove anything that isn't alphanumeric, dash, underscore, dot, or space
    name = re.sub(r"[^\w\-. ]", "_", name)
    # Prevent hidden files
    name = name.lstrip(".")
    return name or "unnamed.pdf"


def _validate_filepath(filepath: Path) -> Path:
    """Validate that a filepath is within UPLOAD_DIR. Prevents path traversal."""
    try:
        resolved = filepath.resolve()
        if not resolved.is_relative_to(UPLOAD_DIR.resolve()):
            raise ValueError(f"Path escapes upload directory: {filepath}")
        if resolved.is_symlink():
            raise ValueError(f"Symlinks not allowed: {filepath}")
        return resolved
    except (OSError, ValueError):
        raise


def _safe_unlink(filepath: Path) -> None:
    """Safely delete a file, handling race conditions (TOCTOU)."""
    try:
        validated = _validate_filepath(filepath)
        validated.unlink()
    except FileNotFoundError:
        pass  # Already deleted — not an error
    except ValueError:
        import logging
        logging.getLogger("pdfhelper").warning("Blocked path traversal attempt: %s", filepath)


PDF_MAGIC_BYTES = b"%PDF-"

ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"}


def _verify_pdf_content(data: bytes) -> bool:
    """Check that the file actually starts with the PDF magic bytes."""
    return data[:5] == PDF_MAGIC_BYTES


def _is_image_file(filename: str) -> bool:
    """Check if a filename has an image extension."""
    return any(filename.lower().endswith(ext) for ext in ALLOWED_IMAGE_EXTENSIONS)


def _image_to_pdf(image_bytes: bytes) -> bytes:
    """Convert an image file to a single-page PDF."""
    from PIL import Image
    import io
    img = Image.open(io.BytesIO(image_bytes))
    if img.mode != "RGB":
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PDF", resolution=200)
    return buf.getvalue()


def _extract_image_base64(image_bytes: bytes) -> str:
    """Return a base64-encoded version of the image for Claude vision."""
    import base64
    return base64.b64encode(image_bytes).decode("ascii")


def _detect_image_media_type(image_bytes: bytes) -> str:
    """Detect image MIME type from bytes."""
    if image_bytes[:8] == b'\x89PNG\r\n\x1a\n':
        return "image/png"
    if image_bytes[:2] == b'\xff\xd8':
        return "image/jpeg"
    if image_bytes[:4] in (b'II*\x00', b'MM\x00*'):
        return "image/tiff"
    if image_bytes[:4] == b'RIFF' and image_bytes[8:12] == b'WEBP':
        return "image/webp"
    if image_bytes[:2] == b'BM':
        return "image/bmp"
    return "image/png"  # fallback


# ---------------------------------------------------------------------------
# Password hashing & JWT helpers
# ---------------------------------------------------------------------------

_HASH_ITERATIONS = 260_000  # OWASP recommended for PBKDF2-SHA256


def _hash_password(password: str) -> str:
    """Hash a password with PBKDF2-SHA256 + random salt."""
    salt = secrets.token_hex(16)
    h = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), _HASH_ITERATIONS)
    return f"{salt}:{h.hex()}"


def _verify_password(password: str, stored_hash: str) -> bool:
    """Verify a password against a stored PBKDF2 hash."""
    try:
        salt, hash_hex = stored_hash.split(":", 1)
        h = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), _HASH_ITERATIONS)
        return secrets.compare_digest(h.hex(), hash_hex)
    except (ValueError, AttributeError):
        return False


def _create_jwt(user_id: str, username: str, is_admin: bool = False) -> str:
    """Create a signed JWT token for authenticated users."""
    import jwt
    payload = {
        "sub": user_id,
        "username": username,
        "admin": is_admin,
        "exp": datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRY_HOURS),
        "iat": datetime.now(timezone.utc),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def _decode_jwt(token: str) -> dict | None:
    """Decode and verify a JWT token. Returns payload or None."""
    import jwt
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except (jwt.ExpiredSignatureError, jwt.InvalidTokenError):
        return None


# ---------------------------------------------------------------------------
# Auth dependency
# ---------------------------------------------------------------------------

async def verify_auth(request: Request):
    """Authenticate via JWT Bearer token OR legacy API key."""
    # Dev mode: skip auth only if no API_KEY, no JWT_SECRET, AND no users exist
    if not API_KEY and os.getenv("JWT_SECRET", "").strip() == "":
        try:
            db = SessionLocal()
            has_users = db.query(DBUser).first() is not None
            db.close()
            if not has_users:
                return
        except Exception:
            return

    auth = request.headers.get("Authorization", "")

    # Try JWT Bearer token first
    if auth.startswith("Bearer "):
        token = auth[7:]
        payload = _decode_jwt(token)
        if payload:
            request.state.user_id = payload.get("sub")
            request.state.username = payload.get("username")
            request.state.is_admin = payload.get("admin", False)
            return

    # Fall back to legacy API key (X-API-Key header)
    api_key_token = request.headers.get("X-API-Key", "")
    if API_KEY and api_key_token and secrets.compare_digest(api_key_token, API_KEY):
        request.state.user_id = None  # API key users have no user_id
        request.state.username = "api_key_user"
        request.state.is_admin = False
        return

    # Also accept Bearer token as API key for backward compatibility
    if API_KEY and auth.startswith("Bearer "):
        token = auth[7:]
        if secrets.compare_digest(token, API_KEY):
            request.state.user_id = None
            request.state.username = "api_key_user"
            request.state.is_admin = False
            return

    log_auth_failure(_get_client_ip(request), request.url.path)
    raise HTTPException(status_code=401, detail="Invalid or missing credentials")


# Backward-compatible alias
verify_api_key = verify_auth


# ---------------------------------------------------------------------------
# DB dependency
# ---------------------------------------------------------------------------

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Encryption helpers
# ---------------------------------------------------------------------------

def _encrypt_and_save(data: bytes, path: Path) -> None:
    """Encrypt data and write to path. Falls back to plaintext if no key."""
    if ENCRYPTION_KEY:
        from encryption import encrypt_bytes
        path.write_bytes(encrypt_bytes(data))
    else:
        path.write_bytes(data)


def _encrypt_text(text: str) -> str:
    """Encrypt a string for database storage. Returns base64-encoded ciphertext."""
    if ENCRYPTION_KEY:
        from encryption import encrypt_bytes
        import base64
        return base64.b64encode(encrypt_bytes(text.encode("utf-8"))).decode("ascii")
    return text


def _decrypt_text(stored: str) -> str:
    """Decrypt a string read from the database."""
    if ENCRYPTION_KEY:
        from encryption import decrypt_bytes
        import base64
        return decrypt_bytes(base64.b64decode(stored)).decode("utf-8")
    return stored


# ---------------------------------------------------------------------------
# PDF processing
# ---------------------------------------------------------------------------

def extract_text_from_bytes(pdf_bytes: bytes) -> list[dict]:
    """Extract text from PDF bytes. Falls back to OCR for scanned PDFs."""
    return extract_text_with_ocr_fallback(pdf_bytes)



# keyword_search and ai_search are imported from search.py


# ---------------------------------------------------------------------------
# Auto-cleanup
# ---------------------------------------------------------------------------

def _run_cleanup(db) -> int:
    """Delete documents older than AUTO_CLEANUP_HOURS. Returns count deleted."""
    if not AUTO_CLEANUP_HOURS:
        return 0
    cutoff = datetime.now(timezone.utc) - timedelta(hours=AUTO_CLEANUP_HOURS)
    old_docs = db.query(DBDocument).filter(DBDocument.uploaded_at < cutoff).all()
    count = 0
    for doc in old_docs:
        _safe_unlink(Path(doc.filepath))
        db.delete(doc)
        count += 1
    if count:
        db.commit()
    return count


def _run_cleanup_background():
    """Run cleanup with its own DB session (safe for background tasks)."""
    db = SessionLocal()
    try:
        _run_cleanup(db)
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class SearchRequest(BaseModel):
    search_terms: list[str] = Field(default=[], max_length=50, description="Exact keywords to search")
    ai_query: str | None = Field(default=None, max_length=2000, description="AI concept search query")
    case_sensitive: bool = False


class AnalyzeRequest(BaseModel):
    compliance_context: str | None = Field(
        default=None,
        max_length=2000,
        description="Optional compliance standard to check against, e.g. 'OSHA 2024', 'HIPAA', 'FDA 21 CFR Part 11'",
    )
    search_terms: list[str] = Field(default=[], max_length=50, description="Optional keywords to search for")
    ai_query: str | None = Field(default=None, max_length=2000, description="Optional AI concept search query")


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"] = Field(description="'user' or 'assistant'")
    content: str = Field(max_length=200000)


class ChatRequest(BaseModel):
    message: str = Field(max_length=10000, description="The user's message")
    doc_ids: list[str] = Field(default=[], max_length=100, description="Document IDs to use as context (empty = all)")
    conversation_history: list[ChatMessage] = Field(default=[], max_length=200, description="Previous messages for context")
    session_id: str | None = Field(default=None, max_length=100, description="Chat session ID to continue (omit to create new)")


class RegisterRequest(BaseModel):
    username: str = Field(min_length=3, max_length=50, pattern=r"^[a-zA-Z0-9_]+$")
    password: str = Field(min_length=8, max_length=128)


class LoginRequest(BaseModel):
    username: str = Field(max_length=50)
    password: str = Field(max_length=128)


class HealthResponse(BaseModel):
    status: str
    version: str
    api_key_required: bool = False
    has_users: bool = False
    warnings: list[str] = []


# Valid work types for isolation requests
_VALID_WORK_TYPES = {
    "MAINTENANCE", "HOT WORK", "CONFINED SPACE ENTRY", "PRESSURE TEST",
    "INSPECTION", "EQUIPMENT REMOVAL", "ELECTRICAL ISOLATION", "INSTRUMENT MAINTENANCE",
}


class IsolationRequest(BaseModel):
    equipment_tag: str = Field(max_length=200, description="Equipment tag, e.g. HB-P-1001A")
    work_description: str = Field(max_length=5000, description="Description of work to be performed")
    work_type: str = Field(max_length=100, description="MAINTENANCE, HOT WORK, CONFINED SPACE ENTRY, PRESSURE TEST, INSPECTION, EQUIPMENT REMOVAL, ELECTRICAL ISOLATION, INSTRUMENT MAINTENANCE")
    fluid_service: str = Field(default="Not specified", max_length=200, description="Fluid service, e.g. Crude Oil, HC Gas, Produced Water")
    special_requirements: str = Field(default="None", max_length=5000, description="Any special requirements")
    facility: str = Field(default="Hebron", max_length=200, description="Facility name")
    regime: str = Field(default="C-NLOPB / C-NLOER", max_length=200, description="Regulatory regime")
    drawing_ids: list[str] = Field(default=[], max_length=20, description="Specific drawing IDs to use (empty = auto-select via Pass 1)")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_upload(file: UploadFile, content: bytes) -> tuple[str, bytes, bool]:
    """Validate an uploaded file (PDF or image). Returns (sanitized_name, file_bytes, is_image).

    Images are converted to PDF for text extraction. The original image bytes
    are returned alongside so they can be stored for Claude vision.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    clean_name = _sanitize_filename(file.filename)
    is_image = _is_image_file(clean_name)

    if not clean_name.lower().endswith(".pdf") and not is_image:
        raise HTTPException(status_code=400,
                            detail=f"Only PDF and image files allowed, got: {clean_name}")

    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"{clean_name} exceeds max size of {MAX_FILE_SIZE // (1024*1024)} MB",
        )

    if is_image:
        # Convert image to PDF for text extraction pipeline
        try:
            pdf_bytes = _image_to_pdf(content)
        except Exception as exc:
            raise HTTPException(status_code=400,
                                detail=f"Could not process image {clean_name}: {exc}")
        return clean_name, pdf_bytes, True
    else:
        # Verify actual PDF content
        if not _verify_pdf_content(content):
            raise HTTPException(status_code=400,
                                detail="File does not appear to be a valid PDF")
        return clean_name, content, False


# ---------------------------------------------------------------------------
# Root UI (loaded from templates/index.html)
# ---------------------------------------------------------------------------

_TEMPLATE_DIR = Path(__file__).parent / "templates"
_cached_html: str | None = None


def _load_root_html() -> str:
    """Load the frontend HTML template. Cached in production to avoid disk reads."""
    global _cached_html
    if IS_PRODUCTION and _cached_html is not None:
        return _cached_html
    html = (_TEMPLATE_DIR / "index.html").read_text(encoding="utf-8")
    if IS_PRODUCTION:
        _cached_html = html
    return html



# ---------------------------------------------------------------------------
# Global error handler for DB failures
# ---------------------------------------------------------------------------

from sqlalchemy.exc import OperationalError as SAOperationalError

@app.exception_handler(SAOperationalError)
async def db_error_handler(request: Request, exc: SAOperationalError):
    """Return a clear error message when the database is unreachable."""
    return JSONResponse(
        status_code=503,
        content={
            "detail": "Database is unavailable. Check that DATABASE_URL is set correctly "
                      "(use the public URL, not the internal .railway.internal hostname)."
        },
    )

@app.exception_handler(Exception)
async def general_error_handler(request: Request, exc: Exception):
    """Catch-all so errors always return JSON, never HTML."""
    import traceback, logging
    logging.getLogger("pdfhelper").error("Unhandled error on %s: %s\n%s", request.url.path, exc, traceback.format_exc())
    if IS_PRODUCTION:
        detail = "Internal server error"
    else:
        detail = f"Internal server error: {type(exc).__name__}: {str(exc)}"
    return JSONResponse(
        status_code=500,
        content={"detail": detail},
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check — also tests current DB connectivity."""
    db_ok = True
    db_err = None
    try:
        from sqlalchemy import text
        db = SessionLocal()
        db.execute(text("SELECT 1"))
        db.close()
    except Exception as e:
        db_ok = False
        # Log full error for operators, but don't expose connection details to clients
        import logging
        logging.getLogger("pdfhelper").error("Health check DB failure: %s", e)
        db_err = "Database connection failed"

    warnings = list(_startup_errors)
    if db_err:
        warnings.append(db_err)

    has_users = False
    if db_ok:
        try:
            db = SessionLocal()
            has_users = db.query(DBUser).first() is not None
            db.close()
        except Exception:
            pass

    status = "ok" if (not warnings and db_ok) else "degraded"
    return {
        "status": status,
        "version": "1.0.0",
        "api_key_required": True,
        "has_users": has_users,
        "warnings": warnings,
    }


@app.get("/verify-key", dependencies=[Depends(verify_api_key)])
async def verify_key():
    """Lightweight API key check — no database required."""
    import asyncio
    from sqlalchemy import text

    def _check_db():
        db = SessionLocal()
        try:
            db.execute(text("SELECT 1"))
        finally:
            db.close()

    db_ok = True
    try:
        await asyncio.wait_for(asyncio.to_thread(_check_db), timeout=3)
    except Exception:
        db_ok = False
    return {"valid": True, "db_ok": db_ok}


@app.get("/", response_class=HTMLResponse)
async def root():
    """Interactive operating interface for PDFHelper."""
    return _load_root_html()


@app.get("/bot")
async def bot_page():
    """Redirect old bot page to main app."""
    from starlette.responses import RedirectResponse
    return RedirectResponse(url="/", status_code=301)


# ---------------------------------------------------------------------------
# User registration & login
# ---------------------------------------------------------------------------

@app.get("/setup-needed")
async def setup_needed(db=Depends(get_db)):
    """Check if initial setup is required (no users exist yet)."""
    any_user = db.query(DBUser).first()
    return {"setup_needed": any_user is None}


@app.post("/setup")
async def setup(body: RegisterRequest, db=Depends(get_db)):
    """First-run setup: create the initial admin account. Only works when no users exist."""
    any_user = db.query(DBUser).first()
    if any_user:
        raise HTTPException(status_code=403, detail="Setup already completed. Use /login instead.")

    user_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    user = DBUser(
        id=user_id,
        username=body.username,
        password_hash=_hash_password(body.password),
        is_admin=True,
        created_at=now,
    )
    db.add(user)
    db.commit()

    token = _create_jwt(user_id, body.username, is_admin=True)
    return {"user_id": user_id, "username": body.username, "token": token}


@app.post("/register")
async def register(body: RegisterRequest, db=Depends(get_db)):
    """Create a new user account. Returns a JWT token."""
    if not ALLOW_REGISTRATION:
        raise HTTPException(status_code=403, detail="Registration is disabled. Contact an admin.")

    existing = db.query(DBUser).filter(DBUser.username == body.username).first()
    if existing:
        raise HTTPException(status_code=409, detail="Username already taken")

    user_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    user = DBUser(
        id=user_id,
        username=body.username,
        password_hash=_hash_password(body.password),
        is_admin=False,
        created_at=now,
    )
    db.add(user)
    db.commit()

    token = _create_jwt(user_id, body.username)
    return {"user_id": user_id, "username": body.username, "token": token}


@app.post("/login")
async def login(body: LoginRequest, db=Depends(get_db)):
    """Authenticate and receive a JWT token."""
    user = db.query(DBUser).filter(DBUser.username == body.username).first()
    if not user or not _verify_password(body.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid username or password")

    token = _create_jwt(user.id, user.username, user.is_admin)
    return {"user_id": user.id, "username": user.username, "token": token}


@app.get("/me", dependencies=[Depends(verify_auth)])
async def get_current_user(request: Request):
    """Return the currently authenticated user's info."""
    return {
        "user_id": getattr(request.state, "user_id", None),
        "username": getattr(request.state, "username", None),
        "is_admin": getattr(request.state, "is_admin", False),
    }


@app.post("/upload", dependencies=[Depends(verify_api_key)])
async def upload_pdfs(
    request: Request,
    files: list[UploadFile] = File(...),
    background_tasks: BackgroundTasks = None,
    db=Depends(get_db),
):
    """Upload one or more PDFs or images for later searching.

    Supported formats: PDF, JPG, PNG, TIFF, BMP, WebP.
    Images are automatically converted to PDF and OCR'd for text extraction.
    Original images are also stored so Claude vision can analyze them in chat.
    """
    if len(files) > MAX_FILES_PER_REQUEST:
        raise HTTPException(status_code=400,
                            detail=f"Max {MAX_FILES_PER_REQUEST} files per request")

    client_ip = _get_client_ip(request)

    # Run cleanup after the response is sent so it doesn't slow down uploads
    if background_tasks:
        background_tasks.add_task(_run_cleanup_background)

    uploaded = []
    for file in files:
        raw_content = await file.read()
        clean_name, pdf_bytes, is_image = validate_upload(file, raw_content)

        # Extract text from PDF bytes (images were converted to PDF above)
        pages = extract_text_from_bytes(pdf_bytes)

        doc_id = str(uuid.uuid4())
        save_path = UPLOAD_DIR / f"{doc_id}.pdf.enc"

        # Encrypt and save the PDF version
        _encrypt_and_save(pdf_bytes, save_path)

        # If it was an image, also save the original for Claude vision
        if is_image:
            img_save_path = UPLOAD_DIR / f"{doc_id}.img.enc"
            _encrypt_and_save(raw_content, img_save_path)

        content_hash = hashlib.sha256(raw_content).hexdigest()

        db_doc = DBDocument(
            id=doc_id,
            filename=_encrypt_text(clean_name),
            filepath=str(save_path),
            page_count=len(pages),
            text_content=_encrypt_text(json.dumps(pages)),
            content_hash=content_hash,
            uploaded_at=datetime.now(timezone.utc),
        )
        db.add(db_doc)
        db.commit()

        log_upload(client_ip, clean_name, doc_id, len(pages))

        uploaded.append({
            "id": doc_id,
            "filename": clean_name,
            "pages": len(pages),
            "type": "image" if is_image else "pdf",
        })

    return {"uploaded": uploaded, "count": len(uploaded)}


@app.post("/search", dependencies=[Depends(verify_api_key)])
async def search_documents(
    request: Request,
    body: SearchRequest,
    doc_ids: list[str] = Query(default=[], description="Document IDs to search (empty = all)"),
    db=Depends(get_db),
):
    """Search uploaded PDFs with keywords and/or AI."""
    if not body.search_terms and not body.ai_query:
        raise HTTPException(status_code=400,
                            detail="Provide search_terms and/or ai_query")

    query = db.query(DBDocument)
    if doc_ids:
        query = query.filter(DBDocument.id.in_(doc_ids))
    documents = query.all()

    if not documents:
        raise HTTPException(status_code=404, detail="No documents found")

    client_ip = _get_client_ip(request)
    all_keyword_results = []
    all_ai_results = []

    for doc in documents:
        pages = json.loads(_decrypt_text(doc.text_content))
        decrypted_name = _decrypt_text(doc.filename)

        if body.search_terms:
            matches = keyword_search(pages, body.search_terms, body.case_sensitive)
            for m in matches:
                m["document_id"] = doc.id
                m["filename"] = decrypted_name
            all_keyword_results.extend(matches)

        if body.ai_query:
            findings = ai_search(pages, body.ai_query, decrypted_name)
            for f in findings:
                f["document_id"] = doc.id
                f["filename"] = decrypted_name
            all_ai_results.extend(findings)

    search_id = str(uuid.uuid4())
    flagged_count = len([r for r in all_ai_results if r.get("needs_review")])

    db_result = DBSearchResult(
        id=search_id,
        search_terms=_encrypt_text(json.dumps(body.search_terms)) if body.search_terms else None,
        ai_query=_encrypt_text(body.ai_query) if body.ai_query else None,
        keyword_results=_encrypt_text(json.dumps(all_keyword_results)),
        ai_results=_encrypt_text(json.dumps(all_ai_results)),
        total_keyword_matches=len(all_keyword_results),
        total_ai_findings=len(all_ai_results),
        flagged_for_review=flagged_count,
        searched_at=datetime.now(timezone.utc),
    )
    db.add(db_result)
    db.commit()

    log_search(client_ip, search_id, body.search_terms, body.ai_query,
               len(documents), len(all_keyword_results) + len(all_ai_results),
               flagged_count)

    return {
        "search_id": search_id,
        "summary": {
            "documents_searched": len(documents),
            "total_keyword_matches": len(all_keyword_results),
            "total_ai_findings": len(all_ai_results),
            "flagged_for_review": flagged_count,
        },
        "keyword_results": all_keyword_results,
        "ai_results": all_ai_results,
    }


@app.get("/documents", dependencies=[Depends(verify_api_key)])
async def list_documents(db=Depends(get_db)):
    """List all uploaded documents."""
    docs = db.query(DBDocument).order_by(DBDocument.uploaded_at.desc()).all()
    return {
        "documents": [
            {
                "id": d.id,
                "filename": _decrypt_text(d.filename),
                "pages": d.page_count,
                "uploaded_at": d.uploaded_at.isoformat(),
            }
            for d in docs
        ]
    }


@app.get("/documents/{doc_id}", dependencies=[Depends(verify_api_key)])
async def get_document(doc_id: str, db=Depends(get_db)):
    """Get details for a specific document."""
    doc = db.query(DBDocument).filter(DBDocument.id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return {
        "id": doc.id,
        "filename": _decrypt_text(doc.filename),
        "pages": doc.page_count,
        "uploaded_at": doc.uploaded_at.isoformat(),
    }


@app.delete("/documents/{doc_id}", dependencies=[Depends(verify_api_key)])
async def delete_document(doc_id: str, request: Request, db=Depends(get_db)):
    """Delete an uploaded document and its encrypted file."""
    doc = db.query(DBDocument).filter(DBDocument.id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    _safe_unlink(Path(doc.filepath))

    log_delete(_get_client_ip(request), doc_id, _decrypt_text(doc.filename))

    db.delete(doc)
    db.commit()
    return {"deleted": doc_id}


@app.get("/history", dependencies=[Depends(verify_api_key)])
async def search_history(limit: int = Query(default=20, le=100), db=Depends(get_db)):
    """Get past search results."""
    results = (
        db.query(DBSearchResult)
        .order_by(DBSearchResult.searched_at.desc())
        .limit(limit)
        .all()
    )
    return {
        "searches": [
            {
                "id": r.id,
                "search_terms": json.loads(_decrypt_text(r.search_terms)) if r.search_terms else [],
                "ai_query": _decrypt_text(r.ai_query) if r.ai_query else None,
                "total_keyword_matches": r.total_keyword_matches,
                "total_ai_findings": r.total_ai_findings,
                "flagged_for_review": r.flagged_for_review,
                "searched_at": r.searched_at.isoformat(),
            }
            for r in results
        ]
    }


@app.get("/history/{search_id}", dependencies=[Depends(verify_api_key)])
async def get_search_result(search_id: str, db=Depends(get_db)):
    """Get full details of a past search."""
    result = db.query(DBSearchResult).filter(DBSearchResult.id == search_id).first()
    if not result:
        raise HTTPException(status_code=404, detail="Search result not found")
    return {
        "id": result.id,
        "search_terms": json.loads(_decrypt_text(result.search_terms)) if result.search_terms else [],
        "ai_query": _decrypt_text(result.ai_query) if result.ai_query else None,
        "keyword_results": json.loads(_decrypt_text(result.keyword_results)),
        "ai_results": json.loads(_decrypt_text(result.ai_results)),
        "summary": {
            "total_keyword_matches": result.total_keyword_matches,
            "total_ai_findings": result.total_ai_findings,
            "flagged_for_review": result.flagged_for_review,
        },
        "searched_at": result.searched_at.isoformat(),
    }


# ---------------------------------------------------------------------------
# Procedure Chatbot
# ---------------------------------------------------------------------------

@app.post("/chat", dependencies=[Depends(verify_api_key)])
async def chat_with_documents(
    request: Request,
    body: ChatRequest,
    db=Depends(get_db),
):
    """Chat with your uploaded documents using AI.

    Sends the user's message along with selected document content to Claude
    and returns a context-aware response with procedure citations.

    If session_id is provided, continues that session (loading history from DB).
    Otherwise creates a new session. All messages are persisted.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not configured on server")

    query = db.query(DBDocument)
    if body.doc_ids:
        query = query.filter(DBDocument.id.in_(body.doc_ids))
    documents = query.all()

    if not documents:
        raise HTTPException(status_code=404, detail="No documents found. Upload documents first.")

    # Resolve or create chat session
    now = datetime.now(timezone.utc)
    current_user_id = getattr(request.state, "user_id", None)
    session = None
    if body.session_id:
        session = db.query(DBChatSession).filter(DBChatSession.id == body.session_id).first()
        # Enforce session ownership: user can only access their own sessions
        if session and current_user_id and session.user_id and session.user_id != current_user_id:
            raise HTTPException(status_code=403, detail="You do not own this chat session")

    if session is None:
        session = DBChatSession(
            id=str(uuid.uuid4()),
            user_id=current_user_id,
            title=body.message[:100],
            doc_ids=json.dumps(body.doc_ids),
            created_at=now,
            updated_at=now,
        )
        db.add(session)
        db.flush()

    # Build procedure context from selected documents
    procedure_parts = []
    image_content_blocks = []  # Claude vision blocks for uploaded images
    for doc in documents:
        decrypted_name = _decrypt_text(doc.filename)
        pages = json.loads(_decrypt_text(doc.text_content))
        full_text = "\n".join(p["text"] for p in pages if p.get("text"))
        if len(full_text) > 80000:
            full_text = full_text[:80000] + "\n\n[... content truncated for context window ...]"
        procedure_parts.append(
            f'--- PROCEDURE: "{decrypted_name}" ---\n{full_text}\n--- END OF "{decrypted_name}" ---'
        )

        # Check if this document has an associated image file for vision
        img_path = Path(doc.filepath.replace(".pdf.enc", ".img.enc"))
        if img_path.exists() and len(image_content_blocks) < 10:  # max 10 images
            try:
                img_bytes = _decrypt_and_load(img_path)
                media_type = _detect_image_media_type(img_bytes)
                b64 = _extract_image_base64(img_bytes)
                image_content_blocks.append({
                    "type": "text",
                    "text": f"[Image: {decrypted_name}]"
                })
                image_content_blocks.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": b64,
                    }
                })
            except Exception:
                pass  # skip if image can't be loaded

    procedure_context = "\n\n".join(procedure_parts)

    # Build conversation from DB history (prefer DB over client-sent history)
    db_messages = (
        db.query(DBChatMessage)
        .filter(DBChatMessage.session_id == session.id)
        .order_by(DBChatMessage.created_at)
        .all()
    )

    if db_messages:
        conversation = [
            {"role": m.role, "content": _decrypt_text(m.content)}
            for m in db_messages[-10:]
        ]
    else:
        conversation = [
            {"role": m.role, "content": m.content}
            for m in body.conversation_history[-10:]
            if m.role in ("user", "assistant")
        ]
    # Build the user message — include images via Claude vision only on the
    # first message of a session so they aren't re-sent on every turn.
    include_images = image_content_blocks and not db_messages
    if include_images:
        user_content = list(image_content_blocks)  # copy
        user_content.append({"type": "text", "text": body.message})
        conversation.append({"role": "user", "content": user_content})
    else:
        conversation.append({"role": "user", "content": body.message})

    # Budget the total context to stay within the model's context window.
    # Reserve chars for the system prompt template, response tokens, and safety margin.
    # Approximate: 1 token ≈ 4 chars.  Model context ≈ 200K tokens ≈ 800K chars.
    # Each image ≈ 1600 tokens, so subtract from budget when included.
    MAX_TOTAL_CHARS = 600000  # leave headroom for response + system template
    if include_images:
        image_char_budget = len(image_content_blocks) // 2 * 6400  # ~1600 tokens * 4 chars per image
        MAX_TOTAL_CHARS -= image_char_budget

    def _msg_text_len(m):
        c = m["content"]
        if isinstance(c, str):
            return len(c)
        if isinstance(c, list):
            return sum(len(b.get("text", "")) for b in c if isinstance(b, dict) and b.get("type") == "text")
        return 0

    conv_chars = sum(_msg_text_len(m) for m in conversation)

    # If conversation history alone is too large, trim older messages (keep latest)
    while conv_chars > 200000 and len(conversation) > 1:
        removed = conversation.pop(0)
        conv_chars -= _msg_text_len(removed)

    budget_for_procedures = MAX_TOTAL_CHARS - conv_chars
    if budget_for_procedures < 10000:
        budget_for_procedures = 10000  # always keep at least some procedure context

    # Truncate procedure context if it exceeds budget
    if len(procedure_context) > budget_for_procedures:
        procedure_context = procedure_context[:budget_for_procedures] + "\n\n[... procedures truncated to fit context window ...]"

    system_prompt = f"""You are a Procedure Knowledge Assistant. You answer questions based on the procedure documents provided below, and when needed you can also search the web for additional information.

RULES:
1. FIRST check the provided procedure documents for relevant information.
2. ALWAYS cite which procedure document your answer comes from by name and section if possible.
3. If the answer cannot be found in the provided procedures, use web search to find relevant information from the internet.
4. When using web search results, clearly indicate which information came from the web vs. from the loaded procedures.
5. Be precise and direct. Quote relevant sections when helpful.
6. If a question spans multiple procedures, reference all relevant ones.
7. Format your answers clearly with procedure references in bold.

LOADED PROCEDURES:
{procedure_context}"""

    from anthropic import Anthropic
    client = Anthropic(api_key=api_key)

    # Save user message now (assistant message saved after stream completes)
    db.add(DBChatMessage(
        id=str(uuid.uuid4()), session_id=session.id,
        role="user", content=_encrypt_text(body.message), created_at=now,
    ))
    db.commit()

    session_id = session.id
    doc_info = [{"id": d.id, "filename": _decrypt_text(d.filename)} for d in documents]

    # Configure tools — optionally include web search
    chat_tools = []
    if CHAT_WEB_SEARCH:
        chat_tools.append({"type": "web_search_20250305"})

    async def stream_chat():
        """Stream the AI response as Server-Sent Events.

        When CHAT_WEB_SEARCH is enabled, the Anthropic API's server-side
        web_search connector automatically searches and returns results
        within a single request — no multi-turn loop needed.
        """
        full_reply = ""

        try:
            # Send session metadata first
            yield f"data: {json.dumps({'type': 'meta', 'session_id': session_id, 'documents_used': doc_info})}\n\n"

            create_kwargs = dict(
                model=CHAT_MODEL,
                max_tokens=CHAT_MAX_TOKENS,
                system=system_prompt,
                messages=conversation,
            )
            if chat_tools:
                create_kwargs["tools"] = chat_tools

            with client.messages.stream(**create_kwargs) as stream:
                for event in stream:
                    # Stream text chunks to the client
                    if hasattr(event, 'type'):
                        if event.type == 'content_block_start':
                            if hasattr(event.content_block, 'type') and event.content_block.type == 'server_tool_use':
                                yield f"data: {json.dumps({'type': 'status', 'message': 'Searching the web...'})}\n\n"
                        elif event.type == 'content_block_delta':
                            if hasattr(event.delta, 'text'):
                                full_reply += event.delta.text
                                yield f"data: {json.dumps({'type': 'chunk', 'text': event.delta.text})}\n\n"

            # Signal completion
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        except Exception as e:
            import logging
            logging.getLogger("pdfhelper").error("Chat AI stream failed: %s", e)
            err_msg = "AI request failed" if IS_PRODUCTION else f"AI request failed: {str(e)}"
            yield f"data: {json.dumps({'type': 'error', 'detail': err_msg})}\n\n"
            full_reply = full_reply or err_msg
        finally:
            # Persist assistant reply after stream completes
            save_db = SessionLocal()
            try:
                save_db.add(DBChatMessage(
                    id=str(uuid.uuid4()), session_id=session_id,
                    role="assistant",
                    content=_encrypt_text(full_reply or "Sorry, I couldn't generate a response."),
                    created_at=datetime.now(timezone.utc),
                ))
                sess = save_db.query(DBChatSession).filter(DBChatSession.id == session_id).first()
                if sess:
                    sess.updated_at = datetime.now(timezone.utc)
                save_db.commit()
            except Exception:
                save_db.rollback()
            finally:
                save_db.close()

    return StreamingResponse(stream_chat(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Chat History Endpoints
# ---------------------------------------------------------------------------

@app.get("/chat/sessions", dependencies=[Depends(verify_api_key)])
async def list_chat_sessions(request: Request, limit: int = Query(default=30, le=100), db=Depends(get_db)):
    """List past chat sessions, most recent first. Filtered to current user."""
    current_user_id = getattr(request.state, "user_id", None)
    query = db.query(DBChatSession)
    if current_user_id:
        query = query.filter(
            (DBChatSession.user_id == current_user_id) | (DBChatSession.user_id.is_(None))
        )
    sessions = query.order_by(DBChatSession.updated_at.desc()).limit(limit).all()
    return {
        "sessions": [
            {
                "id": s.id,
                "title": s.title,
                "doc_ids": json.loads(s.doc_ids),
                "message_count": len(s.messages),
                "created_at": s.created_at.isoformat(),
                "updated_at": s.updated_at.isoformat(),
            }
            for s in sessions
        ]
    }


@app.get("/chat/sessions/{session_id}", dependencies=[Depends(verify_api_key)])
async def get_chat_session(session_id: str, request: Request, db=Depends(get_db)):
    """Get full message history for a chat session."""
    session = db.query(DBChatSession).filter(DBChatSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Chat session not found")
    current_user_id = getattr(request.state, "user_id", None)
    if current_user_id and session.user_id and session.user_id != current_user_id:
        raise HTTPException(status_code=403, detail="You do not own this chat session")
    return {
        "id": session.id,
        "title": session.title,
        "doc_ids": json.loads(session.doc_ids),
        "created_at": session.created_at.isoformat(),
        "updated_at": session.updated_at.isoformat(),
        "messages": [
            {
                "role": m.role,
                "content": _decrypt_text(m.content),
                "created_at": m.created_at.isoformat(),
            }
            for m in session.messages
        ],
    }


@app.delete("/chat/sessions/{session_id}", dependencies=[Depends(verify_api_key)])
async def delete_chat_session(session_id: str, request: Request, db=Depends(get_db)):
    """Delete a chat session and all its messages."""
    session = db.query(DBChatSession).filter(DBChatSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Chat session not found")
    current_user_id = getattr(request.state, "user_id", None)
    if current_user_id and session.user_id and session.user_id != current_user_id:
        raise HTTPException(status_code=403, detail="You do not own this chat session")
    db.delete(session)
    db.commit()
    return {"deleted": session_id}


# ---------------------------------------------------------------------------
# Chat → Word Document Export
# ---------------------------------------------------------------------------

class ExportChatRequest(BaseModel):
    session_id: str = Field(max_length=100, description="Chat session to export")
    format: str = Field(default="docx", pattern=r"^docx$", description="Export format (docx)")


class GenerateDocRequest(BaseModel):
    session_id: str = Field(max_length=100, description="Chat session for context")
    instructions: str = Field(max_length=10000, description="What the document should contain, e.g. 'Create a safety procedure for valve isolation'")
    doc_ids: list[str] = Field(default=[], max_length=100, description="Document IDs to reference")
    title: str = Field(default="Generated Document", max_length=255)


def _markdown_to_docx(text: str, title: str = "Document") -> bytes:
    """Convert markdown-ish AI text to a formatted Word document."""
    from docx import Document as DocxDocument
    from docx.shared import Pt, Inches, RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    import re
    import io

    doc = DocxDocument()

    # Set default font
    style = doc.styles["Normal"]
    font = style.font
    font.name = "Calibri"
    font.size = Pt(11)

    # Add title
    t = doc.add_heading(title, level=0)
    t.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # Add generation date
    from datetime import datetime, timezone
    date_para = doc.add_paragraph()
    date_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = date_para.add_run(f"Generated: {datetime.now(timezone.utc).strftime('%B %d, %Y at %H:%M UTC')}")
    run.font.size = Pt(9)
    run.font.color.rgb = RGBColor(128, 128, 128)
    doc.add_paragraph()  # spacer

    # Parse markdown-like content into Word elements
    lines = text.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Headings
        if stripped.startswith("### "):
            doc.add_heading(stripped[4:], level=3)
        elif stripped.startswith("## "):
            doc.add_heading(stripped[3:], level=2)
        elif stripped.startswith("# "):
            doc.add_heading(stripped[2:], level=1)
        # Bullet points
        elif stripped.startswith("- ") or stripped.startswith("* "):
            p = doc.add_paragraph(stripped[2:], style="List Bullet")
        # Numbered list
        elif re.match(r"^\d+[\.\)]\s", stripped):
            text_content = re.sub(r"^\d+[\.\)]\s", "", stripped)
            p = doc.add_paragraph(text_content, style="List Number")
        # Bold line (like **Section Title**)
        elif stripped.startswith("**") and stripped.endswith("**") and len(stripped) > 4:
            p = doc.add_paragraph()
            run = p.add_run(stripped[2:-2])
            run.bold = True
            run.font.size = Pt(12)
        # Horizontal rule
        elif stripped in ("---", "***", "___"):
            doc.add_paragraph("_" * 50)
        # Empty line
        elif not stripped:
            pass  # skip blank lines (natural spacing from paragraphs)
        # Normal paragraph — handle inline bold/italic
        else:
            p = doc.add_paragraph()
            # Split on **bold** and *italic* markers
            parts = re.split(r"(\*\*.*?\*\*|\*.*?\*)", stripped)
            for part in parts:
                if part.startswith("**") and part.endswith("**"):
                    run = p.add_run(part[2:-2])
                    run.bold = True
                elif part.startswith("*") and part.endswith("*") and len(part) > 2:
                    run = p.add_run(part[1:-1])
                    run.italic = True
                else:
                    p.add_run(part)
        i += 1

    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


@app.post("/chat/export", dependencies=[Depends(verify_api_key)])
async def export_chat_to_docx(body: ExportChatRequest, request: Request, db=Depends(get_db)):
    """Export a chat session's AI responses as a Word document.

    Collects all assistant messages from the session and formats them
    into a downloadable .docx file.
    """
    session = db.query(DBChatSession).filter(DBChatSession.id == body.session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Chat session not found")
    current_user_id = getattr(request.state, "user_id", None)
    if current_user_id and session.user_id and session.user_id != current_user_id:
        raise HTTPException(status_code=403, detail="You do not own this chat session")

    messages = (
        db.query(DBChatMessage)
        .filter(DBChatMessage.session_id == session.id)
        .order_by(DBChatMessage.created_at)
        .all()
    )

    # Build document content from the conversation
    parts = []
    for m in messages:
        content = _decrypt_text(m.content)
        if m.role == "user":
            parts.append(f"**Question:** {content}")
        else:
            parts.append(content)
        parts.append("")  # blank line separator

    full_text = "\n".join(parts)
    title = session.title or "Chat Export"
    docx_bytes = _markdown_to_docx(full_text, title)

    safe_title = re.sub(r'[^\w\s-]', '', title)[:50].strip() or "chat-export"
    filename = f"{safe_title}.docx"

    return Response(
        content=docx_bytes,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.post("/chat/generate-doc", dependencies=[Depends(verify_api_key)])
async def generate_document_from_chat(body: GenerateDocRequest, request: Request, db=Depends(get_db)):
    """Use AI to generate a Word document based on chat context and instructions.

    The AI writes a complete document (procedure, report, summary, etc.)
    using the uploaded procedures and conversation history as context,
    then returns it as a downloadable .docx file.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not configured on server")

    # Load session history for context
    session = db.query(DBChatSession).filter(DBChatSession.id == body.session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Chat session not found")
    current_user_id = getattr(request.state, "user_id", None)
    if current_user_id and session.user_id and session.user_id != current_user_id:
        raise HTTPException(status_code=403, detail="You do not own this chat session")

    # Get conversation context
    db_messages = (
        db.query(DBChatMessage)
        .filter(DBChatMessage.session_id == session.id)
        .order_by(DBChatMessage.created_at)
        .all()
    )
    chat_context = "\n\n".join(
        f"{'User' if m.role == 'user' else 'Assistant'}: {_decrypt_text(m.content)}"
        for m in db_messages[-10:]
    )

    # Get procedure context from selected documents
    query = db.query(DBDocument)
    if body.doc_ids:
        query = query.filter(DBDocument.id.in_(body.doc_ids))
    documents = query.all()

    procedure_parts = []
    for doc in documents:
        decrypted_name = _decrypt_text(doc.filename)
        pages = json.loads(_decrypt_text(doc.text_content))
        full_text = "\n".join(p["text"] for p in pages if p.get("text"))
        if len(full_text) > 40000:
            full_text = full_text[:40000] + "\n[... truncated ...]"
        procedure_parts.append(f'--- "{decrypted_name}" ---\n{full_text}')
    procedure_context = "\n\n".join(procedure_parts) if procedure_parts else "(No procedures loaded)"

    from anthropic import Anthropic
    client = Anthropic(api_key=api_key)

    system = f"""You are a professional document writer. You create well-structured, detailed documents based on the user's instructions and the reference materials provided.

REFERENCE PROCEDURES:
{procedure_context[:200000]}

RECENT CHAT CONTEXT:
{chat_context[:50000]}

INSTRUCTIONS:
- Write the document in clean, professional language
- Use markdown headings (#, ##, ###), bold (**text**), bullet points (- item), and numbered lists (1. item)
- Include all relevant details from the reference procedures
- Structure the document logically with clear sections
- The document should be complete and ready to use — not a draft or outline"""

    create_kwargs = dict(
        model=CHAT_MODEL,
        max_tokens=CHAT_MAX_TOKENS * 2,  # allow longer output for documents
        system=system,
        messages=[{"role": "user", "content": body.instructions}],
    )
    if CHAT_WEB_SEARCH:
        create_kwargs["tools"] = [{"type": "web_search_20250305"}]

    response = client.messages.create(**create_kwargs)
    full_text = ""
    for block in response.content:
        if hasattr(block, "text"):
            full_text += block.text

    if not full_text.strip():
        raise HTTPException(status_code=500, detail="AI failed to generate document content")

    docx_bytes = _markdown_to_docx(full_text, body.title)

    safe_title = re.sub(r'[^\w\s-]', '', body.title)[:50].strip() or "generated-document"
    filename = f"{safe_title}.docx"

    return Response(
        content=docx_bytes,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ---------------------------------------------------------------------------
# Full Analysis Pipeline (multi-agent)
# ---------------------------------------------------------------------------

@app.post("/analyze", dependencies=[Depends(verify_api_key)])
async def analyze_documents(
    request: Request,
    body: AnalyzeRequest,
    doc_ids: list[str] = Query(default=[], description="Document IDs to analyze (empty = all)"),
    db=Depends(get_db),
):
    """Run the full multi-agent analysis pipeline on uploaded documents.

    This runs 4 specialized AI agents:
    1. Document Analyzer — deep analysis of each document
    2. Cross-Reference Checker — finds conflicts between documents
    3. Compliance Checker — flags regulatory/policy issues
    4. Summary Report Generator — produces an actionable executive report

    Optionally also runs keyword and AI search.
    """
    query = db.query(DBDocument)
    if doc_ids:
        query = query.filter(DBDocument.id.in_(doc_ids))
    documents = query.all()

    if not documents:
        raise HTTPException(status_code=404, detail="No documents found")

    client_ip = _get_client_ip(request)

    # Build a cache key from document content hashes + analysis parameters
    # This lets us skip re-analysis when the same documents are analyzed
    # with the same compliance context (search is fast enough to always re-run)
    doc_hashes = sorted(d.content_hash or d.id for d in documents)
    cache_key_input = json.dumps({
        "hashes": doc_hashes,
        "compliance_context": body.compliance_context,
    }, sort_keys=True)
    cache_key = hashlib.sha256(cache_key_input.encode()).hexdigest()

    # Check for a cached analysis with the same content + parameters
    cached_report = (
        db.query(DBAnalysisReport)
        .filter(DBAnalysisReport.cache_key == cache_key)
        .order_by(DBAnalysisReport.analyzed_at.desc())
        .first()
    )
    if cached_report:
        cached_analysis = json.loads(_decrypt_text(cached_report.report_data))
        # Re-run search if requested (cheap), but reuse the cached analysis
        if body.search_terms or body.ai_query:
            docs_for_agents: dict[str, list[dict]] = {}
            for doc in documents:
                decrypted_name = _decrypt_text(doc.filename)
                pages = json.loads(_decrypt_text(doc.text_content))
                docs_for_agents[decrypted_name] = pages
            from search import keyword_search, ai_search
            search_results = {"keyword_results": [], "ai_results": []}
            for filename, pages in docs_for_agents.items():
                if body.search_terms:
                    kw_matches = keyword_search(pages, body.search_terms)
                    for m in kw_matches:
                        m["filename"] = filename
                    search_results["keyword_results"].extend(kw_matches)
                if body.ai_query:
                    import asyncio
                    ai_matches = await asyncio.to_thread(ai_search, pages, body.ai_query, filename)
                    for m in ai_matches:
                        m["filename"] = filename
                    search_results["ai_results"].extend(ai_matches)
            cached_analysis["search_results"] = search_results

        return {
            "report_id": cached_report.id,
            "cached": True,
            "report": cached_analysis.get("report"),
            "document_analyses": cached_analysis.get("document_analyses"),
            "cross_reference_findings": cached_analysis.get("cross_reference_findings"),
            "compliance_findings": cached_analysis.get("compliance_findings"),
            "search_results": cached_analysis.get("search_results"),
        }

    # Build documents dict for the agent pipeline
    docs_for_agents: dict[str, list[dict]] = {}
    for doc in documents:
        decrypted_name = _decrypt_text(doc.filename)
        pages = json.loads(_decrypt_text(doc.text_content))
        docs_for_agents[decrypted_name] = pages

    # Run the full pipeline in a thread pool to avoid blocking the event loop
    # (run_full_analysis makes multiple synchronous Anthropic API calls)
    import asyncio
    from agents import run_full_analysis
    analysis = await asyncio.to_thread(
        run_full_analysis,
        documents=docs_for_agents,
        compliance_context=body.compliance_context,
        search_terms=body.search_terms if body.search_terms else None,
        ai_query=body.ai_query,
    )

    # Save to DB
    report_id = str(uuid.uuid4())
    db_report = DBAnalysisReport(
        id=report_id,
        doc_ids=json.dumps([d.id for d in documents]),
        compliance_context=_encrypt_text(body.compliance_context) if body.compliance_context else None,
        report_data=_encrypt_text(json.dumps(analysis)),
        documents_analyzed=len(documents),
        total_issues=analysis.get("report", {}).get("total_issues_found", 0),
        critical_issues=analysis.get("report", {}).get("critical_issues", 0),
        risk_level=analysis.get("report", {}).get("overall_risk_level", "unknown"),
        cache_key=cache_key,
        analyzed_at=datetime.now(timezone.utc),
    )
    db.add(db_report)
    db.commit()

    log_search(client_ip, report_id, body.search_terms, body.ai_query,
               len(documents), db_report.total_issues, db_report.critical_issues)

    response = {
        "report_id": report_id,
        "cached": False,
        "report": analysis.get("report"),
        "document_analyses": analysis.get("document_analyses"),
        "cross_reference_findings": analysis.get("cross_reference_findings"),
        "compliance_findings": analysis.get("compliance_findings"),
        "search_results": analysis.get("search_results"),
    }
    if analysis.get("warnings"):
        response["warnings"] = analysis["warnings"]
    return response


@app.get("/reports", dependencies=[Depends(verify_api_key)])
async def list_reports(limit: int = Query(default=20, le=100), db=Depends(get_db)):
    """List past analysis reports."""
    reports = (
        db.query(DBAnalysisReport)
        .order_by(DBAnalysisReport.analyzed_at.desc())
        .limit(limit)
        .all()
    )
    return {
        "reports": [
            {
                "id": r.id,
                "documents_analyzed": r.documents_analyzed,
                "total_issues": r.total_issues,
                "critical_issues": r.critical_issues,
                "risk_level": r.risk_level,
                "analyzed_at": r.analyzed_at.isoformat(),
            }
            for r in reports
        ]
    }


@app.get("/reports/{report_id}", dependencies=[Depends(verify_api_key)])
async def get_report(report_id: str, db=Depends(get_db)):
    """Get full details of a past analysis report."""
    report = db.query(DBAnalysisReport).filter(DBAnalysisReport.id == report_id).first()
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    full_data = json.loads(_decrypt_text(report.report_data))
    return {
        "id": report.id,
        "documents_analyzed": report.documents_analyzed,
        "total_issues": report.total_issues,
        "critical_issues": report.critical_issues,
        "risk_level": report.risk_level,
        "analyzed_at": report.analyzed_at.isoformat(),
        **full_data,
    }


# ---------------------------------------------------------------------------
# IsoIntel — P&ID Drawing Management
# ---------------------------------------------------------------------------

@app.post("/drawings/upload", dependencies=[Depends(verify_api_key)])
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


@app.post("/drawings/upload-batch", dependencies=[Depends(verify_api_key)])
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


@app.get("/drawings", dependencies=[Depends(verify_api_key)])
async def list_drawings(db=Depends(get_db)):
    """List all uploaded P&ID drawings."""
    drawings = db.query(DBDrawing).order_by(DBDrawing.uploaded_at.desc()).all()
    return {
        "drawings": [
            {
                "id": d.id,
                "filename": _decrypt_text(d.filename),
                "title": _decrypt_text(d.title) if d.title else None,
                "drawing_number": _decrypt_text(d.drawing_number) if d.drawing_number else None,
                "equipment_tags": _decrypt_text(d.equipment_tags) if d.equipment_tags else None,
                "description": _decrypt_text(d.description) if d.description else None,
                "page_count": d.page_count,
                "uploaded_at": d.uploaded_at.isoformat(),
            }
            for d in drawings
        ]
    }


@app.delete("/drawings/{drawing_id}", dependencies=[Depends(verify_api_key)])
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

@app.post("/isolations/generate", dependencies=[Depends(verify_api_key)])
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


@app.get("/isolations", dependencies=[Depends(verify_api_key)])
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
                "cert_number": _decrypt_text(p.cert_number),
                "equipment_tag": _decrypt_text(p.equipment_tag),
                "work_description": _decrypt_text(p.work_description),
                "work_type": _decrypt_text(p.work_type),
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


@app.get("/isolations/{isolation_id}", dependencies=[Depends(verify_api_key)])
async def get_isolation(isolation_id: str, db=Depends(get_db)):
    """Get the full isolation package by ID."""
    pkg = db.query(DBIsolationPackage).filter(DBIsolationPackage.id == isolation_id).first()
    if not pkg:
        raise HTTPException(status_code=404, detail="Isolation package not found")

    # Try to parse the package data as JSON
    raw_data = _decrypt_text(pkg.package_data)
    try:
        package_json = json.loads(raw_data)
    except (json.JSONDecodeError, ValueError):
        package_json = {"raw": raw_data}

    return {
        "id": pkg.id,
        "cert_number": _decrypt_text(pkg.cert_number),
        "equipment_tag": _decrypt_text(pkg.equipment_tag),
        "work_description": _decrypt_text(pkg.work_description),
        "work_type": _decrypt_text(pkg.work_type),
        "fluid_service": _decrypt_text(pkg.fluid_service) if pkg.fluid_service else None,
        "facility": _decrypt_text(pkg.facility) if pkg.facility else None,
        "regime": _decrypt_text(pkg.regime) if pkg.regime else None,
        "special_requirements": _decrypt_text(pkg.special_requirements) if pkg.special_requirements else None,
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


@app.delete("/isolations/{isolation_id}", dependencies=[Depends(verify_api_key)])
async def delete_isolation(isolation_id: str, db=Depends(get_db)):
    """Delete an isolation package."""
    pkg = db.query(DBIsolationPackage).filter(DBIsolationPackage.id == isolation_id).first()
    if not pkg:
        raise HTTPException(status_code=404, detail="Isolation package not found")
    db.delete(pkg)
    db.commit()
    return {"deleted": isolation_id}


# ---------------------------------------------------------------------------
# PDF Tools — Download, Merge, Split, Annotate
# ---------------------------------------------------------------------------

def _decrypt_and_load(filepath: Path) -> bytes:
    """Load and decrypt any encrypted file, returning raw bytes."""
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    if ENCRYPTION_KEY:
        from encryption import decrypt_file
        return decrypt_file(str(filepath))
    return filepath.read_bytes()


def _load_pdf_bytes(doc) -> bytes:
    """Load and decrypt a stored PDF, returning raw bytes."""
    filepath = Path(doc.filepath)
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="PDF file not found on disk")
    return _decrypt_and_load(filepath)


@app.get("/documents/{doc_id}/download", dependencies=[Depends(verify_api_key)])
async def download_document(doc_id: str, db=Depends(get_db)):
    """Download the original PDF file."""
    doc = db.query(DBDocument).filter(DBDocument.id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    pdf_bytes = _load_pdf_bytes(doc)
    filename = _decrypt_text(doc.filename)
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


class MergeRequest(BaseModel):
    doc_ids: list[str] = Field(..., min_length=2, description="IDs of documents to merge (in order)")
    output_filename: str = Field(default="merged.pdf", max_length=255)


@app.post("/documents/merge", dependencies=[Depends(verify_api_key)])
async def merge_documents(body: MergeRequest, request: Request, db=Depends(get_db)):
    """Merge multiple PDFs into a single new document."""
    import fitz  # PyMuPDF

    docs_db = []
    for did in body.doc_ids:
        doc = db.query(DBDocument).filter(DBDocument.id == did).first()
        if not doc:
            raise HTTPException(status_code=404, detail=f"Document {did} not found")
        docs_db.append(doc)

    merged = fitz.open()
    try:
        for doc in docs_db:
            pdf_bytes = _load_pdf_bytes(doc)
            src = fitz.open(stream=pdf_bytes, filetype="pdf")
            merged.insert_pdf(src)
            src.close()

        merged_bytes = merged.tobytes()
    finally:
        merged.close()

    # Save the merged PDF as a new document
    clean_name = _sanitize_filename(body.output_filename)
    if not clean_name.lower().endswith(".pdf"):
        clean_name += ".pdf"

    doc_id = str(uuid.uuid4())
    save_path = UPLOAD_DIR / f"{doc_id}.pdf.enc"
    _encrypt_and_save(merged_bytes, save_path)

    pages = extract_text_from_bytes(merged_bytes)
    content_hash = hashlib.sha256(merged_bytes).hexdigest()

    db_doc = DBDocument(
        id=doc_id,
        filename=_encrypt_text(clean_name),
        filepath=str(save_path),
        page_count=len(pages),
        text_content=_encrypt_text(json.dumps(pages)),
        content_hash=content_hash,
        uploaded_at=datetime.now(timezone.utc),
    )
    db.add(db_doc)
    db.commit()

    log_upload(_get_client_ip(request), clean_name, doc_id, len(pages))

    return {
        "id": doc_id,
        "filename": clean_name,
        "pages": len(pages),
        "merged_from": body.doc_ids,
    }


class SplitRequest(BaseModel):
    pages: list[int] = Field(..., min_length=1, description="Page numbers to extract (1-based)")
    output_filename: str = Field(default="split.pdf", max_length=255)


@app.post("/documents/{doc_id}/split", dependencies=[Depends(verify_api_key)])
async def split_document(doc_id: str, body: SplitRequest, request: Request, db=Depends(get_db)):
    """Extract specific pages from a PDF into a new document."""
    import fitz

    doc = db.query(DBDocument).filter(DBDocument.id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    pdf_bytes = _load_pdf_bytes(doc)
    src = fitz.open(stream=pdf_bytes, filetype="pdf")

    # Validate page numbers (convert 1-based to 0-based)
    total_pages = src.page_count
    zero_based = []
    for p in body.pages:
        if p < 1 or p > total_pages:
            src.close()
            raise HTTPException(
                status_code=400,
                detail=f"Page {p} out of range (document has {total_pages} pages)",
            )
        zero_based.append(p - 1)

    new_pdf = fitz.open()
    try:
        new_pdf.insert_pdf(src, from_page=-1, to_page=-1)  # empty
        for pg in zero_based:
            new_pdf.insert_pdf(src, from_page=pg, to_page=pg)
        split_bytes = new_pdf.tobytes()
    finally:
        new_pdf.close()
        src.close()

    clean_name = _sanitize_filename(body.output_filename)
    if not clean_name.lower().endswith(".pdf"):
        clean_name += ".pdf"

    new_id = str(uuid.uuid4())
    save_path = UPLOAD_DIR / f"{new_id}.pdf.enc"
    _encrypt_and_save(split_bytes, save_path)

    pages = extract_text_from_bytes(split_bytes)
    content_hash = hashlib.sha256(split_bytes).hexdigest()

    db_doc = DBDocument(
        id=new_id,
        filename=_encrypt_text(clean_name),
        filepath=str(save_path),
        page_count=len(pages),
        text_content=_encrypt_text(json.dumps(pages)),
        content_hash=content_hash,
        uploaded_at=datetime.now(timezone.utc),
    )
    db.add(db_doc)
    db.commit()

    log_upload(_get_client_ip(request), clean_name, new_id, len(pages))

    return {
        "id": new_id,
        "filename": clean_name,
        "pages": len(pages),
        "extracted_from": doc_id,
        "page_numbers": body.pages,
    }


@app.post("/documents/{doc_id}/annotate", dependencies=[Depends(verify_api_key)])
async def annotate_document(
    doc_id: str,
    request: Request,
    db=Depends(get_db),
    text: str = Form(..., description="Text to add"),
    page: int = Form(default=1, description="Page number (1-based)"),
    x: float = Form(default=72, description="X position in points from left"),
    y: float = Form(default=72, description="Y position in points from top"),
    font_size: float = Form(default=12, description="Font size"),
    color: str = Form(default="0,0,0", description="RGB color as 'r,g,b' (0-1 range)"),
    save_as_new: bool = Form(default=True, description="Save as new document instead of overwriting"),
    output_filename: str = Form(default="", description="Output filename (only if save_as_new)"),
):
    """Add text annotation/watermark to a PDF page."""
    import fitz

    doc = db.query(DBDocument).filter(DBDocument.id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    pdf_bytes = _load_pdf_bytes(doc)
    pdf = fitz.open(stream=pdf_bytes, filetype="pdf")

    if page < 1 or page > pdf.page_count:
        pdf.close()
        raise HTTPException(
            status_code=400,
            detail=f"Page {page} out of range (document has {pdf.page_count} pages)",
        )

    # Parse color
    try:
        rgb = tuple(float(c.strip()) for c in color.split(","))
        if len(rgb) != 3:
            raise ValueError
    except (ValueError, TypeError):
        pdf.close()
        raise HTTPException(status_code=400, detail="Color must be 'r,g,b' with values 0-1")

    pg = pdf[page - 1]
    pg.insert_text(
        fitz.Point(x, y),
        text,
        fontsize=font_size,
        color=rgb,
    )

    annotated_bytes = pdf.tobytes()
    pdf.close()

    client_ip = _get_client_ip(request)
    original_name = _decrypt_text(doc.filename)

    if save_as_new:
        clean_name = _sanitize_filename(output_filename or f"annotated_{original_name}")
        if not clean_name.lower().endswith(".pdf"):
            clean_name += ".pdf"
        new_id = str(uuid.uuid4())
        save_path = UPLOAD_DIR / f"{new_id}.pdf.enc"
        _encrypt_and_save(annotated_bytes, save_path)

        pages_data = extract_text_from_bytes(annotated_bytes)
        content_hash = hashlib.sha256(annotated_bytes).hexdigest()

        db_doc = DBDocument(
            id=new_id,
            filename=_encrypt_text(clean_name),
            filepath=str(save_path),
            page_count=len(pages_data),
            text_content=_encrypt_text(json.dumps(pages_data)),
            content_hash=content_hash,
            uploaded_at=datetime.now(timezone.utc),
        )
        db.add(db_doc)
        db.commit()
        log_upload(client_ip, clean_name, new_id, len(pages_data))
        return {"id": new_id, "filename": clean_name, "pages": len(pages_data)}
    else:
        # Overwrite existing document
        save_path = Path(doc.filepath)
        _encrypt_and_save(annotated_bytes, save_path)
        pages_data = extract_text_from_bytes(annotated_bytes)
        doc.page_count = len(pages_data)
        doc.text_content = _encrypt_text(json.dumps(pages_data))
        doc.content_hash = hashlib.sha256(annotated_bytes).hexdigest()
        db.commit()
        return {"id": doc_id, "filename": original_name, "pages": len(pages_data), "updated": True}
