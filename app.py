#!/usr/bin/env python3
"""
PDFHelper — Secure web API for AI-powered PDF search and flagging.

Deployed on Railway. This module creates the FastAPI app, configures
middleware, and mounts all route modules.
"""

import asyncio
import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.gzip import GZipMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from config import (
    ENVIRONMENT, IS_PRODUCTION, API_KEY, ENCRYPTION_KEY,
    ADMIN_USERNAME, ADMIN_PASSWORD,
)
from database import SessionLocal, engine, Base, DBUser
from auth import _hash_password, _get_client_ip
from audit import log_access
from models import HealthResponse

from routes.auth import router as auth_router
from routes.documents import router as documents_router
from routes.chat import router as chat_router
from routes.code_chat import router as code_chat_router
from routes.agents import router as agents_router
from routes.drawings import router as drawings_router
from routes.updater import router as updater_router
from routes.posters import router as posters_router
from routes.procedures import router as procedures_router

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

_startup_errors: list[str] = []


def _run_migrations(logger):
    try:
        from alembic.config import Config
        from alembic import command
        cfg = Config("alembic.ini")
        command.upgrade(cfg, "head")
        logger.info("Alembic migrations applied successfully.")
    except Exception as exc:
        logger.warning("Alembic migration failed: %s — falling back to create_all", exc)
        Base.metadata.create_all(bind=engine)


async def _retry_db_init(logger):
    import asyncio
    for attempt in range(2, 6):
        await asyncio.sleep(2 ** attempt)
        try:
            _run_migrations(logger)
            return
        except Exception as exc:
            logger.warning("Database init attempt %d/5 failed: %s", attempt, exc)
    _startup_errors.append("WARNING: Database initialization failed after 5 attempts.")


@asynccontextmanager
async def lifespan(app: FastAPI):
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

    try:
        _run_migrations(logger)
    except Exception as exc:
        logger.warning("Database init failed on first attempt: %s — retrying in background", exc)
        asyncio.create_task(_retry_db_init(logger))

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
            elif not existing.is_admin:
                existing.is_admin = True
                db.commit()
                logger.info("Existing account '%s' promoted to admin.", ADMIN_USERNAME)
            db.close()
        except Exception as exc:
            logger.warning("Failed to auto-create admin account: %s", exc)

    yield


app = FastAPI(
    title="PDFHelper",
    description="AI-powered PDF search and flagging tool",
    version="1.0.0",
    docs_url="/docs" if not IS_PRODUCTION else None,
    redoc_url=None,
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# Compression
# ---------------------------------------------------------------------------

app.add_middleware(GZipMiddleware, minimum_size=500)

# ---------------------------------------------------------------------------
# Security middleware
# ---------------------------------------------------------------------------

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "").split(",")
ALLOWED_ORIGINS = [o.strip() for o in ALLOWED_ORIGINS if o.strip()]
if IS_PRODUCTION and not ALLOWED_ORIGINS:
    ALLOWED_ORIGINS = []

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS if ALLOWED_ORIGINS else (["*"] if not IS_PRODUCTION else []),
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Authorization", "X-API-Key", "Content-Type"],
)

ALLOWED_HOSTS = os.getenv("ALLOWED_HOSTS", "").split(",")
ALLOWED_HOSTS = [h.strip() for h in ALLOWED_HOSTS if h.strip()]
if ALLOWED_HOSTS:
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=ALLOWED_HOSTS)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        if IS_PRODUCTION:
            response.headers["Strict-Transport-Security"] = "max-age=63072000; includeSubDomains"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' https://cdnjs.cloudflare.com; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: blob:; "
            "connect-src 'self'; "
            "font-src 'self'; "
            "frame-ancestors 'none'"
        )
        if "server" in response.headers:
            del response.headers["server"]
        return response


app.add_middleware(SecurityHeadersMiddleware)


class HTTPSRedirectMiddleware(BaseHTTPMiddleware):
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


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, max_requests: int = 60, window_seconds: int = 60):
        super().__init__(app)
        self.max_requests = max_requests
        self.window = window_seconds
        self.requests: dict[str, list[float]] = {}
        self._last_cleanup = time.time()
        self._cleanup_interval = 300
        self._lock = asyncio.Lock()
        self._auth_paths = {"/login", "/register", "/setup"}
        self._auth_max = 10
        self._auth_requests: dict[str, list[float]] = {}

    async def dispatch(self, request: Request, call_next):
        client_ip = _get_client_ip(request)
        now = time.time()
        window_start = now - self.window

        async with self._lock:
            if now - self._last_cleanup > self._cleanup_interval:
                self.requests = {
                    ip: [t for t in ts if t > window_start]
                    for ip, ts in self.requests.items()
                    if any(t > window_start for t in ts)
                }
                self._auth_requests = {
                    ip: [t for t in ts if t > window_start]
                    for ip, ts in self._auth_requests.items()
                    if any(t > window_start for t in ts)
                }
                self._last_cleanup = now

            if request.url.path in self._auth_paths and request.method == "POST":
                auth_hits = self._auth_requests.get(client_ip, [])
                auth_hits = [t for t in auth_hits if t > window_start]
                if len(auth_hits) >= self._auth_max:
                    return JSONResponse(
                        status_code=429,
                        content={"detail": "Too many authentication attempts. Try again later."},
                    )
                auth_hits.append(now)
                self._auth_requests[client_ip] = auth_hits

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


_rate_limit = 10000 if ENVIRONMENT == "development" else 60
app.add_middleware(RateLimitMiddleware, max_requests=_rate_limit, window_seconds=60)

# ---------------------------------------------------------------------------
# Root UI
# ---------------------------------------------------------------------------

_TEMPLATE_DIR = Path(__file__).parent / "templates"
_cached_html: str | None = None


def _load_root_html() -> str:
    global _cached_html
    if IS_PRODUCTION and _cached_html is not None:
        return _cached_html
    html = (_TEMPLATE_DIR / "index.html").read_text(encoding="utf-8")
    if IS_PRODUCTION:
        _cached_html = html
    return html


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------

from sqlalchemy.exc import OperationalError as SAOperationalError


@app.exception_handler(SAOperationalError)
async def db_error_handler(request: Request, exc: SAOperationalError):
    return JSONResponse(
        status_code=503,
        content={
            "detail": "Database is unavailable. Check that DATABASE_URL is set correctly "
                      "(use the public URL, not the internal .railway.internal hostname)."
        },
    )


@app.exception_handler(Exception)
async def general_error_handler(request: Request, exc: Exception):
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
# Core endpoints (health, root, verify-key)
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
async def health_check():
    db_ok = True
    db_err = None
    try:
        from sqlalchemy import text
        db = SessionLocal()
        db.execute(text("SELECT 1"))
        db.close()
    except Exception as e:
        db_ok = False
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


from auth import verify_api_key
from fastapi import Depends


@app.get("/verify-key", dependencies=[Depends(verify_api_key)])
async def verify_key():
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
    return _load_root_html()


@app.get("/bot")
async def bot_page():
    from starlette.responses import RedirectResponse
    return RedirectResponse(url="/", status_code=301)


# ---------------------------------------------------------------------------
# Mount routers
# ---------------------------------------------------------------------------

app.include_router(auth_router)
app.include_router(documents_router)
app.include_router(chat_router)
app.include_router(code_chat_router)
app.include_router(agents_router)
app.include_router(drawings_router)
app.include_router(updater_router)
app.include_router(posters_router)
app.include_router(procedures_router)
