"""Shared fixtures for PDFHelper test suite.

Environment variables are set BEFORE importing the app so that config.py
and database.py pick up test-specific values at import time.
"""

import os
import pathlib

# ---------------------------------------------------------------------------
# Set test environment variables BEFORE any app imports
# ---------------------------------------------------------------------------
os.environ["PDF_HELPER_API_KEY"] = "test-api-key-12345"
os.environ["JWT_SECRET"] = "test-jwt-secret"
os.environ["ENVIRONMENT"] = "development"
os.environ["DATABASE_URL"] = "sqlite:////tmp/test_pdfhelper.db"

import pytest
import httpx

from app import app
from database import Base, engine

# ---------------------------------------------------------------------------
# Create all database tables before any test runs.  The httpx ASGITransport
# does not trigger ASGI lifespan events, so the startup migrations that the
# app normally runs never execute.  Creating tables here fills that gap.
# ---------------------------------------------------------------------------
Base.metadata.create_all(bind=engine)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def client():
    """Async HTTP client wired directly to the FastAPI app via ASGI transport."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.fixture
def api_headers():
    """Headers containing the test API key."""
    return {"X-API-Key": "test-api-key-12345"}


@pytest.fixture
async def auth_headers(client):
    """Create (or log in as) the test admin and return JWT + API-key headers."""
    r = await client.get("/setup-needed")
    if r.json().get("setup_needed"):
        r = await client.post(
            "/setup",
            json={"username": "testadmin", "password": "testpass123"},
        )
    else:
        r = await client.post(
            "/login",
            json={"username": "testadmin", "password": "testpass123"},
        )
    token = r.json()["token"]
    return {"Authorization": f"Bearer {token}", "X-API-Key": "test-api-key-12345"}


# ---------------------------------------------------------------------------
# Cleanup — remove the test database after the entire test run
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True, scope="session")
def _cleanup_test_db():
    """Remove the test SQLite database after all tests finish."""
    yield
    db_path = pathlib.Path("/tmp/test_pdfhelper.db")
    if db_path.exists():
        db_path.unlink()
