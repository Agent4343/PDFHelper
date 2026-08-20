"""Core API endpoint tests for PDFHelper."""

import io

import pytest


# ---------------------------------------------------------------------------
# Minimal valid PDF used by upload tests
# ---------------------------------------------------------------------------

_MINIMAL_PDF = (
    b"%PDF-1.4\n"
    b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
    b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
    b"3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R>>endobj\n"
    b"xref\n"
    b"0 4\n"
    b"0000000000 65535 f \n"
    b"0000000009 00000 n \n"
    b"0000000058 00000 n \n"
    b"0000000115 00000 n \n"
    b"trailer<</Size 4/Root 1 0 R>>\n"
    b"startxref\n"
    b"190\n"
    b"%%EOF"
)


# ---------------------------------------------------------------------------
# Health & root
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_health(client):
    """GET /health returns 200 with a status field."""
    r = await client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert "status" in data


@pytest.mark.anyio
async def test_root_returns_html(client):
    """GET / returns 200 with HTML content."""
    r = await client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers.get("content-type", "")


# ---------------------------------------------------------------------------
# API-key verification
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_verify_key(client, api_headers):
    """GET /verify-key with a valid API key returns {valid: true}."""
    r = await client.get("/verify-key", headers=api_headers)
    assert r.status_code == 200
    assert r.json()["valid"] is True


@pytest.mark.anyio
async def test_verify_key_no_auth(client):
    """GET /verify-key without credentials returns 401."""
    r = await client.get("/verify-key")
    assert r.status_code == 401


# ---------------------------------------------------------------------------
# Setup & login
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_setup_needed(client):
    """GET /setup-needed returns 200."""
    r = await client.get("/setup-needed")
    assert r.status_code == 200
    assert "setup_needed" in r.json()


@pytest.mark.anyio
async def test_setup_creates_admin(client):
    """POST /setup with username/password creates an admin and returns a token."""
    # Ensure a clean slate — this test must run before test_setup_already_done
    r = await client.get("/setup-needed")
    if not r.json().get("setup_needed"):
        pytest.skip("Admin already exists (test ordering)")

    r = await client.post(
        "/setup",
        json={"username": "testadmin", "password": "testpass123"},
    )
    assert r.status_code == 200
    data = r.json()
    assert "token" in data
    assert data["username"] == "testadmin"


@pytest.mark.anyio
async def test_setup_already_done(client, auth_headers):
    """POST /setup when an admin already exists returns 404 (hidden)."""
    r = await client.post(
        "/setup",
        json={"username": "anotheradmin", "password": "password1234"},
    )
    assert r.status_code == 404


@pytest.mark.anyio
async def test_login(client, auth_headers):
    """POST /login with correct credentials returns a token."""
    # auth_headers fixture guarantees the user exists
    r = await client.post(
        "/login",
        json={"username": "testadmin", "password": "testpass123"},
    )
    assert r.status_code == 200
    assert "token" in r.json()


@pytest.mark.anyio
async def test_login_wrong_password(client, auth_headers):
    """POST /login with a wrong password returns 401."""
    r = await client.post(
        "/login",
        json={"username": "testadmin", "password": "wrongpassword"},
    )
    assert r.status_code == 401


# ---------------------------------------------------------------------------
# /me
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_me_endpoint(client, auth_headers):
    """GET /me with a valid JWT returns user info."""
    r = await client.get("/me", headers=auth_headers)
    assert r.status_code == 200
    data = r.json()
    assert "username" in data
    assert "user_id" in data


# ---------------------------------------------------------------------------
# Document CRUD
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_upload_pdf(client, api_headers):
    """POST /upload with a minimal PDF returns 200 with a document ID."""
    files = [("files", ("test.pdf", io.BytesIO(_MINIMAL_PDF), "application/pdf"))]
    r = await client.post("/upload", headers=api_headers, files=files)
    assert r.status_code == 200
    data = r.json()
    assert data["count"] >= 1
    assert "id" in data["uploaded"][0]


@pytest.mark.anyio
async def test_list_documents(client, api_headers):
    """GET /documents returns a list with a total field."""
    r = await client.get("/documents", headers=api_headers)
    assert r.status_code == 200
    data = r.json()
    assert "documents" in data
    assert "total" in data


@pytest.mark.anyio
async def test_list_documents_pagination(client, api_headers):
    """GET /documents with skip and limit query params works."""
    r = await client.get("/documents?skip=0&limit=10", headers=api_headers)
    assert r.status_code == 200
    data = r.json()
    assert "documents" in data
    assert "total" in data


@pytest.mark.anyio
async def test_delete_document(client, api_headers):
    """DELETE /documents/{doc_id} removes the document and returns 200."""
    # Upload a document first so we have something to delete
    files = [("files", ("deleteme.pdf", io.BytesIO(_MINIMAL_PDF), "application/pdf"))]
    upload_r = await client.post("/upload", headers=api_headers, files=files)
    assert upload_r.status_code == 200
    doc_id = upload_r.json()["uploaded"][0]["id"]

    r = await client.delete(f"/documents/{doc_id}", headers=api_headers)
    assert r.status_code == 200
    assert r.json()["deleted"] == doc_id


# ---------------------------------------------------------------------------
# Chat auth gate
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_chat_requires_auth(client):
    """POST /chat without credentials returns 401."""
    r = await client.post("/chat", json={"message": "hello"})
    assert r.status_code == 401
