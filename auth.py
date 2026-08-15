import hashlib
import secrets
from datetime import datetime, timezone, timedelta

from fastapi import HTTPException, Request

from config import API_KEY, JWT_SECRET, JWT_ALGORITHM, JWT_EXPIRY_HOURS
from database import SessionLocal, DBUser
from audit import log_auth_failure

_HASH_ITERATIONS = 260_000


def _hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    h = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), _HASH_ITERATIONS)
    return f"{salt}:{h.hex()}"


def _verify_password(password: str, stored_hash: str) -> bool:
    try:
        salt, hash_hex = stored_hash.split(":", 1)
        h = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), _HASH_ITERATIONS)
        return secrets.compare_digest(h.hex(), hash_hex)
    except (ValueError, AttributeError):
        return False


def _create_jwt(user_id: str, username: str, is_admin: bool = False) -> str:
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
    import jwt
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except (jwt.ExpiredSignatureError, jwt.InvalidTokenError):
        return None


def _get_client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


import os

async def verify_auth(request: Request):
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

    if auth.startswith("Bearer "):
        token = auth[7:]
        payload = _decode_jwt(token)
        if payload:
            request.state.user_id = payload.get("sub")
            request.state.username = payload.get("username")
            request.state.is_admin = payload.get("admin", False)
            return

    api_key_token = request.headers.get("X-API-Key", "")
    if API_KEY and api_key_token and secrets.compare_digest(api_key_token, API_KEY):
        request.state.user_id = None
        request.state.username = "api_key_user"
        request.state.is_admin = False
        return

    if API_KEY and auth.startswith("Bearer "):
        token = auth[7:]
        if secrets.compare_digest(token, API_KEY):
            request.state.user_id = None
            request.state.username = "api_key_user"
            request.state.is_admin = False
            return

    log_auth_failure(_get_client_ip(request), request.url.path)
    raise HTTPException(status_code=401, detail="Invalid or missing credentials")


verify_api_key = verify_auth


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
