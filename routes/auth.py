import uuid
import time
import threading
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request

from auth import verify_auth, verify_api_key, get_db, _hash_password, _verify_password, _create_jwt, _get_client_ip
from config import ALLOW_REGISTRATION
from database import DBUser
from models import RegisterRequest, LoginRequest
from audit import audit_log

router = APIRouter()

_LOGIN_WINDOW = 900
_MAX_LOGIN_ATTEMPTS = 5
_login_attempts: dict[str, list[float]] = {}
_login_lock = threading.Lock()


def _check_login_rate(ip: str):
    now = time.monotonic()
    with _login_lock:
        attempts = _login_attempts.get(ip, [])
        attempts = [t for t in attempts if t > now - _LOGIN_WINDOW]
        _login_attempts[ip] = attempts
        if len(attempts) >= _MAX_LOGIN_ATTEMPTS:
            audit_log.warning("LOGIN_BLOCKED | ip=%s | reason=rate_limit", ip)
            raise HTTPException(
                status_code=429,
                detail="Too many login attempts. Try again in 15 minutes.",
            )


def _record_failed_login(ip: str):
    with _login_lock:
        _login_attempts.setdefault(ip, []).append(time.monotonic())


def _clear_login_attempts(ip: str):
    with _login_lock:
        _login_attempts.pop(ip, None)


# ---------------------------------------------------------------------------
# User registration & login
# ---------------------------------------------------------------------------

@router.get("/setup-needed")
async def setup_needed(db=Depends(get_db)):
    any_user = db.query(DBUser).first()
    return {"setup_needed": any_user is None}


@router.post("/setup")
async def setup(body: RegisterRequest, request: Request, db=Depends(get_db)):
    any_user = db.query(DBUser).first()
    if any_user:
        raise HTTPException(status_code=404)

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


@router.post("/register")
async def register(body: RegisterRequest, db=Depends(get_db)):
    raise HTTPException(status_code=404)


@router.post("/login")
async def login(body: LoginRequest, request: Request, db=Depends(get_db)):
    ip = _get_client_ip(request)
    _check_login_rate(ip)

    user = db.query(DBUser).filter(DBUser.username == body.username).first()
    if not user or not _verify_password(body.password, user.password_hash):
        _record_failed_login(ip)
        audit_log.warning("LOGIN_FAILED | ip=%s | username=%s", ip, body.username)
        raise HTTPException(status_code=401, detail="Invalid username or password")

    _clear_login_attempts(ip)
    token = _create_jwt(user.id, user.username, user.is_admin)
    return {"user_id": user.id, "username": user.username, "token": token}


@router.get("/me", dependencies=[Depends(verify_auth)])
async def get_current_user(request: Request):
    """Return the currently authenticated user's info."""
    return {
        "user_id": getattr(request.state, "user_id", None),
        "username": getattr(request.state, "username", None),
        "is_admin": getattr(request.state, "is_admin", False),
    }
