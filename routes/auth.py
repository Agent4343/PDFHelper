import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request

from auth import verify_auth, verify_api_key, get_db, _hash_password, _verify_password, _create_jwt
from config import ALLOW_REGISTRATION
from database import DBUser
from models import RegisterRequest, LoginRequest

router = APIRouter()


# ---------------------------------------------------------------------------
# User registration & login
# ---------------------------------------------------------------------------

@router.get("/setup-needed")
async def setup_needed(db=Depends(get_db)):
    """Check if initial setup is required (no users exist yet)."""
    any_user = db.query(DBUser).first()
    return {"setup_needed": any_user is None}


@router.post("/setup")
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


@router.post("/register")
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


@router.post("/login")
async def login(body: LoginRequest, db=Depends(get_db)):
    """Authenticate and receive a JWT token."""
    user = db.query(DBUser).filter(DBUser.username == body.username).first()
    if not user or not _verify_password(body.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid username or password")

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
