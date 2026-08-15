import os
import secrets
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
IS_PRODUCTION = ENVIRONMENT == "production"

API_KEY = (os.getenv("PDF_HELPER_API_KEY") or "").strip() or None
ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY")

MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE_MB", "20")) * 1024 * 1024
MAX_FILES_PER_REQUEST = int(os.getenv("MAX_FILES_PER_REQUEST", "20"))
CHAT_MODEL = os.getenv("CHAT_MODEL", "claude-sonnet-5")
CHAT_MAX_TOKENS = int(os.getenv("CHAT_MAX_TOKENS", "32000"))
AGENT_MAX_TOKENS = int(os.getenv("AGENT_MAX_TOKENS", "12000"))
CHAT_WEB_SEARCH = os.getenv("CHAT_WEB_SEARCH", "false").lower() == "true"
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "/tmp/pdfhelper_uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

AUTO_CLEANUP_HOURS = int(os.getenv("AUTO_CLEANUP_HOURS", "72"))

JWT_SECRET = os.getenv("JWT_SECRET", "").strip() or secrets.token_hex(32)
JWT_ALGORITHM = "HS256"
JWT_EXPIRY_HOURS = int(os.getenv("JWT_EXPIRY_HOURS", "24"))
ALLOW_REGISTRATION = os.getenv("ALLOW_REGISTRATION", "false").lower() == "true"
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "").strip()
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "").strip()

AGENT_MODELS = {
    "sonnet": "claude-sonnet-5",
    "haiku": "claude-haiku-4-5",
}
