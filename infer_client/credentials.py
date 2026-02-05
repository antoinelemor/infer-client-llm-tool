"""Secure local credential storage for infer-client.

Saves API URL and key to ~/.infer_client/credentials.json
with 0o600 file permissions (owner read/write only).
"""

import json
import os
from pathlib import Path
from typing import Optional, Tuple

_CONFIG_DIR = Path.home() / ".infer_client"
_CRED_FILE = _CONFIG_DIR / "credentials.json"


def _ensure_dir() -> None:
    _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    os.chmod(_CONFIG_DIR, 0o700)


def save(url: str, key: str) -> Path:
    """Persist URL and API key to disk."""
    _ensure_dir()
    data = {"url": url, "key": key}
    _CRED_FILE.write_text(json.dumps(data, indent=2) + "\n")
    os.chmod(_CRED_FILE, 0o600)
    return _CRED_FILE


def load() -> Tuple[Optional[str], Optional[str]]:
    """Return (url, key) from saved credentials, or (None, None)."""
    if not _CRED_FILE.exists():
        return None, None
    try:
        data = json.loads(_CRED_FILE.read_text())
        return data.get("url"), data.get("key")
    except (json.JSONDecodeError, KeyError):
        return None, None


def clear() -> bool:
    """Remove saved credentials. Returns True if file existed."""
    if _CRED_FILE.exists():
        _CRED_FILE.unlink()
        return True
    return False


def path() -> Path:
    return _CRED_FILE
