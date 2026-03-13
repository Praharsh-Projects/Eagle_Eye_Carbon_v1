"""Runtime environment checks and lazy imports."""

from __future__ import annotations

from contextlib import contextmanager
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple


def ensure_supported_python() -> None:
    if sys.version_info >= (3, 14):
        raise RuntimeError(
            "Python 3.14 is not supported by Chroma in this setup. "
            "Use Python 3.11 or 3.12."
        )


def import_chromadb() -> Any:
    ensure_supported_python()
    try:
        import chromadb
    except Exception as exc:
        raise RuntimeError(
            "Failed to import chromadb. Ensure dependencies are installed in a Python 3.11/3.12 venv."
        ) from exc
    return chromadb


def _as_bool(value: Optional[str], default: bool = False) -> bool:
    if value is None:
        return default
    token = str(value).strip().lower()
    if token in {"1", "true", "yes", "y", "on"}:
        return True
    if token in {"0", "false", "no", "n", "off"}:
        return False
    return default


REMOTE_VECTOR_ENV_NAMES = (
    "VECTOR_DB_MODE",
    "CHROMA_HOST",
    "CHROMA_PORT",
    "CHROMA_SSL",
    "CHROMA_TENANT",
    "CHROMA_DATABASE",
    "CHROMA_AUTH_TOKEN",
    "CHROMA_AUTH_HEADER",
)


@contextmanager
def force_local_vector_env() -> Iterator[None]:
    """
    Temporarily disable remote vector settings so Chroma opens a local persistent index.
    Useful when a remote Chroma endpoint is configured but unreachable and a local bundle exists.
    """
    previous = {name: os.environ.get(name) for name in REMOTE_VECTOR_ENV_NAMES}
    try:
        os.environ["VECTOR_DB_MODE"] = "local"
        for name in REMOTE_VECTOR_ENV_NAMES:
            if name == "VECTOR_DB_MODE":
                continue
            os.environ.pop(name, None)
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def chroma_remote_settings(config: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    cfg = (config or {}).get("vector_db", {}) if isinstance(config, dict) else {}
    mode = str(os.getenv("VECTOR_DB_MODE", cfg.get("mode", "local"))).strip().lower()
    host = str(os.getenv("CHROMA_HOST", cfg.get("chroma_host", ""))).strip()
    if mode not in {"remote", "http"} and not host:
        return None

    port_raw = os.getenv("CHROMA_PORT", cfg.get("chroma_port", 8000))
    try:
        port = int(port_raw)
    except Exception:
        port = 8000

    ssl_raw = os.getenv("CHROMA_SSL", cfg.get("chroma_ssl", False))
    ssl = _as_bool(str(ssl_raw), default=False)

    tenant = str(os.getenv("CHROMA_TENANT", cfg.get("chroma_tenant", "default_tenant"))).strip()
    database = str(os.getenv("CHROMA_DATABASE", cfg.get("chroma_database", "default_database"))).strip()

    auth_token = str(
        os.getenv("CHROMA_AUTH_TOKEN", cfg.get("chroma_auth_token", ""))
    ).strip()
    auth_header = str(
        os.getenv("CHROMA_AUTH_HEADER", cfg.get("chroma_auth_header", "Authorization"))
    ).strip() or "Authorization"

    headers: Dict[str, str] = {}
    if auth_token:
        token_value = auth_token
        if auth_header.lower() == "authorization" and not auth_token.lower().startswith("bearer "):
            token_value = f"Bearer {auth_token}"
        headers[auth_header] = token_value

    if not host:
        raise RuntimeError(
            "VECTOR_DB_MODE is remote but CHROMA_HOST is not set. "
            "Set CHROMA_HOST (and optionally CHROMA_PORT/CHROMA_SSL)."
        )
    if host == "YOUR_CHROMA_HOST":
        raise RuntimeError(
            "CHROMA_HOST is still the placeholder 'YOUR_CHROMA_HOST'. "
            "Set it to your real Chroma server hostname."
        )

    return {
        "host": host,
        "port": port,
        "ssl": ssl,
        "tenant": tenant or "default_tenant",
        "database": database or "default_database",
        "headers": headers or None,
    }


def create_chroma_client(
    chromadb: Any,
    persist_dir: str | Path,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[Any, str]:
    remote = chroma_remote_settings(config=config)
    if remote:
        client = chromadb.HttpClient(
            host=remote["host"],
            port=int(remote["port"]),
            ssl=bool(remote["ssl"]),
            headers=remote.get("headers"),
            tenant=str(remote.get("tenant", "default_tenant")),
            database=str(remote.get("database", "default_database")),
        )
        return client, "remote_http"

    client = chromadb.PersistentClient(path=str(Path(persist_dir)))
    return client, "local_persist"


def require_openai_api_key() -> str:
    """
    Return OPENAI_API_KEY if present, otherwise raise a clear actionable error.
    """
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Run: export OPENAI_API_KEY='your_key' "
            "before running build/index/app commands."
        )
    return api_key
