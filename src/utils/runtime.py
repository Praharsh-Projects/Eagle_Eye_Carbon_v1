"""Runtime environment checks and lazy imports."""

from __future__ import annotations

import os
import sys
from typing import Any


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
