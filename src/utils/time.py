"""Time parsing and range filtering helpers."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from dateutil import parser


def normalize_timestamp(value: object) -> Optional[datetime]:
    """Parse arbitrary timestamp-like input into a datetime."""
    if value is None:
        return None

    text = str(value).strip()
    if not text or text.lower() in {"nan", "nat", "none"}:
        return None

    try:
        return parser.parse(text)
    except (ValueError, TypeError, OverflowError):
        return None


def to_date_str(ts: Optional[datetime]) -> Optional[str]:
    """Return YYYY-MM-DD date string for datetime."""
    if ts is None:
        return None
    return ts.strftime("%Y-%m-%d")


def country_prefix_from_locode(locode: object) -> str:
    """Extract country prefix (first two chars) from LOCODE."""
    if locode is None:
        return "UNK"
    text = str(locode).strip().upper()
    if len(text) < 2:
        return "UNK"
    return text[:2]


def in_date_range(
    date_value: Optional[str], date_from: Optional[str], date_to: Optional[str]
) -> bool:
    """True when date_value is within inclusive range; None means no lower/upper bound."""
    if not date_from and not date_to:
        return True

    if not date_value:
        return False

    if date_from and date_value < date_from:
        return False
    if date_to and date_value > date_to:
        return False
    return True
