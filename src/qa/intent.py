"""Deterministic question intent classification and entity extraction (A-G taxonomy)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd


DOW_NAMES = [
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
]
DOW_TITLE = {name: name.title() for name in DOW_NAMES}
SPECIAL_DOW_NAMES = {"weekend": "Weekend", "weekday": "Weekday"}

UNSUPPORTED_KEYWORDS = (
    "crane",
    "berth utilization",
    "berth productivity",
    "teu",
    "throughput",
    "gate queue",
    "yard occupancy",
    "terminal gate",
    "container stack",
)

ANOMALY_KEYWORDS = (
    "anomaly",
    "unusual",
    "spike",
    "suspicious",
    "spoof",
    "jump",
    "impossible",
    "teleport",
)

FORECAST_KEYWORDS = (
    "forecast",
    "predict",
    "expected",
    "expect",
    "next",
    "coming",
    "future",
    "will",
)

CARBON_KEYWORDS = (
    "carbon",
    "emission",
    "emissions",
    "co2",
    "co2e",
    "nox",
    "sox",
    "pm",
    "tank-to-wake",
    "ttw",
    "well-to-wake",
    "wtw",
)

COMPARE_KEYWORDS = (
    "compare",
    "vs",
    "versus",
    "more than",
    "less than",
    "which port",
)

DIAGNOSTIC_KEYWORDS = (
    "why",
    "cause",
    "reason",
    "dominated",
    "contributing",
    "contributors",
    "breakdown",
)

TEMPORAL_PATTERN_KEYWORDS = (
    "busiest",
    "usually",
    "pattern",
    "seasonal",
    "weekday",
    "day-of-week",
    "hour",
)

DESCRIPTIVE_KEYWORDS = (
    "how many",
    "count",
    "top",
    "average",
    "mean",
    "median",
    "list",
    "show",
    "what is",
)

PORT_TOKEN_STOPWORDS = {
    "TTW",
    "WTW",
    "CO2",
    "CO2E",
    "NOX",
    "SOX",
    "PM",
    "CARBON",
    "EMISSION",
    "EMISSIONS",
    "POLLUTANT",
    "POLLUTANTS",
    "GHG",
    "AIS",
    "IMO",
    "MMSI",
    "ETA",
    "UTC",
    "CSV",
    "RAG",
    "NIS2",
    "ISPS",
    "BASED",
    "PATTERNS",
    "EXPECTED",
    "PREDICT",
    "PREDICTED",
    "CONGESTION",
    "LIKELY",
    "WHICH",
    "SHOW",
    "SUSPICIOUS",
    "JUMPS",
    "MOVEMENT",
    "CHANGES",
}
NON_PORT_CODE_TOKENS = {
    "TTW",
    "WTW",
    "CO2",
    "CO2E",
    "NOX",
    "SOX",
    "PM",
    "CARBON",
    "EMISSION",
    "EMISSIONS",
    "JANUARY",
    "FEBRUARY",
    "MARCH",
    "APRIL",
    "MAY",
    "JUNE",
    "JULY",
    "AUGUST",
    "SEPTEMBER",
    "OCTOBER",
    "NOVEMBER",
    "DECEMBER",
    "MONDAY",
    "TUESDAY",
    "WEDNESDAY",
    "THURSDAY",
    "FRIDAY",
    "SATURDAY",
    "SUNDAY",
}

LOCODE_RE = re.compile(r"\b([A-Z]{2})\s?([A-Z]{3})\b")
ISO_DATE_RE = re.compile(r"\b(20\d{2}-\d{2}-\d{2})\b")
MONTH_YEAR_RE = re.compile(
    r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+(20\d{2})\b",
    re.IGNORECASE,
)
YEAR_MONTH_RE = re.compile(r"\b(20\d{2})-(0[1-9]|1[0-2])\b")
LAST_WEEKS_RE = re.compile(r"\blast\s+(\d{1,2})\s+weeks?\b", re.IGNORECASE)
HORIZON_WEEKS_RE = re.compile(r"\b(\d{1,2})\s+weeks?\b", re.IGNORECASE)
MMSI_RE = re.compile(r"\bmmsi\s*[:#]?\s*(\d{6,9})\b", re.IGNORECASE)
IMO_RE = re.compile(r"\bimo\s*[:#]?\s*(\d{6,8})\b", re.IGNORECASE)
CALL_ID_RE = re.compile(r"\bcall[_\-\s]?id\s*[:=]?\s*([A-Za-z0-9_\-:.]+)\b", re.IGNORECASE)
LONG_DATE_RE = re.compile(
    r"\b(?:on\s+)?(?:(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\s*,?\s*)?"
    r"(january|february|march|april|may|june|july|august|september|october|november|december)\s+"
    r"(\d{1,2}),?\s+(20\d{2})\b",
    re.IGNORECASE,
)
RELATIVE_DOW_RE = re.compile(
    r"\b(next|this)\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    re.IGNORECASE,
)


@dataclass
class IntentResult:
    intent: str
    entities: Dict[str, Any]
    reason: str


def _extract_days_of_week(question: str) -> List[str]:
    q = question.lower()
    hits: List[tuple[int, str]] = []
    for day in DOW_NAMES:
        idx = q.find(day)
        if idx >= 0:
            hits.append((idx, DOW_TITLE[day]))
    hits.sort(key=lambda x: x[0])
    return [name for _, name in hits]


def _extract_special_dow(question: str) -> Optional[str]:
    q = question.lower()
    for token, label in SPECIAL_DOW_NAMES.items():
        if token in q:
            return label
    return None


def _extract_metric(question: str) -> str:
    q = question.lower()
    if "dwell" in q:
        return "dwell_minutes"
    if "congestion" in q or "busy" in q:
        return "congestion_index"
    if "occupancy" in q:
        return "occupancy_vessels"
    if "arrival" in q or "arrive" in q:
        return "arrivals_vessels"
    if "spike" in q or "anomaly" in q:
        return "arrivals_spike"
    return "arrivals_vessels"


def _extract_aggregation(question: str) -> Optional[str]:
    q = question.lower()
    if any(token in q for token in ("highest", "maximum", "max", "peak")) and any(
        token in q for token in ("day", "date")
    ):
        return "peak_day"
    return None


def _extract_vessel_type(question: str) -> Optional[str]:
    q = question.lower()
    if "tanker" in q:
        return "tanker"
    if "cargo" in q:
        return "cargo ship"
    if "container" in q:
        return "container ship"
    return None


def _extract_carbon_boundary(question: str) -> str:
    q = question.lower()
    if any(token in q for token in ("wtw", "well-to-wake", "well to wake", "lifecycle", "co2e")):
        return "WTW"
    return "TTW"


def _extract_carbon_pollutants(question: str) -> List[str]:
    q = question.lower()
    out: List[str] = []
    if "co2e" in q or "ghg" in q:
        out.append("CO2e")
    if re.search(r"\bco2\b", q) and "CO2" not in out:
        out.append("CO2")
    if "nox" in q:
        out.append("NOx")
    if "sox" in q or "sulfur" in q:
        out.append("SOx")
    if re.search(r"\bpm\b", q) or "particulate" in q:
        out.append("PM")
    if any(token in q for token in ("pollutant", "pollutants", "emissions")) and not out:
        out = ["CO2e", "NOx", "SOx", "PM"]
    if not out:
        out = ["CO2e", "NOx", "SOx", "PM"]
    return out


def _extract_ports(question: str) -> List[str]:
    ports: List[str] = []
    upper = question.upper()

    for c1, c2 in LOCODE_RE.findall(upper):
        locode = f"{c1}{c2}"
        if locode in NON_PORT_CODE_TOKENS:
            continue
        if locode not in PORT_TOKEN_STOPWORDS:
            ports.append(locode)

    upper_tokens = re.findall(r"\b[A-Z]{4,}\b", question)
    for token in upper_tokens:
        if token in PORT_TOKEN_STOPWORDS:
            continue
        if token in {"SHOW", "FIND", "BETWEEN", "DURING", "WEEKS", "FRIDAY", "MONDAY"}:
            continue
        if token.isdigit():
            continue
        # Keep likely destination/port identifiers like LUBECK, GDANSK, RIGA.
        if token not in ports:
            ports.append(token)

    # Phrase-based fallback for mixed-case names.
    phrase_hits = re.findall(
        r"\b(?:at|near|for|to)\s+([A-Za-z][A-Za-z\- ]{2,40})",
        question,
        flags=re.IGNORECASE,
    )
    for phrase in phrase_hits:
        cleaned = re.split(
            r"\b(?:between|from|on|during|next|last|this|in)\b",
            phrase,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0]
        cleaned = cleaned.strip(" ,.;:")
        if not cleaned:
            continue
        if cleaned.upper() in PORT_TOKEN_STOPWORDS:
            continue
        if cleaned.upper() not in ports:
            ports.append(cleaned)

    in_caps_hits = re.findall(r"\bin\s+([A-Z]{4,})\b", question)
    for token in in_caps_hits:
        if token in PORT_TOKEN_STOPWORDS:
            continue
        if token not in ports:
            ports.append(token)

    # Keep first two/three most likely candidates to avoid noise.
    deduped: List[str] = []
    for port in ports:
        if port not in deduped:
            deduped.append(port)
    return deduped[:4]


def _month_start_end(month_name: str, year: int) -> tuple[str, str]:
    ts = pd.Timestamp(year=year, month=pd.Timestamp(month_name).month, day=1)
    month_end = ts + pd.offsets.MonthEnd(0)
    return ts.strftime("%Y-%m-%d"), month_end.strftime("%Y-%m-%d")


def _extract_date_range(question: str) -> tuple[Optional[str], Optional[str], Optional[str]]:
    matches = ISO_DATE_RE.findall(question)
    if len(matches) >= 2:
        start, end = matches[0], matches[1]
        if start > end:
            start, end = end, start
        return start, end, None
    if len(matches) == 1:
        return matches[0], matches[0], None

    month_match = MONTH_YEAR_RE.search(question)
    if month_match:
        month = month_match.group(1)
        year = int(month_match.group(2))
        start, end = _month_start_end(month, year)
        return start, end, None

    ym_match = YEAR_MONTH_RE.search(question)
    if ym_match:
        year = int(ym_match.group(1))
        month = int(ym_match.group(2))
        start = pd.Timestamp(year=year, month=month, day=1)
        end = start + pd.offsets.MonthEnd(0)
        return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), None

    last_weeks_match = LAST_WEEKS_RE.search(question)
    if last_weeks_match:
        weeks = int(last_weeks_match.group(1))
        return None, None, f"last_{weeks}_weeks"

    return None, None, None


def _extract_horizon_weeks(question: str) -> int:
    m = HORIZON_WEEKS_RE.search(question.lower())
    if not m:
        return 4
    value = int(m.group(1))
    return max(1, min(12, value))


def _extract_target_date(question: str) -> Optional[str]:
    # Explicit ISO date like "on 2026-02-20".
    iso_matches = ISO_DATE_RE.findall(question)
    if len(iso_matches) == 1:
        return iso_matches[0]

    # Explicit long date like "on Friday, February 20, 2026".
    long_hit = LONG_DATE_RE.search(question)
    if long_hit:
        month = long_hit.group(2)
        day = long_hit.group(3)
        year = long_hit.group(4)
        ts = pd.to_datetime(f"{month} {day} {year}", errors="coerce")
        if pd.notna(ts):
            return pd.Timestamp(ts).strftime("%Y-%m-%d")

    # Relative weekday like "next Friday" or "this Friday".
    rel_hit = RELATIVE_DOW_RE.search(question)
    if rel_hit:
        mode = rel_hit.group(1).lower()
        day = rel_hit.group(2).lower()
        now = pd.Timestamp.now().floor("D")
        target_idx = DOW_NAMES.index(day)
        delta = (target_idx - now.weekday()) % 7
        if mode == "next":
            if delta == 0:
                delta = 7
        else:  # "this"
            if delta == 0:
                delta = 0
        target = now + pd.Timedelta(days=delta)
        return target.strftime("%Y-%m-%d")

    return None


def classify_question(question: str) -> IntentResult:
    q = question.lower()

    start_date, end_date, window = _extract_date_range(question)
    target_date = _extract_target_date(question)
    if target_date and not start_date and not end_date:
        start_date = target_date
        end_date = target_date
    dows = _extract_days_of_week(question)
    special_dow = _extract_special_dow(question)
    entities: Dict[str, Any] = {
        "ports": _extract_ports(question),
        "port": None,
        "date_from": start_date,
        "date_to": end_date,
        "target_date": target_date,
        "window": window,
        "vessel_type": _extract_vessel_type(question),
        "dow": dows[0] if dows else special_dow,
        "dow_compare": dows[1] if len(dows) > 1 else None,
        "metric": _extract_metric(question),
        "aggregation": _extract_aggregation(question),
        "horizon_weeks": _extract_horizon_weeks(question),
        "mmsi": None,
        "imo": None,
        "call_id": None,
        "boundary": _extract_carbon_boundary(question),
        "pollutants": _extract_carbon_pollutants(question),
    }

    if entities["ports"]:
        entities["port"] = entities["ports"][0]

    mmsi_hit = MMSI_RE.search(question)
    if mmsi_hit:
        entities["mmsi"] = mmsi_hit.group(1)
    imo_hit = IMO_RE.search(question)
    if imo_hit:
        entities["imo"] = imo_hit.group(1)
    call_hit = CALL_ID_RE.search(question)
    if call_hit:
        entities["call_id"] = call_hit.group(1)

    if any(token in q for token in UNSUPPORTED_KEYWORDS):
        return IntentResult(
            intent="G",
            entities=entities,
            reason="Requested metric requires terminal operational data outside AIS/port-call scope.",
        )

    if any(token in q for token in CARBON_KEYWORDS):
        return IntentResult(
            intent="H",
            entities=entities,
            reason="Carbon/emissions inventory request detected.",
        )

    if any(token in q for token in ANOMALY_KEYWORDS):
        return IntentResult(intent="F", entities=entities, reason="Anomaly/suspicious pattern request.")

    if any(token in q for token in FORECAST_KEYWORDS):
        return IntentResult(intent="C", entities=entities, reason="Forecasting language detected.")

    if any(token in q for token in COMPARE_KEYWORDS):
        return IntentResult(intent="D", entities=entities, reason="Comparative phrasing detected.")

    if any(token in q for token in DIAGNOSTIC_KEYWORDS):
        return IntentResult(intent="E", entities=entities, reason="Diagnostic/explanatory phrasing detected.")

    if any(token in q for token in TEMPORAL_PATTERN_KEYWORDS):
        return IntentResult(intent="B", entities=entities, reason="Temporal pattern question detected.")

    if any(token in q for token in DESCRIPTIVE_KEYWORDS):
        return IntentResult(intent="A", entities=entities, reason="Descriptive aggregation request.")

    return IntentResult(intent="A", entities=entities, reason="Defaulting to descriptive analytics.")


def describe_intent(intent: str) -> str:
    names = {
        "A": "Descriptive",
        "B": "Temporal Pattern",
        "C": "Forecasting",
        "D": "Comparative",
        "E": "Diagnostic",
        "F": "Anomaly",
        "G": "Unsupported",
        "H": "Carbon Inventory",
    }
    return names.get(intent, "Unknown")


def required_data_for_intent(intent: str) -> List[str]:
    mapping = {
        "A": ["arrivals_daily.parquet"],
        "B": ["arrivals_daily.parquet", "arrivals_hourly.parquet"],
        "C": ["arrivals_daily.parquet", "congestion_daily.parquet"],
        "D": ["arrivals_daily.parquet", "congestion_daily.parquet"],
        "E": ["arrivals_daily.parquet", "dwell_time.parquet"],
        "F": ["arrivals_daily.parquet"],
        "G": [],
        "H": [
            "carbon_segments.parquet",
            "carbon_emissions_segment.parquet",
            "carbon_emissions_daily_port.parquet",
            "carbon_emissions_call.parquet",
            "carbon_evidence.parquet",
            "carbon_params_version.json",
        ],
    }
    return mapping.get(intent, [])
