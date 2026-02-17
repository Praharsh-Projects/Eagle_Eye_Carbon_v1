"""Serialization helpers for traffic rows (AIS + port-calls) and evidence snippets."""

from __future__ import annotations

import re
from typing import Any, Dict, Optional, Tuple

from .time import country_prefix_from_locode, normalize_timestamp, to_date_str


def _clean(value: Any, fallback: str = "unknown") -> str:
    if value is None:
        return fallback
    text = str(value).strip()
    if not text or text.lower() in {"nan", "nat", "none"}:
        return fallback
    return text


def _pick(row: Dict[str, Any], *names: str, default: str = "unknown") -> str:
    for name in names:
        if name in row:
            cleaned = _clean(row.get(name))
            if cleaned != "unknown":
                return cleaned
    return default


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return None
    try:
        return float(text)
    except ValueError:
        try:
            return float(text.replace(",", "."))
        except ValueError:
            return None


def normalize_destination(destination: str) -> str:
    return re.sub(r"\s+", " ", destination.strip().upper())


def normalize_identifier(identifier: str) -> str:
    text = identifier.strip()
    if re.match(r"^\d+\.0+$", text):
        return text.split(".")[0]
    return text


def _sanitize_id(value: str) -> str:
    return (
        value.replace(" ", "_")
        .replace(":", "-")
        .replace("/", "_")
        .replace("+", "_")
    )


def _date_from_timestamp(raw_timestamp: str) -> str:
    ts = normalize_timestamp(raw_timestamp)
    if ts:
        return to_date_str(ts) or "unknown"
    # Fallback for strings like YYYY-MM-DD HH:MM:SS when parser fails.
    maybe_date = raw_timestamp[:10]
    if re.match(r"^\d{4}-\d{2}-\d{2}$", maybe_date):
        return maybe_date
    return "unknown"


def _normalize_locode(locode: str) -> str:
    return re.sub(r"\s+", "", locode.upper())


def serialize_ais_row(
    row: Dict[str, Any], source_file: str
) -> Optional[Tuple[str, Dict[str, Any], str]]:
    """
    Convert one AIS position row into a telemetry sentence + metadata + stable id.
    Returns None when required fields are missing.
    """
    mmsi = normalize_identifier(_pick(row, "MMSI", "mmsi"))
    time_position = _pick(row, "TimePosition", "timeposition", "timestamp")
    latitude = _to_float(row.get("Latitude", row.get("latitude")))
    longitude = _to_float(row.get("Longitude", row.get("longitude")))

    if mmsi == "unknown" or time_position == "unknown" or latitude is None or longitude is None:
        return None

    imo = normalize_identifier(_pick(row, "IMO", "imo"))
    name = _pick(row, "Name", "name", default=mmsi)
    callsign = _pick(row, "Callsign", "callsign")
    flag = _pick(row, "Flag", "flag")
    vessel_type = _pick(row, "VesselType", "vessel_type", "ship_type")
    nav_status = _pick(row, "NavStatus", "nav_status")
    destination = _pick(row, "Destination", "destination")
    time_eta = _pick(row, "TimeETA", "timeeta", "eta")
    source_position = _pick(row, "SourcePosition", "sourceposition", default="unknown")

    speed = _to_float(row.get("Speed", row.get("speed")))
    course = _to_float(row.get("Course", row.get("course")))
    heading = _to_float(row.get("Heading", row.get("heading")))
    draught = _to_float(row.get("Draught", row.get("draught")))

    ts = normalize_timestamp(time_position)
    timestamp_full = ts.isoformat() if ts else time_position
    timestamp_date = to_date_str(ts) or _date_from_timestamp(time_position)
    destination_norm = normalize_destination(destination) if destination != "unknown" else "UNKNOWN"

    sentence = (
        f"At {timestamp_full}, vessel {name} (MMSI {mmsi}, IMO {imo}, callsign {callsign}, "
        f"flag {flag}, type {vessel_type}) was at ({latitude:.5f}, {longitude:.5f}) moving at "
        f"{speed if speed is not None else 'unknown'} kn, course "
        f"{course if course is not None else 'unknown'} deg, heading "
        f"{heading if heading is not None else 'unknown'} deg. Nav status: {nav_status}. "
        f"Destination: {destination}. ETA: {time_eta}. Draught: "
        f"{draught if draught is not None else 'unknown'} m. Data source: {source_position}."
    )

    stable_id = _sanitize_id(f"{mmsi}_{timestamp_full}_{latitude:.5f}_{longitude:.5f}")

    metadata = {
        "mmsi": mmsi,
        "imo": imo,
        "name": name,
        "callsign": callsign,
        "flag": flag,
        "flag_norm": flag.strip().upper() if flag != "unknown" else "UNKNOWN",
        "vessel_type": vessel_type,
        "vessel_type_norm": vessel_type.strip().lower() if vessel_type != "unknown" else "unknown",
        "nav_status": nav_status,
        "nav_status_norm": nav_status.strip().lower() if nav_status != "unknown" else "unknown",
        "destination": destination,
        "destination_norm": destination_norm,
        "port_name": "unknown",
        "port_name_norm": "unknown",
        "locode": "unknown",
        "locode_norm": "unknown",
        "country_prefix": "UNK",
        "timestamp_date": timestamp_date,
        "timestamp_full": timestamp_full,
        "latitude": latitude,
        "longitude": longitude,
        "speed": speed if speed is not None else -1.0,
        "course": course if course is not None else -1.0,
        "heading": heading if heading is not None else -1.0,
        "draught": draught if draught is not None else -1.0,
        "source_position": source_position,
        "event_kind": "ais_position",
        "source_file": source_file,
        "serialized_excerpt": sentence[:420],
    }
    return sentence, metadata, stable_id


def serialize_port_call_row(
    row: Dict[str, Any], source_file: str
) -> Optional[Tuple[str, Dict[str, Any], str]]:
    """
    Convert one port-call row (PRJ896-style) into serialized text + metadata + stable id.
    Returns None when required fields are missing.
    """
    mmsi = normalize_identifier(_pick(row, "vesselMMSI", "MMSI", "mmsi"))
    imo = normalize_identifier(_pick(row, "vesselIMO", "IMO", "imo"))
    port_arrival = _pick(row, "portArrival", "arrival_time", "arrival")
    port_departure = _pick(row, "portDeparture", "departure_time", "departure")
    locode = _pick(row, "portLocode", "locode", "UNLOCODE")
    port_name = _pick(row, "portName", "port_name", "port")
    vessel_name = _pick(row, "vesselName", "Name", "name", default=mmsi)
    vessel_type = _pick(row, "vesselType", "VesselType", "vessel_type", "ship_type")
    dest_arrival = _pick(row, "vesselDestinationArrival", "destination_arrival", default="unknown")
    dest_departure = _pick(row, "vesselDestinationDeparture", "destination_departure", default="unknown")

    if mmsi == "unknown" or port_arrival == "unknown":
        return None

    arrival_ts = normalize_timestamp(port_arrival)
    departure_ts = normalize_timestamp(port_departure) if port_departure != "unknown" else None
    arrival_full = arrival_ts.isoformat() if arrival_ts else port_arrival
    departure_full = departure_ts.isoformat() if departure_ts else port_departure
    timestamp_date = to_date_str(arrival_ts) or _date_from_timestamp(port_arrival)
    destination_norm = (
        normalize_destination(dest_departure)
        if dest_departure != "unknown"
        else normalize_destination(dest_arrival)
        if dest_arrival != "unknown"
        else "UNKNOWN"
    )

    sentence = (
        f"Vessel {vessel_name} (MMSI {mmsi}, IMO {imo}, type {vessel_type}) called at "
        f"{port_name} ({locode}) arriving at {arrival_full} and departing at {departure_full}. "
        f"Arrival destination: {dest_arrival}. Departure destination: {dest_departure}."
    )

    stable_id = _sanitize_id(f"{mmsi}_{arrival_full}_{locode}_port_call")

    metadata = {
        "mmsi": mmsi,
        "imo": imo,
        "name": vessel_name,
        "callsign": "unknown",
        "flag": "unknown",
        "flag_norm": "UNKNOWN",
        "vessel_type": vessel_type,
        "vessel_type_norm": vessel_type.strip().lower() if vessel_type != "unknown" else "unknown",
        "nav_status": "unknown",
        "nav_status_norm": "unknown",
        "destination": dest_departure if dest_departure != "unknown" else dest_arrival,
        "destination_norm": destination_norm,
        "destination_arrival": dest_arrival,
        "destination_departure": dest_departure,
        "port_name": port_name,
        "port_name_norm": port_name.strip().lower() if port_name != "unknown" else "unknown",
        "locode": locode,
        "locode_norm": _normalize_locode(locode),
        "country_prefix": country_prefix_from_locode(locode),
        "timestamp_date": timestamp_date,
        "timestamp_full": arrival_full,
        "arrival_full": arrival_full,
        "departure_full": departure_full,
        "latitude": None,
        "longitude": None,
        "speed": None,
        "course": None,
        "heading": None,
        "draught": None,
        "source_position": "PORT_CALL_FEED",
        "event_kind": "port_call",
        "source_file": source_file,
        "serialized_excerpt": sentence[:420],
    }
    return sentence, metadata, stable_id


def serialize_traffic_row(
    row: Dict[str, Any], source_file: str
) -> Optional[Tuple[str, Dict[str, Any], str]]:
    """
    Dispatch serializer based on row schema.
    """
    keys = set(row.keys())
    if "TimePosition" in keys or "Latitude" in keys or "Longitude" in keys:
        return serialize_ais_row(row, source_file)
    if "portArrival" in keys or "portDeparture" in keys or "portLocode" in keys:
        return serialize_port_call_row(row, source_file)
    # Fallback heuristics for unknown casing.
    lowered = {k.lower() for k in keys}
    if {"timeposition", "latitude", "longitude"} & lowered:
        return serialize_ais_row(row, source_file)
    if {"portarrival", "portdeparture", "portlocode"} & lowered:
        return serialize_port_call_row(row, source_file)
    return None


def compact_traffic_evidence(metadata: Dict[str, Any], text: str) -> str:
    excerpt = text[:220].strip()
    if metadata.get("event_kind") == "port_call":
        return (
            f"{metadata.get('mmsi', 'unknown')}/{metadata.get('imo', 'unknown')} "
            f"{metadata.get('timestamp_full', 'unknown')} "
            f"{metadata.get('locode', 'unknown')}->{metadata.get('destination_norm', 'UNKNOWN')} :: {excerpt}"
        )
    lat = metadata.get("latitude", "unknown")
    lon = metadata.get("longitude", "unknown")
    destination = metadata.get("destination_norm", "UNKNOWN")
    return (
        f"{metadata.get('mmsi', 'unknown')}/{metadata.get('imo', 'unknown')} "
        f"{metadata.get('timestamp_full', 'unknown')} "
        f"({lat}, {lon}) dest={destination} :: {excerpt}"
    )
