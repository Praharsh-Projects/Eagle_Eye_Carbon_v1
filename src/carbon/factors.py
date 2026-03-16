"""Local factor registry for carbon computations (offline runtime only)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict


DEFAULT_FACTOR_PATH = Path("config/carbon_factors.v1.json")

VESSEL_CLASS_ALIASES = {
    "tanker": "tanker",
    "oil tanker": "tanker",
    "chemical tanker": "tanker",
    "gas tanker": "tanker",
    "cargo ship": "cargo",
    "general cargo": "cargo",
    "bulk carrier": "cargo",
    "container ship": "container",
    "container": "container",
    "ro-ro cargo": "cargo",
    "passenger": "passenger",
    "passenger ship": "passenger",
    "tug": "service",
    "pilot": "service",
    "service": "service",
    "fishing": "service",
    "other": "unknown",
    "unknown": "unknown",
}


@dataclass
class CarbonFactorRegistry:
    raw: Dict[str, Any]
    source_path: Path
    checksum_sha256: str

    @property
    def version(self) -> str:
        return str(self.raw.get("version", "unknown"))

    @property
    def uncertainty_defaults(self) -> Dict[str, float]:
        values = self.raw.get("uncertainty_defaults", {})
        return {k: float(v) for k, v in values.items()}

    @property
    def assumptions(self) -> Dict[str, Any]:
        return dict(self.raw.get("assumptions", {}))

    def factor_payload_hash(self) -> str:
        payload = json.dumps(self.raw, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def resolve_vessel_class(self, vessel_type_norm: str) -> str:
        key = (vessel_type_norm or "").strip().lower()
        if key in VESSEL_CLASS_ALIASES:
            return VESSEL_CLASS_ALIASES[key]
        for alias, cls in VESSEL_CLASS_ALIASES.items():
            if alias in key and alias:
                return cls
        return "unknown"

    def vessel_defaults(self, vessel_class: str) -> Dict[str, Any]:
        defaults = self.raw.get("vessel_defaults", {})
        value = defaults.get(vessel_class, defaults.get("unknown", {}))
        return dict(value)

    def fuel_factors(self, fuel: str) -> Dict[str, float]:
        factors = self.raw.get("fuel_factors", {})
        value = factors.get(fuel, factors.get(self.assumptions.get("default_fuel", "MGO"), {}))
        return {
            "co2_t_per_t_fuel": float(value.get("co2_t_per_t_fuel", 3.206)),
            "wtt_co2e_t_per_t_fuel": float(value.get("wtt_co2e_t_per_t_fuel", 0.62)),
        }

    def mode_aux_power_kw(self, mode: str, vessel_class: str) -> float:
        lookup = self.raw.get("mode_defaults", {}).get("aux_power_kw", {})
        mode_map = lookup.get(mode, {})
        if vessel_class in mode_map:
            return float(mode_map[vessel_class])
        return float(mode_map.get("unknown", 600.0))

    def mode_sfc_main(self, mode: str) -> float:
        values = self.raw.get("mode_defaults", {}).get("sfc_main_g_per_kwh", {})
        return float(values.get(mode, 0.0))

    def mode_sfc_aux(self, mode: str) -> float:
        values = self.raw.get("mode_defaults", {}).get("sfc_aux_g_per_kwh", {})
        return float(values.get(mode, 220.0))

    def mode_sulfur_fraction(self, mode: str) -> float:
        values = self.raw.get("mode_defaults", {}).get("sulfur_fraction", {})
        return float(values.get(mode, 0.0010))

    def nox_factor(self, engine_family: str, mode: str) -> float:
        table = self.raw.get("nox_kg_per_t_fuel_by_engine_mode", {})
        fam = table.get(engine_family, table.get(self.assumptions.get("default_engine_family", "medium_speed_diesel"), {}))
        return float(fam.get(mode, 45.0))

    def pm_factor(self, engine_family: str, mode: str) -> float:
        table = self.raw.get("pm_kg_per_t_fuel_by_engine_mode", {})
        fam = table.get(engine_family, table.get(self.assumptions.get("default_engine_family", "medium_speed_diesel"), {}))
        return float(fam.get(mode, 1.0))


def load_factor_registry(factor_path: str | Path = DEFAULT_FACTOR_PATH) -> CarbonFactorRegistry:
    path = Path(factor_path)
    if not path.exists():
        raise FileNotFoundError(f"Carbon factor file not found: {path}")
    text = path.read_text(encoding="utf-8")
    raw = json.loads(text)
    checksum = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return CarbonFactorRegistry(raw=raw, source_path=path, checksum_sha256=checksum)
