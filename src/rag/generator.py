"""LLM answer generation constrained to retrieved context + evidence formatting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from openai import OpenAI

from src.rag.retriever import EvidenceItem, QueryFilters
from src.utils.config import load_config
from src.utils.runtime import require_openai_api_key
from src.utils.serialization import compact_traffic_evidence


SYSTEM_PROMPT = (
    "You must only answer using provided context. "
    "If context is insufficient, say 'I don't have evidence in the dataset to answer that.' "
    "Always list evidence sources."
)


@dataclass
class GenerationOutput:
    answer: str
    evidence_lines: List[str]
    caveats: List[str]
    refused: bool

    def to_markdown(self) -> str:
        evidence = "\n".join(f"- {line}" for line in self.evidence_lines) or "- No evidence retrieved."
        caveats = "\n".join(f"- {line}" for line in self.caveats)
        return f"Answer\n{self.answer}\n\nEvidence\n{evidence}\n\nCaveats\n{caveats}"


class AnswerGenerator:
    def __init__(self, config_path: str | Path = "config/config.yaml") -> None:
        cfg = load_config(config_path)
        self.model = cfg["models"]["generation_model"]
        self.strict_mode = bool(cfg["retrieval"].get("strict_mode", True))
        self.strict_distance_threshold = float(
            cfg["retrieval"].get("strict_distance_threshold", 0.42)
        )
        self.openai = OpenAI(api_key=require_openai_api_key())

    @staticmethod
    def _format_filters(filters: QueryFilters) -> str:
        active = []
        if filters.mmsi:
            active.append(f"mmsi={filters.mmsi}")
        if filters.imo:
            active.append(f"imo={filters.imo}")
        if filters.locode:
            active.append(f"locode={filters.locode}")
        if filters.port_name:
            active.append(f"port_name={filters.port_name}")
        if filters.vessel_type:
            active.append(f"vessel_type={filters.vessel_type}")
        if filters.flag:
            active.append(f"flag={filters.flag}")
        if filters.destination:
            active.append(f"destination={filters.destination}")
        if filters.nav_status:
            active.append(f"nav_status={filters.nav_status}")
        if filters.date_from:
            active.append(f"date_from={filters.date_from}")
        if filters.date_to:
            active.append(f"date_to={filters.date_to}")
        if filters.lat_min is not None:
            active.append(f"lat_min={filters.lat_min}")
        if filters.lat_max is not None:
            active.append(f"lat_max={filters.lat_max}")
        if filters.lon_min is not None:
            active.append(f"lon_min={filters.lon_min}")
        if filters.lon_max is not None:
            active.append(f"lon_max={filters.lon_max}")
        return ", ".join(active) if active else "none"

    @staticmethod
    def _format_context(evidence_items: Sequence[EvidenceItem], max_items: int = 8) -> str:
        chunks: List[str] = []
        for item in list(evidence_items)[:max_items]:
            excerpt = item.text[:380].replace("\n", " ")
            chunks.append(f"[{item.id}] ({item.source_kind}) {excerpt}")
        return "\n".join(chunks)

    @staticmethod
    def _insufficient_evidence_answer() -> GenerationOutput:
        return GenerationOutput(
            answer="I don't have evidence in the dataset to answer that. Please broaden filters or provide more data.",
            evidence_lines=[],
            caveats=[
                "No sufficiently relevant evidence was retrieved.",
                "Try widening time range, port, or vessel type filters.",
            ],
            refused=True,
        )

    def _strict_gate(self, evidence_items: Sequence[EvidenceItem]) -> bool:
        if not evidence_items:
            return True
        if not self.strict_mode:
            return False
        distances = [x.distance for x in evidence_items if x.distance is not None]
        if not distances:
            return True
        return min(distances) > self.strict_distance_threshold

    def _generate_short_answer(
        self, question: str, filters: QueryFilters, evidence_items: Sequence[EvidenceItem]
    ) -> str:
        context_block = self._format_context(evidence_items)
        user_prompt = (
            "Question:\n"
            f"{question}\n\n"
            "Active filters:\n"
            f"{self._format_filters(filters)}\n\n"
            "Retrieved context snippets (with ids):\n"
            f"{context_block}\n\n"
            "Return a short answer in 3-6 lines. Do not invent facts."
        )
        response = self.openai.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content.strip()

    @staticmethod
    def format_evidence_lines(evidence_items: Sequence[EvidenceItem]) -> List[str]:
        lines: List[str] = []
        for item in evidence_items:
            if item.source_kind == "traffic":
                compact = compact_traffic_evidence(item.metadata, item.text)
                lines.append(f"[traffic] {item.id} | {compact}")
            else:
                source = item.metadata.get("source_file", "unknown.pdf")
                page = item.metadata.get("page", "?")
                source_url = item.metadata.get("source_url")
                snippet = item.text[:180].replace("\n", " ").strip()
                if source_url:
                    lines.append(
                        f"[docs] {item.id} | {source} | {source_url} | \"{snippet}\""
                    )
                else:
                    lines.append(
                        f"[docs] {item.id} | {source} p.{page} | \"{snippet}\""
                    )
        return lines

    def generate(
        self,
        question: str,
        filters: QueryFilters,
        evidence_items: Sequence[EvidenceItem],
        aggregation_result: Optional[Dict[str, Any]] = None,
    ) -> GenerationOutput:
        if aggregation_result is not None:
            analysis_type = aggregation_result.get("analysis_type", "count")
            count = aggregation_result.get("count", 0)
            ids = aggregation_result.get("rows", [])
            evidence_lines = [f"[traffic-count] row_id={row_id}" for row_id in ids[:10]]
            if analysis_type == "jump_detection":
                answer = (
                    f"Detected {count} potential sudden AIS coordinate jumps "
                    f"within the configured time window."
                )
            else:
                answer = f"There are {count} matching AIS position reports in the filtered CSV subset."
            return GenerationOutput(
                answer=answer,
                evidence_lines=evidence_lines,
                caveats=[
                    "Count is computed from the persisted traffic metadata index, not an estimate.",
                    "If this is unexpectedly low/high, broaden filters and rebuild index with more rows.",
                ],
                refused=False,
            )

        if self._strict_gate(evidence_items):
            return self._insufficient_evidence_answer()

        answer = self._generate_short_answer(question, filters, evidence_items)
        return GenerationOutput(
            answer=answer,
            evidence_lines=self.format_evidence_lines(evidence_items),
            caveats=[
                "Answer is constrained to retrieved context only.",
                "Embedding retrieval may miss relevant rows if filters are overly narrow.",
            ],
            refused=False,
        )
