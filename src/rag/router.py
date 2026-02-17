"""Keyword-based router for traffic/docs/both retrieval modes."""

from __future__ import annotations

from typing import List, Literal

from src.rag.retriever import EvidenceItem, QueryFilters, RAGRetriever, RetrievalResult

RouteMode = Literal["traffic", "docs", "both"]


class RAGRouter:
    DOC_KEYWORDS = {
        "regulation",
        "compliance",
        "article",
        "requirement",
        "nis2",
        "isps",
        "solas",
        "directive",
        "law",
        "policy",
        "framework",
        "security level",
    }
    TRAFFIC_KEYWORDS = {
        "traffic",
        "anomaly",
        "vessel",
        "time",
        "mmsi",
        "imo",
        "ais",
        "position",
        "latitude",
        "longitude",
        "speed",
        "course",
        "heading",
        "destination",
        "eta",
        "nav status",
        "vessel type",
        "locode",
        "port",
        "arrival",
        "departure",
        "port call",
    }

    def __init__(self, retriever: RAGRetriever) -> None:
        self.retriever = retriever

    def route(self, question: str) -> RouteMode:
        q = question.lower()
        has_docs = any(token in q for token in self.DOC_KEYWORDS)
        has_traffic = any(token in q for token in self.TRAFFIC_KEYWORDS)
        if has_docs and not has_traffic:
            return "docs"
        if has_traffic and not has_docs:
            return "traffic"
        return "both"

    def retrieve(self, question: str, filters: QueryFilters, top_k: int | None = None) -> RetrievalResult:
        mode = self.route(question)
        if mode == "traffic":
            return self.retriever.query_traffic(question, filters, top_k=top_k)
        if mode == "docs":
            return self.retriever.query_docs(question, top_k=top_k)

        traffic = self.retriever.query_traffic(question, filters, top_k=top_k)
        docs = self.retriever.query_docs(question, top_k=top_k)
        merged: List[EvidenceItem] = sorted(
            [*traffic.evidence, *docs.evidence],
            key=lambda x: x.distance if x.distance is not None else 10.0,
        )
        return RetrievalResult(mode="both", evidence=merged[: (top_k or self.retriever.top_k)], where_filter=traffic.where_filter)
