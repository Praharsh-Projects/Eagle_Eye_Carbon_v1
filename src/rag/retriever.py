"""Retrieval layer for AIS traffic/docs collections with metadata + bbox filters."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pandas as pd
from openai import OpenAI

from src.utils.config import load_config
from src.utils.serialization import normalize_destination, normalize_identifier
from src.utils.time import in_date_range
from src.utils.runtime import import_chromadb, require_openai_api_key


def _normalize_vessel_type(value: str) -> str:
    return value.strip().lower()


def _normalize_nav_status(value: str) -> str:
    return value.strip().lower()


def _normalize_flag(value: str) -> str:
    return value.strip().upper()


def _safe_float(value: Any) -> Optional[float]:
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


def _batched(items: Sequence[str], batch_size: int) -> Iterable[Sequence[str]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _cosine_distance(a: Sequence[float], b: Sequence[float]) -> float:
    dot = 0.0
    mag_a = 0.0
    mag_b = 0.0
    for va, vb in zip(a, b):
        dot += va * vb
        mag_a += va * va
        mag_b += vb * vb
    if mag_a <= 0 or mag_b <= 0:
        return 1.0
    similarity = dot / (math.sqrt(mag_a) * math.sqrt(mag_b))
    return 1.0 - similarity


@dataclass
class QueryFilters:
    mmsi: Optional[str] = None
    imo: Optional[str] = None
    locode: Optional[str] = None
    port_name: Optional[str] = None
    vessel_type: Optional[str] = None
    flag: Optional[str] = None
    destination: Optional[str] = None
    nav_status: Optional[str] = None
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    lat_min: Optional[float] = None
    lat_max: Optional[float] = None
    lon_min: Optional[float] = None
    lon_max: Optional[float] = None

    def normalized(self) -> "QueryFilters":
        return QueryFilters(
            mmsi=normalize_identifier(self.mmsi.strip()) if self.mmsi else None,
            imo=normalize_identifier(self.imo.strip()) if self.imo else None,
            locode=self.locode.strip().upper().replace(" ", "") if self.locode else None,
            port_name=self.port_name.strip().lower() if self.port_name else None,
            vessel_type=_normalize_vessel_type(self.vessel_type)
            if self.vessel_type
            else None,
            flag=_normalize_flag(self.flag) if self.flag else None,
            destination=normalize_destination(self.destination)
            if self.destination
            else None,
            nav_status=_normalize_nav_status(self.nav_status)
            if self.nav_status
            else None,
            date_from=self.date_from,
            date_to=self.date_to,
            lat_min=_safe_float(self.lat_min),
            lat_max=_safe_float(self.lat_max),
            lon_min=_safe_float(self.lon_min),
            lon_max=_safe_float(self.lon_max),
        )


@dataclass
class EvidenceItem:
    id: str
    text: str
    metadata: Dict[str, Any]
    source_kind: str
    distance: Optional[float] = None


@dataclass
class RetrievalResult:
    mode: str
    evidence: List[EvidenceItem]
    where_filter: Optional[Dict[str, Any]]

    @property
    def min_distance(self) -> Optional[float]:
        values = [item.distance for item in self.evidence if item.distance is not None]
        return min(values) if values else None


class RAGRetriever:
    def __init__(
        self,
        persist_dir: str | Path,
        config_path: str | Path = "config/config.yaml",
        top_k: Optional[int] = None,
    ) -> None:
        self.config = load_config(config_path)
        chromadb = import_chromadb()
        self.persist_dir = Path(persist_dir)
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.openai = OpenAI(api_key=require_openai_api_key())
        self.embedding_model = self.config["models"]["embedding_model"]
        self.top_k = int(top_k or self.config["retrieval"].get("top_k", 8))
        self.prefilter_candidate_limit = int(
            self.config["retrieval"].get("bbox_candidate_limit", 20000)
        )
        self.traffic_collection = self.client.get_or_create_collection(
            name=self.config["index"]["traffic_collection"]
        )
        self.docs_collection = self.client.get_or_create_collection(
            name=self.config["index"]["docs_collection"]
        )
        self.metadata_index_path = self.persist_dir / "traffic_metadata_index.csv"
        self._metadata_df: Optional[pd.DataFrame] = None

    def _embed_query(self, question: str) -> List[float]:
        response = self.openai.embeddings.create(model=self.embedding_model, input=[question])
        return response.data[0].embedding

    def _load_metadata_df(self) -> Optional[pd.DataFrame]:
        if self._metadata_df is not None:
            return self._metadata_df
        if not self.metadata_index_path.exists():
            return None
        self._metadata_df = pd.read_csv(self.metadata_index_path, low_memory=False)
        return self._metadata_df

    def _has_bbox_filter(self, filters: QueryFilters) -> bool:
        f = filters.normalized()
        return any(
            value is not None
            for value in (f.lat_min, f.lat_max, f.lon_min, f.lon_max)
        )

    def _build_where(self, filters: QueryFilters) -> Optional[Dict[str, Any]]:
        f = filters.normalized()
        clauses: List[Dict[str, Any]] = []
        if f.mmsi:
            clauses.append({"mmsi": {"$eq": f.mmsi}})
        if f.imo:
            clauses.append({"imo": {"$eq": f.imo}})
        if f.locode:
            clauses.append({"locode_norm": {"$eq": f.locode}})
        if f.port_name:
            clauses.append({"port_name_norm": {"$eq": f.port_name}})
        if f.vessel_type:
            clauses.append({"vessel_type_norm": {"$eq": f.vessel_type}})
        if f.flag:
            clauses.append({"flag_norm": {"$eq": f.flag}})
        if f.destination:
            clauses.append({"destination_norm": {"$eq": f.destination}})
        if f.nav_status:
            clauses.append({"nav_status_norm": {"$eq": f.nav_status}})
        if not clauses:
            return None
        if len(clauses) == 1:
            return clauses[0]
        return {"$and": clauses}

    def _matches_filters(self, metadata: Dict[str, Any], filters: QueryFilters) -> bool:
        f = filters.normalized()
        if f.mmsi and str(metadata.get("mmsi", "")).strip() != f.mmsi:
            return False
        if f.imo and str(metadata.get("imo", "")).strip() != f.imo:
            return False
        if f.locode and str(metadata.get("locode_norm", "")).upper() != f.locode:
            return False
        if f.port_name and str(metadata.get("port_name_norm", "")).lower() != f.port_name:
            return False
        if f.vessel_type and str(metadata.get("vessel_type_norm", "")).lower() != f.vessel_type:
            return False
        if f.flag and str(metadata.get("flag_norm", "")).upper() != f.flag:
            return False
        if f.destination and str(metadata.get("destination_norm", "")).upper() != f.destination:
            return False
        if f.nav_status and str(metadata.get("nav_status_norm", "")).lower() != f.nav_status:
            return False
        if not in_date_range(
            str(metadata.get("timestamp_date", "")),
            date_from=f.date_from,
            date_to=f.date_to,
        ):
            return False

        lat = _safe_float(metadata.get("latitude"))
        lon = _safe_float(metadata.get("longitude"))
        if f.lat_min is not None and (lat is None or lat < f.lat_min):
            return False
        if f.lat_max is not None and (lat is None or lat > f.lat_max):
            return False
        if f.lon_min is not None and (lon is None or lon < f.lon_min):
            return False
        if f.lon_max is not None and (lon is None or lon > f.lon_max):
            return False
        return True

    def _prefilter_candidate_ids(self, filters: QueryFilters) -> Optional[List[str]]:
        df = self._load_metadata_df()
        if df is None or df.empty:
            return None

        f = filters.normalized()
        mask = pd.Series(True, index=df.index)
        if f.mmsi and "mmsi" in df.columns:
            mask &= df["mmsi"].astype(str).str.strip() == f.mmsi
        if f.imo and "imo" in df.columns:
            mask &= df["imo"].astype(str).str.strip() == f.imo
        if f.locode and "locode_norm" in df.columns:
            mask &= (
                df["locode_norm"].astype(str).str.upper().str.replace(" ", "", regex=False)
                == f.locode
            )
        if f.port_name and "port_name_norm" in df.columns:
            mask &= df["port_name_norm"].astype(str).str.lower() == f.port_name
        if f.vessel_type and "vessel_type_norm" in df.columns:
            mask &= df["vessel_type_norm"].astype(str).str.lower() == f.vessel_type
        if f.flag and "flag_norm" in df.columns:
            mask &= df["flag_norm"].astype(str).str.upper() == f.flag
        if f.destination and "destination_norm" in df.columns:
            mask &= df["destination_norm"].astype(str).str.upper() == f.destination
        if f.nav_status and "nav_status_norm" in df.columns:
            mask &= df["nav_status_norm"].astype(str).str.lower() == f.nav_status
        if f.date_from and "timestamp_date" in df.columns:
            mask &= df["timestamp_date"].astype(str) >= f.date_from
        if f.date_to and "timestamp_date" in df.columns:
            mask &= df["timestamp_date"].astype(str) <= f.date_to
        if f.lat_min is not None and "latitude" in df.columns:
            mask &= pd.to_numeric(df["latitude"], errors="coerce") >= f.lat_min
        if f.lat_max is not None and "latitude" in df.columns:
            mask &= pd.to_numeric(df["latitude"], errors="coerce") <= f.lat_max
        if f.lon_min is not None and "longitude" in df.columns:
            mask &= pd.to_numeric(df["longitude"], errors="coerce") >= f.lon_min
        if f.lon_max is not None and "longitude" in df.columns:
            mask &= pd.to_numeric(df["longitude"], errors="coerce") <= f.lon_max

        filtered = df[mask]
        if filtered.empty or "stable_id" not in filtered.columns:
            return []
        ids = filtered["stable_id"].astype(str).tolist()
        return ids[: self.prefilter_candidate_limit]

    def _rank_candidates_by_similarity(
        self, query_embedding: Sequence[float], candidate_ids: Sequence[str], top_k: int
    ) -> List[EvidenceItem]:
        ranked: List[EvidenceItem] = []
        for id_batch in _batched(list(candidate_ids), batch_size=512):
            got = self.traffic_collection.get(
                ids=list(id_batch),
                include=["documents", "metadatas", "embeddings"],
            )
            ids = got.get("ids", [])
            docs = got.get("documents", [])
            metas = got.get("metadatas", [])
            embeddings = got.get("embeddings", [])
            for idx, doc_id in enumerate(ids):
                emb = embeddings[idx] if idx < len(embeddings) else None
                if emb is None:
                    continue
                distance = _cosine_distance(query_embedding, emb)
                ranked.append(
                    EvidenceItem(
                        id=doc_id,
                        text=docs[idx] if idx < len(docs) else "",
                        metadata=metas[idx] if idx < len(metas) else {},
                        source_kind="traffic",
                        distance=distance,
                    )
                )
        ranked.sort(key=lambda x: x.distance if x.distance is not None else 10.0)
        return ranked[:top_k]

    def query_traffic(
        self, question: str, filters: QueryFilters, top_k: Optional[int] = None
    ) -> RetrievalResult:
        where = self._build_where(filters)
        available = self.traffic_collection.count()
        if available == 0:
            return RetrievalResult(mode="traffic", evidence=[], where_filter=where)

        requested_top_k = int(top_k or self.top_k)
        query_embedding = self._embed_query(question)

        # Bounding-box filtering is enforced through pandas prefilter before ranking.
        if self._has_bbox_filter(filters):
            candidate_ids = self._prefilter_candidate_ids(filters)
            if candidate_ids is None:
                # Metadata index unavailable; fallback to vector query + post-filtering.
                n_results = min(requested_top_k * 5, available)
                try:
                    result = self.traffic_collection.query(
                        query_embeddings=[query_embedding],
                        n_results=n_results,
                        where=where,
                        include=["documents", "metadatas", "distances"],
                    )
                except Exception:
                    result = self.traffic_collection.query(
                        query_embeddings=[query_embedding],
                        n_results=n_results,
                        where=None,
                        include=["documents", "metadatas", "distances"],
                    )
                hits = self._to_evidence(result, source_kind="traffic")
                filtered = [item for item in hits if self._matches_filters(item.metadata, filters)]
                filtered.sort(key=lambda x: x.distance if x.distance is not None else 10.0)
                return RetrievalResult(
                    mode="traffic",
                    evidence=filtered[:requested_top_k],
                    where_filter={"prefilter_candidates": "metadata-index-missing"},
                )
            if not candidate_ids:
                return RetrievalResult(
                    mode="traffic",
                    evidence=[],
                    where_filter={"prefilter_candidates": 0},
                )
            evidence = self._rank_candidates_by_similarity(
                query_embedding, candidate_ids, requested_top_k
            )
            evidence = [item for item in evidence if self._matches_filters(item.metadata, filters)]
            return RetrievalResult(
                mode="traffic",
                evidence=evidence,
                where_filter={"prefilter_candidates": len(candidate_ids)},
            )

        n_results = min(requested_top_k * 3, available)
        try:
            result = self.traffic_collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        except Exception:
            result = self.traffic_collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=None,
                include=["documents", "metadatas", "distances"],
            )
        hits = self._to_evidence(result, source_kind="traffic")
        filtered = [item for item in hits if self._matches_filters(item.metadata, filters)]
        filtered.sort(key=lambda x: x.distance if x.distance is not None else 10.0)
        return RetrievalResult(
            mode="traffic",
            evidence=filtered[:requested_top_k],
            where_filter=where,
        )

    def query_docs(self, question: str, top_k: Optional[int] = None) -> RetrievalResult:
        available = self.docs_collection.count()
        if available == 0:
            return RetrievalResult(mode="docs", evidence=[], where_filter=None)

        query_embedding = self._embed_query(question)
        n_results = min(int(top_k or self.top_k), available)
        result = self.docs_collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )
        hits = self._to_evidence(result, source_kind="docs")
        hits.sort(key=lambda x: x.distance if x.distance is not None else 10.0)
        return RetrievalResult(
            mode="docs",
            evidence=hits[: int(top_k or self.top_k)],
            where_filter=None,
        )

    def _to_evidence(self, result: Dict[str, Any], source_kind: str) -> List[EvidenceItem]:
        ids = (result.get("ids") or [[]])[0]
        docs = (result.get("documents") or [[]])[0]
        metas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        evidence: List[EvidenceItem] = []
        for idx, doc_id in enumerate(ids):
            evidence.append(
                EvidenceItem(
                    id=doc_id,
                    text=docs[idx] if idx < len(docs) else "",
                    metadata=metas[idx] if idx < len(metas) else {},
                    source_kind=source_kind,
                    distance=distances[idx] if idx < len(distances) else None,
                )
            )
        return evidence

    @staticmethod
    def is_aggregation_question(question: str) -> bool:
        q = question.lower()
        keywords = (
            "how many",
            "count",
            "number of",
            "total",
            "aggregate",
            "sum of",
        )
        return any(token in q for token in keywords)

    @staticmethod
    def is_jump_detection_question(question: str) -> bool:
        q = question.lower()
        keywords = ("sudden", "jump", "teleport", "coordinate jump", "suspicious ais")
        return any(token in q for token in keywords)

    def compute_traffic_count(
        self, filters: QueryFilters, question: str, max_ids: int = 200
    ) -> Optional[Dict[str, Any]]:
        if not self.is_aggregation_question(question):
            return None
        df = self._load_metadata_df()
        if df is None or df.empty:
            return {"analysis_type": "count", "count": 0, "rows": []}

        f = filters.normalized()
        mask = pd.Series(True, index=df.index)
        if f.mmsi and "mmsi" in df.columns:
            mask &= df["mmsi"].astype(str).str.strip() == f.mmsi
        if f.imo and "imo" in df.columns:
            mask &= df["imo"].astype(str).str.strip() == f.imo
        if f.locode and "locode_norm" in df.columns:
            mask &= (
                df["locode_norm"].astype(str).str.upper().str.replace(" ", "", regex=False)
                == f.locode
            )
        if f.port_name and "port_name_norm" in df.columns:
            mask &= df["port_name_norm"].astype(str).str.lower() == f.port_name
        if f.vessel_type and "vessel_type_norm" in df.columns:
            mask &= df["vessel_type_norm"].astype(str).str.lower() == f.vessel_type
        if f.flag and "flag_norm" in df.columns:
            mask &= df["flag_norm"].astype(str).str.upper() == f.flag
        if f.destination and "destination_norm" in df.columns:
            mask &= df["destination_norm"].astype(str).str.upper() == f.destination
        if f.nav_status and "nav_status_norm" in df.columns:
            mask &= df["nav_status_norm"].astype(str).str.lower() == f.nav_status
        if f.date_from and "timestamp_date" in df.columns:
            mask &= df["timestamp_date"].astype(str) >= f.date_from
        if f.date_to and "timestamp_date" in df.columns:
            mask &= df["timestamp_date"].astype(str) <= f.date_to
        if f.lat_min is not None and "latitude" in df.columns:
            mask &= pd.to_numeric(df["latitude"], errors="coerce") >= f.lat_min
        if f.lat_max is not None and "latitude" in df.columns:
            mask &= pd.to_numeric(df["latitude"], errors="coerce") <= f.lat_max
        if f.lon_min is not None and "longitude" in df.columns:
            mask &= pd.to_numeric(df["longitude"], errors="coerce") >= f.lon_min
        if f.lon_max is not None and "longitude" in df.columns:
            mask &= pd.to_numeric(df["longitude"], errors="coerce") <= f.lon_max

        filtered = df[mask]
        ids = filtered["stable_id"].astype(str).tolist()[:max_ids]
        return {"analysis_type": "count", "count": int(len(filtered)), "rows": ids}

    def detect_sudden_jumps(
        self, filters: QueryFilters, max_minutes: int = 30, km_threshold: float = 80.0
    ) -> Dict[str, Any]:
        """
        Detect likely suspicious jumps for a single MMSI within time window.
        """
        df = self._load_metadata_df()
        if df is None or df.empty:
            return {"analysis_type": "jump_detection", "count": 0, "rows": []}

        f = filters.normalized()
        work = df.copy()
        if f.mmsi:
            work = work[work["mmsi"].astype(str).str.strip() == f.mmsi]
        if f.date_from:
            work = work[work["timestamp_date"].astype(str) >= f.date_from]
        if f.date_to:
            work = work[work["timestamp_date"].astype(str) <= f.date_to]
        work["latitude"] = pd.to_numeric(work["latitude"], errors="coerce")
        work["longitude"] = pd.to_numeric(work["longitude"], errors="coerce")
        work["timestamp_dt"] = pd.to_datetime(work["timestamp_full"], errors="coerce")
        work = work.dropna(subset=["timestamp_dt", "latitude", "longitude", "mmsi"])
        if work.empty:
            return {"analysis_type": "jump_detection", "count": 0, "rows": []}

        work = work.sort_values(["mmsi", "timestamp_dt"])
        jump_ids: List[str] = []
        for _, group in work.groupby("mmsi"):
            prev = group.shift(1)
            dt_minutes = (group["timestamp_dt"] - prev["timestamp_dt"]).dt.total_seconds() / 60.0
            dlat = group["latitude"] - prev["latitude"]
            dlon = group["longitude"] - prev["longitude"]
            # Approx rough km distance on Earth surface.
            dist_km = ((dlat * 111.0) ** 2 + (dlon * 111.0) ** 2) ** 0.5
            mask = (dt_minutes > 0) & (dt_minutes <= max_minutes) & (dist_km >= km_threshold)
            ids = group.loc[mask, "stable_id"].astype(str).tolist()
            jump_ids.extend(ids)
        return {
            "analysis_type": "jump_detection",
            "count": len(jump_ids),
            "rows": jump_ids[:200],
        }
