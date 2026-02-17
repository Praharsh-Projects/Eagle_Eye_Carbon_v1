"""Web page ingestion for public regulatory context sources."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List
from urllib.parse import urlparse


@dataclass
class WebIngestResult:
    ids: List[str]
    texts: List[str]
    metadatas: List[Dict[str, Any]]


def _chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    if not text:
        return []
    chunks: List[str] = []
    stride = max(1, chunk_size - chunk_overlap)
    start = 0
    while start < len(text):
        chunks.append(text[start : start + chunk_size])
        start += stride
    return chunks


def _text_from_html(html: str) -> str:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "header", "footer"]):
        tag.decompose()
    text = soup.get_text(separator=" ", strip=True)
    return " ".join(text.split())


def ingest_web_urls(
    urls: Iterable[str],
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
    timeout_sec: int = 30,
) -> WebIngestResult:
    import requests

    ids: List[str] = []
    texts: List[str] = []
    metadatas: List[Dict[str, Any]] = []
    headers = {"User-Agent": "Portathon-RAG-Demo/1.0"}

    for url in urls:
        url = url.strip()
        if not url:
            continue
        try:
            response = requests.get(url, headers=headers, timeout=timeout_sec)
            response.raise_for_status()
            text = _text_from_html(response.text)
        except requests.RequestException:
            continue
        chunks = _chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        parsed = urlparse(url)
        short_hash = hashlib.sha1(url.encode("utf-8")).hexdigest()[:10]
        source_label = f"{parsed.netloc}{parsed.path}"
        for idx, chunk in enumerate(chunks):
            stable_id = f"web_{short_hash}_c{idx}"
            metadata: Dict[str, Any] = {
                "source_file": source_label,
                "source_url": url,
                "page": 1,
                "chunk_index": idx,
                "stable_id": stable_id,
                "timestamp_date": "unknown",
            }
            ids.append(stable_id)
            texts.append(chunk)
            metadatas.append(metadata)

    return WebIngestResult(ids=ids, texts=texts, metadatas=metadatas)
