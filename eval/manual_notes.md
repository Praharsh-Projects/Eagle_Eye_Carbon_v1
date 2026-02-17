# Supervisor Demo Notes

## Dataset Scope
- PRJ912 AIS telemetry CSV + PRJ896 port-call CSV indexed together (subset/full mode).
- NIS2 PDF + optional public ISPS explanatory docs indexed for compliance framing.

## Demo Flow
- Run index build command and confirm persisted collections.
- Ask one AIS traffic question (destination/nav status + date + bbox).
- Ask one port-call question (locode or port name + vessel type + date).
- Ask one compliance question and show document/page/url citations.
- Ask one impossible question to demonstrate strict refusal.

## What to Observe
- Answer starts with short summary.
- Evidence includes stable AIS row ids (traffic) or doc/page/url snippets (docs).
- Filters (including bbox) are reflected in evidence metadata.
- Low-evidence questions are refused instead of hallucinated.

## Risks / Follow-up
- PDF extraction quality varies by source formatting.
- Web page layout changes may affect text extraction quality.
- Similarity threshold may need tuning per embedding model.
- Full CSV indexing latency depends on row count and API quota.
