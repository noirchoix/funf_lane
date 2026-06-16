# Fünf RAG — Five-Lane Interactive RAG Workbench

Fünf RAG is the productized ChemRAG Pro layer: a general five-lane RAG workbench with document upload, lane-aware chatbot retrieval, hybrid dense+sparse search, reciprocal-rank fusion, reranking, trace inspection and CI retrieval evaluation.

## Lanes

Default lanes:

- `research`
- `technical_docs`
- `policy_compliance`
- `product_business`
- `custom`
- optional seeded `chemrag_demo`

All five generic lanes use the shared Qdrant collection `funf_rag_chunks` with `rag_lane` metadata filtering.

## Backend

```bash
docker compose up -d qdrant
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

## Frontend

```bash
cd apps/web
npm install --registry=https://registry.npmjs.org/
cp .env.example .env
npm run dev -- --host 0.0.0.0
```

`.env`:

```env
VITE_API_BASE_URL=http://127.0.0.1:8000
VITE_SERVICE_API_KEY=8b7d3b2e6b1a4d2f9a3c1c7f0a2e9c44
```

## New product routes

```text
GET  /v1/funf/lanes
POST /v1/funf/upload
POST /v1/funf/chat
```

Existing ChemRAG Pro routes remain available:

```text
POST /v1/pro/query
POST /v1/pro/eval/run
GET  /v1/pro/capabilities
```
