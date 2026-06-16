# ChemRAG

ChemRAG is an independent FastAPI microservice that provides citation-backed retrieval, semantic memory, and planner-style explanation around the phytochemistry wheel system.

ChemRAG is **not** the numerical compute layer. It does not replace `reaction_framework`, `formulation_engine`, FoodDB, COCO taxonomy, or DESS. It retrieves evidence, explains compute outputs, preserves decisions, and provides planner-level orchestration.

## Current Version

```text
ChemRAG v2
= structured phytochemistry export ingestion
+ lane-routed retrieval
+ artifact-aware planner orchestration
+ semantic memory
+ compute-output explanation
+ multi-module citation support
```

## Architecture

```text
Compute truth layer
  reaction_framework
  formulation_engine
  FoodDB
  COCO
  DESS

Knowledge/explanation layer
  ChemRAG

Future orchestration/enrichment layer
  CrewAI / agency-agents
  ml-intern validation runners
  LightRAG-style graph retrieval
```

## Implemented Capabilities

- FastAPI service with authenticated routes
- Gemini-first generation with DeepSeek fallback
- Gemini-first embeddings with Voyage fallback
- Qdrant vector storage
- SQLite structured memory audit log
- PDF/text ingestion
- structured phytochemistry export ingestion
- lane-routed retrieval across multiple Qdrant collections
- artifact-aware planner prompt
- semantic memory search and compaction
- planner memory write-back

## RAG Lanes

ChemRAG v2 uses one Qdrant instance with multiple collections:

```text
rag_phytochemistry_context   structured subsystem export from reaction/DESS/taxonomy/FoodDB/formulation lanes
rag_reaction_orgchem         organic chemistry, reaction mechanisms, rxnutils/reaction evidence
rag_quality_control          formulation QC, safety, stability, regulatory-style checks
rag_physical_chem            DESS explanation, oxidation, volatility, physical chemistry
rag_internal_notes           perfumery notes, accords, note wheel conventions, internal notes
chem_memory_v1               semantic memory events
```

## Repository Structure

```text
chemrag/
  app/
    main.py
    settings.py
    schemas.py
    deps.py
    memory/
      db.py
      store.py
      policy.py
      summarize.py
  rag/
    collections.py
    embed_gemini.py
    embed_router.py
    embed_voyage.py
    llm_gemini.py
    llm_deepseek.py
    llm_router.py
    qdrant_store.py
    pipeline.py
    phytochemistry_context.py
    retrieval_enhancements.py
    langchain_loaders.py
    intent.py
    prompting.py
    compute_client.py
  routes/
    ingest.py
    ingest_pdf.py
    ingest_phytochemistry.py
    query.py
    memory.py
    chat.py
    health.py
    metrics.py
```

## Environment Variables

Create `.env`:

```env
SERVICE_API_KEY=change_me

GEMINI_API_KEY=
DEEPSEEK_API_KEY=
VOYAGEAI_API_KEY=

DEFAULT_GENERATION_PROVIDER=gemini
FALLBACK_GENERATION_PROVIDER=deepseek
GEMINI_MODEL=gemini-2.0-flash
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat

DEFAULT_EMBED_PROVIDER=gemini
FALLBACK_EMBED_PROVIDER=voyage
GEMINI_EMBED_MODEL=text-embedding-004
VOYAGE_EMBED_MODEL=voyage-3

QDRANT_URL=http://127.0.0.1:6333
QDRANT_API_KEY=

MEMORY_DB_PATH=data/memory.db
MEMORY_COLLECTION=chem_memory_v1
```

## Run Locally

```powershell
docker compose up -d
uvicorn app.main:app --reload
```

Health checks:

```powershell
Invoke-WebRequest http://127.0.0.1:6333/healthz -UseBasicParsing
Invoke-WebRequest http://127.0.0.1:8000/health -UseBasicParsing
```

API docs:

```text
http://127.0.0.1:8000/docs
```

## Ingest PDF into a Lane

```powershell
curl.exe -X POST "http://127.0.0.1:8000/v1/ingest/pdf" `
  -H "X-API-Key: change_me" `
  -F "doc_id=orgchem_p50_70" `
  -F "title=Organic Chemistry pages 50-70" `
  -F "source_uri=local:advanced-organic-chemistry-jerry-march.pdf" `
  -F "tags=organic,mechanism,oxidation,stability" `
  -F "rag_lane=reaction_orgchem" `
  -F "module_relevance=rxnutils,reaction_framework" `
  -F "evidence_type=textbook" `
  -F "source_quality=secondary" `
  -F "start_page=50" `
  -F "max_pages=20" `
  -F "file=@C:\path\to\book.pdf;type=application/pdf"
```

## Ingest Structured Phytochemistry Export

This endpoint ingests `latest_subsystem_rag_export.json` and splits it into lane-aware records.

```powershell
$headers = @{ "X-API-Key" = "change_me" }

$payload = @{
  path = "C:\path\to\latest_subsystem_rag_export.json"
  project_id = "phytoquery"
  tags = @("wheel-system", "runtime-export")
  artifact_versions = @{ food_chemistry="v1"; dess_physics="v1"; rxnutils_templates="v1" }
} | ConvertTo-Json -Depth 10

Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/v1/ingest/phytochemistry-export" `
  -Method Post `
  -Headers $headers `
  -ContentType "application/json" `
  -Body $payload | ConvertTo-Json -Depth 20
```

## Query with Lane Override

```powershell
$headers = @{ "X-API-Key" = "change_me" }

$payload = @{
  query = "Why is limonene oxidation relevant to perfume stability?"
  top_k = 8
  retrieval_lanes = @("physical_chem", "reaction_orgchem", "phytochemistry_context")
} | ConvertTo-Json -Depth 10

Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/v1/query" `
  -Method Post `
  -Headers $headers `
  -ContentType "application/json" `
  -Body $payload | ConvertTo-Json -Depth 20
```

## Planner Query

```powershell
$payload = @{
  query = "Explain why the formulation engine suggested cardamom as a substitute for clove."
  project_id = "phytoquery"
  user_id = "u_123"
  session_id = "s_456"
  runtime_context = @{ target="perfume"; constraints=@("low oxidation risk") }
  artifact_versions = @{ food_chemistry="v1"; dess_physics="v1"; taxonomy_coconut="v1" }
  compute_result = @{
    source="formulation_engine"
    recommended_substitutions=@(@{ replace="clove"; with="cardamom"; reason="similar FoodDB vector profile with lower oxidation-risk profile" })
  }
  top_k_docs = 8
  top_k_memory = 6
} | ConvertTo-Json -Depth 20

Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/v1/chat/planner" `
  -Method Post `
  -Headers $headers `
  -ContentType "application/json" `
  -Body $payload | ConvertTo-Json -Depth 20
```

Planner response includes:

```json
{
  "intent": "artifact_explanation",
  "answer": "...",
  "provider": "gemini",
  "selected_lanes": ["phytochemistry_context", "quality_control", "physical_chem", "reaction_orgchem"],
  "artifact_versions_used": {},
  "citations": [],
  "memories_used": [],
  "compute_result": {}
}
```

## Memory Event Types

ChemRAG v2 supports general and artifact-aware events:

```text
decision
deterministic
ml_output
system
summary
formulation_score
recommendation_generated
recommendation_accepted
recommendation_rejected
reaction_hypothesis
stability_warning
taxonomy_fallback_used
fooddb_similarity_used
dess_physics_support
calibration_update
anchor_formulation_selected
artifact_explanation
```

## Development Notes

- Use `curl.exe` on Windows because PowerShell aliases `curl` to `Invoke-WebRequest`.
- Keep compute modules external. ChemRAG explains their outputs; it should not duplicate them.
- Keep `memory_v1`/`chem_memory_v1` event-based and separate from document RAG collections.
- LightRAG/CrewAI/agency-agents/ml-intern should be added only after lane routing and artifact-aware planner behavior are stable.
