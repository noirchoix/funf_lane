# Fünf RAG

A production-oriented retrieval-augmented generation and information-retrieval platform built around hybrid search, domain-specific retrieval lanes, reranking, provider failover, persistent memory, structured ingestion, and retrieval-quality evaluation.

The system is designed to make retrieval behavior inspectable: it records how evidence is retrieved, fused, reranked, selected, and ultimately passed into generation.

## Core Capabilities

- FastAPI service with API-key protected routes
- Qdrant-backed dense retrieval across domain-specific collections
- Custom BM25 sparse retrieval
- Reciprocal Rank Fusion (RRF)
- Optional cross-encoder reranking
- Lane balancing and retrieval quotas
- Dense, sparse, fused, reranked, and final-context retrieval traces
- Gemini embeddings with Voyage fallback
- Gemini generation with DeepSeek fallback
- SQLite event/audit memory
- Qdrant semantic memory
- PDF and text ingestion
- Structured phytochemistry export ingestion
- Intent-aware lane routing
- Artifact-aware query planning
- Citation-grounded responses
- Retrieval evaluation using Recall@K, MRR, and nDCG@K
- Citation/no-citation evaluation
- Docker and Docker Compose
- GitHub Actions evaluation workflow
- Metrics endpoints
- OpenTelemetry instrumentation foundations
- SvelteKit/Svelte 5 frontend workbench

## Retrieval Architecture

The retrieval pipeline combines multiple retrieval signals rather than relying on a single vector-search result.

```text
User query
    |
    v
Intent / lane selection
    |
    +-------------------+
    |                   |
    v                   v
Dense retrieval     BM25 retrieval
    |                   |
    +---------+---------+
              |
              v
     Reciprocal Rank Fusion
              |
              v
      Optional cross-encoder
             reranking
              |
              v
        Context selection
              |
              v
     Citation-grounded answer
```

Each retrieval stage can emit trace information so the final context can be inspected and evaluated.

## Retrieval Lanes

A single Qdrant instance is divided into domain-specific collections so that retrieval can be routed and balanced by intent.

```text
rag_phytochemistry_context
rag_reaction_orgchem
rag_quality_control
rag_physical_chem
rag_internal_notes
chem_memory_v1
```

The lane router can select one or more collections based on the query and apply lane quotas so that larger corpora do not automatically dominate more specialized evidence.

## Provider Routing

### Generation

Primary:

```text
Gemini
```

Fallback:

```text
DeepSeek
```

### Embeddings

Primary:

```text
Gemini embeddings
```

Fallback:

```text
Voyage AI
```

Provider routing allows the system to continue operating when a preferred provider is unavailable or unsuitable for a request.

## Persistent Memory

Fünf uses two complementary persistence layers.

### SQLite

Stores structured memory and audit events.

Examples include:

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

### Qdrant

Stores semantic memory that can be retrieved alongside document evidence.

Memory can be searched, compacted, and written back during supported workflows.

## Ingestion

The system supports:

- general text ingestion
- PDF ingestion
- structured phytochemistry exports

Ingested content is chunked and assigned to an appropriate retrieval lane with metadata such as:

- document ID
- title
- source URI
- tags
- evidence type
- source quality
- module relevance
- artifact version

## Structured Phytochemistry Ingestion

Structured subsystem exports can be ingested and split into lane-aware evidence records.

Example:

```powershell
$headers = @{ "X-API-Key" = "change_me" }

$payload = @{
  path = "C:\path\to\latest_subsystem_rag_export.json"
  project_id = "phytoquery"
  tags = @("wheel-system", "runtime-export")
  artifact_versions = @{
    food_chemistry = "v1"
    dess_physics = "v1"
    rxnutils_templates = "v1"
  }
} | ConvertTo-Json -Depth 10

Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/v1/ingest/phytochemistry-export" `
  -Method Post `
  -Headers $headers `
  -ContentType "application/json" `
  -Body $payload
```

## PDF Ingestion

Example:

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

## Querying

Queries can either allow automatic lane selection or explicitly specify retrieval lanes.

Example:

```powershell
$headers = @{ "X-API-Key" = "change_me" }

$payload = @{
  query = "Why is limonene oxidation relevant to perfume stability?"
  top_k = 8
  retrieval_lanes = @(
    "physical_chem",
    "reaction_orgchem",
    "phytochemistry_context"
  )
} | ConvertTo-Json -Depth 10

Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/v1/query" `
  -Method Post `
  -Headers $headers `
  -ContentType "application/json" `
  -Body $payload
```

## Artifact-Aware Planner

The planner endpoint accepts structured runtime context and externally computed results, retrieves supporting evidence, and generates a citation-backed explanation.

Example:

```powershell
$payload = @{
  query = "Explain why the formulation engine suggested cardamom as a substitute for clove."
  project_id = "phytoquery"
  user_id = "u_123"
  session_id = "s_456"

  runtime_context = @{
    target = "perfume"
    constraints = @("low oxidation risk")
  }

  artifact_versions = @{
    food_chemistry = "v1"
    dess_physics = "v1"
    taxonomy_coconut = "v1"
  }

  compute_result = @{
    source = "formulation_engine"
    recommended_substitutions = @(
      @{
        replace = "clove"
        with = "cardamom"
        reason = "similar FoodDB vector profile with lower oxidation-risk profile"
      }
    )
  }

  top_k_docs = 8
  top_k_memory = 6
} | ConvertTo-Json -Depth 20

Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/v1/chat/planner" `
  -Method Post `
  -Headers $headers `
  -ContentType "application/json" `
  -Body $payload
```

A planner response can include:

```json
{
  "intent": "artifact_explanation",
  "answer": "...",
  "provider": "gemini",
  "selected_lanes": [
    "phytochemistry_context",
    "quality_control",
    "physical_chem",
    "reaction_orgchem"
  ],
  "artifact_versions_used": {},
  "citations": [],
  "memories_used": [],
  "compute_result": {}
}
```

## Retrieval Evaluation

The repository includes retrieval-quality evaluation for:

- Recall@K
- Mean Reciprocal Rank
- nDCG@K
- citation coverage
- no-citation behavior
- retrieval pass/fail thresholds

Evaluation scripts can be used for regression checks so retrieval changes can be measured rather than judged only from generated answers.

## Repository Structure

```text
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

apps/web/
  ...
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

Start Qdrant:

```powershell
docker compose up -d
```

Start the API:

```powershell
uvicorn app.main:app --reload
```

Health checks:

```powershell
Invoke-WebRequest http://127.0.0.1:6333/healthz -UseBasicParsing
Invoke-WebRequest http://127.0.0.1:8000/health -UseBasicParsing
```

API documentation:

```text
http://127.0.0.1:8000/docs
```

## Reliability and Scope

The system implements production-oriented retrieval architecture, evaluation, provider fallback, persistence, tracing, containerization, and observability foundations.

Further production hardening include externally enforced secrets, dependency-aware readiness checks, fully wired telemetry, representative checked-in retrieval regression datasets, and broader conventional unit/integration coverage alongside the retrieval evaluation suite.
