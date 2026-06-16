<script lang="ts">
  import {
    getLanes,
    runEval,
    runFunfChat,
    uploadLaneDocument,
    type LaneInfo,
    type RetrievalMode
  } from '$lib/api';

  const modes: RetrievalMode[] = ['vector', 'sparse', 'hybrid', 'hybrid_rerank'];
  const traceTabs = ['final_context', 'reranked_hits', 'fused_hits', 'dense_hits', 'sparse_hits'];

  let lanes = $state<LaneInfo[]>([
    { lane: 'research', name: 'Research', description: 'Papers and evidence documents.', collection: 'funf_rag_chunks', editable: true },
    { lane: 'technical_docs', name: 'Technical Docs', description: 'READMEs, API specs and runbooks.', collection: 'funf_rag_chunks', editable: true },
    { lane: 'policy_compliance', name: 'Policy / Compliance', description: 'Policies, procedures and standards.', collection: 'funf_rag_chunks', editable: true },
    { lane: 'product_business', name: 'Product / Business', description: 'Requirements, strategy and operating docs.', collection: 'funf_rag_chunks', editable: true },
    { lane: 'custom', name: 'Custom', description: 'Flexible fifth knowledge lane.', collection: 'funf_rag_chunks', editable: true }
  ]);

  let selectedLanes = $state<string[]>(['research', 'technical_docs']);
  let selectedMode = $state<RetrievalMode>('hybrid_rerank');
  let topK = $state(8);
  let query = $state('Summarize the strongest evidence across the selected lanes and cite the retrieved chunks.');
  let result = $state<any>(null);
  let error = $state('');
  let busy = $state(false);
  let uploadBusy = $state(false);
  let uploadMessage = $state('');
  let activeTrace = $state('final_context');
  let selectedUploadLane = $state('research');
  let uploadTitle = $state('');
  let uploadText = $state('');
  let uploadFile = $state<File | null>(null);
  let activePanel = $state<'ingest' | 'eval'>('ingest');
  let evalInput = $state(`{
  "cases": [
    {
      "id": "case_001",
      "question": "What does the selected corpus say about the topic?",
      "expected_doc_ids": [],
      "expected_chunk_ids": [],
      "retrieval_lanes": ["research"]
    }
  ],
  "top_k": 8,
  "retrieval_mode": "hybrid_rerank",
  "generate_answers": false
}`);
  let evalResult = $state<any>(null);

  $effect(() => {
    getLanes().then((data) => {
      lanes = data.lanes || lanes;
      if (!lanes.find((l) => l.lane === selectedUploadLane)) selectedUploadLane = lanes[0]?.lane || 'research';
    }).catch(() => {});
  });

  function toggleLane(lane: string) {
    selectedLanes = selectedLanes.includes(lane)
      ? selectedLanes.filter((item) => item !== lane)
      : [...selectedLanes, lane];
  }

  function onFileChange(event: Event) {
    const input = event.target as HTMLInputElement;
    uploadFile = input.files?.[0] || null;
    if (uploadFile && !uploadTitle) uploadTitle = uploadFile.name.replace(/\.[^.]+$/, '');
  }

  async function handleUpload() {
    error = '';
    uploadMessage = '';
    uploadBusy = true;
    try {
      const form = new FormData();
      form.append('lane', selectedUploadLane);
      form.append('title', uploadTitle || uploadFile?.name || 'Untitled document');
      if (uploadText.trim()) form.append('text', uploadText.trim());
      if (uploadFile) form.append('file', uploadFile);
      const response = await uploadLaneDocument(form);
      uploadMessage = `Indexed ${response.chunks || 0} chunks into ${laneName(response.rag_lane || selectedUploadLane)}.`;
      if (!selectedLanes.includes(selectedUploadLane)) selectedLanes = [...selectedLanes, selectedUploadLane];
    } catch (e) {
      error = e instanceof Error ? e.message : String(e);
    } finally {
      uploadBusy = false;
    }
  }

  async function ask() {
    error = '';
    result = null;
    busy = true;
    try {
      result = await runFunfChat({
        query,
        top_k: topK,
        retrieval_mode: selectedMode,
        selected_lanes: selectedLanes,
        generate_answer: true,
        return_trace: true
      });
    } catch (e) {
      error = e instanceof Error ? e.message : String(e);
    } finally {
      busy = false;
    }
  }

  async function submitEval() {
    error = '';
    evalResult = null;
    try {
      evalResult = await runEval(JSON.parse(evalInput));
    } catch (e) {
      error = e instanceof Error ? e.message : String(e);
    }
  }

  function traceRows() {
    return result?.trace?.[activeTrace] || [];
  }

  function laneName(lane: string) {
    return lanes.find((l) => l.lane === lane)?.name || lane;
  }

  function traceCount(tab: string) {
    return result?.trace?.[tab]?.length || 0;
  }
</script>

<svelte:head>
  <title>Fünf RAG</title>
</svelte:head>

<main class="workspace">
  <aside class="sidebar">
    <div class="brand">
      <div class="mark">F</div>
      <div>
        <strong>Fünf RAG</strong>
        <span>five-lane retrieval workbench</span>
      </div>
    </div>

    <div class="sidebar-section">
      <div class="section-label">Active lanes</div>
      <div class="lane-list">
        {#each lanes as lane}
          <button class:active={selectedLanes.includes(lane.lane)} class="lane-row" onclick={() => toggleLane(lane.lane)}>
            <span class="dot"></span>
            <span>
              <strong>{lane.name}</strong>
              <small>{lane.description}</small>
            </span>
          </button>
        {/each}
      </div>
    </div>

    <div class="sidebar-section controls">
      <label>Retrieval mode
        <select bind:value={selectedMode}>
          {#each modes as mode}
            <option value={mode}>{mode}</option>
          {/each}
        </select>
      </label>
      <label>Top K
        <input type="number" min="1" max="30" bind:value={topK} />
      </label>
    </div>
  </aside>

  <section class="center">
    <header class="topbar">
      <div>
        <p>Production RAG</p>
        <h1>Ask across selected knowledge lanes.</h1>
      </div>
      <button class="primary" onclick={ask} disabled={busy || !selectedLanes.length}>{busy ? 'Searching…' : 'Run query'}</button>
    </header>

    {#if error}
      <div class="alert">{error}</div>
    {/if}

    <section class="query-card">
      <div class="selected-strip">
        {#each selectedLanes as lane}
          <span>{laneName(lane)}</span>
        {/each}
        {#if !selectedLanes.length}<em>Select at least one lane.</em>{/if}
      </div>
      <textarea bind:value={query} rows="4" aria-label="Question"></textarea>
      <div class="query-footer">
        <span>{selectedMode.replace('_', ' + ')} · top {topK}</span>
        <button onclick={ask} disabled={busy || !selectedLanes.length}>{busy ? 'Retrieving evidence…' : 'Ask Fünf'}</button>
      </div>
    </section>

    <section class="answer-card">
      <div class="card-head">
        <div>
          <p>Grounded answer</p>
          <h2>Response</h2>
        </div>
        {#if result}
          <span class="provider">{result.provider || 'rag'}</span>
        {/if}
      </div>

      {#if result}
        <article>{result.answer}</article>
        <div class="citation-grid">
          {#each (result.citations || []).slice(0, 10) as hit}
            <span>{hit.doc_id || 'document'} · {laneName(hit.rag_lane || '')} · #{hit.rank || '?'}</span>
          {/each}
        </div>
      {:else}
        <div class="empty-answer">
          <strong>No answer yet.</strong>
          <span>Upload context or use an existing lane, then ask a question. The trace panel will show dense, sparse, fused and reranked evidence.</span>
        </div>
      {/if}
    </section>
  </section>

  <aside class="right-panel">
    <div class="tabs">
      <button class:active={activePanel === 'ingest'} onclick={() => (activePanel = 'ingest')}>Ingest</button>
      <button class:active={activePanel === 'eval'} onclick={() => (activePanel = 'eval')}>Eval</button>
    </div>

    {#if activePanel === 'ingest'}
      <section class="tool-card">
        <div class="card-head compact">
          <div>
            <p>Lane indexing</p>
            <h2>Add context</h2>
          </div>
        </div>
        <label>Lane
          <select bind:value={selectedUploadLane}>
            {#each lanes as lane}
              <option value={lane.lane}>{lane.name}</option>
            {/each}
          </select>
        </label>
        <label>Title
          <input bind:value={uploadTitle} placeholder="Document title" />
        </label>
        <label>File
          <input type="file" accept=".pdf,.txt,.md,.markdown" onchange={onFileChange} />
        </label>
        <label>Paste text
          <textarea bind:value={uploadText} rows="6" placeholder="Paste README, notes, policy excerpts or a small document."></textarea>
        </label>
        <button class="primary full" onclick={handleUpload} disabled={uploadBusy}>{uploadBusy ? 'Indexing…' : 'Index document'}</button>
        {#if uploadMessage}<p class="success">{uploadMessage}</p>{/if}
      </section>
    {:else}
      <section class="tool-card">
        <div class="card-head compact">
          <div>
            <p>Release gate</p>
            <h2>Golden QA eval</h2>
          </div>
        </div>
        <textarea class="mono" bind:value={evalInput} rows="13"></textarea>
        <button class="primary full" onclick={submitEval}>Run eval</button>
        {#if evalResult}
          <div class:pass={evalResult.ok} class:fail={!evalResult.ok} class="gate">{evalResult.ok ? 'PASS' : 'FAIL'}</div>
          <pre>{JSON.stringify(evalResult.summary, null, 2)}</pre>
        {/if}
      </section>
    {/if}

    <section class="trace-card">
      <div class="card-head compact">
        <div>
          <p>Retrieval trace</p>
          <h2>Evidence pipeline</h2>
        </div>
      </div>
      <div class="trace-tabs">
        {#each traceTabs as tab}
          <button class:active={activeTrace === tab} onclick={() => (activeTrace = tab)}>
            <span>{tab.replace('_', ' ')}</span>
            <strong>{traceCount(tab)}</strong>
          </button>
        {/each}
      </div>

      {#if result && traceRows().length}
        <div class="hits">
          {#each traceRows() as hit}
            <div class="hit">
              <div class="hit-top">
                <strong>#{hit.rank || '?'}</strong>
                <span>{laneName(hit.rag_lane || '')}</span>
              </div>
              <p>{hit.chunk}</p>
              <small>score {Number(hit.score || 0).toFixed(4)} · dense {Number(hit.dense_score || 0).toFixed(4)} · sparse {Number(hit.sparse_score || 0).toFixed(4)} · rerank {Number(hit.rerank_score || 0).toFixed(4)}</small>
            </div>
          {/each}
        </div>
      {:else}
        <div class="trace-empty">Run a query to inspect retrieval behavior.</div>
      {/if}
    </section>
  </aside>
</main>

<style>
  .workspace {
    min-height: 100vh;
    padding: 18px;
    display: grid;
    grid-template-columns: 300px minmax(420px, 1fr) 390px;
    gap: 18px;
  }

  .sidebar, .center, .right-panel {
    min-height: calc(100vh - 36px);
  }

  .sidebar {
    border-radius: 24px;
    background: #111827;
    color: #f9fafb;
    padding: 18px;
    display: flex;
    flex-direction: column;
    gap: 20px;
    box-shadow: 0 24px 70px rgba(17, 24, 39, .18);
  }

  .brand {
    display: flex;
    gap: 12px;
    align-items: center;
    padding: 4px 4px 14px;
    border-bottom: 1px solid rgba(255,255,255,.08);
  }

  .mark {
    width: 42px;
    height: 42px;
    border-radius: 14px;
    display: grid;
    place-items: center;
    background: linear-gradient(135deg, #d7f56f, #a8f5c6);
    color: #111827;
    font-weight: 900;
  }

  .brand strong { display: block; font-size: 15px; }
  .brand span, .section-label { color: #9ca3af; font-size: 12px; }

  .sidebar-section { display: grid; gap: 10px; }
  .lane-list { display: grid; gap: 8px; }

  .lane-row {
    border: 1px solid rgba(255,255,255,.07);
    background: rgba(255,255,255,.035);
    color: #e5e7eb;
    border-radius: 16px;
    padding: 12px;
    display: grid;
    grid-template-columns: 10px 1fr;
    gap: 10px;
    text-align: left;
    align-items: start;
  }

  .lane-row.active {
    border-color: rgba(215, 245, 111, .8);
    background: rgba(215, 245, 111, .12);
  }

  .dot {
    width: 8px;
    height: 8px;
    margin-top: 5px;
    border-radius: 50%;
    background: #6b7280;
  }
  .lane-row.active .dot { background: #d7f56f; box-shadow: 0 0 0 4px rgba(215,245,111,.12); }
  .lane-row strong { display: block; font-size: 13px; margin-bottom: 4px; }
  .lane-row small { color: #9ca3af; line-height: 1.35; }

  .controls {
    margin-top: auto;
    padding-top: 16px;
    border-top: 1px solid rgba(255,255,255,.08);
  }

  .center {
    display: grid;
    grid-template-rows: auto auto minmax(0, 1fr);
    gap: 14px;
  }

  .topbar, .query-card, .answer-card, .tool-card, .trace-card {
    border: 1px solid #dfe3ea;
    background: rgba(255,255,255,.88);
    box-shadow: 0 18px 50px rgba(20, 28, 44, .08);
  }

  .topbar {
    border-radius: 24px;
    padding: 20px 22px;
    display: flex;
    justify-content: space-between;
    gap: 18px;
    align-items: center;
  }

  .topbar p, .card-head p {
    margin: 0 0 5px;
    color: #64748b;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: .08em;
    text-transform: uppercase;
  }

  h1, h2 { margin: 0; color: #0f172a; letter-spacing: -.03em; }
  h1 { font-size: clamp(22px, 2.1vw, 31px); line-height: 1.08; }
  h2 { font-size: 16px; }

  .primary, .query-footer button {
    border: 0;
    border-radius: 12px;
    background: #111827;
    color: #fff;
    padding: 11px 15px;
    font-weight: 800;
    font-size: 13px;
  }
  .primary.full { width: 100%; }

  .query-card { border-radius: 24px; padding: 16px; display: grid; gap: 12px; }
  .selected-strip { display: flex; flex-wrap: wrap; gap: 8px; min-height: 28px; align-items: center; }
  .selected-strip span, .citation-grid span {
    border-radius: 999px;
    background: #eef2ff;
    color: #364152;
    padding: 7px 10px;
    font-size: 12px;
    font-weight: 700;
  }
  .selected-strip em { color: #64748b; font-size: 13px; }

  textarea, input, select {
    width: 100%;
    border: 1px solid #d7dde8;
    border-radius: 12px;
    background: #ffffff;
    color: #111827;
    padding: 11px 12px;
    outline: none;
  }

  textarea:focus, input:focus, select:focus { border-color: #111827; box-shadow: 0 0 0 3px rgba(17,24,39,.08); }
  textarea { resize: vertical; line-height: 1.5; }

  .query-card textarea {
    min-height: 112px;
    border: 0;
    background: #f8fafc;
    font-size: 15px;
  }

  .query-footer {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 12px;
  }
  .query-footer span { color: #64748b; font-size: 12px; font-weight: 700; }

  .answer-card { border-radius: 24px; padding: 20px; min-height: 430px; }
  .card-head { display: flex; justify-content: space-between; gap: 12px; align-items: flex-start; margin-bottom: 14px; }
  .card-head.compact { margin-bottom: 12px; }
  .provider { border-radius: 999px; padding: 6px 10px; background: #d7f56f; color: #111827; font-size: 11px; font-weight: 900; }
  .answer-card article { white-space: pre-wrap; line-height: 1.68; color: #202938; font-size: 14px; }
  .citation-grid { margin-top: 18px; display: flex; flex-wrap: wrap; gap: 8px; }

  .empty-answer {
    min-height: 320px;
    border: 1px dashed #cfd7e3;
    border-radius: 18px;
    display: grid;
    place-items: center;
    align-content: center;
    gap: 8px;
    text-align: center;
    color: #64748b;
    padding: 28px;
  }
  .empty-answer strong { color: #0f172a; }
  .empty-answer span { max-width: 430px; line-height: 1.5; }

  .right-panel {
    display: grid;
    grid-template-rows: auto auto minmax(0, 1fr);
    gap: 14px;
  }

  .tabs {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
    padding: 6px;
    border-radius: 18px;
    background: #e8edf4;
    border: 1px solid #dbe2ec;
  }
  .tabs button {
    border: 0;
    border-radius: 13px;
    padding: 10px;
    color: #475569;
    background: transparent;
    font-weight: 800;
  }
  .tabs button.active { background: #fff; color: #111827; box-shadow: 0 8px 22px rgba(20, 28, 44, .08); }

  .tool-card, .trace-card { border-radius: 24px; padding: 18px; }
  .tool-card { display: grid; gap: 12px; }
  label { display: grid; gap: 7px; color: #475569; font-size: 12px; font-weight: 800; }
  .success { margin: 0; color: #166534; font-size: 12px; line-height: 1.4; }
  .mono, pre { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12px; }

  .trace-card { min-height: 0; display: grid; grid-template-rows: auto auto minmax(0, 1fr); }
  .trace-tabs { display: grid; gap: 8px; margin-bottom: 12px; }
  .trace-tabs button {
    border: 1px solid #dfe3ea;
    background: #f8fafc;
    border-radius: 14px;
    padding: 10px 12px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    text-transform: capitalize;
    color: #475569;
    font-weight: 800;
    text-align: left;
  }
  .trace-tabs button.active { background: #111827; color: #fff; border-color: #111827; }
  .trace-tabs strong { font-size: 12px; opacity: .8; }

  .hits { display: grid; gap: 10px; overflow: auto; padding-right: 4px; }
  .hit { border-radius: 16px; border: 1px solid #e2e8f0; background: #fbfdff; padding: 12px; }
  .hit-top { display: flex; justify-content: space-between; gap: 10px; margin-bottom: 8px; }
  .hit-top strong { color: #4f46e5; }
  .hit-top span { color: #64748b; font-size: 12px; font-weight: 800; }
  .hit p { margin: 0 0 8px; color: #334155; font-size: 12px; line-height: 1.5; max-height: 96px; overflow: hidden; }
  .hit small { color: #64748b; line-height: 1.45; }
  .trace-empty { border: 1px dashed #cfd7e3; border-radius: 16px; color: #64748b; padding: 18px; text-align: center; font-size: 13px; }

  .alert {
    border-radius: 16px;
    border: 1px solid #fecaca;
    background: #fff1f2;
    color: #9f1239;
    padding: 12px 14px;
    font-size: 13px;
  }

  .gate { display: inline-block; margin-top: 12px; border-radius: 999px; padding: 7px 12px; font-weight: 900; }
  .pass { background: #dcfce7; color: #166534; }
  .fail { background: #fee2e2; color: #991b1b; }

  @media (max-width: 1240px) {
    .workspace { grid-template-columns: 270px minmax(0, 1fr); }
    .right-panel { grid-column: 1 / -1; grid-template-columns: 360px minmax(0, 1fr); min-height: auto; }
    .tabs { grid-column: 1 / -1; }
  }

  @media (max-width: 820px) {
    .workspace { grid-template-columns: 1fr; padding: 10px; }
    .sidebar, .center, .right-panel { min-height: auto; }
    .right-panel { grid-template-columns: 1fr; }
    .topbar, .query-footer { display: grid; }
  }
</style>
