const API_BASE = (import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000').replace(/\/$/, '');
const API_KEY = import.meta.env.VITE_SERVICE_API_KEY || '';

async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  const headers: Record<string, string> = {
    'X-API-Key': API_KEY,
    ...((options.headers as Record<string, string>) || {})
  };
  const res = await fetch(`${API_BASE}${path}`, { ...options, headers });
  const text = await res.text();
  let data: any = null;
  try { data = text ? JSON.parse(text) : null; } catch { data = text; }
  if (!res.ok) {
    throw new Error(typeof data?.detail === 'string' ? data.detail : data?.detail ? JSON.stringify(data.detail) : `HTTP ${res.status}`);
  }
  return data as T;
}

async function jsonRequest<T>(path: string, payload?: unknown, method = 'POST'): Promise<T> {
  return request<T>(path, {
    method,
    headers: { 'Content-Type': 'application/json' },
    body: payload === undefined ? undefined : JSON.stringify(payload)
  });
}

export type RetrievalMode = 'vector' | 'sparse' | 'hybrid' | 'hybrid_rerank';

export interface LaneInfo {
  lane: string;
  name: string;
  description: string;
  collection: string;
  editable: boolean;
}

export interface ProQueryPayload {
  query: string;
  top_k: number;
  retrieval_mode: RetrievalMode;
  retrieval_lanes: string[];
  generate_answer: boolean;
  return_trace: boolean;
}

export interface FunfChatPayload {
  query: string;
  top_k: number;
  retrieval_mode: RetrievalMode;
  selected_lanes: string[];
  generate_answer: boolean;
  return_trace: boolean;
}

export async function getLanes() {
  return request<{ product: string; lanes: LaneInfo[] }>('/v1/funf/lanes');
}

export async function uploadLaneDocument(form: FormData) {
  return request<any>('/v1/funf/upload', { method: 'POST', body: form });
}

export async function runFunfChat(payload: FunfChatPayload) {
  return jsonRequest<any>('/v1/funf/chat', payload);
}

export async function runProQuery(payload: ProQueryPayload) {
  return jsonRequest<any>('/v1/pro/query', payload);
}

export async function getCapabilities() {
  return request<any>('/v1/pro/capabilities');
}

export async function runEval(payload: any) {
  return jsonRequest<any>('/v1/pro/eval/run', payload);
}
