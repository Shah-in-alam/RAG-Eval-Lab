// Shapes returned by the FastAPI server (api/server.py).

export type Method = "dense" | "bm25" | "hybrid" | "hybrid_rerank";

export interface ChunkerSpec {
  type: "fixed" | "heading" | "paragraph";
  size?: number;
  overlap?: number;
  name?: string;
}

export interface Health {
  status: string;
  llm_available: boolean;
  llm_model: string | null;
  embedders: string[];
  n_documents: number;
  n_uploads: number;
  upload_types: string[];
  max_upload_mb: number;
}

export type Collection = "all" | "builtin" | "uploads";

export interface UploadMeta {
  doc_id: string;
  title: string;
  filename: string;
  kind: string;
  size_bytes: number;
  n_words: number;
  n_chars: number;
  source: "upload" | "link";
  url: string | null;
  uploaded_at: string;
}

export interface UploadResult {
  added: UploadMeta[];
  errors: { filename: string; error: string }[];
}

export interface Options {
  chunkers: (ChunkerSpec & { name: string })[];
  embedders: string[];
  methods: Method[];
  ks: number[];
}

export interface Hit {
  rank: number;
  chunk_id: string;
  doc_id: string;
  title: string;
  text: string;
  score: number;
  start: number;
  end: number;
  relevant: boolean | null;
}

export interface Answer {
  text: string;
  citations: number[];
  generator: "llm" | "extractive";
  latency_ms: number;
  input_tokens: number;
  output_tokens: number;
  model: string | null;
  notes: string[];
}

export interface EvalItem {
  id: string;
  question: string;
  answer: string;
  evidence: string;
  doc_id: string;
  source: "manual" | "generated";
  type: "factoid" | "explanatory";
}

export interface QueryRequest {
  question: string;
  chunker: ChunkerSpec;
  embedder: string;
  method: Method;
  k: number;
  generate: boolean;
  collection: Collection;
}

export interface QueryResponse {
  question: string;
  config: { chunker: string; embedder: string; method: Method; k: number; collection: Collection };
  hits: Hit[];
  answer: Answer | null;
  eval_item: EvalItem | null;
  timings: { index_ms: number; retrieval_ms: number; generation_ms: number };
}

export interface ResultRow {
  [key: string]: unknown;
  run_id: string;
  chunker: string;
  embedder: string;
  method: Method;
  k: number;
  n_questions: number;
  n_chunks: number;
  avg_chunk_words: number;
  "recall@k": number;
  mrr: number;
  "recall@1": number;
  recall_by_type: Record<string, number>;
  faithfulness: number | null;
  correctness: number | null;
  generator: string | null;
  judge: string | null;
  index_time_s: number;
  latency_ms: number;
  retrieval_latency_ms: number;
  cost_per_1k_usd: number;
  index_cost_usd: number;
  timestamp: string;
}

export interface PerQuestion {
  id: string;
  type: string;
  recall: number;
  rr: number;
  first_relevant_rank: number | null;
  hits: string[];
  answer?: string;
  faithfulness?: number;
  correctness?: number;
}

export interface ResultDetail extends ResultRow {
  per_question: PerQuestion[];
}

export interface DocumentInfo {
  doc_id: string;
  title: string;
  n_words: number;
  n_chars: number;
  collection: "builtin" | "uploads";
  kind?: string;
  filename?: string;
}

export interface ExperimentStatus {
  running: boolean;
  suite: string | null;
  done: number;
  total: number;
  error: string | null;
  finished_at: string | null;
}

export interface ConfigInfo {
  name: string;
  description: string;
  chunkers: number;
  embedders: string[];
  methods: string[];
  ks: number[];
}
