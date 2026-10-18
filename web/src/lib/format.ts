export function formatMetric(v: number | null | undefined): string {
  return v === null || v === undefined ? "—" : v.toFixed(3);
}

export function formatMs(ms: number): string {
  return ms >= 1000 ? `${(ms / 1000).toFixed(2)} s` : `${ms.toFixed(1)} ms`;
}

export function formatUsd(v: number): string {
  if (v === 0) return "$0";
  return v < 1 ? `$${Number(v.toPrecision(2))}` : `$${v.toFixed(2)}`;
}

const METHOD_LABELS: Record<string, string> = {
  dense: "Dense",
  bm25: "BM25",
  hybrid: "Hybrid",
  hybrid_rerank: "Hybrid + rerank",
};

export function methodLabel(m: string): string {
  return METHOD_LABELS[m] ?? m;
}

export function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / (1024 * 1024)).toFixed(1)} MB`;
}
