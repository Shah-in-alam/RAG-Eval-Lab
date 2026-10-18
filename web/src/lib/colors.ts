import type { Method } from "../types";

// Fixed slot per method: colour follows the entity, never its rank or filter state.
export const METHOD_ORDER: Method[] = ["dense", "bm25", "hybrid", "hybrid_rerank"];
export const METHOD_COLOR: Record<Method, string> = {
  dense: "var(--chart-1)",
  bm25: "var(--chart-2)",
  hybrid: "var(--chart-3)",
  hybrid_rerank: "var(--chart-4)",
};
