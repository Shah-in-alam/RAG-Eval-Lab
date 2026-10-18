import { describe, expect, it } from "vitest";
import { bestBy, filterRows, uniqueValues } from "./group";

const rows = [
  { method: "dense", embedder: "minilm", mrr: 0.5 },
  { method: "dense", embedder: "bge-small", mrr: 0.7 },
  { method: "bm25", embedder: "none", mrr: 0.6 },
];

describe("group helpers", () => {
  it("returns the best row per key", () => {
    const best = bestBy(rows, "method", "mrr");
    expect(best.map((r) => [r.method, r.mrr])).toEqual([["dense", 0.7], ["bm25", 0.6]]);
  });
  it("lists unique values in first-seen order", () => {
    expect(uniqueValues(rows, "method")).toEqual(["dense", "bm25"]);
  });
  it("filters by several fields, empty filter means all", () => {
    expect(filterRows(rows, { method: "dense", embedder: "" })).toHaveLength(2);
    expect(filterRows(rows, { method: "dense", embedder: "minilm" })).toHaveLength(1);
    expect(filterRows(rows, {})).toHaveLength(3);
  });
});
