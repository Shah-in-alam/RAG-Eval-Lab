import { describe, expect, it } from "vitest";
import { sortRows } from "./sort";

const rows = [
  { id: "a", mrr: 0.5, faithfulness: null },
  { id: "b", mrr: 0.9, faithfulness: 0.7 },
  { id: "c", mrr: 0.1, faithfulness: 0.9 },
];

describe("sortRows", () => {
  it("sorts numerically descending", () => {
    expect(sortRows(rows, "mrr", "desc").map((r) => r.id)).toEqual(["b", "a", "c"]);
  });
  it("sorts ascending", () => {
    expect(sortRows(rows, "mrr", "asc").map((r) => r.id)).toEqual(["c", "a", "b"]);
  });
  it("puts missing values last in both directions", () => {
    expect(sortRows(rows, "faithfulness", "desc").map((r) => r.id)).toEqual(["c", "b", "a"]);
    expect(sortRows(rows, "faithfulness", "asc").map((r) => r.id)).toEqual(["b", "c", "a"]);
  });
  it("does not mutate the input", () => {
    const copy = [...rows];
    sortRows(rows, "mrr", "asc");
    expect(rows).toEqual(copy);
  });
  it("sorts strings alphabetically", () => {
    const s = [{ n: "b" }, { n: "a" }];
    expect(sortRows(s, "n", "asc").map((r) => r.n)).toEqual(["a", "b"]);
  });
});
