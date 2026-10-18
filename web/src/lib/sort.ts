export type SortDir = "asc" | "desc";

/** Sort a copy of `rows` by `key`. Numbers compare numerically, strings alphabetically;
 * null/undefined always go last regardless of direction. */
export function sortRows<T extends Record<string, unknown>>(rows: T[], key: keyof T, dir: SortDir): T[] {
  const sign = dir === "asc" ? 1 : -1;
  return [...rows].sort((a, b) => {
    const x = a[key];
    const y = b[key];
    const xMissing = x === null || x === undefined;
    const yMissing = y === null || y === undefined;
    if (xMissing || yMissing) return xMissing === yMissing ? 0 : xMissing ? 1 : -1;
    if (typeof x === "number" && typeof y === "number") return (x - y) * sign;
    return String(x).localeCompare(String(y)) * sign;
  });
}
