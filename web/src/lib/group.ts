/** Best row (highest `metric`) for each distinct value of `key`, in first-seen order. */
export function bestBy<T extends Record<string, unknown>>(rows: T[], key: keyof T, metric: keyof T): T[] {
  const best = new Map<unknown, T>();
  for (const r of rows) {
    const cur = best.get(r[key]);
    if (!cur || Number(r[metric] ?? -Infinity) > Number(cur[metric] ?? -Infinity)) best.set(r[key], r);
  }
  return [...best.values()];
}

export function uniqueValues<T extends Record<string, unknown>>(rows: T[], key: keyof T): string[] {
  return [...new Set(rows.map((r) => String(r[key])))];
}

/** Keep rows matching every non-empty filter value. */
export function filterRows<T extends Record<string, unknown>>(rows: T[], filters: Partial<Record<keyof T, string>>): T[] {
  const active = Object.entries(filters).filter(([, v]) => v) as [keyof T, string][];
  return rows.filter((r) => active.every(([k, v]) => String(r[k]) === v));
}
