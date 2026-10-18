import { ArrowDown, ArrowUp, ArrowUpDown, FlaskConical, Gauge, Loader2, Play, RotateCcw, Timer, Trophy } from "lucide-react";
import { useCallback, useEffect, useMemo, useState } from "react";
import { toast } from "sonner";
import { api } from "@/api";
import { MetricByKChart, RankedBarChart, type KPoint, type RankedBar } from "@/components/app/charts";
import { EmptyState, FieldSelect, LoadingCards, MethodSwatch, PageHeader, StatCard } from "@/components/app/common";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardAction, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Progress } from "@/components/ui/progress";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Sheet, SheetContent, SheetDescription, SheetHeader, SheetTitle } from "@/components/ui/sheet";
import { Switch } from "@/components/ui/switch";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { METHOD_COLOR, METHOD_ORDER } from "@/lib/colors";
import { formatMetric, formatMs, formatUsd, methodLabel } from "@/lib/format";
import { bestBy, filterRows, uniqueValues } from "@/lib/group";
import { sortRows, type SortDir } from "@/lib/sort";
import { cn } from "@/lib/utils";
import type { ConfigInfo, ExperimentStatus, Method, ResultDetail, ResultRow } from "@/types";

type MetricKey = "mrr" | "recall@k" | "recall@1" | "correctness" | "faithfulness" | "latency_ms" | "cost_per_1k_usd";
const METRICS: { key: MetricKey; label: string; higherBetter: boolean; unit: boolean }[] = [
  { key: "mrr", label: "MRR", higherBetter: true, unit: true },
  { key: "recall@k", label: "Recall@k", higherBetter: true, unit: true },
  { key: "recall@1", label: "Recall@1", higherBetter: true, unit: true },
  { key: "correctness", label: "Answer correctness", higherBetter: true, unit: true },
  { key: "faithfulness", label: "Faithfulness", higherBetter: true, unit: true },
  { key: "latency_ms", label: "Query latency", higherBetter: false, unit: false },
  { key: "cost_per_1k_usd", label: "Cost per 1k queries", higherBetter: false, unit: false },
];
const FILTERS = [
  { key: "chunker", label: "Chunking" }, { key: "embedder", label: "Embedding" },
  { key: "method", label: "Retrieval" }, { key: "k", label: "Top-k" },
] as const;
type FilterKey = (typeof FILTERS)[number]["key"];
const ALL = "__all";
const NO_FILTERS: Record<FilterKey, string> = { chunker: ALL, embedder: ALL, method: ALL, k: ALL };

function fmt(key: MetricKey, v: number | null): string {
  if (v === null) return "—";
  if (key === "latency_ms") return formatMs(v);
  if (key === "cost_per_1k_usd") return formatUsd(v);
  return formatMetric(v);
}

function RunControls({ onDone }: { onDone: () => void }) {
  const [configs, setConfigs] = useState<ConfigInfo[]>([]);
  const [choice, setChoice] = useState("quick");
  const [status, setStatus] = useState<ExperimentStatus | null>(null);

  useEffect(() => {
    api.configs().then(setConfigs).catch((e) => toast.error(e.message));
    api.status().then(setStatus).catch(() => undefined);
  }, []);
  useEffect(() => {
    if (!status?.running) return;
    const id = window.setInterval(async () => {
      try {
        const s = await api.status();
        setStatus(s);
        if (!s.running) {
          if (s.error) toast.error(`Suite ${s.suite} failed: ${s.error}`);
          else toast.success(`Suite ${s.suite} finished: ${s.total} runs`);
          onDone();
        }
      } catch { /* server restarting; keep polling */ }
    }, 1000);
    return () => window.clearInterval(id);
  }, [status?.running, onDone]);

  async function start() {
    try { setStatus(await api.run(choice)); toast(`Started suite ${choice}`); } catch (e) { toast.error((e as Error).message); }
  }
  const pct = status && status.total ? Math.round((status.done / status.total) * 100) : 0;
  const selected = configs.find((c) => c.name === choice);
  return (
    <div className="flex flex-col items-end gap-1.5">
      <div className="flex items-center gap-2">
        <Select value={choice} onValueChange={setChoice} disabled={status?.running}>
          <SelectTrigger className="w-44" aria-label="Experiment suite"><SelectValue /></SelectTrigger>
          <SelectContent>
            {configs.map((c) => <SelectItem key={c.name} value={c.name}>Suite: {c.name}</SelectItem>)}
          </SelectContent>
        </Select>
        <Button onClick={start} disabled={status?.running || !configs.length}>
          {status?.running ? <><Loader2 className="animate-spin" />Running {pct}%</> : <><Play />Run suite</>}
        </Button>
      </div>
      {status?.running
        ? <div className="w-60"><Progress value={pct} className="h-1.5" /><p className="mt-1 text-right text-xs text-muted-foreground">{status.suite}: {status.done} / {status.total || "?"} runs</p></div>
        : selected && <p className="max-w-80 text-right text-xs text-muted-foreground">{selected.description}</p>}
    </div>
  );
}

function RunSheet({ runId, onClose }: { runId: string | null; onClose: () => void }) {
  const [detail, setDetail] = useState<ResultDetail | null>(null);
  const [questions, setQuestions] = useState<Record<string, string>>({});
  const [onlyMisses, setOnlyMisses] = useState(true);

  useEffect(() => {
    if (!runId) return;
    setDetail(null);
    api.result(runId).then(setDetail).catch((e) => toast.error(e.message));
  }, [runId]);
  useEffect(() => {
    api.evalItems().then((items) => setQuestions(Object.fromEntries(items.map((i) => [i.id, i.question])))).catch(() => undefined);
  }, []);
  const rows = detail?.per_question.filter((q) => !onlyMisses || (q.first_relevant_rank ?? 99) > 1) ?? [];

  return (
    <Sheet open={!!runId} onOpenChange={(o) => !o && onClose()}>
      <SheetContent className="w-full gap-0 sm:max-w-2xl">
        <SheetHeader className="border-b">
          <SheetTitle>Run details</SheetTitle>
          <SheetDescription className="font-mono text-xs">{runId}</SheetDescription>
        </SheetHeader>
        <div className="grid gap-4 overflow-y-auto p-4">
          {!detail ? <LoadingCards /> : (
            <>
              <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
                <StatCard label={`Recall@${detail.k}`} value={formatMetric(detail["recall@k"])} detail={`${detail.n_questions} questions`} />
                <StatCard label="MRR" value={formatMetric(detail.mrr)} detail={`Recall@1 ${formatMetric(detail["recall@1"])}`} />
                {Object.entries(detail.recall_by_type).map(([t, v]) => (
                  <StatCard key={t} label={`Recall · ${t}`} value={formatMetric(v)} detail={`${detail.per_question.filter((q) => q.type === t).length} questions`} />
                ))}
              </div>
              <Card className="gap-0 py-0">
                <CardHeader className="py-4">
                  <CardTitle>Per-question results</CardTitle>
                  <CardAction className="flex items-center gap-2">
                    <Switch id="misses" checked={onlyMisses} onCheckedChange={setOnlyMisses} />
                    <Label htmlFor="misses" className="text-xs font-normal">Not ranked first only</Label>
                  </CardAction>
                </CardHeader>
                <Table>
                  <TableHeader><TableRow>
                    <TableHead className="pl-4">ID</TableHead><TableHead>Question{detail.faithfulness !== null && " · answer"}</TableHead>
                    <TableHead>Type</TableHead><TableHead className="text-right">Rank</TableHead>
                    {detail.faithfulness !== null && <TableHead className="pr-4 text-right">Correct</TableHead>}
                  </TableRow></TableHeader>
                  <TableBody>
                    {rows.map((q) => (
                      <TableRow key={q.id}>
                        <TableCell className="pl-4 align-top font-mono text-xs">{q.id}</TableCell>
                        <TableCell className="whitespace-normal">
                          <div>{questions[q.id]}</div>
                          {q.answer && <div className="mt-1 text-xs text-muted-foreground">{q.answer}</div>}
                        </TableCell>
                        <TableCell className="align-top"><Badge variant="outline">{q.type}</Badge></TableCell>
                        <TableCell className="text-right align-top font-mono">
                          {q.first_relevant_rank ?? <Badge className="bg-warning-soft text-warning">miss</Badge>}
                        </TableCell>
                        {detail.faithfulness !== null && <TableCell className="pr-4 text-right align-top font-mono">{formatMetric(q.correctness ?? null)}</TableCell>}
                      </TableRow>
                    ))}
                    {!rows.length && <TableRow><TableCell colSpan={5} className="pl-4 text-muted-foreground">Every question had its evidence at rank 1.</TableCell></TableRow>}
                  </TableBody>
                </Table>
              </Card>
            </>
          )}
        </div>
      </SheetContent>
    </Sheet>
  );
}

const COLUMNS: { key: keyof ResultRow; label: string; num?: boolean; render?: (r: ResultRow) => React.ReactNode }[] = [
  { key: "chunker", label: "Chunking", render: (r) => <span className="font-mono text-xs">{r.chunker}</span> },
  { key: "embedder", label: "Embedding", render: (r) => <span className="font-mono text-xs">{r.embedder}</span> },
  { key: "method", label: "Retrieval", render: (r) => <span className="flex items-center gap-2"><MethodSwatch color={METHOD_COLOR[r.method]} />{methodLabel(r.method)}</span> },
  { key: "k", label: "k", num: true },
  { key: "recall@k", label: "Recall@k", num: true, render: (r) => formatMetric(r["recall@k"]) },
  { key: "mrr", label: "MRR", num: true, render: (r) => formatMetric(r.mrr) },
  { key: "recall@1", label: "Recall@1", num: true, render: (r) => formatMetric(r["recall@1"]) },
  { key: "correctness", label: "Correct", num: true, render: (r) => formatMetric(r.correctness) },
  { key: "latency_ms", label: "Latency", num: true, render: (r) => formatMs(r.latency_ms) },
  { key: "index_time_s", label: "Index", num: true, render: (r) => `${r.index_time_s.toFixed(1)} s` },
  { key: "cost_per_1k_usd", label: "$ / 1k", num: true, render: (r) => formatUsd(r.cost_per_1k_usd) },
  { key: "n_chunks", label: "Chunks", num: true },
];

function RunsTable({ rows, selected, onSelect }: { rows: ResultRow[]; selected: string | null; onSelect: (id: string) => void }) {
  const [sort, setSort] = useState<{ key: keyof ResultRow; dir: SortDir }>({ key: "mrr", dir: "desc" });
  const sorted = sortRows(rows, sort.key, sort.dir);
  return (
    <div className="max-h-[560px] overflow-auto">
      <Table>
        <TableHeader className="sticky top-0 z-10 bg-card">
          <TableRow>
            {COLUMNS.map((c, i) => (
              <TableHead key={String(c.key)} className={cn(c.num && "text-right", i === 0 && "pl-6", i === COLUMNS.length - 1 && "pr-6")}
                aria-sort={sort.key === c.key ? (sort.dir === "asc" ? "ascending" : "descending") : "none"}>
                <Button variant="ghost" size="xs" className={cn("-mx-2 text-muted-foreground", c.num && "flex-row-reverse")}
                  onClick={() => setSort((s) => (s.key === c.key ? { key: c.key, dir: s.dir === "desc" ? "asc" : "desc" } : { key: c.key, dir: c.num ? "desc" : "asc" }))}>
                  {c.label}
                  {sort.key === c.key ? (sort.dir === "asc" ? <ArrowUp /> : <ArrowDown />) : <ArrowUpDown className="opacity-40" />}
                </Button>
              </TableHead>
            ))}
          </TableRow>
        </TableHeader>
        <TableBody>
          {sorted.map((r) => (
            <TableRow key={r.run_id} data-state={selected === r.run_id ? "selected" : undefined} className="cursor-pointer"
              onClick={() => onSelect(r.run_id)} tabIndex={0} onKeyDown={(e) => e.key === "Enter" && onSelect(r.run_id)}>
              {COLUMNS.map((c, i) => (
                <TableCell key={String(c.key)} className={cn(c.num && "text-right font-mono tabular", i === 0 && "pl-6", i === COLUMNS.length - 1 && "pr-6")}>
                  {c.render ? c.render(r) : String(r[c.key] ?? "—")}
                </TableCell>
              ))}
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  );
}

export function Results() {
  const [rows, setRows] = useState<ResultRow[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [metric, setMetric] = useState<MetricKey>("mrr");
  const [filters, setFilters] = useState(NO_FILTERS);
  const [selected, setSelected] = useState<string | null>(null);

  const load = useCallback(() => { api.results().then(setRows).catch((e) => setError(e.message)); }, []);
  useEffect(load, [load]);

  const active = useMemo(() => Object.fromEntries(Object.entries(filters).map(([k, v]) => [k, v === ALL ? "" : v])), [filters]);
  const filtered = useMemo(() => filterRows(rows ?? [], active), [rows, active]);
  const m = METRICS.find((x) => x.key === metric)!;
  const dir: SortDir = m.higherBetter ? "desc" : "asc";

  const bars: RankedBar[] = useMemo(() => sortRows(filtered.filter((r) => r[metric] !== null && r[metric] !== undefined), metric, dir)
    .slice(0, 12).map((r) => ({
      id: r.run_id,
      label: `${r.chunker} · ${r.method === "bm25" ? "bm25" : r.embedder} · k=${r.k}`,
      title: `${r.chunker} · ${r.method === "bm25" ? "—" : r.embedder} · ${methodLabel(r.method)} · k=${r.k}`,
      value: Number(r[metric]),
      method: r.method,
      details: ([[m.label, fmt(metric, Number(r[metric]))], ["MRR", formatMetric(r.mrr)], [`Recall@${r.k}`, formatMetric(r["recall@k"])],
        ["Query latency", formatMs(r.latency_ms)], ["Chunks", String(r.n_chunks)]] as [string, string][])
        .filter(([label], i) => i === 0 || label !== m.label),
    })), [filtered, metric, m, dir]);

  const ks = useMemo(() => [...new Set(filtered.map((r) => r.k))].sort((a, b) => a - b), [filtered]);
  const methods = useMemo(() => METHOD_ORDER.filter((x) => filtered.some((r) => r.method === x && r[metric] !== null)), [filtered, metric]);
  const kData: KPoint[] = useMemo(() => ks.map((k) => {
    const point: KPoint = { k };
    for (const meth of methods) {
      const atK = filtered.filter((r) => r.k === k && r.method === meth && r[metric] !== null);
      if (atK.length) point[meth] = Number(sortRows(atK, metric, dir)[0][metric]);
    }
    return point;
  }), [ks, methods, filtered, metric, dir]);

  const header = (
    <PageHeader title="Experiments"
      description="Every combination of chunking, embedding, retrieval method and top-k, scored on the labelled question set."
      actions={<RunControls onDone={load} />} />
  );
  if (error) return <div className="grid gap-6">{header}<Alert variant="destructive"><AlertDescription>{error}</AlertDescription></Alert></div>;
  if (!rows) return <div className="grid gap-6">{header}<LoadingCards count={3} /></div>;
  if (!rows.length) {
    return (
      <div className="grid gap-6">{header}
        <Card><EmptyState icon={<FlaskConical />} title="No experiment results yet">
          Pick a suite in the top right and click Run suite. "quick" finishes in about 15 seconds.
        </EmptyState></Card>
      </div>
    );
  }

  const best = sortRows(rows, "mrr", "desc")[0];
  const bestPerMethod = bestBy(rows, "method", "mrr");
  const fastest = sortRows(rows.filter((r) => r.mrr >= best.mrr - 0.02), "latency_ms", "asc")[0];
  const filtersActive = Object.values(filters).some((v) => v !== ALL);

  return (
    <div className="grid gap-6">
      {header}
      <div className="grid grid-cols-2 gap-4 md:grid-cols-3 xl:grid-cols-6">
        <StatCard label="Best MRR" icon={<Trophy />} value={formatMetric(best.mrr)}
          detail={`${best.chunker} · ${best.method === "bm25" ? "" : `${best.embedder} · `}${methodLabel(best.method)} · k=${best.k}`} />
        {METHOD_ORDER.map((meth) => bestPerMethod.find((r) => r.method === meth)).filter((r): r is ResultRow => !!r).map((r) => (
          <StatCard key={r.method} label={methodLabel(r.method)} icon={<MethodSwatch color={METHOD_COLOR[r.method as Method]} />}
            value={formatMetric(r.mrr)} detail={`${r.chunker}${r.method !== "bm25" ? ` · ${r.embedder}` : ""} · k=${r.k}`} />
        ))}
        {fastest && <StatCard label="Fastest near best" icon={<Timer />} value={formatMs(fastest.latency_ms)}
          detail={`${fastest.chunker} · ${methodLabel(fastest.method)} · MRR ${formatMetric(fastest.mrr)}`} />}
      </div>

      <Card className="py-4">
        <CardContent className="grid grid-cols-2 items-end gap-3 px-4 md:grid-cols-3 xl:grid-cols-[repeat(5,minmax(0,1fr))_auto]">
          <FieldSelect id="metric" label="Metric" value={metric} onChange={(v) => setMetric(v as MetricKey)}
            options={METRICS.map((x) => ({ value: x.key, label: x.label }))} />
          {FILTERS.map(({ key, label }) => (
            <FieldSelect key={key} id={`f-${key}`} label={label} value={filters[key]} onChange={(v) => setFilters({ ...filters, [key]: v })}
              options={[{ value: ALL, label: "All" }, ...uniqueValues(rows, key)
                .sort((a, b) => (key === "k" ? Number(a) - Number(b) : a.localeCompare(b)))
                .map((v) => ({ value: v, label: key === "method" ? methodLabel(v) : v }))]} />
          ))}
          <Button variant="ghost" onClick={() => setFilters(NO_FILTERS)} disabled={!filtersActive}><RotateCcw />Reset</Button>
        </CardContent>
      </Card>

      <div className="grid gap-6 xl:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Top configurations · {m.label}</CardTitle>
            <CardDescription>{m.higherBetter ? "Higher" : "Lower"} is better · click a bar for per-question details</CardDescription>
            <CardAction><Badge variant="secondary" className="font-mono">{filtered.length} runs</Badge></CardAction>
          </CardHeader>
          <CardContent>
            {bars.length
              ? <RankedBarChart bars={bars} max={m.unit ? 1 : undefined} format={(v) => fmt(metric, v)} onSelect={setSelected} />
              : <EmptyState icon={<Gauge />} title="No data for this metric">Answer metrics need a suite with with_answers: true.</EmptyState>}
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle>{m.label} by top-k</CardTitle>
            <CardDescription>Best configuration per retrieval method at each k</CardDescription>
          </CardHeader>
          <CardContent><MetricByKChart data={kData} methods={methods} format={(v) => fmt(metric, v)} /></CardContent>
        </Card>
      </div>

      <Card className="gap-0 pb-0">
        <CardHeader className="pb-4">
          <CardTitle>All runs</CardTitle>
          <CardDescription>Sort by any column · click a row for per-question details</CardDescription>
          <CardAction><span className="font-mono text-xs text-muted-foreground">{filtered.length} of {rows.length}</span></CardAction>
        </CardHeader>
        <div className="border-t"><RunsTable rows={filtered} selected={selected} onSelect={setSelected} /></div>
      </Card>
      <RunSheet runId={selected} onClose={() => setSelected(null)} />
    </div>
  );
}
