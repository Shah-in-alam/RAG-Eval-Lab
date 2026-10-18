import { CheckCircle2, CircleSlash, FileSearch, Info, Loader2, Search, Shuffle, Sparkles } from "lucide-react";
import { useEffect, useState } from "react";
import { api } from "@/api";
import { EmptyState, FieldSelect, LoadingCards, PageHeader } from "@/components/app/common";
import { HitCard } from "@/components/app/hit-card";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardAction, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Separator } from "@/components/ui/separator";
import { Switch } from "@/components/ui/switch";
import { Textarea } from "@/components/ui/textarea";
import { formatMs, methodLabel } from "@/lib/format";
import type { Collection, EvalItem, Health, Method, Options, QueryResponse } from "@/types";

const COLLECTION_LABEL: Record<Collection, string> = { all: "All documents", uploads: "My documents", builtin: "Wikipedia set" };

function AnswerText({ text }: { text: string }) {
  return (
    <p className="text-[15px] leading-7">
      {text.split(/(\[\d+\])/g).map((p, i) => {
        const m = p.match(/^\[(\d+)\]$/);
        return m ? (
          <button key={i} aria-label={`Go to passage ${m[1]}`}
            onClick={() => document.getElementById(`hit-${m[1]}`)?.scrollIntoView({ behavior: "smooth", block: "center" })}
            className="mx-0.5 inline-flex h-5 min-w-5 items-center justify-center rounded bg-primary/10 px-1 align-[1px] font-mono text-[11px] font-semibold text-primary hover:bg-primary/20">
            {m[1]}
          </button>
        ) : <span key={i}>{p}</span>;
      })}
    </p>
  );
}

export function Ask({ options, health }: { options: Options; health: Health | null }) {
  const nUploads = health?.n_uploads ?? 0;
  const [question, setQuestion] = useState("");
  const [collection, setCollection] = useState<Collection>("all");
  const [chunker, setChunker] = useState(options.chunkers.find((c) => c.name === "fixed-200-o15")?.name ?? options.chunkers[0].name);
  const [embedder, setEmbedder] = useState(options.embedders.includes("bge-small") ? "bge-small" : options.embedders[0]);
  const [method, setMethod] = useState<Method>("hybrid");
  const [k, setK] = useState("5");
  const [generate, setGenerate] = useState(true);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<QueryResponse | null>(null);
  const [examples, setExamples] = useState<EvalItem[]>([]);

  useEffect(() => { api.evalItems().then(setExamples).catch(() => setExamples([])); }, []);

  async function submit(e?: React.FormEvent) {
    e?.preventDefault();
    if (!question.trim()) { setError("Type a question first."); return; }
    setLoading(true);
    setError(null);
    try {
      const spec = options.chunkers.find((c) => c.name === chunker)!;
      setResult(await api.query({ question, chunker: spec, embedder, method, k: Number(k), generate, collection }));
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setLoading(false);
    }
  }

  const cited = new Set(result?.answer?.citations ?? []);
  const firstRelevant = result?.hits.find((h) => h.relevant)?.rank;
  const maxScore = Math.max(...(result?.hits.map((h) => h.score) ?? [0]));
  const t = result?.timings;
  const total = t ? t.index_ms + t.retrieval_ms + t.generation_ms : 0;

  return (
    <div className="grid gap-6">
      <PageHeader title="Ask"
        description={nUploads
          ? `Search your ${nUploads} uploaded document${nUploads === 1 ? "" : "s"} and the 14-article Wikipedia set, and see exactly which passages the answer came from.`
          : "Query 14 Wikipedia articles on search, embeddings and language models, or upload your own on the Documents page."} />

      <Card>
        <CardContent>
          <form onSubmit={submit} aria-label="Search" className="grid gap-4">
            <div className="flex flex-col gap-2 sm:flex-row">
              <div className="relative flex-1">
                <Search className="pointer-events-none absolute left-3 top-3 size-4 text-muted-foreground" />
                <Label htmlFor="q" className="sr-only">Question</Label>
                <Textarea id="q" rows={1} value={question} onChange={(e) => setQuestion(e.target.value)}
                  placeholder="Ask a question, e.g. What value is the BM25 parameter b usually set to?"
                  className="min-h-10 resize-none py-2.5 pl-9 text-[15px]"
                  onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); submit(); } }} />
              </div>
              <Button type="submit" size="lg" className="h-10 px-5" disabled={loading}>
                {loading ? <><Loader2 className="animate-spin" />Searching</> : <><Search />Search</>}
              </Button>
            </div>
            <div className="grid grid-cols-2 gap-3 md:grid-cols-3 xl:grid-cols-6 xl:items-end">
              <FieldSelect id="collection" label="Search in" value={collection} onChange={(v) => setCollection(v as Collection)} options={[
                { value: "all", label: "All documents" },
                { value: "uploads", label: `My documents${nUploads ? ` (${nUploads})` : " (none yet)"}`, disabled: !nUploads },
                { value: "builtin", label: "Wikipedia set" },
              ]} />
              <FieldSelect id="chunker" label="Chunking" value={chunker} onChange={setChunker}
                options={options.chunkers.map((c) => ({ value: c.name, label: c.name }))} />
              <FieldSelect id="method" label="Retrieval" value={method} onChange={(v) => setMethod(v as Method)}
                options={options.methods.map((m) => ({ value: m, label: methodLabel(m) }))} />
              <FieldSelect id="embedder" label="Embedding" value={embedder} onChange={setEmbedder} disabled={method === "bm25"}
                options={options.embedders.map((m) => ({ value: m, label: m }))} />
              <FieldSelect id="k" label="Top-k" value={k} onChange={setK} options={options.ks.map((x) => ({ value: String(x), label: String(x) }))} />
              <div className="flex h-9 items-center gap-2">
                <Switch id="generate" checked={generate} onCheckedChange={setGenerate} />
                <Label htmlFor="generate" className="text-sm font-normal">Generate answer</Label>
              </div>
            </div>
            <div className="flex flex-wrap items-center justify-between gap-2">
              <Button type="button" variant="link" size="sm" className="h-auto px-0" disabled={!examples.length}
                onClick={() => setQuestion(examples[Math.floor(Math.random() * examples.length)].question)}>
                <Shuffle />Try a question from the eval set
              </Button>
              <span className="text-xs text-muted-foreground">Enter to search · Shift+Enter for a new line · a new chunking/embedding builds its index on first use</span>
            </div>
          </form>
        </CardContent>
      </Card>

      <div aria-live="polite">
        {error && <Alert variant="destructive"><Info /><AlertDescription>{error}</AlertDescription></Alert>}
        {loading && !result && <LoadingCards />}
        {!result && !error && !loading && (
          <Card><EmptyState icon={<FileSearch />} title="No search yet">
            Results show the generated answer, the ranked passages behind it, and how long each step took.
          </EmptyState></Card>
        )}
        {result && (
          <div className={`grid gap-6 lg:grid-cols-[minmax(0,1fr)_320px] lg:items-start ${loading ? "opacity-60" : ""}`}>
            <div className="grid gap-6">
              {result.answer && (
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2"><Sparkles className="size-4 text-primary" />Answer</CardTitle>
                    <CardAction>
                      <Badge variant={result.answer.generator === "llm" ? "default" : "outline"}>
                        {result.answer.generator === "llm" ? result.answer.model : "Extractive"}
                      </Badge>
                    </CardAction>
                  </CardHeader>
                  <CardContent className="grid gap-3">
                    <AnswerText text={result.answer.text} />
                    {result.answer.generator === "extractive" && health && !health.llm_available && (
                      <Alert><Info /><AlertDescription>
                        No Claude API key is configured, so this is the best-matching sentence from the passages. Set
                        <code className="mx-1 font-mono">ANTHROPIC_API_KEY</code>and restart the API for generated answers with citations.
                      </AlertDescription></Alert>
                    )}
                    {result.answer.notes.map((n) => <Alert key={n}><Info /><AlertDescription>{n}</AlertDescription></Alert>)}
                  </CardContent>
                </Card>
              )}
              <Card className="gap-0 pb-0">
                <CardHeader className="pb-4">
                  <CardTitle>Retrieved passages</CardTitle>
                  <CardDescription>Top {result.hits.length} from {COLLECTION_LABEL[result.config.collection].toLowerCase()}, best first</CardDescription>
                </CardHeader>
                <div className="border-t">
                  {result.hits.map((h) => (
                    <HitCard key={h.chunk_id} hit={h} question={result.question} cited={cited.has(h.rank)}
                      evidence={result.eval_item?.evidence} maxScore={maxScore} />
                  ))}
                </div>
              </Card>
            </div>

            <div className="grid gap-6">
              {result.eval_item && (
                <Card>
                  <CardHeader>
                    <CardTitle>Eval check</CardTitle>
                    <CardDescription>Question {result.eval_item.id} · {result.eval_item.type}</CardDescription>
                  </CardHeader>
                  <CardContent className="grid gap-3 text-sm">
                    <div>{firstRelevant
                      ? <Badge className="bg-success-soft text-success"><CheckCircle2 />Evidence at rank {firstRelevant}</Badge>
                      : <Badge className="bg-warning-soft text-warning"><CircleSlash />Evidence not in top {result.config.k}</Badge>}</div>
                    <div>
                      <div className="mb-1 text-xs text-muted-foreground">Reference answer</div>
                      <div>{result.eval_item.answer}</div>
                    </div>
                    <div className="text-xs text-muted-foreground">Source: {result.eval_item.doc_id}</div>
                  </CardContent>
                </Card>
              )}
              <Card>
                <CardHeader><CardTitle>Run details</CardTitle></CardHeader>
                <CardContent className="grid gap-4 text-sm">
                  <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-2">
                    {([["Searched", COLLECTION_LABEL[result.config.collection]], ["Chunking", result.config.chunker],
                      ["Retrieval", methodLabel(result.config.method)], ["Embedding", result.config.method === "bm25" ? "—" : result.config.embedder],
                      ["Top-k", String(result.config.k)]] as const).map(([k2, v]) => (
                      <div key={k2} className="contents"><dt className="text-muted-foreground">{k2}</dt><dd className="text-right font-mono text-xs leading-5">{v}</dd></div>
                    ))}
                  </dl>
                  {t && (
                    <>
                      <Separator />
                      <div className="flex h-1.5 gap-0.5 overflow-hidden rounded-full bg-muted" aria-hidden>
                        {total > 0 && ([["index_ms", "var(--muted-foreground)"], ["retrieval_ms", "var(--chart-1)"], ["generation_ms", "var(--chart-3)"]] as const)
                          .map(([key, c]) => <span key={key} style={{ width: `${(t[key] / total) * 100}%`, background: c }} />)}
                      </div>
                      <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-2">
                        {([["Index lookup", t.index_ms, "var(--muted-foreground)"], ["Retrieval", t.retrieval_ms, "var(--chart-1)"],
                          ["Answer", t.generation_ms, "var(--chart-3)"]] as const).map(([label, v, c]) => (
                          <div key={label} className="contents">
                            <dt className="flex items-center gap-2 text-muted-foreground"><span className="size-2 rounded-sm" style={{ background: c }} />{label}</dt>
                            <dd className="text-right font-mono text-xs leading-5">{formatMs(v)}</dd>
                          </div>
                        ))}
                      </dl>
                    </>
                  )}
                </CardContent>
              </Card>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
