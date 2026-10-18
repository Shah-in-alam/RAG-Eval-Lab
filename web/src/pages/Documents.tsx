import { BookOpen, ExternalLink, FileText, FolderOpen, HelpCircle, Search, Trash2, Upload } from "lucide-react";
import { useCallback, useEffect, useMemo, useState } from "react";
import { toast } from "sonner";
import { api } from "@/api";
import { EmptyState, LoadingCards, PageHeader, StatCard } from "@/components/app/common";
import { UploadPanel } from "@/components/app/upload-panel";
import {
  AlertDialog, AlertDialogAction, AlertDialogCancel, AlertDialogContent, AlertDialogDescription, AlertDialogFooter,
  AlertDialogHeader, AlertDialogTitle, AlertDialogTrigger,
} from "@/components/ui/alert-dialog";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardAction, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { formatBytes } from "@/lib/format";
import { cn } from "@/lib/utils";
import type { DocumentInfo, EvalItem, Health, UploadMeta } from "@/types";

/** Plain text with `== Heading ==` lines rendered as headings; optionally highlight a passage. */
function ArticleText({ text, highlight }: { text: string; highlight: string | null }) {
  const idx = highlight ? text.indexOf(highlight) : -1;
  const render = (chunk: string, keyBase: string) => chunk.split(/^(=+ .+? =+)$/m).map((part, i) =>
    /^=+ .+ =+$/.test(part)
      ? <span key={`${keyBase}${i}`} className="mt-3 block text-sm font-semibold text-foreground">{part.replace(/=/g, "").trim()}</span>
      : part);
  if (idx < 0) return <>{render(text, "a")}</>;
  return (
    <>
      {render(text.slice(0, idx), "a")}
      <mark ref={(el) => el?.scrollIntoView({ block: "center" })}>{highlight}</mark>
      {render(text.slice(idx + (highlight?.length ?? 0)), "b")}
    </>
  );
}

function DeleteDocument({ name, onConfirm }: { name: string; onConfirm: () => void }) {
  return (
    <AlertDialog>
      <AlertDialogTrigger asChild>
        <Button variant="ghost" size="icon-sm" aria-label={`Delete ${name}`} onClick={(e) => e.stopPropagation()}><Trash2 /></Button>
      </AlertDialogTrigger>
      <AlertDialogContent onClick={(e) => e.stopPropagation()}>
        <AlertDialogHeader>
          <AlertDialogTitle>Delete {name}?</AlertDialogTitle>
          <AlertDialogDescription>
            The extracted text is removed from this computer and the document disappears from search. The original file is not affected.
          </AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogFooter>
          <AlertDialogCancel>Cancel</AlertDialogCancel>
          <AlertDialogAction variant="destructive" onClick={onConfirm}>Delete</AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
}

function LibraryButton({ doc, active, meta, onClick }: { doc: DocumentInfo; active: boolean; meta: string; onClick: () => void }) {
  return (
    <button onClick={onClick} aria-current={active}
      className={cn("grid w-full gap-0.5 rounded-md px-2.5 py-2 text-left text-sm transition-colors hover:bg-muted", active && "bg-primary/10 hover:bg-primary/10")}>
      <span className={cn("truncate font-medium", active && "text-primary")}>{doc.title}</span>
      <span className="text-xs text-muted-foreground">{meta}</span>
    </button>
  );
}

export function Documents({ health, onChanged }: { health: Health | null; onChanged: () => void }) {
  const [docs, setDocs] = useState<DocumentInfo[] | null>(null);
  const [uploads, setUploads] = useState<UploadMeta[]>([]);
  const [items, setItems] = useState<EvalItem[]>([]);
  const [current, setCurrent] = useState<string | null>(null);
  const [text, setText] = useState("");
  const [query, setQuery] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [highlight, setHighlight] = useState<string | null>(null);

  const reload = useCallback(async (select?: string) => {
    try {
      const [d, u, e] = await Promise.all([api.documents(), api.uploads(), api.evalItems()]);
      setDocs(d); setUploads(u); setItems(e);
      setCurrent((cur) => select ?? (cur && d.some((x) => x.doc_id === cur) ? cur : d[0]?.doc_id ?? null));
    } catch (err) {
      setError((err as Error).message);
    }
  }, []);
  useEffect(() => { reload(); }, [reload]);
  useEffect(() => {
    if (current) api.document(current).then((d) => setText(d.text)).catch((e) => setError(e.message));
  }, [current]);

  const shown = useMemo(() => {
    const q = query.trim().toLowerCase();
    return items.filter((it) => (q ? it.question.toLowerCase().includes(q) || it.answer.toLowerCase().includes(q) : it.doc_id === current));
  }, [items, query, current]);
  const counts = useMemo(() => {
    const c: Record<string, number> = {};
    for (const it of items) c[it.doc_id] = (c[it.doc_id] ?? 0) + 1;
    return c;
  }, [items]);

  async function remove(u: UploadMeta) {
    try {
      await api.deleteUpload(u.doc_id);
      toast.success(`Deleted ${u.filename}`);
      setHighlight(null);
      await reload();
      onChanged();
    } catch (e) {
      toast.error((e as Error).message);
    }
  }

  const header = <PageHeader title="Documents"
    description="Add your own files or links, browse everything that can be searched, and see the labelled questions experiments are scored on." />;
  if (!docs) return <div className="grid gap-6">{header}{error ? <Alert variant="destructive"><AlertDescription>{error}</AlertDescription></Alert> : <LoadingCards count={3} />}</div>;

  const builtin = docs.filter((d) => d.collection === "builtin");
  const mine = docs.filter((d) => d.collection === "uploads");
  const currentDoc = docs.find((d) => d.doc_id === current);
  const words = (list: { n_words: number }[]) => list.reduce((s, d) => s + d.n_words, 0).toLocaleString();
  const select = (id: string) => { setCurrent(id); setHighlight(null); };
  const openQuestion = (it: EvalItem) => { setCurrent(it.doc_id); setHighlight(it.evidence); };

  return (
    <div className="grid gap-6">
      {header}
      {error && <Alert variant="destructive"><AlertDescription>{error}</AlertDescription></Alert>}
      <div className="grid grid-cols-2 gap-4 xl:grid-cols-4">
        <StatCard label="My documents" icon={<FolderOpen />} value={uploads.length} detail={uploads.length ? `${words(uploads)} words` : "none uploaded yet"} />
        <StatCard label="Wikipedia set" icon={<FileText />} value={builtin.length} detail={`${words(builtin)} words · CC BY-SA 4.0`} />
        <StatCard label="Eval questions" icon={<HelpCircle />} value={items.length}
          detail={`${items.filter((i) => i.type === "factoid").length} factoid · ${items.filter((i) => i.type === "explanatory").length} explanatory`} />
        <StatCard label="Searchable total" icon={<BookOpen />} value={docs.length} detail={`${words(docs)} words`} />
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2"><Upload className="size-4" />Add documents</CardTitle>
          <CardDescription>Uploaded documents are indexed on first search and can be searched on the Ask page.</CardDescription>
        </CardHeader>
        <CardContent>
          <UploadPanel types={health?.upload_types ?? []} maxMb={health?.max_upload_mb ?? 25}
            onAdded={(added) => { reload(added[0]?.doc_id); setHighlight(null); onChanged(); }} />
        </CardContent>
      </Card>

      <Card className="gap-0 pb-0">
        <CardHeader className="pb-4">
          <CardTitle>My documents</CardTitle>
          <CardDescription>Stored on this computer in data/uploads</CardDescription>
          <CardAction><Badge variant="secondary" className="font-mono">{uploads.length}</Badge></CardAction>
        </CardHeader>
        <div className="border-t">
          {uploads.length ? (
            <Table>
              <TableHeader><TableRow>
                <TableHead className="pl-6">Name</TableHead><TableHead>Type</TableHead><TableHead className="text-right">Words</TableHead>
                <TableHead className="text-right">Size</TableHead><TableHead>Source</TableHead><TableHead>Added</TableHead><TableHead className="w-12 pr-6" />
              </TableRow></TableHeader>
              <TableBody>
                {uploads.map((u) => (
                  <TableRow key={u.doc_id} data-state={current === u.doc_id ? "selected" : undefined} className="cursor-pointer"
                    onClick={() => select(u.doc_id)} tabIndex={0} onKeyDown={(e) => e.key === "Enter" && select(u.doc_id)}>
                    <TableCell className="max-w-80 truncate pl-6 font-medium" title={u.filename}>{u.filename}</TableCell>
                    <TableCell><Badge variant="outline" className="font-mono">{u.kind.toUpperCase()}</Badge></TableCell>
                    <TableCell className="text-right font-mono tabular">{u.n_words.toLocaleString()}</TableCell>
                    <TableCell className="text-right font-mono tabular">{formatBytes(u.size_bytes)}</TableCell>
                    <TableCell>{u.source === "link"
                      ? <a href={u.url ?? "#"} target="_blank" rel="noreferrer" onClick={(e) => e.stopPropagation()}
                          className="inline-flex items-center gap-1 text-primary hover:underline">Link<ExternalLink className="size-3" /></a>
                      : <span className="text-muted-foreground">Upload</span>}</TableCell>
                    <TableCell className="font-mono text-xs text-muted-foreground">{u.uploaded_at.replace("T", " ").slice(0, 16)}</TableCell>
                    <TableCell className="pr-6 text-right"><DeleteDocument name={u.filename} onConfirm={() => remove(u)} /></TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          ) : <EmptyState icon={<FolderOpen />} title="No documents yet">Upload files or import a link above.</EmptyState>}
        </div>
      </Card>

      <div className="grid gap-6 lg:grid-cols-[280px_minmax(0,1fr)] lg:items-start">
        <Card className="gap-0 py-0">
          <CardHeader className="py-4"><CardTitle>Library</CardTitle></CardHeader>
          <ScrollArea className="h-[560px] border-t">
            <div className="grid gap-0.5 p-2">
              {mine.length > 0 && <p className="px-2.5 pb-1 pt-2 text-xs font-medium text-muted-foreground">My documents</p>}
              {mine.map((d) => <LibraryButton key={d.doc_id} doc={d} active={current === d.doc_id} onClick={() => select(d.doc_id)}
                meta={`${d.kind?.toUpperCase()} · ${d.n_words.toLocaleString()} words`} />)}
              <p className="px-2.5 pb-1 pt-2 text-xs font-medium text-muted-foreground">Wikipedia set</p>
              {builtin.map((d) => <LibraryButton key={d.doc_id} doc={d} active={current === d.doc_id}
                onClick={() => { select(d.doc_id); setQuery(""); }} meta={`${d.n_words.toLocaleString()} words · ${counts[d.doc_id] ?? 0} questions`} />)}
            </div>
          </ScrollArea>
        </Card>
        <Card className="gap-0 py-0">
          <CardHeader className="py-4">
            <CardTitle className="truncate">{currentDoc?.title}</CardTitle>
            <CardDescription className="font-mono text-xs">
              {currentDoc && `${currentDoc.n_words.toLocaleString()} words · ${currentDoc.collection === "uploads" ? currentDoc.filename : `${currentDoc.doc_id}.txt`}`}
            </CardDescription>
            <CardAction>
              {highlight ? <Badge>Evidence highlighted</Badge> : currentDoc?.collection === "uploads" ? <Badge variant="secondary">Your document</Badge> : null}
            </CardAction>
          </CardHeader>
          <ScrollArea className="h-[560px] border-t">
            <div className="whitespace-pre-wrap break-words px-6 py-4 text-sm leading-7 text-muted-foreground"><ArticleText text={text} highlight={highlight} /></div>
          </ScrollArea>
        </Card>
      </div>

      <Card className="gap-0 pb-0">
        <CardHeader className="pb-4">
          <CardTitle>Evaluation questions</CardTitle>
          <CardDescription>
            {query ? `Matching "${query}" across all documents`
              : currentDoc?.collection === "uploads" ? "Uploaded documents have no labelled questions; search all questions instead"
              : `Questions for ${currentDoc?.title ?? "this document"} · click one to highlight its evidence`}
          </CardDescription>
          <CardAction>
            <div className="relative w-64">
              <Search className="pointer-events-none absolute left-2.5 top-2.5 size-4 text-muted-foreground" />
              <Input type="search" aria-label="Search questions" value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Search all questions" className="pl-8" />
            </div>
          </CardAction>
        </CardHeader>
        <div className="max-h-[480px] overflow-auto border-t">
          <Table>
            <TableHeader className="sticky top-0 bg-card"><TableRow>
              <TableHead className="pl-6">ID</TableHead><TableHead>Question</TableHead><TableHead>Reference answer</TableHead>
              <TableHead>Type</TableHead><TableHead>Source</TableHead><TableHead className="pr-6">Document</TableHead>
            </TableRow></TableHeader>
            <TableBody>
              {shown.map((it) => (
                <TableRow key={it.id} className="cursor-pointer" onClick={() => openQuestion(it)} tabIndex={0}
                  onKeyDown={(e) => e.key === "Enter" && openQuestion(it)} data-eval-row="">
                  <TableCell className="pl-6 align-top font-mono text-xs">{it.id}</TableCell>
                  <TableCell className="min-w-64 whitespace-normal align-top">{it.question}</TableCell>
                  <TableCell className="min-w-64 whitespace-normal align-top text-muted-foreground">{it.answer}</TableCell>
                  <TableCell className="align-top"><Badge variant="outline">{it.type}</Badge></TableCell>
                  <TableCell className="align-top"><Badge variant={it.source === "manual" ? "default" : "secondary"}>{it.source}</Badge></TableCell>
                  <TableCell className="pr-6 align-top font-mono text-xs text-muted-foreground">{it.doc_id}</TableCell>
                </TableRow>
              ))}
              {!shown.length && <TableRow><TableCell colSpan={6} className="pl-6 text-muted-foreground">No questions match.</TableCell></TableRow>}
            </TableBody>
          </Table>
        </div>
      </Card>
    </div>
  );
}
