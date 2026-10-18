import { CheckCircle2, CloudUpload, Link2, Loader2, XCircle } from "lucide-react";
import { useRef, useState } from "react";
import { toast } from "sonner";
import { api } from "@/api";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Progress } from "@/components/ui/progress";
import { formatBytes } from "@/lib/format";
import { cn } from "@/lib/utils";
import type { UploadMeta } from "@/types";

interface QueueItem { id: string; name: string; size: number; status: "uploading" | "done" | "error"; progress: number; message?: string }

export function UploadPanel({ types, maxMb, onAdded }: { types: string[]; maxMb: number; onAdded: (docs: UploadMeta[]) => void }) {
  const input = useRef<HTMLInputElement>(null);
  const [drag, setDrag] = useState(false);
  const [queue, setQueue] = useState<QueueItem[]>([]);
  const [url, setUrl] = useState("");
  const [importing, setImporting] = useState(false);
  const update = (id: string, patch: Partial<QueueItem>) => setQueue((q) => q.map((it) => (it.id === id ? { ...it, ...patch } : it)));

  async function upload(files: File[]) {
    if (!files.length) return;
    const limit = maxMb * 1024 * 1024;
    const items: QueueItem[] = files.map((f, i) => ({ id: `${Date.now()}-${i}`, name: f.name, size: f.size, status: "uploading", progress: 0 }));
    setQueue((q) => [...items, ...q].slice(0, 12));
    items.filter((it) => it.size > limit).forEach((it) => update(it.id, { status: "error", message: `Larger than ${maxMb} MB` }));
    const okFiles = files.filter((f) => f.size <= limit);
    const okItems = items.filter((it) => it.size <= limit);
    if (!okFiles.length) return;
    try {
      const res = await api.uploadFiles(okFiles, (p) => okItems.forEach((it) => update(it.id, { progress: p })));
      okItems.forEach((it) => {
        const err = res.errors.find((e) => e.filename === it.name);
        const added = res.added.find((a) => a.filename === it.name);
        update(it.id, err ? { status: "error", message: err.error }
          : { status: "done", progress: 1, message: added ? `${added.n_words.toLocaleString()} words indexed` : undefined });
      });
      if (res.added.length) { onAdded(res.added); toast.success(`Added ${res.added.length} document${res.added.length > 1 ? "s" : ""}`); }
      if (res.errors.length) toast.error(`${res.errors.length} file${res.errors.length > 1 ? "s" : ""} could not be read`);
    } catch (e) {
      okItems.forEach((it) => update(it.id, { status: "error", message: (e as Error).message }));
    }
  }

  async function importLink(e: React.FormEvent) {
    e.preventDefault();
    if (!url.trim()) return;
    setImporting(true);
    try {
      const doc = await api.importUrl(url.trim());
      toast.success(`Imported ${doc.filename}`, { description: `${doc.n_words.toLocaleString()} words indexed` });
      setUrl("");
      onAdded([doc]);
    } catch (err) {
      toast.error("Import failed", { description: (err as Error).message });
    } finally {
      setImporting(false);
    }
  }

  return (
    <div className="grid gap-4 lg:grid-cols-[minmax(0,1.2fr)_minmax(0,1fr)]">
      <div role="button" tabIndex={0} aria-label="Upload files: drop files here or press Enter to browse"
        onClick={() => input.current?.click()}
        onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && input.current?.click()}
        onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
        onDragLeave={() => setDrag(false)}
        onDrop={(e) => { e.preventDefault(); setDrag(false); upload([...e.dataTransfer.files]); }}
        className={cn("flex cursor-pointer flex-col items-center justify-center gap-1.5 rounded-xl border-2 border-dashed bg-muted/40 px-6 py-8 text-center transition-colors hover:border-primary/60 hover:bg-primary/5 focus-visible:outline-none focus-visible:ring-[3px] focus-visible:ring-ring",
          drag && "border-primary bg-primary/10")}>
        <div className="mb-2 flex size-11 items-center justify-center rounded-xl border bg-background text-primary shadow-xs"><CloudUpload className="size-5" /></div>
        <p className="text-sm font-medium">Drop files here or <span className="text-primary">browse</span></p>
        <p className="text-xs text-muted-foreground">PDF, Word, PowerPoint, Excel, CSV, TXT, Markdown, HTML, RTF, EPUB, JSON · up to {maxMb} MB each</p>
        <p className="text-xs text-muted-foreground">Files in OneDrive, Google Drive or Dropbox folders on this computer work too.</p>
        <input ref={input} type="file" multiple accept={types.join(",")} hidden
          onChange={(e) => { upload([...(e.target.files ?? [])]); e.target.value = ""; }} />
      </div>

      <form onSubmit={importLink} className="grid content-start gap-3">
        <div className="grid gap-1.5">
          <Label htmlFor="import-url">Import from a link</Label>
          <div className="flex gap-2">
            <div className="relative flex-1">
              <Link2 className="pointer-events-none absolute left-2.5 top-2.5 size-4 text-muted-foreground" />
              <Input id="import-url" inputMode="url" value={url} onChange={(e) => setUrl(e.target.value)}
                placeholder="https://drive.google.com/file/d/…" className="pl-8" />
            </div>
            <Button type="submit" variant="secondary" disabled={importing || !url.trim()}>
              {importing ? <><Loader2 className="animate-spin" />Importing</> : "Import"}
            </Button>
          </div>
        </div>
        <p className="text-xs leading-relaxed text-muted-foreground">
          Google Drive, Docs, Sheets and Slides, OneDrive, SharePoint and Dropbox share links set to
          "Anyone with the link", plus any direct link to a file or web page.
        </p>
      </form>

      {queue.length > 0 && (
        <ul className="divide-y rounded-lg border lg:col-span-2" aria-live="polite">
          {queue.map((it) => (
            <li key={it.id} className="flex items-start gap-3 px-3 py-2.5">
              <span className="pt-0.5">
                {it.status === "uploading" && <Loader2 className="size-4 animate-spin text-muted-foreground" />}
                {it.status === "done" && <CheckCircle2 className="size-4 text-success" />}
                {it.status === "error" && <XCircle className="size-4 text-destructive" />}
              </span>
              <span className="grid min-w-0 flex-1 gap-1">
                <span className="truncate text-sm font-medium">{it.name}</span>
                <span className={cn("text-xs", it.status === "error" ? "text-destructive" : "text-muted-foreground")}>
                  {it.status === "uploading" ? `${Math.round(it.progress * 100)}% · ${formatBytes(it.size)}` : it.message ?? formatBytes(it.size)}
                </span>
                {it.status === "uploading" && <Progress value={it.progress * 100} className="h-1" />}
              </span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
