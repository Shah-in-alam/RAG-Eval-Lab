import { CheckCircle2, ChevronDown, ChevronUp, Quote } from "lucide-react";
import { useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { cn } from "@/lib/utils";
import type { Hit } from "@/types";

const STOP = new Set("a an and are as at be by can did do does for from has have how in is it its of on or that the this to was what when where which who why with".split(" "));

/** Highlight the eval evidence when known, otherwise the question's content words. */
function highlight(text: string, question: string, evidence?: string): React.ReactNode[] {
  if (evidence) {
    const i = text.indexOf(evidence);
    if (i >= 0) return [text.slice(0, i), <mark key="ev">{evidence}</mark>, text.slice(i + evidence.length)];
  }
  const terms = [...new Set(question.toLowerCase().match(/[a-z0-9]+/g) ?? [])].filter((t) => t.length > 2 && !STOP.has(t));
  if (!terms.length) return [text];
  const re = new RegExp(`\\b(${terms.map((t) => t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("|")})\\b`, "gi");
  return text.split(re).map((part, i) => (i % 2 === 1 ? <mark key={i}>{part}</mark> : part));
}

export function HitCard({ hit, question, cited, evidence, maxScore }: {
  hit: Hit; question: string; cited: boolean; evidence?: string; maxScore: number;
}) {
  const [open, setOpen] = useState(false);
  const pct = maxScore > 0 ? Math.max(0, Math.min(100, (hit.score / maxScore) * 100)) : 0;
  return (
    <article id={`hit-${hit.rank}`} className={cn("border-t px-6 py-4 first:border-t-0", cited && "shadow-[inset_3px_0_0_var(--primary)]")}>
      <div className="mb-2 flex flex-wrap items-center gap-2">
        <span className="flex size-6 items-center justify-center rounded-md bg-muted font-mono text-xs font-semibold">{hit.rank}</span>
        <span className="text-sm font-medium">{hit.title}</span>
        <span className="font-mono text-xs text-muted-foreground">{hit.chunk_id}</span>
        {hit.relevant === true && <Badge className="bg-success-soft text-success"><CheckCircle2 />Contains the answer</Badge>}
        {cited && <Badge variant="secondary"><Quote />Cited</Badge>}
        <span className="ml-auto flex items-center gap-2 text-xs text-muted-foreground" title="Retrieval score relative to the top result">
          <Progress value={pct} className="h-1 w-12" />
          <span className="font-mono tabular">{hit.score.toFixed(3)}</span>
        </span>
      </div>
      <p className={cn("whitespace-pre-wrap break-words text-sm leading-relaxed text-muted-foreground", !open && "line-clamp-5")}>
        {highlight(hit.text, question, evidence)}
      </p>
      {hit.text.length > 420 && (
        <Button variant="link" size="sm" className="mt-1 h-auto px-0" onClick={() => setOpen((o) => !o)}>
          {open ? <><ChevronUp />Show less</> : <><ChevronDown />Show full passage</>}
        </Button>
      )}
    </article>
  );
}
