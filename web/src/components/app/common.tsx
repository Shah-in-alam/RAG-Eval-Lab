import type { ReactNode } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import { cn } from "@/lib/utils";

export function PageHeader({ title, description, actions }: { title: string; description?: ReactNode; actions?: ReactNode }) {
  return (
    <div className="flex flex-wrap items-end justify-between gap-4">
      <div className="space-y-1">
        <h1 className="text-2xl font-semibold tracking-tight">{title}</h1>
        {description && <p className="max-w-2xl text-sm text-muted-foreground">{description}</p>}
      </div>
      {actions && <div className="flex flex-wrap items-center gap-2">{actions}</div>}
    </div>
  );
}

export function StatCard({ label, value, detail, icon }: { label: string; value: ReactNode; detail?: ReactNode; icon?: ReactNode }) {
  return (
    <Card className="gap-2 py-4">
      <CardHeader className="px-4">
        <CardDescription className="flex items-center justify-between text-xs font-medium">
          <span>{label}</span>
          <span className="text-muted-foreground [&_svg]:size-4">{icon}</span>
        </CardDescription>
        <CardTitle className="font-mono text-2xl font-semibold tabular">{value}</CardTitle>
      </CardHeader>
      {detail && <CardContent className="truncate px-4 text-xs text-muted-foreground" title={typeof detail === "string" ? detail : undefined}>{detail}</CardContent>}
    </Card>
  );
}

export function EmptyState({ icon, title, children, action, className }: {
  icon: ReactNode; title: string; children?: ReactNode; action?: ReactNode; className?: string;
}) {
  return (
    <div className={cn("flex flex-col items-center gap-2 px-6 py-12 text-center", className)}>
      <div className="mb-1 flex size-11 items-center justify-center rounded-xl bg-muted text-muted-foreground [&_svg]:size-5">{icon}</div>
      <h3 className="text-sm font-semibold">{title}</h3>
      {children && <p className="max-w-md text-sm text-muted-foreground">{children}</p>}
      {action && <div className="mt-2">{action}</div>}
    </div>
  );
}

export interface Option { value: string; label: ReactNode; disabled?: boolean }

export function FieldSelect({ id, label, value, onChange, options, disabled, className }: {
  id: string; label: string; value: string; onChange: (v: string) => void; options: Option[]; disabled?: boolean; className?: string;
}) {
  return (
    <div className={cn("grid min-w-0 gap-1.5", className)}>
      <Label htmlFor={id} className="text-xs text-muted-foreground">{label}</Label>
      <Select value={value} onValueChange={onChange} disabled={disabled}>
        <SelectTrigger id={id} className="w-full">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          {options.map((o) => <SelectItem key={o.value} value={o.value} disabled={o.disabled}>{o.label}</SelectItem>)}
        </SelectContent>
      </Select>
    </div>
  );
}

export function LoadingCards({ count = 2 }: { count?: number }) {
  return (
    <div className="grid gap-4" aria-busy="true">
      {Array.from({ length: count }, (_, i) => (
        <Card key={i}>
          <CardContent className="space-y-3">
            <Skeleton className="h-4 w-1/3" />
            <Skeleton className="h-3 w-full" />
            <Skeleton className="h-3 w-5/6" />
            <Skeleton className="h-3 w-2/3" />
          </CardContent>
        </Card>
      ))}
    </div>
  );
}

export function MethodSwatch({ color }: { color: string }) {
  return <span className="inline-block size-2.5 shrink-0 rounded-[3px]" style={{ background: color }} aria-hidden />;
}
