import { Bar, BarChart, CartesianGrid, Cell, Line, LineChart, XAxis, YAxis } from "recharts";
import { ChartContainer, ChartLegend, ChartLegendContent, ChartTooltip, type ChartConfig } from "@/components/ui/chart";
import { METHOD_COLOR, METHOD_ORDER } from "@/lib/colors";
import { methodLabel } from "@/lib/format";
import type { Method } from "@/types";
import { MethodSwatch } from "./common";

const methodConfig: ChartConfig = Object.fromEntries(
  METHOD_ORDER.map((m) => [m, { label: methodLabel(m), color: METHOD_COLOR[m] }]),
);

export interface RankedBar {
  id: string;
  label: string;
  title: string;
  value: number;
  method: Method;
  details: [string, string][];
}

function TooltipBox({ title, rows }: { title: string; rows: [React.ReactNode, string][] }) {
  return (
    <div className="grid min-w-44 gap-1.5 rounded-lg border bg-popover px-3 py-2 text-xs shadow-xl">
      <div className="font-medium text-popover-foreground">{title}</div>
      {rows.map(([k, v], i) => (
        <div key={i} className="flex items-center justify-between gap-4 text-muted-foreground">
          <span className="flex items-center gap-1.5">{k}</span>
          <span className="font-mono text-foreground tabular">{v}</span>
        </div>
      ))}
    </div>
  );
}

/** Ranked horizontal bars, coloured by retrieval method. Click a bar to open the run. */
export function RankedBarChart({ bars, max, format, onSelect }: {
  bars: RankedBar[]; max?: number; format: (v: number) => string; onSelect?: (id: string) => void;
}) {
  const present = METHOD_ORDER.filter((m) => bars.some((b) => b.method === m));
  return (
    <div>
      {present.length > 1 && (
        <div className="mb-3 flex flex-wrap gap-4 text-xs text-muted-foreground">
          {present.map((m) => <span key={m} className="flex items-center gap-1.5"><MethodSwatch color={METHOD_COLOR[m]} />{methodLabel(m)}</span>)}
        </div>
      )}
      <ChartContainer config={methodConfig} className="aspect-auto w-full" style={{ height: Math.max(160, bars.length * 30 + 20) }}>
        <BarChart data={bars} layout="vertical" margin={{ left: 0, right: 48, top: 0, bottom: 0 }} barCategoryGap={6}>
          <CartesianGrid horizontal={false} />
          <XAxis type="number" domain={[0, max ?? "auto"]} hide />
          <YAxis type="category" dataKey="label" width={200} tickLine={false} axisLine={false} interval={0}
            tick={({ x, y, payload }) => (
              <text x={Number(x) - 6} y={Number(y)} dy={4} textAnchor="end" className="fill-muted-foreground" fontSize={11}>
                {String(payload.value).length > 34 ? `${String(payload.value).slice(0, 33)}…` : payload.value}
              </text>
            )} />
          <ChartTooltip cursor={{ fill: "var(--muted)", opacity: 0.6 }}
            content={({ active, payload }) => {
              const b = active && payload?.[0]?.payload as RankedBar | undefined;
              return b ? <TooltipBox title={b.title} rows={b.details} /> : null;
            }} />
          <Bar dataKey="value" radius={[0, 4, 4, 0]} className="cursor-pointer"
            onClick={(d) => onSelect?.((d as unknown as { payload: RankedBar }).payload.id)}
            label={{ position: "right", fontSize: 11, className: "fill-foreground font-mono", formatter: (v: unknown) => format(Number(v)) }}>
            {bars.map((b) => <Cell key={b.id} fill={METHOD_COLOR[b.method]} />)}
          </Bar>
        </BarChart>
      </ChartContainer>
    </div>
  );
}

export interface KPoint { k: number; [method: string]: number }

/** Best metric value per method at each top-k. One y-axis. */
export function MetricByKChart({ data, methods, format }: { data: KPoint[]; methods: Method[]; format: (v: number) => string }) {
  const ys = data.flatMap((d) => methods.map((m) => d[m]).filter((v): v is number => typeof v === "number"));
  if (!ys.length) return <p className="py-10 text-center text-sm text-muted-foreground">No data for this view.</p>;
  const lo = Math.max(0, Math.floor((Math.min(...ys) - 0.03) * 20) / 20);
  const hi = Math.max(...ys) <= 1 ? Math.min(1, Math.ceil((Math.max(...ys) + 0.01) * 20) / 20) : "auto";
  return (
    <ChartContainer config={methodConfig} className="aspect-auto h-[300px] w-full">
      <LineChart data={data} margin={{ left: 4, right: 16, top: 8 }}>
        <CartesianGrid vertical={false} />
        <XAxis dataKey="k" tickLine={false} axisLine={false} tickMargin={8} tickFormatter={(k) => `k=${k}`} />
        <YAxis domain={[lo, hi]} tickLine={false} axisLine={false} width={44} tickFormatter={(v: number) => format(v)} />
        <ChartTooltip content={({ active, payload, label }) => active && payload?.length ? (
          <TooltipBox title={`top-k = ${label}`} rows={METHOD_ORDER.filter((m) => methods.includes(m)).flatMap((m) => {
            const p = payload.find((x) => x.dataKey === m);
            return p ? [[<><MethodSwatch color={METHOD_COLOR[m]} />{methodLabel(m)}</>, format(Number(p.value))] as [React.ReactNode, string]] : [];
          })} />
        ) : null} />
        <ChartLegend content={<ChartLegendContent />} />
        {METHOD_ORDER.filter((m) => methods.includes(m)).map((m) => (
          <Line key={m} dataKey={m} type="monotone" stroke={`var(--color-${m})`} strokeWidth={2}
            dot={{ r: 4, strokeWidth: 2, stroke: "var(--card)" }} activeDot={{ r: 5 }} />
        ))}
      </LineChart>
    </ChartContainer>
  );
}
