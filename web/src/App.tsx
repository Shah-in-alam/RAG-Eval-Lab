import { BarChart3, Check, Cpu, Database, FolderOpen, Layers, Library, Monitor, Moon, Search, Sparkles, Sun } from "lucide-react";
import { ThemeProvider, useTheme } from "next-themes";
import { useCallback, useEffect, useState } from "react";
import { api } from "@/api";
import { LoadingCards } from "@/components/app/common";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "@/components/ui/dropdown-menu";
import { Separator } from "@/components/ui/separator";
import {
  Sidebar, SidebarContent, SidebarFooter, SidebarGroup, SidebarGroupContent, SidebarGroupLabel, SidebarHeader,
  SidebarInset, SidebarMenu, SidebarMenuBadge, SidebarMenuButton, SidebarMenuItem, SidebarProvider, SidebarRail, SidebarTrigger,
} from "@/components/ui/sidebar";
import { Toaster } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { Ask } from "@/pages/Ask";
import { Documents } from "@/pages/Documents";
import { Results } from "@/pages/Results";
import type { Health, Options } from "@/types";

type Tab = "ask" | "results" | "documents";
const TABS: { id: Tab; label: string; icon: typeof Search }[] = [
  { id: "ask", label: "Ask", icon: Search },
  { id: "results", label: "Experiments", icon: BarChart3 },
  { id: "documents", label: "Documents", icon: Library },
];

function tabFromHash(): Tab {
  const h = window.location.hash.replace("#", "").replace("dataset", "documents");
  return (TABS.some((t) => t.id === h) ? h : "ask") as Tab;
}

function ThemeMenu() {
  const { theme, setTheme } = useTheme();
  const options = [["light", "Light", Sun], ["dark", "Dark", Moon], ["system", "System", Monitor]] as const;
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button variant="ghost" size="icon" aria-label="Change theme">
          <Sun className="dark:hidden" /><Moon className="hidden dark:block" />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end">
        {options.map(([value, label, Icon]) => (
          <DropdownMenuItem key={value} onClick={() => setTheme(value)}>
            <Icon />{label}{theme === value && <Check className="ml-auto" />}
          </DropdownMenuItem>
        ))}
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

function AppSidebar({ tab, onTab, health }: { tab: Tab; onTab: (t: Tab) => void; health: Health | null }) {
  return (
    <Sidebar collapsible="icon">
      <SidebarHeader>
        <SidebarMenu>
          <SidebarMenuItem>
            <SidebarMenuButton size="lg" className="pointer-events-none">
              <div className="flex aspect-square size-8 items-center justify-center rounded-lg bg-sidebar-primary text-sidebar-primary-foreground">
                <Layers className="size-4" />
              </div>
              <div className="grid flex-1 text-left leading-tight">
                <span className="truncate font-semibold">RAG Eval Lab</span>
                <span className="truncate text-xs text-muted-foreground">Retrieval experiments</span>
              </div>
            </SidebarMenuButton>
          </SidebarMenuItem>
        </SidebarMenu>
      </SidebarHeader>
      <SidebarContent>
        <SidebarGroup>
          <SidebarGroupLabel>Workspace</SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>
              {TABS.map(({ id, label, icon: Icon }) => (
                <SidebarMenuItem key={id}>
                  <SidebarMenuButton isActive={tab === id} tooltip={label} onClick={() => onTab(id)}>
                    <Icon /><span>{label}</span>
                  </SidebarMenuButton>
                  {id === "documents" && !!health?.n_uploads && <SidebarMenuBadge>{health.n_uploads}</SidebarMenuBadge>}
                </SidebarMenuItem>
              ))}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>
      </SidebarContent>
      <SidebarFooter>
        {health && (
          <SidebarMenu>
            <SidebarMenuItem>
              <SidebarMenuButton tooltip={health.llm_available ? `Answers by ${health.llm_model}` : "Set ANTHROPIC_API_KEY to enable Claude answers"}
                className="cursor-default">
                <Sparkles />
                <span className="flex-1">Answer model</span>
                <span className={health.llm_available ? "text-success" : "text-muted-foreground"}>{health.llm_available ? "Claude" : "Offline"}</span>
              </SidebarMenuButton>
            </SidebarMenuItem>
            <SidebarMenuItem>
              <SidebarMenuButton tooltip="Embedding models" className="cursor-default">
                <Cpu /><span className="flex-1">Embedding models</span><span className="font-mono">{health.embedders.length}</span>
              </SidebarMenuButton>
            </SidebarMenuItem>
            <SidebarMenuItem>
              <SidebarMenuButton tooltip="My documents" className="cursor-default">
                <FolderOpen /><span className="flex-1">My documents</span><span className="font-mono">{health.n_uploads}</span>
              </SidebarMenuButton>
            </SidebarMenuItem>
            <SidebarMenuItem>
              <SidebarMenuButton tooltip="Wikipedia set" className="cursor-default">
                <Database /><span className="flex-1">Wikipedia set</span><span className="font-mono">{health.n_documents}</span>
              </SidebarMenuButton>
            </SidebarMenuItem>
          </SidebarMenu>
        )}
      </SidebarFooter>
      <SidebarRail />
    </Sidebar>
  );
}

function Shell() {
  const [tab, setTab] = useState<Tab>(tabFromHash);
  const [health, setHealth] = useState<Health | null>(null);
  const [options, setOptions] = useState<Options | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const onHash = () => setTab(tabFromHash());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);
  useEffect(() => {
    Promise.all([api.health(), api.options()])
      .then(([h, o]) => { setHealth(h); setOptions(o); })
      .catch((e) => setError(e.message));
  }, []);
  useEffect(() => { document.title = `${TABS.find((t) => t.id === tab)?.label} · RAG Eval Lab`; }, [tab]);
  const refreshHealth = useCallback(() => { api.health().then(setHealth).catch(() => undefined); }, []);
  const go = (t: Tab) => { window.location.hash = t; setTab(t); };
  const current = TABS.find((t) => t.id === tab)!;

  return (
    <SidebarProvider>
      <AppSidebar tab={tab} onTab={go} health={health} />
      <SidebarInset>
        <header className="sticky top-0 z-20 flex h-14 shrink-0 items-center gap-2 border-b bg-background/80 px-4 backdrop-blur">
          <SidebarTrigger className="-ml-1" />
          <Separator orientation="vertical" className="mr-1 data-[orientation=vertical]:h-4" />
          <nav aria-label="Breadcrumb" className="flex items-center gap-1.5 text-sm">
            <span className="text-muted-foreground">RAG Eval Lab</span>
            <span className="text-muted-foreground">/</span>
            <span className="font-medium">{current.label}</span>
          </nav>
          <div className="ml-auto flex items-center gap-1"><ThemeMenu /></div>
        </header>
        <main className="mx-auto w-full max-w-7xl flex-1 p-4 md:p-6 lg:p-8">
          {error && (
            <Alert variant="destructive">
              <AlertTitle>Can't reach the API</AlertTitle>
              <AlertDescription>
                {error} From the project folder run <code className="font-mono">python -m uvicorn api.server:app --port 8000</code>, then reload.
              </AlertDescription>
            </Alert>
          )}
          {!error && !options && <LoadingCards count={3} />}
          {options && tab === "ask" && <Ask options={options} health={health} />}
          {options && tab === "results" && <Results />}
          {options && tab === "documents" && <Documents health={health} onChanged={refreshHealth} />}
        </main>
      </SidebarInset>
    </SidebarProvider>
  );
}

export function App() {
  return (
    <ThemeProvider attribute="class" defaultTheme="dark" enableSystem disableTransitionOnChange>
      <TooltipProvider delayDuration={200}>
        <Shell />
        <Toaster richColors position="bottom-right" />
      </TooltipProvider>
    </ThemeProvider>
  );
}
