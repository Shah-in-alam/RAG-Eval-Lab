import type {
  ConfigInfo, DocumentInfo, EvalItem, ExperimentStatus, Health, Options, QueryRequest, QueryResponse,
  ResultDetail, ResultRow, UploadMeta, UploadResult,
} from "@/types";

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(path, { headers: { "Content-Type": "application/json" }, ...init });
  } catch {
    throw new ApiError(0, "Cannot reach the API server. Start it with: uvicorn api.server:app --port 8000");
  }
  if (!res.ok) {
    let message = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (typeof body.detail === "string") message = body.detail;
      else if (Array.isArray(body.detail)) message = body.detail.map((d: { msg: string }) => d.msg).join("; ");
    } catch {
      /* keep status text */
    }
    throw new ApiError(res.status, message);
  }
  return res.json() as Promise<T>;
}

/** Multipart upload with progress (fetch has no upload progress events). */
function uploadFiles(files: File[], onProgress?: (fraction: number) => void): Promise<UploadResult> {
  return new Promise((resolve, reject) => {
    const form = new FormData();
    files.forEach((f) => form.append("files", f, f.name));
    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/api/uploads");
    xhr.upload.onprogress = (e) => e.lengthComputable && onProgress?.(e.loaded / e.total);
    xhr.onload = () => {
      let body: unknown = null;
      try { body = JSON.parse(xhr.responseText); } catch { /* not json */ }
      if (xhr.status >= 200 && xhr.status < 300) resolve(body as UploadResult);
      else reject(new ApiError(xhr.status, (body as { detail?: string })?.detail ?? `Upload failed (${xhr.status})`));
    };
    xhr.onerror = () => reject(new ApiError(0, "Upload failed: cannot reach the API server."));
    xhr.send(form);
  });
}

export const api = {
  health: () => request<Health>("/api/health"),
  options: () => request<Options>("/api/options"),
  documents: () => request<DocumentInfo[]>("/api/documents"),
  document: (id: string) => request<{ doc_id: string; title: string; text: string; collection: string; meta?: UploadMeta }>(`/api/documents/${encodeURIComponent(id)}`),
  uploads: () => request<UploadMeta[]>("/api/uploads"),
  uploadFiles,
  importUrl: (url: string) => request<UploadMeta>("/api/uploads/url", { method: "POST", body: JSON.stringify({ url }) }),
  deleteUpload: async (id: string) => {
    const res = await fetch(`/api/uploads/${encodeURIComponent(id)}`, { method: "DELETE" });
    if (!res.ok && res.status !== 204) throw new ApiError(res.status, "Could not delete the document.");
  },
  evalItems: () => request<EvalItem[]>("/api/eval"),
  query: (body: QueryRequest) => request<QueryResponse>("/api/query", { method: "POST", body: JSON.stringify(body) }),
  results: () => request<ResultRow[]>("/api/results"),
  result: (runId: string) => request<ResultDetail>(`/api/results/${encodeURIComponent(runId)}`),
  configs: () => request<ConfigInfo[]>("/api/experiments/configs"),
  status: () => request<ExperimentStatus>("/api/experiments/status"),
  run: (config: string) => request<ExperimentStatus>("/api/experiments/run", { method: "POST", body: JSON.stringify({ config }) }),
};
