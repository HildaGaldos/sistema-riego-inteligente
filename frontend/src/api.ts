export const API_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

export type DatasetStatus = { available: boolean; filename?: string; message?: string; quality?: Record<string, unknown>; warnings?: string[] };
export type Prediction = { probability_irrigation: number; predicted_class: number; recommendation: string; threshold: number; model_name: string; model_version: string };
export type PipelineStage = { id: string; label: string };
export type PipelineStatus = { status: string; stage: string; progress: number; message: string; detail?: string; elapsed_seconds?: number; eta_seconds?: number | null; stages?: PipelineStage[]; history?: { stage: string; progress: number; message: string; detail?: string; at?: string }[]; error?: string; result?: { model_name?: string; rows?: number; excluded_rows?: number; folds?: number } };

function authHeaders(): HeadersInit {
  const token = sessionStorage.getItem("irrigation_token");
  return token ? { Authorization: `Bearer ${token}` } : {};
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const response = await fetch(`${API_URL}${path}`, { ...init, headers: { ...authHeaders(), ...(init.headers ?? {}) } });
  if (!response.ok) {
    if (response.status === 401 && path !== "/auth/token") {
      sessionStorage.removeItem("irrigation_token");
      throw new Error("AUTH_EXPIRED");
    }
    const detail = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(detail.detail ?? "Error de API");
  }
  return response.json();
}

export async function login(username: string, password: string) {
  const body = new URLSearchParams({ username, password });
  const result = await request<{ access_token: string }>("/auth/token", { method: "POST", headers: { "Content-Type": "application/x-www-form-urlencoded" }, body });
  sessionStorage.setItem("irrigation_token", result.access_token);
}
export const me = () => request<{ username: string; is_admin: boolean }>("/auth/me");
export const status = () => request<DatasetStatus>("/data/status");
export const upload = (file: File) => { const form = new FormData(); form.append("file", file); return request<Record<string, unknown>>("/data/upload", { method: "POST", body: form }); };
export const train = (fast: boolean, folds = 5, tuningTrials?: number) => request<Record<string, unknown>>(`/train?fast=${fast}&folds=${folds}${tuningTrials ? `&tuning_trials=${tuningTrials}` : ""}`, { method: "POST" });
export const runPipeline = (fast: boolean, folds = 5, tuningTrials?: number) => request<Record<string, unknown>>(`/pipeline/run?fast=${fast}&folds=${folds}${tuningTrials ? `&tuning_trials=${tuningTrials}` : ""}`, { method: "POST" });
export const cancelPipeline = () => request<PipelineStatus>("/pipeline/cancel", { method: "POST" });
export const trainStatus = () => request<PipelineStatus>("/pipeline/status");
export const predict = (payload: object) => request<Prediction>("/predict", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
export const predictBatch = (file: File) => { const form = new FormData(); form.append("file", file); return request<{ rows: Record<string, unknown>[] }>("/predict/batch", { method: "POST", body: form }); };
export const metrics = () => request<{ available: boolean; models: Record<string, unknown>[] }>("/metrics/summary");
export const eda = () => request<Record<string, unknown>>("/eda/summary");
export const cv = () => request<{ available: boolean; rows: Record<string, unknown>[] }>("/metrics/cv");
export const tuning = () => request<Record<string, unknown>>("/tuning/summary");
export const statistics = () => request<Record<string, unknown>>("/statistics/summary");
export const figures = () => request<{ figures: { filename: string; size: number }[] }>("/eda/figures");
export async function downloadFigure(filename: string): Promise<Blob> {
  const response = await fetch(`${API_URL}/eda/figures/${encodeURIComponent(filename)}`, { headers: authHeaders() });
  if (!response.ok) throw new Error("No fue posible cargar la figura");
  return response.blob();
}
export const modelMetadata = () => request<Record<string, unknown>>("/model/metadata");
export async function downloadModel(filename = "best_irrigation_model.h5"): Promise<Blob> {
  const response = await fetch(`${API_URL}/model/artifacts/${encodeURIComponent(filename)}`, { headers: authHeaders() });
  if (!response.ok) throw new Error("No fue posible descargar el modelo");
  return response.blob();
}
export const reports = () => request<{ reports: { filename: string; size: number }[] }>("/reports");
export async function downloadReport(filename: string): Promise<Blob> {
  const response = await fetch(`${API_URL}/reports/${encodeURIComponent(filename)}`, { headers: authHeaders() });
  if (!response.ok) throw new Error("No fue posible descargar el reporte");
  return response.blob();
}
export const logout = () => sessionStorage.removeItem("irrigation_token");
