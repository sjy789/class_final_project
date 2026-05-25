export type FeatureMap = Record<string, number>;

export type Candidate = {
  id: string;
  x: number;
  y: number;
  w: number;
  h: number;
  nx: number;
  ny: number;
  nw: number;
  nh: number;
  score: number;
  label: string;
  reason: string;
  features: FeatureMap;
  preview?: string | null;
};

export type RecommendResult = {
  mode: "auto";
  background: { width: number; height: number };
  foreground: {
    width: number;
    height: number;
    mask_source: string;
    mask_quality: number;
    original_width: number;
    original_height: number;
  };
  top: Candidate[];
  candidates: Candidate[];
  composite: string;
  heatmap?: string | null;
  output_dir: string;
};

export type ManualResult = {
  mode: "manual";
  background: { width: number; height: number };
  foreground: {
    width: number;
    height: number;
    mask_source: string;
    mask_quality: number;
  };
  candidate: Candidate;
  composite: string;
  heatmap?: string | null;
};

export type Placement = {
  x: number;
  y: number;
  scale: number;
};

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";

type RequestBase = {
  background: File;
  foreground: File;
  scale: number;
  harmonize: boolean;
};

function appendBase(form: FormData, request: RequestBase) {
  form.append("background", request.background);
  form.append("foreground", request.foreground);
  form.append("scale", String(request.scale));
  form.append("harmonize", String(request.harmonize));
}

async function readJson<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let message = response.statusText;
    try {
      const payload = await response.json();
      message = payload.detail ?? message;
    } catch {
      // Keep the HTTP status text.
    }
    throw new Error(message);
  }
  return response.json() as Promise<T>;
}

export async function requestRecommendations(
  request: RequestBase & { topK: number; explain: boolean },
) {
  const form = new FormData();
  appendBase(form, request);
  form.append("top_k", String(request.topK));
  form.append("explain", String(request.explain));
  const response = await fetch(`${API_BASE}/api/recommend`, {
    method: "POST",
    body: form,
  });
  return readJson<RecommendResult>(response);
}

export async function requestManualEvaluation(
  request: RequestBase & { placement: Placement; explain: boolean },
) {
  const form = new FormData();
  appendBase(form, request);
  form.append("x", String(request.placement.x));
  form.append("y", String(request.placement.y));
  form.append("explain", String(request.explain));
  const response = await fetch(`${API_BASE}/api/evaluate`, {
    method: "POST",
    body: form,
  });
  return readJson<ManualResult>(response);
}
