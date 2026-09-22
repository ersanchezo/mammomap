/**
 * PathVision API client
 * Talks to the Modal backend (or localhost in dev)
 *
 * Usage: import { uploadImage, runFilter, runPipeline, getResults } from './api'
 */

const BASE = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

async function post(path, body, isFormData = false) {
  const res = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: isFormData ? undefined : { "Content-Type": "application/json" },
    body: isFormData ? body : JSON.stringify(body),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`${res.status} ${path}: ${text}`);
  }
  return res.json();
}

async function get(path) {
  const res = await fetch(`${BASE}${path}`);
  if (!res.ok) throw new Error(`${res.status} ${path}`);
  return res.json();
}

/** Upload an H&E image file. Returns { analysis_id } */
export async function uploadImage(file) {
  const form = new FormData();
  form.append("file", file);
  return post("/api/v1/upload", form, true);
}

/**
 * Run tissue filter. Returns { result: { grid, kept_count, tissue_avg, mask_b64 } }
 * @param {string} analysisId
 * @param {{ method: string, threshold_adj: number, min_pct: number }} opts
 */
export async function runFilter(analysisId, { method = "otsu", threshold_adj = 0, min_pct = 30 } = {}) {
  const params = new URLSearchParams({ method, threshold_adj, min_pct });
  const res = await fetch(`${BASE}/api/v1/filter/${analysisId}?${params}`, { method: "POST" });
  if (!res.ok) throw new Error(`filter failed: ${await res.text()}`);
  return res.json();
}

/**
 * Run the full ML pipeline: IHC + segmentation + spatial.
 * This is a synchronous call on Modal — it blocks until done (~20-90s on T4).
 * Returns { result: { ihc, segmentation, spatial, metrics } }
 * @param {string} analysisId
 * @param {{ stains: string[] }} opts
 */
export async function runPipeline(analysisId, { stains = ["HER2", "Ki67", "ER", "PR"] } = {}) {
  const params = new URLSearchParams({ stains: stains.join(",") });
  const res = await fetch(`${BASE}/api/v1/pipeline/${analysisId}?${params}`, { method: "POST" });
  if (!res.ok) throw new Error(`pipeline failed: ${await res.text()}`);
  return res.json();
}

/** Fetch stored results for an analysis */
export async function getResults(analysisId) {
  return get(`/api/v1/results/${analysisId}`);
}

/** Health check */
export async function ping() {
  return get("/health");
}

// ─── Drop-in replacements for the mock runner functions in the JSX ─────────────
//
// In tissue_analysis_platform.jsx, replace the runner bodies with these:
//
// const handleFile = async (file) => {
//   const url = URL.createObjectURL(file);            // keep local preview
//   const img = new Image();
//   img.onload = async () => {
//     setImgDims({ w: img.width, h: img.height });
//     setImage(url);
//     const { analysis_id } = await uploadImage(file); // upload to API
//     setAnalysisId(analysis_id);                      // store in state
//     setStep(1);
//   };
//   img.src = url;
// };
//
// const applyFilterReal = async (method, tAdj, mp) => {
//   const { result } = await runFilter(analysisId, {
//     method, threshold_adj: tAdj, min_pct: mp,
//   });
//   // result.mask_b64 → paint onto filterCanvas
//   // result.grid     → paint patch grid onto gridCanvas
//   setFResult(result);
//   setFReady(true);
// };
//
// const runPipelineReal = async () => {
//   setRunning(true); setProgress(10); setProgLabel("Uploading to GPU…");
//   const { result } = await runPipeline(analysisId, { stains: [...ihcSelStains] });
//   //  result.ihc.HER2.intensity_map → ihcData
//   //  result.segmentation.regions   → segMask
//   //  result.spatial.tiles          → spatialData
//   //  result.metrics                → metrics
//   setIhcResults(result.ihc);
//   setSegMask(result.segmentation.regions);
//   setSpatialData(result.spatial.tiles);
//   setMetrics(result.metrics);
//   setRunning(false); setStep(5);
// };
