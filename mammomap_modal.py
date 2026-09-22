"""
mammomap — Modal.com serverless GPU backend
Tier 2: Vercel (frontend) + Modal (GPU inference) + Supabase (data) + Upstash (Redis)

Deploy:
    pip install modal
    modal setup
    modal deploy mammomap_modal.py
"""

import io
import uuid
import json
import base64
import modal
from modal import App, Image, Secret

# ─── Modal app ─────────────────────────────────────────────────────────────────
app = App("mammomap")

# ─── Container image (built once, cached) ─────────────────────────────────────
weights_volume = modal.Volume.from_name("mammomap-weights", create_if_missing=True)

ml_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0", "libgomp1", "libvips-dev")
    .pip_install(
        # API
        "fastapi==0.111.0",
        "python-multipart==0.0.9",
        "uvicorn==0.30.0",
        # ML — core
        "torch==2.2.1",
        "torchvision==0.17.1",
        "timm==1.0.3",
        "numpy==1.26.4",
        "Pillow==10.3.0",
        "opencv-python-headless==4.9.0.80",
        "scikit-learn==1.5.0",
        "scikit-image==0.22.0",
        "scipy==1.13.0",
        # Virchow2 + ATM segmentation
        "peft==0.10.0",
        "monai==1.3.0",
        "huggingface-hub==0.23.0",
        # Storage / DB
        "supabase==2.5.0",
        "redis==5.0.4",
        "httpx==0.27.0",
    )
    # Bundle the segmentation module into the container image
    .add_local_file("virchow_seg.py", "/root/virchow_seg.py")
)

# ─── Secrets (set via: modal secret create mammomap-secrets KEY=val …) ──────
secrets = [Secret.from_name("mammomap-secrets")]

# ══════════════════════════════════════════════════════════════════════════════
# CPU FUNCTION — tissue filter (cheap, no GPU needed)
# ══════════════════════════════════════════════════════════════════════════════
@app.function(image=ml_image, cpu=2.0, memory=2048, timeout=60, secrets=secrets)
@modal.concurrent(max_inputs=10)
def tissue_filter(image_bytes: bytes, method: str, threshold_adj: int, min_pct: int) -> dict:
    import numpy as np
    import cv2
    from PIL import Image as PILImage

    img = PILImage.open(io.BytesIO(image_bytes)).convert("RGB")
    img_np = np.array(img)
    h, w = img_np.shape[:2]

    # ── Score pixels ───────────────────────────────────────────────────────────
    if method == "otsu":
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        t, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        t = int(np.clip(t + threshold_adj, 10, 245))
        scores = (gray < t).astype(np.float32)

    elif method == "hsv":
        hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
        sat = hsv[:, :, 1] / 255.0
        ms = max(0.01, 0.06 - threshold_adj * 0.002)
        scores = (sat > ms).astype(np.float32)

    else:  # laplacian
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        lap = np.abs(cv2.Laplacian(gray.astype(np.float64), cv2.CV_64F))
        lap_norm = lap / (lap.max() + 1e-6)
        t, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        t = int(np.clip(t + threshold_adj, 10, 245))
        scores = ((gray < t) | (lap_norm > 0.15)).astype(np.float32)

    # ── Patch grid ────────────────────────────────────────────────────────────
    GRID = 18
    ph, pw = h // GRID, w // GRID
    grid = []
    for gy in range(GRID):
        for gx in range(GRID):
            cell = scores[gy * ph:(gy + 1) * ph, gx * pw:(gx + 1) * pw]
            grid.append({"gx": gx, "gy": gy, "pct": float(cell.mean())})

    kept = [g for g in grid if g["pct"] * 100 >= min_pct]

    # ── Mask preview image (darken background) ────────────────────────────────
    overlay = img_np.copy()
    bg = scores < 0.25
    overlay[bg] = (overlay[bg] * 0.12).astype(np.uint8)
    _, buf = cv2.imencode(".jpg", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
                          [cv2.IMWRITE_JPEG_QUALITY, 80])
    mask_b64 = base64.b64encode(buf.tobytes()).decode()

    return {
        "grid":        grid,
        "kept_count":  len(kept),
        "total_count": len(grid),
        "kept_pct":    round(len(kept) / len(grid) * 100, 1),
        "tissue_avg":  round(float(scores.mean() * 100), 1),
        "mask_b64":    mask_b64,
        "min_pct":     min_pct,
    }


# ══════════════════════════════════════════════════════════════════════════════
# GPU FUNCTION — IHC + segmentation + spatial (runs on T4)
# ══════════════════════════════════════════════════════════════════════════════
@app.function(
    image=ml_image,
    gpu="T4",
    memory=12288,   # Virchow2 ViT-H needs ~5 GB weights + activations
    timeout=300,
    secrets=secrets,
    volumes={"/weights": weights_volume},
)
def run_ml_pipeline(image_bytes: bytes, config: dict) -> dict:
    import numpy as np
    import cv2
    import torch
    import torchvision.transforms as T
    from torchvision.models import resnet50, ResNet50_Weights
    from sklearn.decomposition import PCA
    from PIL import Image as PILImage

    img = PILImage.open(io.BytesIO(image_bytes)).convert("RGB")
    img_np = np.array(img)
    H, W = img_np.shape[:2]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results: dict = {}

    # ── 1. IHC virtual staining — Macenko stain deconvolution ─────────────────
    # Real signal: separates hematoxylin (nuclear) and eosin (cytoplasmic) OD
    def macenko(rgb: np.ndarray):
        eps = 1e-6
        OD = -np.log((rgb.astype(np.float64) + 1) / 256.0)
        OD_hat = OD.reshape(-1, 3)
        OD_hat = OD_hat[OD_hat.min(axis=1) > 0.1]
        if len(OD_hat) < 100:
            return np.zeros((H, W)), np.zeros((H, W))
        _, V = np.linalg.eigh(np.cov(OD_hat.T))
        V = V[:, [2, 1]]
        angles = np.arctan2(OD_hat @ V[:, 1], OD_hat @ V[:, 0])
        v1 = V @ [np.cos(np.percentile(angles, 1)),  np.sin(np.percentile(angles, 1))]
        v2 = V @ [np.cos(np.percentile(angles, 99)), np.sin(np.percentile(angles, 99))]
        HE = np.array([v1, v2] if v1[0] > v2[0] else [v2, v1])
        HE /= (np.linalg.norm(HE, axis=1, keepdims=True) + eps)
        Y, *_ = np.linalg.lstsq(HE.T, OD.reshape(-1, 3).T, rcond=None)
        H_ch = np.clip(Y[0].reshape(H, W), 0, None)
        E_ch = np.clip(Y[1].reshape(H, W), 0, None)
        return H_ch / (H_ch.max() + eps), E_ch / (E_ch.max() + eps)

    H_chan, E_chan = macenko(img_np)

    # Edge map (membrane proxy for HER2)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    edges = np.abs(cv2.Laplacian(gray.astype(np.float64), cv2.CV_64F))
    edges /= (edges.max() + 1e-6)

    stain_bases = {
        "HER2":  np.clip(H_chan * 0.55 + edges * 0.45, 0, 1),  # membrane signal
        "Ki67":  H_chan,                                          # nuclear proliferation
        "ER":    np.clip(H_chan * 1.1, 0, 1),                   # nuclear ER
        "PR":    np.clip(H_chan * 0.9 + E_chan * 0.1, 0, 1),    # nuclear PR
    }

    ihc_results = {}
    IHCG = 12
    gh, gw = H // IHCG, W // IHCG

    for stain in config.get("stains", list(stain_bases.keys())):
        base = stain_bases.get(stain, H_chan)
        smoothed = cv2.GaussianBlur(base, (21, 21), 0)
        intensity_map = []
        for gy in range(IHCG):
            for gx in range(IHCG):
                cell = smoothed[gy * gh:(gy + 1) * gh, gx * gw:(gx + 1) * gw]
                intensity_map.append(round(float(cell.mean()), 4))
        pct_pos = float(np.mean(np.array(intensity_map) > 0.25) * 100)
        if stain == "HER2":
            grade = "3+" if pct_pos > 50 else ("2+" if pct_pos > 20 else ("1+" if pct_pos > 5 else "0"))
            score_val = grade
        else:
            allred = min(8, max(0, int(pct_pos / 12.5)))
            grade = f"Allred {allred}/8"
            score_val = f"{pct_pos:.1f}%"
        ihc_results[stain] = {
            "intensity_map": intensity_map,
            "positive_pct":  round(pct_pos, 1),
            "grade":         grade,
            "score_value":   score_val,
        }
    results["ihc"] = ihc_results

    # ── 2. Segmentation — Virchow2 (ViT-H/14) + Cascaded ATM decoder ──────────────
    import sys
    sys.path.insert(0, "/root")
    from virchow_seg import load_virchow_seg_model, run_virchow_seg

    seg_model = load_virchow_seg_model(device=device)
    results["segmentation"] = run_virchow_seg(seg_model, image_bytes, device=device)
    # results["segmentation"] keys: regions, metrics, model

    # Derive class_areas for downstream clinical metrics
    class_areas = results["segmentation"]["metrics"]

    # ── 3. Spatial features — ResNet50 patch features (SEQUOIA proxy) ─────────
    # Production: swap for SEQUOIA with UNI features (set HF_TOKEN in secrets)
    backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    backbone.fc = torch.nn.Identity()
    backbone = backbone.to(device).eval()

    tfm = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    SGRID = 14
    ph_s, pw_s = H // SGRID, W // SGRID
    tensors, coords = [], []
    for gy in range(SGRID):
        for gx in range(SGRID):
            patch = img_np[gy * ph_s:(gy + 1) * ph_s, gx * pw_s:(gx + 1) * pw_s]
            tensors.append(tfm(PILImage.fromarray(patch)))
            coords.append([gx / SGRID * 100, gy / SGRID * 100])

    all_feats = []
    with torch.no_grad():
        for i in range(0, len(tensors), 32):
            batch = torch.stack(tensors[i:i + 32]).to(device)
            all_feats.append(backbone(batch).cpu().numpy())
    feats = np.vstack(all_feats)  # (196, 2048)

    genes = ["ESR1", "PGR", "ERBB2", "MKI67", "TP53", "BRCA1", "CDH1", "PIK3CA", "CCND1", "GATA3"]
    pca = PCA(n_components=len(genes))
    reduced = pca.fit_transform(feats)  # (196, 10)
    # Normalise each gene component to [0, 1]
    gene_min = reduced.min(axis=0)
    gene_max = reduced.max(axis=0)
    gene_norm = (reduced - gene_min) / (gene_max - gene_min + 1e-6)

    tiles = []
    for i, (coord, row) in enumerate(zip(coords, gene_norm)):
        pred = {g: round(float(row[j]), 4) for j, g in enumerate(genes)}
        tiles.append({"cx": coord[0], "cy": coord[1], "predictions": pred,
                      "expr": pred["ESR1"]})

    gene_means = {g: round(float(gene_norm[:, j].mean()), 4) for j, g in enumerate(genes)}
    results["spatial"] = {
        "tiles":      tiles,
        "gene_stats": gene_means,
        "method":     "ResNet50 (set HF_TOKEN in secrets for SEQUOIA/UNI)",
    }

    # ── 4. Clinical summary ───────────────────────────────────────────────────
    tp    = results["segmentation"]["metrics"]["tumor_purity"]
    til   = results["segmentation"]["metrics"]["til_score"]
    her2  = ihc_results.get("HER2", {}).get("positive_pct", 0)
    er_al = int(ihc_results.get("ER", {}).get("score_value", "0%")
                .replace("%", "").split()[0].split("/")[0]) if "ER" in ihc_results else 0

    if er_al >= 6 and her2 < 20:
        subtype, risk = "Luminal A", "Low"
    elif er_al >= 4 and her2 >= 20:
        subtype, risk = "Luminal B / HER2+", "Intermediate"
    elif er_al < 3 and her2 >= 30:
        subtype, risk = "HER2-enriched", "High"
    else:
        subtype, risk = "Triple-negative", "High"

    results["metrics"] = {
        "tumor_purity":     tp,
        "til_score":        til,
        "stroma_ratio":     results["segmentation"]["metrics"]["stroma_ratio"],
        "necrosis_ratio":   results["segmentation"]["metrics"]["necrosis_ratio"],
        "recurrence_risk":  risk,
        "molecular_subtype":subtype,
        "top_genes": sorted(
            [{"name": g, "mean": v} for g, v in gene_means.items()],
            key=lambda x: x["mean"], reverse=True
        )[:5],
    }

    return results


# ══════════════════════════════════════════════════════════════════════════════
# FASTAPI ASGI APP  — all HTTP endpoints
# ══════════════════════════════════════════════════════════════════════════════
@app.function(
    image=ml_image,
    secrets=secrets,
    memory=512,
)
@modal.concurrent(max_inputs=20)
@modal.asgi_app()
def fastapi_app():
    import os
    from fastapi import FastAPI, UploadFile, File, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from supabase import create_client, Client

    api = FastAPI(title="mammomap API", version="1.0.0")
    api.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],   # lock down to your Vercel domain in production
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def sb() -> Client:
        return create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"])

    # ── health ─────────────────────────────────────────────────────────────────
    @api.get("/health")
    def health():
        return {"status": "ok", "version": "1.0.0"}

    # ── upload ─────────────────────────────────────────────────────────────────
    @api.post("/api/v1/upload")
    async def upload(file: UploadFile = File(...)):
        analysis_id = str(uuid.uuid4())
        data = await file.read()
        path = f"images/{analysis_id}/original.jpg"
        sb().storage.from_("mammomap").upload(path, data, {"content-type": "image/jpeg"})
        sb().table("analyses").insert({"id": analysis_id, "status": "uploaded",
                                       "original_path": path}).execute()
        return {"analysis_id": analysis_id, "status": "uploaded"}

    # ── tissue filter ─────────────────────────────────────────────────────────
    @api.post("/api/v1/filter/{analysis_id}")
    def run_filter(
        analysis_id: str,
        method: str = "otsu",
        threshold_adj: int = 0,
        min_pct: int = 30,
    ):
        row = sb().table("analyses").select("original_path").eq("id", analysis_id).execute()
        if not row.data:
            raise HTTPException(404, "Analysis not found")
        image_bytes = sb().storage.from_("mammomap").download(row.data[0]["original_path"])
        result = tissue_filter.remote(image_bytes, method, threshold_adj, min_pct)
        sb().table("analyses").update({"filter_result": result, "status": "filtered"}) \
            .eq("id", analysis_id).execute()
        return {"status": "complete", "result": result}

    # ── full ML pipeline (IHC + seg + spatial) ────────────────────────────────
    @api.post("/api/v1/pipeline/{analysis_id}")
    def run_pipeline(
        analysis_id: str,
        stains: str = "HER2,Ki67,ER,PR",
    ):
        row = sb().table("analyses").select("original_path").eq("id", analysis_id).execute()
        if not row.data:
            raise HTTPException(404, "Analysis not found")
        image_bytes = sb().storage.from_("mammomap").download(row.data[0]["original_path"])
        sb().table("analyses").update({"status": "processing"}).eq("id", analysis_id).execute()
        try:
            result = run_ml_pipeline.remote(image_bytes, {"stains": stains.split(",")})
            sb().table("analyses").update({"pipeline_result": result, "status": "complete"}) \
                .eq("id", analysis_id).execute()
            return {"status": "complete", "result": result}
        except Exception as exc:
            sb().table("analyses").update({"status": "failed", "error": str(exc)}) \
                .eq("id", analysis_id).execute()
            raise HTTPException(500, detail=str(exc))

    # ── results ───────────────────────────────────────────────────────────────
    @api.get("/api/v1/results/{analysis_id}")
    def get_results(analysis_id: str):
        row = sb().table("analyses").select("*").eq("id", analysis_id).execute()
        if not row.data:
            raise HTTPException(404, "Not found")
        return row.data[0]

    return api


# ══════════════════════════════════════════════════════════════════════════════
# CASE-LEVEL ENDPOINTS  — multi-ROI support
# ══════════════════════════════════════════════════════════════════════════════

@app.function(image=ml_image, secrets=secrets, cpu=1, memory=512)
def aggregate_case_fn(case_id: str) -> dict:
    """
    Runs after all ROIs in a case have completed the ML pipeline.
    Reads each ROI's pipeline_result from Supabase, aggregates, writes back.
    """
    import os, sys
    sys.path.insert(0, "/root")
    from supabase import create_client
    from case_aggregation import aggregate_case

    sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"])

    rows = (
        sb.table("analyses")
        .select("roi_index, roi_label, pipeline_result")
        .eq("case_id", case_id)
        .eq("status", "complete")
        .order("roi_index")
        .execute()
        .data
    )

    if not rows:
        raise ValueError(f"No completed ROIs for case {case_id}")

    roi_results = [r["pipeline_result"] for r in rows]
    agg = aggregate_case(roi_results)

    sb.table("case_aggregations").upsert({"case_id": case_id, **agg}).execute()
    sb.table("cases").update({"status": "complete"}).eq("id", case_id).execute()
    return agg


# ── Additional FastAPI routes (add these inside fastapi_app()) ──────────────
# Paste the block below inside the `def fastapi_app():` function above,
# before the `return api` line.
#
#   @api.post("/api/v1/cases")
#   def create_case(patient_code: str = "", case_label: str = ""):
#       row = sb().table("cases").insert({
#           "patient_code": patient_code, "case_label": case_label,
#       }).execute()
#       return {"case_id": row.data[0]["id"]}
#
#   @api.post("/api/v1/cases/{case_id}/upload")
#   async def upload_roi(
#       case_id: str,
#       roi_index: int,
#       roi_label: str = "unspecified",
#       file: UploadFile = File(...),
#   ):
#       analysis_id = str(uuid.uuid4())
#       data = await file.read()
#       path = f"images/{case_id}/roi_{roi_index}.jpg"
#       sb().storage.from_("mammomap").upload(path, data, {"content-type":"image/jpeg"})
#       sb().table("analyses").insert({
#           "id": analysis_id, "case_id": case_id,
#           "roi_index": roi_index, "roi_label": roi_label,
#           "status": "uploaded", "original_path": path,
#       }).execute()
#       sb().table("cases").update({"n_rois": roi_index}).eq("id", case_id).execute()
#       return {"analysis_id": analysis_id, "case_id": case_id}
#
#   @api.post("/api/v1/cases/{case_id}/run")
#   def run_case_pipeline(case_id: str, stains: str = "HER2,Ki67,ER,PR"):
#       """Kick off pipeline for every ROI, then aggregate."""
#       rows = sb().table("analyses").select("id,original_path") \
#           .eq("case_id", case_id).eq("status", "uploaded").execute().data
#       results = []
#       for row in rows:
#           img_bytes = sb().storage.from_("mammomap").download(row["original_path"])
#           result = run_ml_pipeline.remote(img_bytes, {"stains": stains.split(",")})
#           sb().table("analyses").update({
#               "pipeline_result": result, "status": "complete"
#           }).eq("id", row["id"]).execute()
#           results.append(result)
#       agg = aggregate_case_fn.remote(case_id)
#       return {"status": "complete", "n_rois": len(results), "aggregation": agg}
#
#   @api.get("/api/v1/cases/{case_id}")
#   def get_case(case_id: str):
#       agg  = sb().table("case_aggregations").select("*").eq("case_id", case_id).execute().data
#       rois = sb().table("analyses").select("roi_index,roi_label,status,pipeline_result") \
#           .eq("case_id", case_id).order("roi_index").execute().data
#       return {"aggregation": agg[0] if agg else None, "rois": rois}
