import { useState, useRef, useEffect, useCallback, useReducer, useMemo } from "react";

// ─── Design tokens ────────────────────────────────────────────────────────────
const T = {
  bg0:"#080b12", bg1:"#0d1018", bg2:"#121620", bg3:"#181d2b", bg4:"#1e2435",
  bdr:"#1f2840", bdr2:"#171c2c",
  txt:"#dde2f0", txt2:"#7a8aaa", txt3:"#48566e",
  teal:"#00d4aa", red:"#e84c6a", amber:"#f5a94a",
  purple:"#8b6de8", blue:"#3c8cdc", green:"#3cc878",
};

// ─── Gene overlay config ──────────────────────────────────────────────────────
const MAX_GENES = 15;
const PALETTE = ["#00d4aa","#e84c6a","#f5a94a","#8b6de8","#3c8cdc","#3cc878","#ff6fae","#ffd166","#06b6d4","#a3e635","#fb7185","#818cf8","#f472b6","#4ade80","#facc15"];
const SHAPES = ["circle","square","triangle","diamond","star"];
const DEFAULT_GENES = ["ESR1","ERBB2","PGR","MKI67"];

// ─── Segmentation legend config ───────────────────────────────────────────────
// Colors are the exact RGB values pulled from the real cached mask PNGs
// (verified via pixel sampling, not a colormap function — the palette turned
// out to be a hand-picked discrete set, not matplotlib's jet as hypothesized:
// jet's endpoints are dark navy/maroon, but the actual masks are all bright
// and fully saturated, and include a pure green jet can't produce). Class
// assignment (which color = which tissue) was confirmed by cross-referencing
// mask regions against their matching H&E: red = Other (dominant background
// stroma/fat), blue = Invasive (confluent tumor sheets/nests), green =
// Non-invasive (smaller, often bordering the invasive regions), yellow =
// Necrosis (matches hemorrhagic/necrotic-debris regions on H&E). Still
// editable in the UI in case a given ROI's mask differs.
const SEG_CLASSES = [
  { id: 1, label: "Other",        color: "#ff3232" },
  { id: 2, label: "Non-invasive", color: "#32ff32" },
  { id: 3, label: "Invasive",     color: "#3296ff" },
  { id: 4, label: "Necrosis",     color: "#ffc800" },
];

// ─── Virtual IHC config ───────────────────────────────────────────────────────
const IHC_MARKERS = ["ER", "PR", "HER2", "KI67"];
const IHC_MARKER_LABELS = { ER: "ER", PR: "PR", HER2: "HER2", KI67: "Ki-67" };
// Matching transcript in the SEQUOIA gene panel, used as a secondary,
// non-clinical cross-reference next to the image-quantified protein call.
const IHC_GENE_MAP = { ER: "ESR1", PR: "PGR", HER2: "ERBB2", KI67: "MKI67" };

// ─── Backend analysis hook (for user-uploaded ROIs) ───────────────────────────
// Point this at your Modal `analyze_roi` endpoint to let uploaded ROIs run
// through segmentation / SEQUOIA / virtual-IHC inference. Left blank by
// default — uploaded ROIs display the raw H&E only until this is set, rather
// than silently failing against a fake URL.
const ANALYSIS_API_URL = process.env.NEXT_PUBLIC_ANALYSIS_API_URL || "";

// ─── Mitosis detection hook (Nottingham grade — mitotic count component) ─────
// Points at a Modal endpoint wrapping TIAToolbox's pretrained KongNet mitosis
// detector (model "KongNet_Det_MIDOG_1"), run in patch_mode since MammoMap
// works with cropped ROIs, not whole slide images. See mitosis_detection.py.
// Left blank by default, same as ANALYSIS_API_URL.
const MITOSIS_API_URL = process.env.NEXT_PUBLIC_MITOSIS_API_URL || "";
// Approximate conversion of the classic Nottingham/Elston-Ellis per-10-HPF
// mitotic-count cutoffs (≤11 / 12–22 / ≥23 per 10 HPF) to mitoses-per-mm²,
// assuming a 0.55 mm field diameter (10 fields ≈ 2.37 mm²) — a commonly cited
// convention, NOT a universal constant. Field diameter varies by microscope
// and by lab, so these are editable in the UI rather than hardcoded truth.
const MITOSIS_SCORE_BREAKS_DEFAULT = { low: 4.6, high: 9.7 };

// ─── Pre-computed Demo Patient Data ───────────────────────────────────────────
const DEMO_PATIENTS = [
  {
    id: "PAT-320", code: "BRCA-2026-320", subtype: "Luminal A", risk: "Low", age: 54,
    rois: [
      { label: "Tumor Core", src: "/patient320/patient320_wsi1_roi1.png", maskSrc: "/patient320/patient320_wsi1_roi1_mask.png", spatialSrc: "/patient320/patient320_wsi1_roi1_spatial.csv",
        metrics: { invasive_purity: 62.1, non_invasive_ratio: 12.4, other_ratio: 20.3, necrosis_ratio: 5.2 },
        ihc: {
          ER:   { src:"/patient320/patient320_wsi1_roi1_ER.png",   positivity:92, hscore:270, status:"Positive" },
          PR:   { src:"/patient320/patient320_wsi1_roi1_PR.png",   positivity:78, hscore:210, status:"Positive" },
          HER2: { src:"/patient320/patient320_wsi1_roi1_HER2.png", score:"1+", ish:"Negative", status:"Negative" },
          KI67: { src:"/patient320/patient320_wsi1_roi1_KI67.png", positivity:12, status:"Low" },
        } },
      { label: "Invasive Front", src: "/patient320/patient320_wsi1_roi2.png", maskSrc: "/patient320/patient320_wsi1_roi2_mask.png", spatialSrc: "/patient320/patient320_wsi1_roi2_spatial.csv",
        metrics: { invasive_purity: 45.0, non_invasive_ratio: 25.1, other_ratio: 28.0, necrosis_ratio: 1.9 },
        ihc: {
          ER:   { src:"/patient320/patient320_wsi1_roi2_ER.png",   positivity:85, hscore:250, status:"Positive" },
          PR:   { src:"/patient320/patient320_wsi1_roi2_PR.png",   positivity:70, hscore:190, status:"Positive" },
          HER2: { src:"/patient320/patient320_wsi1_roi2_HER2.png", score:"1+", ish:"Negative", status:"Negative" },
          KI67: { src:"/patient320/patient320_wsi1_roi2_KI67.png", positivity:18, status:"Low" },
        } },
      { label: "Stroma", src: "/patient320/patient320_wsi1_roi3.png", maskSrc: "/patient320/patient320_wsi1_roi3_mask.png", spatialSrc: "/patient320/patient320_wsi1_roi3_spatial.csv",
        metrics: { invasive_purity: 10.0, non_invasive_ratio: 5.1, other_ratio: 82.0, necrosis_ratio: 2.9 },
        ihc: {
          ER:   { src:"/patient320/patient320_wsi1_roi3_ER.png",   positivity:5, hscore:10, status:"Negative" },
          PR:   { src:"/patient320/patient320_wsi1_roi3_PR.png",   positivity:3, hscore:5, status:"Negative" },
          HER2: { src:"/patient320/patient320_wsi1_roi3_HER2.png", score:"0", ish:"Negative", status:"Negative" },
          KI67: { src:"/patient320/patient320_wsi1_roi3_KI67.png", positivity:2, status:"Low" },
        } }
    ]
  },
  {
    id: "PAT-321", code: "BRCA-2026-321", subtype: "HER2-enriched", risk: "High", age: 61,
    rois: [
      { label: "Tumor Core", src: "/patient321/patient321_wsi1_roi1.png", maskSrc: "/patient321/patient321_wsi1_roi1_mask.png", spatialSrc: "/patient321/patient321_wsi1_roi1_spatial.csv",
        metrics: { invasive_purity: 78.5, non_invasive_ratio: 5.2, other_ratio: 10.1, necrosis_ratio: 6.2 },
        ihc: {
          ER:   { src:"/patient321/patient321_wsi1_roi1_ER.png",   positivity:15, hscore:40, status:"Negative" },
          PR:   { src:"/patient321/patient321_wsi1_roi1_PR.png",   positivity:8, hscore:20, status:"Negative" },
          HER2: { src:"/patient321/patient321_wsi1_roi1_HER2.png", score:"3+", ish:"Positive", status:"Positive" },
          KI67: { src:"/patient321/patient321_wsi1_roi1_KI67.png", positivity:62, status:"High" },
        } },
      { label: "Stroma", src: "/patient321/patient321_wsi1_roi2.png", maskSrc: "/patient321/patient321_wsi1_roi2_mask.png", spatialSrc: "/patient321/patient321_wsi1_roi2_spatial.csv",
        metrics: { invasive_purity: 15.2, non_invasive_ratio: 10.0, other_ratio: 72.5, necrosis_ratio: 2.3 },
        ihc: {
          ER:   { src:"/patient321/patient321_wsi1_roi2_ER.png",   positivity:5, hscore:10, status:"Negative" },
          PR:   { src:"/patient321/patient321_wsi1_roi2_PR.png",   positivity:3, hscore:6, status:"Negative" },
          HER2: { src:"/patient321/patient321_wsi1_roi2_HER2.png", score:"0", ish:"Negative", status:"Negative" },
          KI67: { src:"/patient321/patient321_wsi1_roi2_KI67.png", positivity:4, status:"Low" },
        } },
      { label: "Necrosis Focus", src: "/patient321/patient321_wsi1_roi3.png", maskSrc: "/patient321/patient321_wsi1_roi3_mask.png", spatialSrc: "/patient321/patient321_wsi1_roi3_spatial.csv",
        metrics: { invasive_purity: 35.4, non_invasive_ratio: 4.1, other_ratio: 15.5, necrosis_ratio: 45.0 },
        ihc: {
          ER:   { src:"/patient321/patient321_wsi1_roi3_ER.png",   positivity:2, hscore:5, status:"Negative" },
          PR:   { src:"/patient321/patient321_wsi1_roi3_PR.png",   positivity:1, hscore:2, status:"Negative" },
          HER2: { src:"/patient321/patient321_wsi1_roi3_HER2.png", score:"0", ish:"Negative", status:"Negative" },
          KI67: { src:"/patient321/patient321_wsi1_roi3_KI67.png", positivity:8, status:"Low" },
        } },
      { label: "Invasive Front", src: "/patient321/patient321_wsi1_roi4.png", maskSrc: "/patient321/patient321_wsi1_roi4_mask.png", spatialSrc: "/patient321/patient321_wsi1_roi4_spatial.csv",
        metrics: { invasive_purity: 45.4, non_invasive_ratio: 14.1, other_ratio: 35.5, necrosis_ratio: 5.0 },
        ihc: {
          ER:   { src:"/patient321/patient321_wsi1_roi4_ER.png",   positivity:20, hscore:55, status:"Negative" },
          PR:   { src:"/patient321/patient321_wsi1_roi4_PR.png",   positivity:10, hscore:25, status:"Negative" },
          HER2: { src:"/patient321/patient321_wsi1_roi4_HER2.png", score:"3+", ish:"Positive", status:"Positive" },
          KI67: { src:"/patient321/patient321_wsi1_roi4_KI67.png", positivity:58, status:"High" },
        } }
    ]
  },
  {
    id: "PAT-322", code: "BRCA-2026-322", subtype: "Triple-negative", risk: "High", age: 48,
    rois: [
      { label: "Tumor Core", src: "/patient322/patient322_wsi1_roi1.png", maskSrc: "/patient322/patient322_wsi1_roi1_mask.png", spatialSrc: "/patient322/patient322_wsi1_roi1_spatial.csv",
        metrics: { invasive_purity: 82.0, non_invasive_ratio: 2.1, other_ratio: 8.4, necrosis_ratio: 7.5 },
        ihc: {
          ER:   { src:"/patient322/patient322_wsi1_roi1_ER.png",   positivity:2, hscore:5, status:"Negative" },
          PR:   { src:"/patient322/patient322_wsi1_roi1_PR.png",   positivity:1, hscore:3, status:"Negative" },
          HER2: { src:"/patient322/patient322_wsi1_roi1_HER2.png", score:"0", ish:"Negative", status:"Negative" },
          KI67: { src:"/patient322/patient322_wsi1_roi1_KI67.png", positivity:75, status:"High" },
        } },
      { label: "Invasive Front", src: "/patient322/patient322_wsi1_roi2.png", maskSrc: "/patient322/patient322_wsi1_roi2_mask.png", spatialSrc: "/patient322/patient322_wsi1_roi2_spatial.csv",
        metrics: { invasive_purity: 55.6, non_invasive_ratio: 12.0, other_ratio: 30.1, necrosis_ratio: 2.3 },
        ihc: {
          ER:   { src:"/patient322/patient322_wsi1_roi2_ER.png",   positivity:3, hscore:6, status:"Negative" },
          PR:   { src:"/patient322/patient322_wsi1_roi2_PR.png",   positivity:2, hscore:4, status:"Negative" },
          HER2: { src:"/patient322/patient322_wsi1_roi2_HER2.png", score:"0", ish:"Negative", status:"Negative" },
          KI67: { src:"/patient322/patient322_wsi1_roi2_KI67.png", positivity:68, status:"High" },
        } }
    ]
  }
];

// ─── Layer groups reducer ─────────────────────────────────────────────────────
const mkGroups = () => [
  {id:"base",  label:"Base Image",    expanded:true,  composite:"normal",   layers:[{id:"he", label:"H&E ROI", vis:true, op:100, color:T.txt, locked:true}]},
  {id:"seg",   label:"Segmentation",  expanded:true,  composite:"normal",   layers:[
    {id:"seg_mask", label:"Model Prediction", vis:true, op:60, color:T.teal, locked:false}
  ]},
  {id:"ihc",   label:"Virtual IHC",   expanded:true,  composite:"normal",  layers:[
    {id:"ihc_overlay", label:"IHC Overlay", vis:false, op:70, color:T.purple, locked:false}
  ]},
  {id:"mitosis", label:"Mitotic Count", expanded:true, composite:"normal", layers:[
    {id:"mitosis_pts", label:"Detected Mitoses", vis:true, op:100, color:T.red, locked:false}
  ]},
  {id:"spatial",label:"SEQUOIA Spatial",expanded:true, composite:"normal", layers:[
    {id:"spatial_expr", label:"Gene Expression", vis:true, op:80, color:T.amber, locked:false}
  ]}
];
function gRed(st,a) {
  const c=st.map(g=>({...g,layers:g.layers.map(l=>({...l}))}));
  if(a.type==="TG") return c.map(g=>g.id===a.gid?{...g,expanded:!g.expanded}:g);
  if(a.type==="TL") return c.map(g=>({...g,layers:g.layers.map(l=>l.id===a.lid?{...l,vis:!l.vis}:l)}));
  if(a.type==="OP") return c.map(g=>({...g,layers:g.layers.map(l=>l.id===a.lid?{...l,op:a.val}:l)}));
  return c;
}

const Eye=({on,toggle,color="#00d4aa"})=>(<div onClick={e=>{e.stopPropagation();toggle();}} title={on?"Hide":"Show"} style={{width:22,height:22,display:"flex",alignItems:"center",justifyContent:"center",cursor:"pointer",color:on?color:T.txt3,flexShrink:0,borderRadius:4}} onMouseEnter={e=>e.currentTarget.style.background=T.bg4} onMouseLeave={e=>e.currentTarget.style.background="transparent"}>
  {on?<svg width="14" height="10" viewBox="0 0 14 10" fill="none"><path d="M7 1C4 1 1.5 4 1.5 5S4 9 7 9 12.5 6 12.5 5 10 1 7 1Z" stroke="currentColor" strokeWidth="1.3"/><circle cx="7" cy="5" r="2" fill="currentColor"/></svg>
    :<svg width="14" height="12" viewBox="0 0 14 12" fill="none"><path d="M1 1L13 11M7 2C4.5 2 2.5 4 1.5 5.5M7 9C9.5 9 11.5 7 12.5 5.5" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round"/></svg>}
</div>);

// ─── CSV Parser ───────────────────────────────────────────────────────────────
async function fetchSpatialCSV(url) {
  try {
    const res = await fetch(url);
    if (!res.ok) return null;
    const text = await res.text();
    const lines = text.trim().split('\n');
    const headers = lines[0].split(',');
    return lines.slice(1).map(line => {
      const values = line.split(',');
      const row = {};
      headers.forEach((h, i) => row[h.trim()] = parseFloat(values[i]));
      return row;
    });
  } catch (err) {
    console.error("Failed to load spatial data:", err);
    return null;
  }
}

// ─── Coordinate mapping ───────────────────────────────────────────────────────
// Spatial CSVs can encode cx/cy in different conventions (fraction 0-1,
// percentage 0-100, or raw grid/pixel coordinates like the sliding-window
// script that produced this data). We detect which convention the file is
// using from its own value range, so the overlay always lines up with the
// H&E image regardless of how the coordinates were exported.
function computeBounds(data) {
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  data.forEach(pt => {
    if (Number.isFinite(pt.cx)) { if (pt.cx < minX) minX = pt.cx; if (pt.cx > maxX) maxX = pt.cx; }
    if (Number.isFinite(pt.cy)) { if (pt.cy < minY) minY = pt.cy; if (pt.cy > maxY) maxY = pt.cy; }
  });
  return { minX, maxX, minY, maxY };
}

function mapCoordToImage(pt, bounds, imgW, imgH) {
  const { minX, maxX, minY, maxY } = bounds;
  let fx, fy;
  if (maxX <= 1.001 && minX >= -0.001 && maxY <= 1.001 && minY >= -0.001) {
    fx = pt.cx; fy = pt.cy;
  } else if (maxX <= 100.001 && minX >= -0.001 && maxY <= 100.001 && minY >= -0.001) {
    fx = pt.cx / 100; fy = pt.cy / 100;
  } else {
    const rangeX = (maxX - minX) || 1;
    const rangeY = (maxY - minY) || 1;
    fx = (pt.cx - minX) / rangeX;
    fy = (pt.cy - minY) / rangeY;
  }
  return { px: fx * imgW, py: fy * imgH };
}

function hexToRgba(hex, alpha) {
  const h = hex.replace('#', '');
  const full = h.length === 3 ? h.split('').map(c => c + c).join('') : h;
  const bigint = parseInt(full, 16);
  const r = (bigint >> 16) & 255, g = (bigint >> 8) & 255, b = bigint & 255;
  return `rgba(${r},${g},${b},${Math.max(0, Math.min(alpha, 1))})`;
}

function drawStar(ctx, cx, cy, r) {
  const spikes = 5, outerR = r, innerR = r / 2.4;
  let rot = (Math.PI / 2) * 3;
  const step = Math.PI / spikes;
  ctx.beginPath();
  ctx.moveTo(cx, cy - outerR);
  for (let i = 0; i < spikes; i++) {
    let x = cx + Math.cos(rot) * outerR, y = cy + Math.sin(rot) * outerR;
    ctx.lineTo(x, y); rot += step;
    x = cx + Math.cos(rot) * innerR; y = cy + Math.sin(rot) * innerR;
    ctx.lineTo(x, y); rot += step;
  }
  ctx.closePath();
  ctx.fill();
}

function drawShape(ctx, shape, x, y, size, color, alpha) {
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.fillStyle = color;
  switch (shape) {
    case "square":
      ctx.fillRect(x - size, y - size, size * 2, size * 2);
      break;
    case "triangle":
      ctx.beginPath();
      ctx.moveTo(x, y - size);
      ctx.lineTo(x + size, y + size);
      ctx.lineTo(x - size, y + size);
      ctx.closePath();
      ctx.fill();
      break;
    case "diamond":
      ctx.beginPath();
      ctx.moveTo(x, y - size);
      ctx.lineTo(x + size, y);
      ctx.lineTo(x, y + size);
      ctx.lineTo(x - size, y);
      ctx.closePath();
      ctx.fill();
      break;
    case "star":
      drawStar(ctx, x, y, size);
      break;
    default: // circle
      ctx.beginPath();
      ctx.arc(x, y, size, 0, Math.PI * 2);
      ctx.fill();
  }
  ctx.restore();
}

// Small shape swatch used in gene rows / legend
function ShapeIcon({ shape, color, size = 12 }) {
  const s = size;
  const common = { width: s, height: s, display: "inline-block", flexShrink: 0 };
  if (shape === "circle") return <span style={{ ...common, borderRadius: "50%", background: color }} />;
  if (shape === "square") return <span style={{ ...common, background: color }} />;
  if (shape === "diamond") return <span style={{ ...common, background: color, transform: "rotate(45deg)" }} />;
  if (shape === "triangle") return (
    <svg width={s} height={s} viewBox="0 0 10 10"><polygon points="5,0 10,10 0,10" fill={color} /></svg>
  );
  if (shape === "star") return (
    <svg width={s} height={s} viewBox="0 0 20 20"><polygon points="10,0 13,7 20,7 14,11 16,19 10,14 4,19 6,11 0,7 7,7" fill={color} /></svg>
  );
  return <span style={{ ...common, borderRadius: "50%", background: color }} />;
}

// Ringed crosshair marker for detected mitotic figures — visually distinct
// from the filled gene-overlay shapes so the two overlays don't get confused.
function drawMitosisMarker(ctx, x, y, r, color) {
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = Math.max(1, r * 0.35);
  ctx.beginPath();
  ctx.arc(x, y, r, 0, Math.PI * 2);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(x - r * 0.6, y); ctx.lineTo(x + r * 0.6, y);
  ctx.moveTo(x, y - r * 0.6); ctx.lineTo(x, y + r * 0.6);
  ctx.stroke();
  ctx.restore();
}


// Replaces the earlier hardcoded demo positivity/H-score with numbers actually
// measured from the cached virtual-IHC PNG, using the same H-score method real
// digital-pathology tools use: classify each tissue pixel's DAB (brown)
// staining intensity into 0/1+/2+/3+ bands, then
//   positivity% = %(1+) + %(2+) + %(3+)
//   H-score     = 1·%(1+) + 2·%(2+) + 3·%(3+)   (range 0–300)
// This is a color-based heuristic, not a validated pathology assay — see the
// caveats returned alongside each status.
// Re-encodes a loaded <img> (whether it came from a URL or a data: URL) as a
// PNG data URL, so the mitosis-detection call works the same way for cached
// demo ROIs and uploaded ones. Same-origin assumption as the rest of the app.
function imageElementToBase64(img) {
  const canvas = document.createElement("canvas");
  canvas.width = img.naturalWidth || img.width;
  canvas.height = img.naturalHeight || img.height;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(img, 0, 0);
  return canvas.toDataURL("image/png");
}

function quantifyIHCImage(img, maxDim = 300) {
  try {
    const canvas = document.createElement("canvas");
    const scale = Math.min(1, maxDim / Math.max(img.naturalWidth || img.width, img.naturalHeight || img.height));
    canvas.width = Math.max(1, Math.round((img.naturalWidth || img.width) * scale));
    canvas.height = Math.max(1, Math.round((img.naturalHeight || img.height) * scale));
    const ctx = canvas.getContext("2d");
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    const data = ctx.getImageData(0, 0, canvas.width, canvas.height).data;

    let tissue = 0, b1 = 0, b2 = 0, b3 = 0;
    for (let i = 0; i < data.length; i += 4) {
      const r = data[i], g = data[i + 1], b = data[i + 2], a = data[i + 3];
      if (a < 20) continue;
      const lum = (r + g + b) / 3;
      if (lum > 235 || lum < 15) continue; // background / mounting-medium / artifact
      tissue++;
      const brown = (r - b) / 255; // DAB (positive) reads brown; hematoxylin (negative) reads blue
      if (brown < 0.05) continue;  // counterstain only → band 0, not counted as positive
      const darkness = 1 - lum / 255;
      const intensity = Math.min(1, brown) * 0.6 + darkness * 0.4;
      if (intensity < 0.28) b1++;
      else if (intensity < 0.5) b2++;
      else b3++;
    }
    if (tissue === 0) return null;
    const p1 = (b1 / tissue) * 100, p2 = (b2 / tissue) * 100, p3 = (b3 / tissue) * 100;
    return { positivity: p1 + p2 + p3, hscore: p1 * 1 + p2 * 2 + p3 * 3 };
  } catch (err) {
    // Most likely a tainted canvas (cross-origin image without CORS headers)
    console.warn("IHC quantification skipped:", err);
    return null;
  }
}

function deriveIHCStatus(marker, positivity, hscore) {
  if (marker === "ER" || marker === "PR") {
    // ASCO/CAP guideline: ≥1% tumor nuclei staining is called positive.
    return positivity >= 1
      ? { status: "Positive", caveat: "≥1% stained (ASCO/CAP threshold)" }
      : { status: "Negative", caveat: "<1% stained" };
  }
  if (marker === "KI67") {
    const status = positivity < 10 ? "Low" : positivity < 30 ? "Intermediate" : "High";
    return { status, caveat: "No single universal cutoff — bands shown are for reference only" };
  }
  if (marker === "HER2") {
    let status;
    if (hscore < 30) status = "Negative (0/1+)";
    else if (hscore < 120) status = "Equivocal (2+) — confirm with ISH";
    else status = "Positive (3+)";
    return { status, caveat: "Approximate — real HER2 scoring needs membrane-pattern assessment, not just stain color" };
  }
  return { status: "Unknown", caveat: "" };
}


function MiniChart({ type, data, xLabel, yLabel }) {
  const W = 300, H = 170, P = 30;
  if (!data || data.length === 0) return <div style={{ fontSize: 11, color: T.txt3 }}>No data to plot.</div>;

  if (type === "scatter") {
    const xs = data.map(d => d.x), ys = data.map(d => d.y);
    const xMin = Math.min(...xs), xMax = Math.max(...xs), yMin = Math.min(...ys), yMax = Math.max(...ys);
    const sx = v => P + ((v - xMin) / ((xMax - xMin) || 1)) * (W - 2 * P);
    const sy = v => H - P - ((v - yMin) / ((yMax - yMin) || 1)) * (H - 2 * P);
    return (
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: H }}>
        <line x1={P} y1={H - P} x2={W - P} y2={H - P} stroke={T.bdr} />
        <line x1={P} y1={P} x2={P} y2={H - P} stroke={T.bdr} />
        {data.map((d, i) => <circle key={i} cx={sx(d.x)} cy={sy(d.y)} r={4} fill={T.teal} opacity={0.85} />)}
        <text x={W / 2} y={H - 6} textAnchor="middle" fontSize="9" fill={T.txt3}>{xLabel}</text>
        <text x={10} y={H / 2} textAnchor="middle" fontSize="9" fill={T.txt3} transform={`rotate(-90 10 ${H / 2})`}>{yLabel}</text>
      </svg>
    );
  }

  if (type === "line") {
    const ys = data.map(d => d.value);
    const yMin = Math.min(...ys, 0), yMax = Math.max(...ys);
    const stepX = (W - 2 * P) / ((data.length - 1) || 1);
    const sy = v => H - P - ((v - yMin) / ((yMax - yMin) || 1)) * (H - 2 * P);
    const pts = data.map((d, i) => `${P + i * stepX},${sy(d.value)}`).join(" ");
    return (
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: H }}>
        <line x1={P} y1={H - P} x2={W - P} y2={H - P} stroke={T.bdr} />
        <polyline points={pts} fill="none" stroke={T.amber} strokeWidth="2" />
        {data.map((d, i) => <circle key={i} cx={P + i * stepX} cy={sy(d.value)} r={3.5} fill={T.amber} />)}
        {data.map((d, i) => <text key={"l" + i} x={P + i * stepX} y={H - P + 13} textAnchor="middle" fontSize="8" fill={T.txt3}>{d.label.slice(0, 10)}</text>)}
      </svg>
    );
  }

  // bar & ranked-bar
  const sorted = type === "ranked-bar" ? [...data].sort((a, b) => b.value - a.value) : data;
  const maxV = Math.max(...sorted.map(d => d.value), 0.0001);
  const gap = (W - 2 * P) / sorted.length;
  const barW = gap * 0.6;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: H }}>
      <line x1={P} y1={H - P} x2={W - P} y2={H - P} stroke={T.bdr} />
      {sorted.map((d, i) => {
        const h = (d.value / maxV) * (H - 2 * P);
        const x = P + i * gap + (gap - barW) / 2;
        const y = H - P - h;
        return (
          <g key={i}>
            <rect x={x} y={y} width={barW} height={h} fill={i === 0 && type === "ranked-bar" ? T.teal : T.blue} rx={2} />
            <text x={x + barW / 2} y={H - P + 13} textAnchor="middle" fontSize="8" fill={T.txt3}>{d.label.slice(0, 9)}</text>
            <text x={x + barW / 2} y={y - 4} textAnchor="middle" fontSize="8" fill={T.txt}>{d.value.toFixed(2)}</text>
          </g>
        );
      })}
    </svg>
  );
}

// ─── Lightweight agent (rule-based intent parsing + on-demand data fetch) ─────
// This is intentionally NOT calling a hosted LLM: doing that from the browser
// would mean shipping an API key to the client. It's a small deterministic
// planner instead — parse → decide chart type → fetch any spatial data it
// doesn't have yet → compute → explain. Swap `planIntent` for a call to your
// own backend (which can itself call Claude) if you want smarter parsing.
const METRIC_KEYWORDS = [
  { key: "invasive_purity", terms: ["invasive purity", "invasive"] },
  { key: "non_invasive_ratio", terms: ["non-invasive", "non invasive", "noninvasive"] },
  { key: "other_ratio", terms: ["stroma", "other tissue", "other"] },
  { key: "necrosis_ratio", terms: ["necrosis", "necrotic"] },
];
const IHC_ALIASES = {
  ER: ["er", "estrogen"],
  PR: ["pr", "progesterone"],
  HER2: ["her2", "her-2"],
  KI67: ["ki67", "ki-67", "proliferation"],
};
const EXAMPLE_PROMPTS = [
  "Rank ROIs by MKI67 expression",
  "Compare necrosis vs invasive purity",
  "Show ER positivity across ROIs",
  "ESR1 vs ERBB2",
];

function findAllMatches(lower, availableGenes) {
  const found = [];
  METRIC_KEYWORDS.forEach(m => m.terms.forEach(t => {
    const idx = lower.indexOf(t);
    if (idx !== -1) found.push({ type: "metric", key: m.key, label: t, idx });
  }));
  Object.entries(IHC_ALIASES).forEach(([marker, aliases]) => aliases.forEach(t => {
    const idx = lower.indexOf(t);
    if (idx !== -1) found.push({ type: "ihc", key: marker, label: IHC_MARKER_LABELS[marker], idx });
  }));
  (availableGenes || []).forEach(g => {
    const idx = lower.indexOf(g.toLowerCase());
    if (idx !== -1) found.push({ type: "gene", key: g, label: g, idx });
  });
  const seen = new Set(), uniq = [];
  found.sort((a, b) => a.idx - b.idx).forEach(f => {
    const sig = f.type + ":" + f.key;
    if (!seen.has(sig)) { seen.add(sig); uniq.push(f); }
  });
  return uniq;
}

function detectChartType(lower, subjectCount) {
  if (subjectCount >= 2 && /(vs\.?|versus|correlat|relationship| and )/.test(lower)) return "scatter";
  if (/trend|progress|across roi|over the roi/.test(lower)) return "line";
  if (/top|highest|rank|most|least|lowest/.test(lower)) return "ranked-bar";
  return "bar";
}

async function getSubjectValueForROI(subject, roi, idx, ensureSpatialLoaded) {
  if (subject.type === "metric") return roi.metrics ? roi.metrics[subject.key] : null;
  if (subject.type === "ihc") {
    const entry = roi.ihc && roi.ihc[subject.key];
    if (!entry) return null;
    return Number.isFinite(entry.positivity) ? entry.positivity : (Number.isFinite(entry.hscore) ? entry.hscore : null);
  }
  if (subject.type === "gene") {
    const rows = await ensureSpatialLoaded(idx);
    if (!rows || rows.length === 0) return null;
    const vals = rows.map(r => r[subject.key]).filter(v => Number.isFinite(v));
    if (!vals.length) return null;
    return vals.reduce((a, b) => a + b, 0) / vals.length;
  }
  return null;
}

async function runAgent(query, ctx) {
  const lower = query.toLowerCase();
  const subjects = findAllMatches(lower, ctx.availableGenes);
  if (subjects.length === 0) {
    return { chart: null, explanation: "I couldn't match that to a gene, IHC marker, or segmentation class. Try something like \u201ccompare MKI67 across ROIs\u201d, \u201cnecrosis vs invasive purity\u201d, or \u201crank ROIs by ER positivity\u201d. (Gene names are only recognized once you've opened an ROI with spatial data.)" };
  }
  const chartType = detectChartType(lower, subjects.length);

  if (chartType === "scatter" && subjects.length >= 2) {
    const [sA, sB] = subjects;
    const points = [];
    for (let i = 0; i < ctx.roiList.length; i++) {
      const roi = ctx.roiList[i];
      const vx = await getSubjectValueForROI(sA, roi, i, ctx.ensureSpatialLoaded);
      const vy = await getSubjectValueForROI(sB, roi, i, ctx.ensureSpatialLoaded);
      if (Number.isFinite(vx) && Number.isFinite(vy)) points.push({ x: vx, y: vy, label: roi.label });
    }
    if (!points.length) return { chart: null, explanation: `No ROIs have both ${sA.label} and ${sB.label} data available yet.` };
    return {
      chart: { type: "scatter", data: points, xLabel: sA.label, yLabel: sB.label, title: `${sA.label} vs ${sB.label}` },
      explanation: `Plotted ${sA.label} against ${sB.label} across ${points.length} ROI${points.length > 1 ? "s" : ""} with data for both.`
    };
  }

  const s = subjects[0];
  const rows = [];
  for (let i = 0; i < ctx.roiList.length; i++) {
    const roi = ctx.roiList[i];
    const v = await getSubjectValueForROI(s, roi, i, ctx.ensureSpatialLoaded);
    if (Number.isFinite(v)) rows.push({ label: roi.label, value: v });
  }
  if (!rows.length) return { chart: null, explanation: `No ROIs currently have data for ${s.label}. Upload/analyze an ROI or open one with spatial data first.` };

  const sorted = [...rows].sort((a, b) => b.value - a.value);
  const explanation = `${sorted[0].label} has the highest ${s.label} (${sorted[0].value.toFixed(2)})${sorted.length > 1 ? `, ${sorted[sorted.length - 1].label} the lowest (${sorted[sorted.length - 1].value.toFixed(2)})` : ""}.`;

  return {
    chart: { type: chartType, data: rows, title: `${s.label} across ROIs` },
    explanation
  };
}

// ─── Logo ──────────────────────────────────────────────────────────────────
// Tries a real logo file first (drop one at /logo.png in your public folder)
// and falls back to a built-in mark + wordmark so the app never ships with a
// broken image while you don't have a final logo yet.
function Logo({ size = 30 }) {
  const [imgOk, setImgOk] = useState(true);
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
      {imgOk ? (
        <img
          src="/logo.png"
          alt="MammoMap"
          style={{ height: size, width: size, objectFit: "contain" }}
          onError={() => setImgOk(false)}
        />
      ) : (
        <svg width={size} height={size} viewBox="0 0 32 32" fill="none">
          <path d="M16 2 L29 9 V23 L16 30 L3 23 V9 Z" stroke={T.teal} strokeWidth="2" fill="rgba(0,212,170,0.1)" />
          <circle cx="16" cy="16" r="6" fill="none" stroke={T.teal} strokeWidth="2" />
          <circle cx="16" cy="16" r="2" fill={T.teal} />
        </svg>
      )}
      <span style={{ fontSize: Math.round(size * 0.6), fontWeight: 800, letterSpacing: 0.2, color: T.txt }}>MammoMap</span>
    </div>
  );
}

// ─── Home = Gallery (cached cases + a functional "upload your own" tile) ────
function Gallery({ patients, onSelectPatient, onUploadPatient }) {
  const fileInputRef = useRef();
  const handleUploadClick = () => fileInputRef.current?.click();

  const handleFilesSelected = (e) => {
    const files = Array.from(e.target.files || []);
    if (!files.length) return;
    Promise.all(files.map(file => new Promise(resolve => {
      const reader = new FileReader();
      reader.onload = () => resolve({
        label: file.name.replace(/\.[a-zA-Z0-9]+$/, "") || "Uploaded ROI",
        src: reader.result,
        maskSrc: null, spatialSrc: null, ihc: null, metrics: null,
        custom: true, analysisStatus: "idle"
      });
      reader.readAsDataURL(file);
    }))).then(rois => onUploadPatient(rois));
    e.target.value = "";
  };

  return (
    <div style={{background:T.bg1,minHeight:"100vh",color:T.txt,fontFamily:"system-ui",padding:"32px 40px"}}>
      <div style={{maxWidth: 1040, margin: "0 auto"}}>
        <Logo />
        <p style={{color:T.txt2, fontSize:13, margin:"8px 0 28px"}}>Breast pathology case gallery — open a cached case below, or upload your own ROIs to start a new one.</p>

        <div style={{display:"grid",gridTemplateColumns:"repeat(auto-fill, minmax(280px, 1fr))",gap:20}}>
          {patients.map(p => (
            <div key={p.id} onClick={() => onSelectPatient(p.rois)} style={{background:T.bg2, border:`1px solid ${T.bdr}`, borderRadius:10, padding:20, cursor:"pointer"}}>
              <div style={{fontWeight:700, fontSize:16, marginBottom: 5}}>{p.code}</div>
              <div style={{fontSize:12, color:T.txt2, marginBottom:15}}>{p.subtype} • {p.rois.length} ROI{p.rois.length > 1 ? "s" : ""}</div>
              <div style={{display:"flex", gap:10, overflowX:"auto"}}>
                {p.rois.slice(0, 4).map((r, i) => (
                  <img key={i} src={r.src} alt={r.label} style={{width:64, height:64, objectFit:"cover", borderRadius:6, flexShrink:0}} onError={e => { e.currentTarget.style.display = "none"; }} />
                ))}
              </div>
            </div>
          ))}

          <div
            onClick={handleUploadClick}
            style={{background:T.bg2, border:`1.5px dashed ${T.bdr}`, borderRadius:10, padding:20, cursor:"pointer", display:"flex", flexDirection:"column", alignItems:"center", justifyContent:"center", textAlign:"center", minHeight:148, gap:8, color:T.txt2, transition:"border-color .15s"}}
            onMouseEnter={e => e.currentTarget.style.borderColor = T.teal}
            onMouseLeave={e => e.currentTarget.style.borderColor = T.bdr}
          >
            <div style={{fontSize:26, color:T.teal, lineHeight:1}}>+</div>
            <div style={{fontSize:13, fontWeight:600, color:T.txt}}>Upload Patient ROIs</div>
            <div style={{fontSize:11, color:T.txt3}}>Start a new case from your own images</div>
          </div>
        </div>

        <input ref={fileInputRef} type="file" accept="image/*" multiple onChange={handleFilesSelected} style={{display:"none"}} />
      </div>
    </div>
  );
}

// ─── 3. Analysis Viewer (Multi-ROI + Segmentation + IHC + Spatial + Insights) ─
function AnalysisViewer({ rois, onBack }) {
  const [activeIdx, setActiveIdx] = useState("aggregate");
  const [groups, dispatch] = useReducer(gRed, null, mkGroups);
  const [panelTab, setPanelTab] = useState("layers"); // "layers" | "insights"

  // ROI list is local state so uploaded ROIs and analysis results can be appended
  const [roiList, setRoiList] = useState(rois);
  const fileInputRef = useRef();

  // SEQUOIA spatial state
  const [spatialData, setSpatialData] = useState(null);
  const [availableGenes, setAvailableGenes] = useState([]);
  const [geneLayers, setGeneLayers] = useState([]);
  const [geneSearch, setGeneSearch] = useState("");
  const spatialCache = useRef({}); // idx -> parsed rows, used by the agent

  // Segmentation legend (editable in case the mask PNG's real colors differ)
  const [segLegend, setSegLegend] = useState(SEG_CLASSES);
  const updateSegColor = (id, color) => setSegLegend(prev => prev.map(c => c.id === id ? { ...c, color } : c));

  // Virtual IHC state
  const [activeIHCMarker, setActiveIHCMarker] = useState("ER");
  const [ihcImage, setIhcImage] = useState(null);
  const [ihcImageStatus, setIhcImageStatus] = useState("idle"); // idle | loading | loaded | missing
  const ihcImgCache = useRef({});
  const [computedIHC, setComputedIHC] = useState({}); // `${idx}_${marker}` -> { positivity, hscore, status, caveat }

  // Mitosis detection state (Nottingham grade — mitotic count component)
  const [mitosisResults, setMitosisResults] = useState({}); // idx -> { points, count, status, error }
  const [roiMpp, setRoiMpp] = useState({}); // idx -> microns-per-pixel the user has entered for that ROI
  const [scoreBreaks, setScoreBreaks] = useState(MITOSIS_SCORE_BREAKS_DEFAULT); // mitoses/mm² cutoffs, editable

  // Agentic insights panel
  const [query, setQuery] = useState("");
  const [agentBusy, setAgentBusy] = useState(false);
  const [agentResult, setAgentResult] = useState(null);

  // View transform (pan/zoom)
  const tf = useRef({ x: 20, y: 20, scale: 1 });
  const [scaleDisplay, setScaleDisplay] = useState(1);
  const [isPanning, setIsPanning] = useState(false);
  const panState = useRef({ startX: 0, startY: 0, origX: 0, origY: 0 });

  const cvsMain = useRef(), cont = useRef();
  const [images, setImages] = useState({ base: null, mask: null });

  const activeRoi = activeIdx === "aggregate" ? null : roiList[activeIdx];
  const bounds = useMemo(() => (spatialData && spatialData.length ? computeBounds(spatialData) : null), [spatialData]);

  // Merge the image-measured IHC result over the demo placeholder (when present),
  // and compute a secondary, non-clinical transcript cross-reference.
  const ihcCacheKey = activeIdx !== "aggregate" ? `${activeIdx}_${activeIHCMarker}` : null;
  const displayIHC = useMemo(() => {
    if (!activeRoi) return null;
    const demo = activeRoi.ihc && activeRoi.ihc[activeIHCMarker];
    const measured = ihcCacheKey ? computedIHC[ihcCacheKey] : null;
    if (measured) {
      return {
        positivity: measured.positivity, hscore: measured.hscore, status: measured.status, caveat: measured.caveat,
        score: demo?.score, ish: demo?.ish, // HER2's ISH result is a separate assay, not derivable from the stain image
        method: "measured"
      };
    }
    return demo ? { ...demo, method: "placeholder" } : null;
  }, [activeRoi, activeIHCMarker, computedIHC, ihcCacheKey]);

  const transcriptGene = IHC_GENE_MAP[activeIHCMarker];
  const transcriptMean = useMemo(() => {
    if (!spatialData || !transcriptGene) return null;
    const vals = spatialData.map(r => r[transcriptGene]).filter(v => Number.isFinite(v));
    if (!vals.length) return null;
    return vals.reduce((a, b) => a + b, 0) / vals.length;
  }, [spatialData, transcriptGene]);

  // Mitotic density (per mm²) and Nottingham mitotic-count score, only
  // computable once the ROI's µm/pixel is known — otherwise we'd be turning a
  // pixel count into a fake physical measurement.
  const activeMitosis = activeIdx !== "aggregate" ? mitosisResults[activeIdx] : null;
  const activeMpp = activeIdx !== "aggregate" ? roiMpp[activeIdx] : null;
  const mitosisDensity = useMemo(() => {
    if (!activeMitosis?.count || !activeMpp || !images.base) return null;
    const areaMm2 = (images.base.width * activeMpp / 1000) * (images.base.height * activeMpp / 1000);
    if (!areaMm2) return null;
    return activeMitosis.count / areaMm2;
  }, [activeMitosis, activeMpp, images.base]);
  const mitosisScore = useMemo(() => {
    if (mitosisDensity == null) return null;
    if (mitosisDensity <= scoreBreaks.low) return 1;
    if (mitosisDensity <= scoreBreaks.high) return 2;
    return 3;
  }, [mitosisDensity, scoreBreaks]);

  // 1. Load base + mask images and spatial data when the active ROI changes
  useEffect(() => {
    if (activeIdx === "aggregate" || !activeRoi) {
      setImages({ base: null, mask: null });
      setSpatialData(null);
      return;
    }

    let cancelled = false;
    const baseImg = new Image();
    baseImg.src = activeRoi.originalImageBase64 || activeRoi.src;

    const maskUrl = activeRoi.maskBase64 || activeRoi.maskSrc;
    const maskImg = maskUrl ? new Image() : null;
    if (maskImg) maskImg.src = maskUrl;

    let baseDone = false, maskDone = !maskImg;
    const finish = () => { if (!cancelled && baseDone && maskDone) setImages({ base: baseImg, mask: maskImg }); };
    baseImg.onload = () => { baseDone = true; finish(); };
    baseImg.onerror = () => { baseDone = true; finish(); };
    if (maskImg) {
      maskImg.onload = () => { maskDone = true; finish(); };
      maskImg.onerror = () => { maskDone = true; finish(); };
    }

    if (Array.isArray(activeRoi.spatialData)) {
      setSpatialData(activeRoi.spatialData);
    } else if (activeRoi.spatialSrc) {
      fetchSpatialCSV(activeRoi.spatialSrc).then(data => { if (!cancelled) setSpatialData(data); });
    } else {
      setSpatialData(null);
    }

    return () => { cancelled = true; };
  }, [activeIdx, activeRoi]);

  // 2. Load the active marker's virtual-IHC overlay image (lazy + cached),
  // then quantify it so Positivity/H-score/Status are measured, not fabricated.
  useEffect(() => {
    if (activeIdx === "aggregate" || !activeRoi) { setIhcImage(null); setIhcImageStatus("idle"); return; }
    const entry = activeRoi.ihc && activeRoi.ihc[activeIHCMarker];
    const url = entry && (entry.base64 || entry.src);
    if (!url) { setIhcImage(null); setIhcImageStatus("missing"); return; }

    const cacheKey = `${activeIdx}_${activeIHCMarker}`;
    const quantifyAndStore = (imgEl) => {
      if (computedIHC[cacheKey]) return;
      const q = quantifyIHCImage(imgEl);
      if (!q) return;
      const { status, caveat } = deriveIHCStatus(activeIHCMarker, q.positivity, q.hscore);
      setComputedIHC(prev => ({ ...prev, [cacheKey]: { ...q, status, caveat } }));
    };

    if (ihcImgCache.current[cacheKey]) {
      setIhcImage(ihcImgCache.current[cacheKey]);
      setIhcImageStatus("loaded");
      quantifyAndStore(ihcImgCache.current[cacheKey]);
      return;
    }
    setIhcImageStatus("loading");
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => {
      ihcImgCache.current[cacheKey] = img;
      setIhcImage(img);
      setIhcImageStatus("loaded");
      quantifyAndStore(img);
    };
    img.onerror = () => { setIhcImage(null); setIhcImageStatus("missing"); };
    img.src = url;
  }, [activeIdx, activeRoi, activeIHCMarker]);

  // 3. Seed available genes + default gene layers whenever a new spatial file loads
  useEffect(() => {
    if (!spatialData || spatialData.length === 0) {
      setAvailableGenes([]);
      setGeneLayers([]);
      return;
    }
    const headers = Object.keys(spatialData[0]).filter(k => k !== "cx" && k !== "cy");
    setAvailableGenes(headers);
    if (activeIdx !== "aggregate") spatialCache.current[activeIdx] = spatialData;
    const defaults = DEFAULT_GENES.filter(g => headers.includes(g));
    const seeded = (defaults.length ? defaults : headers.slice(0, 4)).slice(0, MAX_GENES);
    setGeneLayers(seeded.map((g, i) => ({
      gene: g, color: PALETTE[i % PALETTE.length], shape: SHAPES[i % SHAPES.length], cutoff: 0.8, mode: "points", visible: true
    })));
    setGeneSearch("");
  }, [spatialData, activeIdx]);

  const addGene = (gene) => {
    setGeneLayers(prev => {
      if (prev.find(g => g.gene === gene) || prev.length >= MAX_GENES) return prev;
      return [...prev, { gene, color: PALETTE[prev.length % PALETTE.length], shape: SHAPES[prev.length % SHAPES.length], cutoff: 0.8, mode: "points", visible: true }];
    });
    setGeneSearch("");
  };
  const removeGene = (gene) => setGeneLayers(prev => prev.filter(g => g.gene !== gene));
  const updateGene = (gene, patch) => setGeneLayers(prev => prev.map(g => g.gene === gene ? { ...g, ...patch } : g));

  const filteredAvailableGenes = useMemo(() => {
    if (!geneSearch) return [];
    const q = geneSearch.toLowerCase();
    const selected = new Set(geneLayers.map(g => g.gene));
    return availableGenes.filter(g => g.toLowerCase().includes(q) && !selected.has(g));
  }, [geneSearch, availableGenes, geneLayers]);

  // Lets the agent fetch spatial data for ANY ROI on demand, not just the open one
  const ensureSpatialLoaded = useCallback(async (idx) => {
    if (spatialCache.current[idx]) return spatialCache.current[idx];
    const roi = roiList[idx];
    if (!roi) return null;
    if (Array.isArray(roi.spatialData)) { spatialCache.current[idx] = roi.spatialData; return roi.spatialData; }
    if (roi.spatialSrc) {
      const rowsData = await fetchSpatialCSV(roi.spatialSrc);
      if (rowsData) spatialCache.current[idx] = rowsData;
      return rowsData;
    }
    return null;
  }, [roiList]);

  const handleAsk = async (qOverride) => {
    const q = (qOverride ?? query).trim();
    if (!q) return;
    setAgentBusy(true);
    try {
      const result = await runAgent(q, { roiList, ensureSpatialLoaded, availableGenes });
      setAgentResult(result);
    } catch (err) {
      setAgentResult({ chart: null, explanation: `Something went wrong answering that: ${err.message || err}` });
    } finally {
      setAgentBusy(false);
    }
  };

  // ── Upload a custom ROI ─────────────────────────────────────────────────────
  const handleUploadClick = () => fileInputRef.current?.click();
  const handleFileChange = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      const newRoi = {
        label: file.name.replace(/\.[a-zA-Z0-9]+$/, "") || `Custom ROI ${roiList.length + 1}`,
        src: reader.result,
        maskSrc: null, spatialSrc: null, ihc: null, metrics: null,
        custom: true, analysisStatus: "idle"
      };
      setRoiList(prev => {
        const next = [...prev, newRoi];
        setActiveIdx(next.length - 1);
        return next;
      });
    };
    reader.readAsDataURL(file);
    e.target.value = "";
  };

  const runAnalysis = async (idx) => {
    const roi = roiList[idx];
    if (!roi || !roi.custom) return;
    if (!ANALYSIS_API_URL) {
      setRoiList(prev => prev.map((r, i) => i === idx ? { ...r, analysisStatus: "unconfigured" } : r));
      return;
    }
    setRoiList(prev => prev.map((r, i) => i === idx ? { ...r, analysisStatus: "running" } : r));
    try {
      const res = await fetch(ANALYSIS_API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image_base64: roi.src })
      });
      if (!res.ok) throw new Error(`Server responded ${res.status}`);
      const result = await res.json();
      setRoiList(prev => prev.map((r, i) => i === idx ? {
        ...r,
        maskBase64: result.maskBase64 || null,
        spatialData: result.spatialData || null,
        ihc: result.ihc || null,
        metrics: result.metrics || r.metrics,
        analysisStatus: "done"
      } : r));
    } catch (err) {
      console.error("ROI analysis failed:", err);
      setRoiList(prev => prev.map((r, i) => i === idx ? { ...r, analysisStatus: "error", analysisError: String(err.message || err) } : r));
    }
  };

  const runMitosisDetection = async (idx) => {
    const roi = roiList[idx];
    if (!roi) return;
    if (!MITOSIS_API_URL) {
      setMitosisResults(prev => ({ ...prev, [idx]: { ...(prev[idx] || {}), status: "unconfigured" } }));
      return;
    }
    setMitosisResults(prev => ({ ...prev, [idx]: { ...(prev[idx] || {}), status: "running" } }));
    try {
      // Reuse the already-loaded image element for the active ROI; otherwise
      // load it fresh so this still works for a non-active ROI.
      let imgEl = idx === activeIdx ? images.base : null;
      if (!imgEl) {
        imgEl = await new Promise((resolve, reject) => {
          const im = new Image();
          im.onload = () => resolve(im);
          im.onerror = reject;
          im.src = roi.originalImageBase64 || roi.src;
        });
      }
      const imageBase64 = imageElementToBase64(imgEl);
      const res = await fetch(MITOSIS_API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image_base64: imageBase64, min_confidence: 0.5 })
      });
      if (!res.ok) throw new Error(`Server responded ${res.status}`);
      const result = await res.json();
      const points = result.points || [];
      setMitosisResults(prev => ({ ...prev, [idx]: { points, count: result.count ?? points.length, status: "done" } }));
    } catch (err) {
      console.error("Mitosis detection failed:", err);
      setMitosisResults(prev => ({ ...prev, [idx]: { ...(prev[idx] || {}), status: "error", error: String(err.message || err) } }));
    }
  };

  // 4. Main Draw Loop
  const draw = useCallback(() => {
    if (!images.base || !cvsMain.current || activeIdx === "aggregate") return;
    const ctx = cvsMain.current.getContext("2d");
    const W = cvsMain.current.width, H = cvsMain.current.height;
    const { x, y, scale: sc } = tf.current;

    ctx.clearRect(0, 0, W, H);
    ctx.save(); ctx.translate(x, y); ctx.scale(sc, sc);

    const baseLayer = groups.find(g => g.id === "base").layers[0];
    if (baseLayer.vis && images.base.complete) {
      ctx.globalAlpha = baseLayer.op / 100;
      ctx.drawImage(images.base, 0, 0, images.base.width, images.base.height);
    }

    const maskLayer = groups.find(g => g.id === "seg").layers[0];
    if (maskLayer.vis && images.mask && images.mask.complete) {
      ctx.globalAlpha = maskLayer.op / 100;
      ctx.drawImage(images.mask, 0, 0, images.base.width, images.base.height);
    }

    const ihcLayer = groups.find(g => g.id === "ihc")?.layers[0];
    if (ihcLayer?.vis && ihcImage && ihcImage.complete) {
      // "multiply" blend treats the virtual-IHC image like a real stain laid
      // over the tissue: its white/background pixels let the H&E show through
      // untouched, while stained (darker) pixels tint the tissue underneath —
      // much closer to how a pathologist reads an actual IHC slide than a
      // flat opaque overlay would look.
      ctx.globalAlpha = ihcLayer.op / 100;
      ctx.globalCompositeOperation = "multiply";
      ctx.drawImage(ihcImage, 0, 0, images.base.width, images.base.height);
      ctx.globalCompositeOperation = "source-over";
    }

    const mitosisLayer = groups.find(g => g.id === "mitosis")?.layers[0];
    const mitosisPoints = mitosisResults[activeIdx]?.points;
    if (mitosisLayer?.vis && mitosisPoints && mitosisPoints.length) {
      // Detections are already in this ROI's own pixel space (returned by the
      // detector against the exact image we sent it), so they're drawn 1:1 —
      // no coordinate-convention guessing needed, unlike the spatial CSV.
      ctx.globalAlpha = mitosisLayer.op / 100;
      const r = Math.max(4, images.base.width * 0.008);
      mitosisPoints.forEach(pt => drawMitosisMarker(ctx, pt.x, pt.y, r, mitosisLayer.color));
    }

    const spatialLayer = groups.find(g => g.id === "spatial")?.layers[0];
    if (spatialLayer?.vis && spatialData && bounds) {
      ctx.globalAlpha = spatialLayer.op / 100;
      ctx.globalCompositeOperation = "source-over";

      const heatmapLayers = geneLayers.filter(g => g.visible && g.mode === "heatmap");
      const pointLayers = geneLayers.filter(g => g.visible && g.mode === "points");

      heatmapLayers.forEach(gl => {
        const radius = images.base.width / 14;
        spatialData.forEach(pt => {
          const expr = pt[gl.gene] || 0;
          if (expr < gl.cutoff) return;
          const { px, py } = mapCoordToImage(pt, bounds, images.base.width, images.base.height);
          const grad = ctx.createRadialGradient(px, py, 0, px, py, radius);
          grad.addColorStop(0, hexToRgba(gl.color, Math.min(expr, 1)));
          grad.addColorStop(1, hexToRgba(gl.color, 0));
          ctx.beginPath();
          ctx.arc(px, py, radius, 0, Math.PI * 2);
          ctx.fillStyle = grad;
          ctx.fill();
        });
      });

      pointLayers.forEach(gl => {
        spatialData.forEach(pt => {
          const expr = pt[gl.gene] || 0;
          if (expr < gl.cutoff) return;
          const { px, py } = mapCoordToImage(pt, bounds, images.base.width, images.base.height);
          const size = images.base.width * 0.003 * (0.6 + expr * 1.4);
          drawShape(ctx, gl.shape, px, py, size, gl.color, Math.min(0.4 + expr * 0.6, 1));
        });
      });
    }

    ctx.restore();
  }, [groups, images, activeIdx, spatialData, bounds, geneLayers, ihcImage, mitosisResults]);

  useEffect(() => { draw(); }, [draw]);

  // 5. Fit-to-screen once per newly loaded image (never runs on plain resize,
  // so it doesn't clobber the user's zoom/pan while they're working)
  useEffect(() => {
    if (!images.base || !cont.current) return;
    const { width, height } = cont.current.getBoundingClientRect();
    const scale = Math.min((width - 40) / images.base.width, (height - 40) / images.base.height);
    tf.current = { x: 20, y: 20, scale };
    setScaleDisplay(scale);
    draw();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [images.base]);

  useEffect(() => {
    const ro = new ResizeObserver(e => {
      const { width, height } = e[0].contentRect;
      if (cvsMain.current) { cvsMain.current.width = width; cvsMain.current.height = height; }
      draw();
    });
    if (cont.current) ro.observe(cont.current);
    return () => ro.disconnect();
  }, [draw]);

  // ── Zoom & pan interaction ──────────────────────────────────────────────────
  const zoomBy = useCallback((factor, centerX, centerY) => {
    if (!cvsMain.current) return;
    const rect = cvsMain.current.getBoundingClientRect();
    const cx = centerX ?? rect.width / 2, cy = centerY ?? rect.height / 2;
    const { x, y, scale } = tf.current;
    const newScale = Math.max(0.1, Math.min(scale * factor, 12));
    const imgX = (cx - x) / scale, imgY = (cy - y) / scale;
    const newX = cx - imgX * newScale, newY = cy - imgY * newScale;
    tf.current = { x: newX, y: newY, scale: newScale };
    setScaleDisplay(newScale);
    draw();
  }, [draw]);

  const onWheel = (e) => {
    if (activeIdx === "aggregate") return;
    e.preventDefault();
    const rect = cvsMain.current.getBoundingClientRect();
    const mx = e.clientX - rect.left, my = e.clientY - rect.top;
    const factor = e.deltaY < 0 ? 1.12 : 1 / 1.12;
    zoomBy(factor, mx, my);
  };

  const onMouseDown = (e) => {
    if (activeIdx === "aggregate") return;
    setIsPanning(true);
    panState.current = { startX: e.clientX, startY: e.clientY, origX: tf.current.x, origY: tf.current.y };
  };
  const onMouseMove = (e) => {
    if (!isPanning) return;
    const dx = e.clientX - panState.current.startX;
    const dy = e.clientY - panState.current.startY;
    tf.current = { ...tf.current, x: panState.current.origX + dx, y: panState.current.origY + dy };
    draw();
  };
  const stopPanning = () => setIsPanning(false);

  const resetView = () => {
    if (!images.base || !cont.current) return;
    const { width, height } = cont.current.getBoundingClientRect();
    const scale = Math.min((width - 40) / images.base.width, (height - 40) / images.base.height);
    tf.current = { x: 20, y: 20, scale };
    setScaleDisplay(scale);
    draw();
  };

  const btnStyle = { background:T.bg2, border:`1px solid ${T.bdr}`, color:T.txt, borderRadius:6, width:30, height:30, cursor:"pointer", fontSize:15, display:"flex", alignItems:"center", justifyContent:"center" };
  const needsAnalysis = activeRoi?.custom && activeRoi.analysisStatus !== "done";

  return (
    <div style={{display:"grid",gridTemplateColumns:"240px 1fr 290px",height:"100vh",background:T.bg1,fontFamily:"system-ui",color:T.txt}}>

      {/* LEFT PANEL */}
      <div style={{borderRight:`1px solid ${T.bdr}`,display:"flex",flexDirection:"column",background:T.bg2}}>
        <div style={{padding:"15px",borderBottom:`1px solid ${T.bdr}`}}>
          <button onClick={onBack} style={{background:"transparent",border:"none",color:T.txt2,cursor:"pointer",fontSize:11,marginBottom:15}}>← Close Case</button>
          <div style={{fontSize:14,fontWeight:700}}>Case Analysis</div>
        </div>

        <div style={{padding:"10px", flex:1, overflowY:"auto"}}>
          <div onClick={() => setActiveIdx("aggregate")} style={{padding:"10px", background: activeIdx==="aggregate" ? `${T.teal}22` : T.bg3, border:`1px solid ${activeIdx==="aggregate"?T.teal:T.bdr}`, borderRadius:6, cursor:"pointer", marginBottom:10}}>
            <div style={{fontSize:12, fontWeight:600, color: activeIdx==="aggregate" ? T.teal : T.txt}}>⬡ Aggregate View</div>
          </div>

          <div style={{display:"flex", justifyContent:"space-between", alignItems:"center", marginTop:15, marginBottom:10}}>
            <div style={{fontSize:10,color:T.txt3,textTransform:"uppercase",letterSpacing:1}}>Individual ROIs</div>
            <button onClick={handleUploadClick} title="Upload your own ROI image" style={{background:"transparent", border:`1px solid ${T.bdr}`, color:T.teal, borderRadius:4, fontSize:10, padding:"3px 7px", cursor:"pointer"}}>+ Upload</button>
          </div>
          <input ref={fileInputRef} type="file" accept="image/*" onChange={handleFileChange} style={{display:"none"}} />

          {roiList.map((roi, idx) => (
            <div key={idx} onClick={() => setActiveIdx(idx)} style={{padding:"10px", background: activeIdx===idx ? `${T.blue}22` : T.bg3, border:`1px solid ${activeIdx===idx?T.blue:T.bdr}`, borderRadius:6, cursor:"pointer", marginBottom:8}}>
              <div style={{fontSize:12, fontWeight:600, color: activeIdx===idx ? T.blue : T.txt, display:"flex", alignItems:"center", gap:6}}>
                <span>ROI {idx + 1}: {roi.label}</span>
                {roi.custom && <span style={{fontSize:8, color:T.purple, border:`1px solid ${T.purple}`, borderRadius:3, padding:"1px 4px"}}>custom</span>}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* CENTER CANVAS */}
      <div style={{display:"flex",flexDirection:"column",overflow:"hidden"}}>
        <div ref={cont} style={{flex:1, background:T.bg0, position:"relative", display:"flex", alignItems:"center", justifyContent:"center"}}>
          {activeIdx === "aggregate" ? (
            <div style={{textAlign:"center", color:T.txt3}}>
              <div style={{fontSize:32, marginBottom:10}}>⬡</div>
              <p style={{fontSize:16, fontWeight:600, color:T.txt}}>Select an ROI from the left</p>
            </div>
          ) : (
            <>
              <canvas
                ref={cvsMain}
                onWheel={onWheel}
                onMouseDown={onMouseDown}
                onMouseMove={onMouseMove}
                onMouseUp={stopPanning}
                onMouseLeave={stopPanning}
                style={{display:"block",width:"100%",height:"100%", cursor: isPanning ? "grabbing" : "grab"}}
              />

              {/* Zoom toolbar */}
              <div style={{position:"absolute", top:12, right:12, display:"flex", gap:6}}>
                <button onClick={()=>zoomBy(0.8)} style={btnStyle} title="Zoom out">−</button>
                <div style={{...btnStyle, cursor:"default", width:56}}>{Math.round(scaleDisplay*100)}%</div>
                <button onClick={()=>zoomBy(1.25)} style={btnStyle} title="Zoom in">+</button>
                <button onClick={resetView} style={{...btnStyle, width:34, fontSize:12}} title="Reset view">⤢</button>
              </div>

              {/* IHC badge */}
              {groups.find(g=>g.id==="ihc").layers[0].vis && (
                <div style={{position:"absolute", top:12, left:12, background:"rgba(13,16,24,0.88)", border:`1px solid ${T.bdr}`, borderRadius:8, padding:"6px 10px", fontSize:11, color:T.purple, fontWeight:600}}>
                  IHC: {IHC_MARKER_LABELS[activeIHCMarker]}
                  {displayIHC?.status ? ` • ${displayIHC.status}` : ""}
                  {displayIHC?.method === "measured" && <span style={{color:T.green, fontWeight:400}}> (measured)</span>}
                  {ihcImageStatus === "missing" && <span style={{color:T.amber, fontWeight:400}}> (no image, scores only)</span>}
                </div>
              )}

              {/* Mitosis count badge */}
              {groups.find(g=>g.id==="mitosis").layers[0].vis && activeMitosis?.status === "done" && (
                <div style={{position:"absolute", top: groups.find(g=>g.id==="ihc").layers[0].vis ? 50 : 12, left:12, background:"rgba(13,16,24,0.88)", border:`1px solid ${T.bdr}`, borderRadius:8, padding:"6px 10px", fontSize:11, color:T.red, fontWeight:600}}>
                  Mitoses: {activeMitosis.count}
                  {mitosisDensity != null && <span style={{fontWeight:400}}> • {mitosisDensity.toFixed(1)}/mm² • Score {mitosisScore}</span>}
                  {mitosisDensity == null && <span style={{color:T.amber, fontWeight:400}}> (enter µm/px for density)</span>}
                </div>
              )}

              {/* Segmentation legend */}
              {groups.find(g=>g.id==="seg").layers[0].vis && (
                <div style={{position:"absolute", bottom:12, right:12, background:"rgba(13,16,24,0.9)", border:`1px solid ${T.bdr}`, borderRadius:8, padding:"9px 11px", maxWidth:200, zIndex:4}}>
                  <div style={{fontSize:9, color:T.txt3, marginBottom:7, textTransform:"uppercase", letterSpacing:0.5}}>Segmentation</div>
                  {segLegend.map(c => (
                    <div key={c.id} style={{display:"flex", alignItems:"center", gap:8, marginBottom:5}}>
                      <input
                        type="color"
                        value={c.color}
                        onChange={e => updateSegColor(c.id, e.target.value)}
                        title="Adjust if this doesn't match the mask color"
                        style={{width:16, height:16, padding:0, border:`1px solid ${T.bdr}`, borderRadius:3, background:"transparent", cursor:"pointer", flexShrink:0}}
                      />
                      <span style={{fontSize:11, color:T.txt}}>{c.label}</span>
                    </div>
                  ))}
                </div>
              )}

              {/* Gene legend */}
              {geneLayers.some(g => g.visible) && groups.find(g=>g.id==="spatial").layers[0].vis && (
                <div style={{position:"absolute", bottom:12, left:12, background:"rgba(13,16,24,0.88)", border:`1px solid ${T.bdr}`, borderRadius:8, padding:"8px 10px", maxWidth:230}}>
                  <div style={{fontSize:9, color:T.txt3, marginBottom:6, textTransform:"uppercase", letterSpacing:0.5}}>Gene Legend</div>
                  {geneLayers.filter(g => g.visible).map(g => (
                    <div key={g.gene} style={{display:"flex", alignItems:"center", gap:7, marginBottom:4, fontSize:11}}>
                      <ShapeIcon shape={g.mode==="heatmap" ? "circle" : g.shape} color={g.color} />
                      <span>{g.gene}</span>
                      <span style={{color:T.txt3, marginLeft:"auto", fontSize:9}}>{g.mode==="heatmap"?"heat":"pts"} ≥{g.cutoff.toFixed(2)}</span>
                    </div>
                  ))}
                </div>
              )}
            </>
          )}
        </div>
      </div>

      {/* RIGHT PANEL (Layers & Insights) */}
      <div style={{borderLeft:`1px solid ${T.bdr}`,display:"flex",flexDirection:"column",background:T.bg2, overflowY:"auto"}}>
        <div style={{display:"flex",borderBottom:`1px solid ${T.bdr}`,padding:"6px 8px 0",background:T.bg1}}>
          {[["layers","Visual Layers"],["insights","Insights"]].map(([key,label]) => (
            <button key={key} onClick={()=>setPanelTab(key)} style={{fontSize:11,fontWeight:600,padding:"6px 12px",borderRadius:"4px 4px 0 0",border:"none",background: panelTab===key ? T.bg2 : "transparent",color: panelTab===key ? T.txt : T.txt2,borderBottom: panelTab===key ? `2px solid ${T.teal}` : "2px solid transparent", cursor:"pointer"}}>{label}</button>
          ))}
        </div>

        {panelTab === "layers" && (
          activeIdx === "aggregate" ? (
            <div style={{padding:16, fontSize:11, color:T.txt3}}>Select an ROI to configure its layers.</div>
          ) : (
            <div style={{padding:"10px",display:"flex",flexDirection:"column",gap:10}}>

              {needsAnalysis && (
                <div style={{background:T.bg3, border:`1px solid ${T.bdr}`, borderRadius:6, padding:10}}>
                  <div style={{fontSize:11, fontWeight:600, marginBottom:6}}>Custom ROI</div>
                  <div style={{fontSize:11, color:T.txt2, marginBottom:8}}>
                    Uploaded ROIs need to run through the inference backend before segmentation, IHC, and spatial layers are available.
                  </div>
                  <button
                    onClick={()=>runAnalysis(activeIdx)}
                    disabled={activeRoi.analysisStatus==="running"}
                    style={{width:"100%", background:T.teal, color:"#04241d", border:"none", borderRadius:6, padding:"8px", fontSize:11, fontWeight:700, cursor: activeRoi.analysisStatus==="running" ? "default":"pointer", opacity: activeRoi.analysisStatus==="running"?0.7:1}}
                  >
                    {activeRoi.analysisStatus==="running" ? "Analyzing…" : "Run Analysis"}
                  </button>
                  {activeRoi.analysisStatus === "unconfigured" && (
                    <div style={{fontSize:10, color:T.amber, marginTop:6}}>Set ANALYSIS_API_URL at the top of this file to your Modal endpoint to enable this.</div>
                  )}
                  {activeRoi.analysisStatus === "error" && (
                    <div style={{fontSize:10, color:T.red, marginTop:6}}>Analysis failed: {activeRoi.analysisError}</div>
                  )}
                </div>
              )}

              {groups.map(g=>(
                <div key={g.id} style={{background:T.bg3,borderRadius:6,padding:"8px"}}>
                  <div style={{fontSize:11,fontWeight:600,marginBottom:8}}>{g.label}</div>
                  {g.layers.map(l=>(
                    <div key={l.id} style={{display:"flex",flexDirection:"column",gap:8}}>
                      <div style={{display:"flex",alignItems:"center",gap:8}}>
                        <Eye on={l.vis} toggle={()=>dispatch({type:"TL",lid:l.id})} color={l.color}/>
                        <span style={{fontSize:11}}>{l.label}</span>
                      </div>
                      {!l.locked && (
                        <input type="range" min={0} max={100} value={l.op} onChange={e=>dispatch({type:"OP",lid:l.id,val:+e.target.value})} style={{accentColor:T.teal}}/>
                      )}
                    </div>
                  ))}

                  {/* Virtual IHC marker picker + score card */}
                  {g.id === "ihc" && (
                    <div style={{marginTop:10, paddingTop:10, borderTop:`1px solid ${T.bdr}`}}>
                      <div style={{fontSize:10, color:T.txt3, marginBottom:6}}>Marker</div>
                      <div style={{display:"flex", gap:5, marginBottom:8, flexWrap:"wrap"}}>
                        {IHC_MARKERS.map(m => (
                          <button key={m} onClick={()=>setActiveIHCMarker(m)}
                            style={{fontSize:10, padding:"5px 9px", borderRadius:5, cursor:"pointer",
                              border:`1px solid ${activeIHCMarker===m ? T.purple : T.bdr}`,
                              background: activeIHCMarker===m ? hexToRgba(T.purple,0.18) : "transparent",
                              color: activeIHCMarker===m ? T.purple : T.txt2}}>
                            {IHC_MARKER_LABELS[m]}
                          </button>
                        ))}
                      </div>

                      {displayIHC ? (
                        <div style={{background:T.bg1, border:`1px solid ${T.bdr}`, borderRadius:6, padding:8, fontSize:11}}>
                          <div style={{display:"flex", justifyContent:"space-between", marginBottom:6}}>
                            <span style={{fontSize:9, color: displayIHC.method === "measured" ? T.green : T.amber, textTransform:"uppercase", letterSpacing:0.4}}>
                              {displayIHC.method === "measured" ? "Measured from image" : "Demo placeholder"}
                            </span>
                          </div>
                          {Object.entries(displayIHC).filter(([k])=>!["src","base64","method","caveat"].includes(k) && displayIHC[k] !== undefined).map(([k,v])=>(
                            <div key={k} style={{display:"flex", justifyContent:"space-between", marginBottom:3}}>
                              <span style={{color:T.txt3, textTransform:"capitalize"}}>{k.replace(/_/g," ")}</span>
                              <span>{typeof v === "number" ? v.toFixed(1) : String(v)}</span>
                            </div>
                          ))}
                          {displayIHC.method === "measured" && displayIHC.caveat && (
                            <div style={{fontSize:9, color:T.txt3, marginTop:6, lineHeight:1.4}}>{displayIHC.caveat}</div>
                          )}
                          {transcriptMean != null && (
                            <div style={{display:"flex", justifyContent:"space-between", marginTop:7, paddingTop:6, borderTop:`1px solid ${T.bdr2}`}}>
                              <span style={{color:T.txt3}}>Transcript ({transcriptGene}, mean)</span>
                              <span style={{color:T.amber}}>{transcriptMean.toFixed(2)}</span>
                            </div>
                          )}
                        </div>
                      ) : (
                        <div style={{fontSize:10, color:T.txt3}}>No {IHC_MARKER_LABELS[activeIHCMarker]} data for this ROI yet.</div>
                      )}
                    </div>
                  )}

                  {/* Mitotic count controls */}
                  {g.id === "mitosis" && (
                    <div style={{marginTop:10, paddingTop:10, borderTop:`1px solid ${T.bdr}`}}>
                      <button
                        onClick={()=>runMitosisDetection(activeIdx)}
                        disabled={activeMitosis?.status === "running"}
                        style={{width:"100%", background:T.red, color:"#2b0509", border:"none", borderRadius:6, padding:"8px", fontSize:11, fontWeight:700, cursor: activeMitosis?.status==="running" ? "default":"pointer", opacity: activeMitosis?.status==="running"?0.7:1, marginBottom:8}}
                      >
                        {activeMitosis?.status === "running" ? "Detecting…" : "Detect Mitoses"}
                      </button>

                      {activeMitosis?.status === "unconfigured" && (
                        <div style={{fontSize:10, color:T.amber, marginBottom:8}}>Set MITOSIS_API_URL at the top of this file to your KongNet endpoint to enable this.</div>
                      )}
                      {activeMitosis?.status === "error" && (
                        <div style={{fontSize:10, color:T.red, marginBottom:8}}>Detection failed: {activeMitosis.error}</div>
                      )}

                      {activeMitosis?.status === "done" && (
                        <div style={{background:T.bg1, border:`1px solid ${T.bdr}`, borderRadius:6, padding:8, fontSize:11, marginBottom:8}}>
                          <div style={{display:"flex", justifyContent:"space-between", marginBottom:3}}>
                            <span style={{color:T.txt3}}>Mitotic figures detected</span>
                            <span>{activeMitosis.count}</span>
                          </div>
                          {mitosisDensity != null ? (
                            <>
                              <div style={{display:"flex", justifyContent:"space-between", marginBottom:3}}>
                                <span style={{color:T.txt3}}>Density</span>
                                <span>{mitosisDensity.toFixed(2)}/mm²</span>
                              </div>
                              <div style={{display:"flex", justifyContent:"space-between"}}>
                                <span style={{color:T.txt3}}>Mitotic count score</span>
                                <span style={{color:T.red, fontWeight:700}}>{mitosisScore} / 3</span>
                              </div>
                            </>
                          ) : (
                            <div style={{fontSize:10, color:T.amber}}>Enter this ROI's µm/pixel below to convert the count to density and a score.</div>
                          )}
                        </div>
                      )}

                      <label style={{fontSize:9, color:T.txt3, display:"block", marginBottom:8}}>
                        µm / pixel for this ROI
                        <input
                          type="number" step="0.01" min="0"
                          placeholder="e.g. 0.25 (40x)"
                          value={roiMpp[activeIdx] ?? ""}
                          onChange={e => setRoiMpp(prev => ({ ...prev, [activeIdx]: e.target.value === "" ? undefined : +e.target.value }))}
                          style={{display:"block", width:"100%", boxSizing:"border-box", marginTop:3, background:T.bg1, border:`1px solid ${T.bdr}`, borderRadius:4, padding:"5px 7px", fontSize:11, color:T.txt, outline:"none"}}
                        />
                      </label>

                      <div style={{fontSize:9, color:T.txt3, marginBottom:4}}>Score cutoffs (mitoses/mm²) — edit to match your lab's field-diameter convention</div>
                      <div style={{display:"grid", gridTemplateColumns:"1fr 1fr", gap:8, marginBottom:8}}>
                        <label style={{fontSize:9, color:T.txt3}}>
                          Score 1→2 at
                          <input type="number" step="0.1" value={scoreBreaks.low}
                            onChange={e=>setScoreBreaks(prev=>({...prev, low:+e.target.value}))}
                            style={{display:"block", width:"100%", boxSizing:"border-box", marginTop:2, background:T.bg1, border:`1px solid ${T.bdr}`, borderRadius:4, padding:"4px 6px", fontSize:10, color:T.txt}} />
                        </label>
                        <label style={{fontSize:9, color:T.txt3}}>
                          Score 2→3 at
                          <input type="number" step="0.1" value={scoreBreaks.high}
                            onChange={e=>setScoreBreaks(prev=>({...prev, high:+e.target.value}))}
                            style={{display:"block", width:"100%", boxSizing:"border-box", marginTop:2, background:T.bg1, border:`1px solid ${T.bdr}`, borderRadius:4, padding:"4px 6px", fontSize:10, color:T.txt}} />
                        </label>
                      </div>

                      <div style={{fontSize:9, color:T.txt3, lineHeight:1.4}}>
                        Mitotic count is one of three Nottingham grade components — tubule formation and nuclear pleomorphism still need separate scoring. Default cutoffs approximate the classic per-10-HPF thresholds at a 0.55mm field diameter; field size varies by microscope, so verify against your lab's convention.
                      </div>
                    </div>
                  )}


                  {g.id === "spatial" && spatialData && (
                    <div style={{marginTop: 10, paddingTop: 10, borderTop: `1px solid ${T.bdr}`}}>
                      <div style={{fontSize:10, color:T.txt3, marginBottom:6, display:"flex", justifyContent:"space-between"}}>
                        <span>Genes shown</span><span>{geneLayers.length}/{MAX_GENES}</span>
                      </div>

                      <input
                        type="text"
                        placeholder={geneLayers.length >= MAX_GENES ? `Max ${MAX_GENES} genes reached` : "Search gene to add…"}
                        value={geneSearch}
                        disabled={geneLayers.length >= MAX_GENES}
                        onChange={e => setGeneSearch(e.target.value)}
                        style={{width:"100%", boxSizing:"border-box", background:T.bg1, border:`1px solid ${T.bdr}`, borderRadius:4, padding:"6px 7px", fontSize:11, color:T.txt, outline:"none", marginBottom:8}}
                      />

                      {geneSearch && (
                        <div style={{maxHeight:130, overflowY:"auto", background:T.bg1, border:`1px solid ${T.bdr}`, borderRadius:4, marginBottom:8}}>
                          {filteredAvailableGenes.length === 0 && (
                            <div style={{padding:"6px 8px", fontSize:11, color:T.txt3}}>No matches</div>
                          )}
                          {filteredAvailableGenes.slice(0, 40).map(gn => (
                            <div
                              key={gn}
                              onClick={() => addGene(gn)}
                              style={{padding:"5px 8px", fontSize:11, cursor:"pointer", borderBottom:`1px solid ${T.bdr2}`}}
                              onMouseEnter={e=>e.currentTarget.style.background=T.bg3}
                              onMouseLeave={e=>e.currentTarget.style.background="transparent"}
                            >
                              {gn}
                            </div>
                          ))}
                        </div>
                      )}

                      <div style={{display:"flex", flexDirection:"column", gap:8}}>
                        {geneLayers.map(gl => (
                          <div key={gl.gene} style={{background:T.bg1, border:`1px solid ${T.bdr}`, borderRadius:6, padding:8}}>
                            <div style={{display:"flex", alignItems:"center", justifyContent:"space-between", marginBottom:7}}>
                              <div style={{display:"flex", alignItems:"center", gap:7}}>
                                <Eye on={gl.visible} toggle={()=>updateGene(gl.gene, {visible: !gl.visible})} color={gl.color} />
                                <ShapeIcon shape={gl.shape} color={gl.color} />
                                <span style={{fontSize:11, fontWeight:600}}>{gl.gene}</span>
                              </div>
                              <button onClick={()=>removeGene(gl.gene)} title="Remove gene" style={{background:"transparent", border:"none", color:T.txt3, cursor:"pointer", fontSize:14, lineHeight:1}}>×</button>
                            </div>

                            <div style={{display:"grid", gridTemplateColumns:"1fr 1fr", gap:8, marginBottom:7}}>
                              <label style={{fontSize:9, color:T.txt3}}>
                                Color
                                <input
                                  type="color"
                                  value={gl.color}
                                  onChange={e=>updateGene(gl.gene, {color: e.target.value})}
                                  style={{display:"block", width:"100%", height:22, marginTop:2, border:`1px solid ${T.bdr}`, borderRadius:4, background:"transparent", cursor:"pointer"}}
                                />
                              </label>
                              <label style={{fontSize:9, color:T.txt3}}>
                                Shape
                                <select
                                  value={gl.shape}
                                  onChange={e=>updateGene(gl.gene, {shape: e.target.value})}
                                  style={{display:"block", width:"100%", marginTop:2, background:T.bg2, color:T.txt, border:`1px solid ${T.bdr}`, borderRadius:4, fontSize:10, padding:"4px"}}
                                >
                                  {SHAPES.map(s => <option key={s} value={s}>{s}</option>)}
                                </select>
                              </label>
                            </div>

                            <div style={{marginBottom:7}}>
                              <label style={{fontSize:9, color:T.txt3, display:"flex", justifyContent:"space-between", marginBottom:3}}>
                                <span>Cutoff (min expression)</span><span>{gl.cutoff.toFixed(2)}</span>
                              </label>
                              <input
                                type="range" min={0} max={1} step={0.01}
                                value={gl.cutoff}
                                onChange={e=>updateGene(gl.gene, {cutoff: +e.target.value})}
                                style={{width:"100%", accentColor: gl.color}}
                              />
                            </div>

                            <div style={{display:"flex", gap:6}}>
                              {["points","heatmap"].map(m => (
                                <button
                                  key={m}
                                  onClick={()=>updateGene(gl.gene, {mode: m})}
                                  style={{
                                    flex:1, fontSize:10, padding:"5px", borderRadius:4, cursor:"pointer",
                                    border:`1px solid ${gl.mode===m ? gl.color : T.bdr}`,
                                    background: gl.mode===m ? hexToRgba(gl.color, 0.15) : "transparent",
                                    color: gl.mode===m ? gl.color : T.txt2
                                  }}
                                >
                                  {m === "points" ? "Points" : "Heatmap"}
                                </button>
                              ))}
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </div>
          )
        )}

        {panelTab === "insights" && (
          <div style={{padding:10, display:"flex", flexDirection:"column", gap:10}}>
            <div style={{fontSize:10, color:T.txt3, lineHeight:1.4}}>
              Ask about segmentation ratios, spatial genes, or IHC markers. This is a small rule-based agent — it parses your question, fetches any spatial data it needs, then charts the result. (Gene names need an ROI with spatial data opened at least once.)
            </div>
            <div style={{display:"flex", gap:6}}>
              <input
                value={query}
                onChange={e=>setQuery(e.target.value)}
                onKeyDown={e=>{ if(e.key==="Enter") handleAsk(); }}
                placeholder="e.g. compare MKI67 across ROIs"
                style={{flex:1, background:T.bg1, border:`1px solid ${T.bdr}`, borderRadius:4, padding:"7px 8px", fontSize:11, color:T.txt, outline:"none"}}
              />
              <button onClick={()=>handleAsk()} disabled={agentBusy || !query.trim()} style={{background:T.teal, color:"#04241d", border:"none", borderRadius:4, padding:"0 12px", fontSize:11, fontWeight:700, cursor:"pointer", opacity: agentBusy?0.6:1}}>
                {agentBusy ? "…" : "Ask"}
              </button>
            </div>
            <div style={{display:"flex", flexWrap:"wrap", gap:5}}>
              {EXAMPLE_PROMPTS.map(p => (
                <button key={p} onClick={()=>{ setQuery(p); handleAsk(p); }} style={{fontSize:9, background:T.bg1, border:`1px solid ${T.bdr}`, color:T.txt2, borderRadius:10, padding:"4px 8px", cursor:"pointer"}}>{p}</button>
              ))}
            </div>

            {agentResult && (
              <div style={{background:T.bg1, border:`1px solid ${T.bdr}`, borderRadius:6, padding:10}}>
                {agentResult.chart && (
                  <>
                    <div style={{fontSize:10, fontWeight:600, marginBottom:6, color:T.txt2}}>{agentResult.chart.title || "Result"}</div>
                    <MiniChart {...agentResult.chart} />
                  </>
                )}
                <div style={{fontSize:11, color:T.txt, marginTop: agentResult.chart ? 8 : 0, lineHeight:1.4}}>{agentResult.explanation}</div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export default function App() {
  const [view, setView] = useState("gallery"); // "gallery" (home) | "viewer"
  const [caseRois, setCaseRois] = useState([]);
  const [customPatients, setCustomPatients] = useState([]); // patients created via the upload tile

  const handleUploadPatient = (rois) => {
    const patient = { id: `custom-${Date.now()}`, code: `Custom Case ${customPatients.length + 1}`, subtype: "Uploaded", risk: "Unknown", age: null, rois };
    setCustomPatients(prev => [...prev, patient]);
    setCaseRois(rois);
    setView("viewer");
  };

  if (view === "viewer" && caseRois.length > 0) return <AnalysisViewer rois={caseRois} onBack={() => setView("gallery")}/>;
  return (
    <Gallery
      patients={[...DEMO_PATIENTS, ...customPatients]}
      onSelectPatient={(rois) => { setCaseRois(rois); setView("viewer"); }}
      onUploadPatient={handleUploadPatient}
    />
  );
}
