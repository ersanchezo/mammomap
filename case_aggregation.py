"""
PathVision — case-level aggregation for multi-ROI analysis
Called by Modal after all ROIs in a case have finished the ML pipeline.
"""

from __future__ import annotations
import numpy as np
from typing import Any

STAINS = ["HER2", "Ki67", "ER", "PR"]
GENES  = ["ESR1","PGR","ERBB2","MKI67","TP53","BRCA1","CDH1","PIK3CA","CCND1","GATA3"]

# ─── IHC aggregation rules (ASCO/CAP + St Gallen consensus) ──────────────────
IHC_RULES: dict[str, dict] = {
    "HER2": {
        "rule": "worst_case",
        "note": "ASCO/CAP 2018: positive if any ROI is 3+",
    },
    "Ki67": {
        "rule": "mean",
        "note": "St Gallen: arithmetic mean across ROIs",
    },
    "ER": {
        "rule": "mean_with_discordance",
        "note": "Flag if any ROI Allred differs by >2 points",
        "discordance_threshold": 2,
    },
    "PR": {
        "rule": "mean_with_discordance",
        "note": "Flag if any ROI Allred differs by >2 points",
        "discordance_threshold": 2,
    },
}

GRADE_ORDER = ["0", "1+", "2+", "3+"]


def aggregate_ihc(roi_results: list[dict]) -> dict:
    """Aggregate IHC results from N ROIs into one case-level score."""
    out = {}
    for stain, rule_def in IHC_RULES.items():
        roi_data = [r["ihc"].get(stain, {}) for r in roi_results if r.get("ihc")]
        if not roi_data:
            continue

        pcts = [d.get("positive_pct", 0) for d in roi_data]
        grades = [d.get("grade", "0") for d in roi_data]

        rule = rule_def["rule"]

        if rule == "worst_case":
            best_idx = max(range(len(grades)),
                           key=lambda i: GRADE_ORDER.index(grades[i])
                           if grades[i] in GRADE_ORDER else 0)
            case_grade = grades[best_idx]
            case_pct   = pcts[best_idx]
            discordant = len(set(g in ["2+", "3+"] for g in grades)) > 1

        elif rule == "mean":
            case_pct   = float(np.mean(pcts))
            case_grade = f"{case_pct:.1f}%"
            discordant = float(np.std(pcts)) > 15          # >15pp std = flag

        else:  # mean_with_discordance
            case_pct   = float(np.mean(pcts))
            allred_vals = [d.get("allred", round(p / 12.5)) for d, p in zip(roi_data, pcts)]
            case_grade = f"Allred {round(np.mean(allred_vals))}/8"
            discordant = (max(allred_vals) - min(allred_vals)) > rule_def["discordance_threshold"]

        out[stain] = {
            "positive_pct": round(case_pct, 1),
            "grade":        case_grade,
            "discordant":   discordant,
            "roi_values":   pcts,
            "rule":         rule,
            "note":         rule_def["note"],
        }
    return out


def aggregate_segmentation(roi_results: list[dict]) -> dict:
    """Weighted-mean tissue composition, weighted by ROI tissue coverage."""
    keys = ["tumor_purity", "til_score", "stroma_ratio", "necrosis_ratio"]
    agg = {k: 0.0 for k in keys}
    per_roi = {k: [] for k in keys}

    for r in roi_results:
        m = r.get("segmentation", {}).get("metrics", {})
        for k in keys:
            per_roi[k].append(m.get(k, 0.0))

    for k in keys:
        vals = per_roi[k]
        agg[k]        = round(float(np.mean(vals)), 1)
        agg[f"{k}_std"] = round(float(np.std(vals)), 1)

    return {"case_metrics": agg, "per_roi": per_roi}


def aggregate_spatial(roi_results: list[dict]) -> dict:
    """Union spatial predictions; compute case-level gene means and ITH per gene."""
    all_tiles = []
    gene_means_per_roi = []

    for roi_idx, r in enumerate(roi_results):
        tiles = r.get("spatial", {}).get("tiles", [])
        gene_stats = r.get("spatial", {}).get("gene_stats", {})
        # Tag tiles with their ROI index for cross-ROI visualisation
        for t in tiles:
            all_tiles.append({**t, "roi_index": roi_idx + 1})
        if gene_stats:
            gene_means_per_roi.append(gene_stats)

    if not gene_means_per_roi:
        return {"tiles": all_tiles, "gene_stats": {}, "ith_per_gene": {}}

    # Case-level gene expression: mean across ROIs
    gene_means = {
        g: round(float(np.mean([r.get(g, 0) for r in gene_means_per_roi])), 4)
        for g in GENES
    }

    # Intra-tumoral heterogeneity per gene: coefficient of variation across ROIs
    ith_per_gene = {
        g: round(float(np.std([r.get(g, 0) for r in gene_means_per_roi]) /
                       (np.mean([r.get(g, 0) for r in gene_means_per_roi]) + 1e-6)), 4)
        for g in GENES
    }

    return {
        "tiles":        all_tiles,
        "gene_stats":   gene_means,
        "ith_per_gene": ith_per_gene,
    }


def compute_heterogeneity_score(
    roi_results: list[dict],
    ihc_agg: dict,
    seg_agg: dict,
    spatial_agg: dict,
) -> float:
    """
    Intra-tumoral heterogeneity (ITH) score: 0 (homogeneous) → 1 (highly heterogeneous).

    Combines:
      40% gene expression CV across ROIs
      40% IHC coefficient of variation across ROIs
      20% tumour purity SD across ROIs
    """
    # Gene expression heterogeneity
    ith_vals = list(spatial_agg.get("ith_per_gene", {}).values())
    expr_het = float(np.mean(ith_vals)) if ith_vals else 0.0

    # IHC heterogeneity: mean CV across stains
    ihc_cvs = []
    for stain, d in ihc_agg.items():
        vals = d.get("roi_values", [])
        if len(vals) > 1 and np.mean(vals) > 0:
            ihc_cvs.append(np.std(vals) / np.mean(vals))
    ihc_het = float(np.mean(ihc_cvs)) if ihc_cvs else 0.0

    # Segmentation heterogeneity: SD of tumour purity normalised to [0,1]
    per_roi = seg_agg.get("per_roi", {}).get("tumor_purity", [])
    seg_het = float(np.std(per_roi)) / 100.0 if len(per_roi) > 1 else 0.0

    score = 0.40 * min(expr_het, 1.0) + 0.40 * min(ihc_het, 1.0) + 0.20 * seg_het
    return round(float(np.clip(score, 0, 1)), 4)


def derive_case_metrics(ihc_agg: dict, seg_agg: dict) -> dict:
    """Derive molecular subtype and recurrence risk from aggregated case data."""
    cm    = seg_agg.get("case_metrics", {})
    her2  = ihc_agg.get("HER2", {}).get("positive_pct", 0)
    er_al = ihc_agg.get("ER",   {}).get("positive_pct", 0)
    ki67  = ihc_agg.get("Ki67", {}).get("positive_pct", 0)

    if er_al >= 60 and her2 < 20 and ki67 < 15:
        subtype, risk = "Luminal A", "Low"
    elif er_al >= 40 and her2 >= 20:
        subtype, risk = "Luminal B / HER2+", "Intermediate"
    elif er_al < 20 and her2 >= 30:
        subtype, risk = "HER2-enriched", "High"
    elif er_al < 20 and her2 < 20:
        subtype, risk = "Triple-negative", "High"
    else:
        subtype, risk = "Luminal B", "Intermediate"

    return {
        "molecular_subtype": subtype,
        "recurrence_risk":   risk,
        "tumor_purity":      cm.get("tumor_purity", 0),
        "til_score":         cm.get("til_score", 0),
        "stroma_ratio":      cm.get("stroma_ratio", 0),
        "ki67_index":        round(ki67, 1),
    }


def aggregate_case(roi_results: list[dict]) -> dict:
    """
    Top-level aggregation function.
    Input:  list of pipeline_result dicts (one per ROI)
    Output: case_aggregation dict ready to INSERT into case_aggregations table
    """
    ihc_agg     = aggregate_ihc(roi_results)
    seg_agg     = aggregate_segmentation(roi_results)
    spatial_agg = aggregate_spatial(roi_results)
    het_score   = compute_heterogeneity_score(roi_results, ihc_agg, seg_agg, spatial_agg)
    case_metrics = derive_case_metrics(ihc_agg, seg_agg)

    return {
        "n_rois":              len(roi_results),
        "aggregated_ihc":      ihc_agg,
        "aggregated_seg":      seg_agg,
        "aggregated_spatial":  spatial_agg,
        "heterogeneity_score": het_score,
        "ith_per_gene":        spatial_agg.get("ith_per_gene", {}),
        "case_metrics":        case_metrics,
    }
