"""
═══════════════════════════════════════════════════════════════════════════════
  Points2Regions + SEQUOIA Integration for PathVision

  This module:
  1. Takes SEQUOIA predicted gene expression spots
  2. Runs Points2Regions to cluster them into tissue regions
  3. Uses an agent to auto-annotate clusters with cell types
  4. Adjusts region sizes based on cluster composition
═══════════════════════════════════════════════════════════════════════════════
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import json

# ═══════════════════════════════════════════════════════════════════════════════
#  1. SEQUOIA → Points2Regions Data Adapter
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SpatialSpot:
    """Single SEQUOIA-predicted spot with gene expression."""
    x: float          # Microns or pixels
    y: float
    gene: str         # Gene name (e.g., "ESR1", "MKI67")
    expr: float       # Predicted expression level (0-1)

@dataclass
class Region:
    """Points2Regions output region with cell type annotation."""
    region_id: int
    label: str                    # e.g., "Tumor epithelium", "Stroma"
    cell_type: str                # e.g., "Luminal A epithelial"
    confidence: float             # 0-1
    color: str                    # Hex color for visualization
    spots: List[SpatialSpot]      # Spots belonging to this region
    boundary: List[Tuple[float, float]]  # Polygon boundary
    area_um2: float
    centroid: Tuple[float, float]
    # Cluster composition stats
    gene_composition: Dict[str, float]   # Gene → avg expression
    marker_score: Dict[str, float]       # Cell type marker enrichment


class SequoiaToPoints2Regions:
    """
    Converts SEQUOIA output (gene expression spots) into the format
    Points2Regions expects: a DataFrame with X, Y, Genes columns.

    SEQUOIA predicts gene expression at spatial locations.
    Points2Regions clusters spatial points by their categorical composition.
    The trick: we treat HIGH-EXPRESSION genes at each spot as "markers".
    """

    def __init__(self, expr_threshold: float = 0.5, top_n_genes: int = 5):
        self.expr_threshold = expr_threshold
        self.top_n_genes = top_n_genes

    def convert(self, sequoia_spots: List[SpatialSpot]) -> pd.DataFrame:
        """
        Convert SEQUOIA spots to Points2Regions format.

        For each spot, we keep only genes with expression > threshold.
        These become the "categorical markers" for Points2Regions.

        Returns DataFrame with columns: ['X', 'Y', 'Genes']
        where each row is one (gene, spot) pair — same as ISS/MERFISH data.
        """
        rows = []

        for spot in sequoia_spots:
            # Only include genes with meaningful expression
            if spot.expr >= self.expr_threshold:
                rows.append({
                    'X': spot.x,
                    'Y': spot.y,
                    'Genes': spot.gene,
                    'Expression': spot.expr  # Optional: for weighted CV
                })

        df = pd.DataFrame(rows)
        print(f"Converted {len(sequoia_spots)} spots → {len(df)} marker points")
        return df

    def convert_weighted(self, sequoia_spots: List[SpatialSpot]) -> pd.DataFrame:
        """
        Alternative: treat expression level as "intensity" of the marker.
        Higher expression = more copies of that mRNA at that location.
        We duplicate rows proportional to expression (quantized).
        """
        rows = []

        for spot in sequoia_spots:
            # Quantize expression into 1-10 copies
            n_copies = max(1, int(spot.expr * 10))
            for _ in range(n_copies):
                rows.append({
                    'X': spot.x,
                    'Y': spot.y,
                    'Genes': spot.gene
                })

        df = pd.DataFrame(rows)
        print(f"Weighted conversion: {len(sequoia_spots)} spots → {len(df)} points")
        return df


# ═══════════════════════════════════════════════════════════════════════════════
#  2. Points2Regions Clustering Wrapper
# ═══════════════════════════════════════════════════════════════════════════════

class Points2RegionsWrapper:
    """
    Wrapper around the Points2Regions Python package.
    Handles hyperparameter tuning for different biological scales.
    """

    def __init__(self, pixel_width: float = 1.0):
        """
        Args:
            pixel_width: Microns per pixel. SEQUOIA spots are typically
                        on a grid where each spot = ~55µm (Visium) or ~1-10µm (ISS).
                        For SEQUOIA on H&E patches, use the patch resolution.
        """
        self.pixel_width = pixel_width
        self.p2r = None

    def fit(self, df: pd.DataFrame, 
            num_clusters: int = 10,
            pixel_smoothing: float = 5.0,
            min_markers_per_bin: int = 5) -> 'Points2RegionsWrapper':
        """
        Run Points2Regions clustering.

        Args:
            num_clusters: Number of regions to find. Start with 10, adjust based on tissue.
            pixel_smoothing: Spatial smoothing distance in pixels. 
                           Larger = bigger regions (tissue niches).
                           Smaller = finer regions (cell types).
                           Rule of thumb: 2-5× expected cell diameter.
            min_markers_per_bin: Density threshold. Bins with fewer markers are discarded.
        """
        try:
            from points2regions import Points2Regions
        except ImportError:
            raise ImportError("pip install points2regions")

        self.p2r = Points2Regions(
            df[['X', 'Y']].values,
            df['Genes'].values,
            pixel_width=self.pixel_width,
            pixel_smoothing=pixel_smoothing
        )

        self.p2r.fit(
            num_clusters=num_clusters,
            min_num_markers_per_bin=min_markers_per_bin
        )

        return self

    def get_regions(self, output_format: str = 'connected') -> Tuple[np.ndarray, dict]:
        """
        Get clustered regions.

        Returns:
            label_mask: 2D array where each pixel = region ID
            transform: Affine transform to map pixels back to microns
            (or connected components info)
        """
        if output_format == 'connected':
            return self.p2r.predict(output='connected')
        elif output_format == 'pixel':
            return self.p2r.predict(output='pixel')
        else:
            return self.p2r.predict(output='marker')

    def get_cluster_per_marker(self) -> np.ndarray:
        """Get cluster assignment for each input marker point."""
        return self.p2r.predict(output='marker')


# ═══════════════════════════════════════════════════════════════════════════════
#  3. Cell Type Annotation Agent
# ═══════════════════════════════════════════════════════════════════════════════

# Breast cancer marker gene database for cell type annotation
BREAST_CANCER_MARKERS = {
    # Epithelial / Tumor
    "Luminal epithelial": {
        "markers": ["ESR1", "PGR", "GATA3", "FOXA1", "KRT8", "KRT18"],
        "confidence_threshold": 0.6
    },
    "Basal epithelial": {
        "markers": ["KRT5", "KRT14", "TP63", "EGFR"],
        "confidence_threshold": 0.5
    },
    "HER2+ epithelial": {
        "markers": ["ERBB2", "GRB7", "STARD3"],
        "confidence_threshold": 0.5
    },
    "Proliferating epithelial": {
        "markers": ["MKI67", "TOP2A", "AURKA", "CCNB1"],
        "confidence_threshold": 0.5
    },

    # Stromal
    "Fibroblast": {
        "markers": ["COL1A1", "COL3A1", "VIM", "PDGFRA", "FAP"],
        "confidence_threshold": 0.5
    },
    "Endothelial": {
        "markers": ["PECAM1", "CDH5", "VWF", "ENG"],
        "confidence_threshold": 0.4
    },
    "Pericyte": {
        "markers": ["ACTA2", "RGS5", "PDGFRB"],
        "confidence_threshold": 0.4
    },

    # Immune
    "T cell": {
        "markers": ["CD3D", "CD3E", "CD8A", "CD4", "IL7R"],
        "confidence_threshold": 0.5
    },
    "B cell": {
        "markers": ["CD19", "CD79A", "MS4A1"],
        "confidence_threshold": 0.4
    },
    "Plasma cell": {
        "markers": ["MZB1", "JCHAIN", "IGHG1"],
        "confidence_threshold": 0.4
    },
    "Macrophage": {
        "markers": ["CD68", "CD14", "CSF1R", "LYZ"],
        "confidence_threshold": 0.5
    },
    "NK cell": {
        "markers": ["NKG7", "GNLY", "KLRD1"],
        "confidence_threshold": 0.4
    },

    # Other
    "Adipocyte": {
        "markers": ["ADIPOQ", "LEP", "PPARG"],
        "confidence_threshold": 0.3
    },
    "Nerve": {
        "markers": ["S100B", "NGFR"],
        "confidence_threshold": 0.3
    }
}


class CellTypeAnnotationAgent:
    """
    Agent that automatically assigns cell types to Points2Regions clusters
    based on marker gene enrichment.

    This replaces manual expert annotation with a rule-based + scoring system.
    """

    def __init__(self, marker_db: Dict = BREAST_CANCER_MARKERS):
        self.marker_db = marker_db

    def annotate(self, 
                 spots: List[SpatialSpot],
                 cluster_labels: np.ndarray) -> List[Region]:
        """
        Annotate each Points2Regions cluster with a cell type.

        Algorithm:
        1. For each cluster, compute average expression of all genes
        2. For each cell type in marker DB, compute enrichment score
        3. Assign the cell type with highest score above threshold
        4. If no cell type passes threshold, label as "Unknown/Mixed"

        Returns:
            List of Region objects with cell_type and confidence
        """
        df = pd.DataFrame([
            {'x': s.x, 'y': s.y, 'gene': s.gene, 'expr': s.expr, 'cluster': c}
            for s, c in zip(spots, cluster_labels)
        ])

        regions = []

        for cluster_id in sorted(df['cluster'].unique()):
            cluster_df = df[df['cluster'] == cluster_id]

            # Compute gene composition (average expression per gene)
            gene_comp = cluster_df.groupby('gene')['expr'].mean().to_dict()

            # Score each cell type
            scores = {}
            for cell_type, info in self.marker_db.items():
                markers = info['markers']
                threshold = info['confidence_threshold']

                # Compute mean expression of markers in this cluster
                marker_exprs = [gene_comp.get(m, 0.0) for m in markers]
                score = np.mean(marker_exprs)

                # Penalize if non-markers are also high (specificity check)
                non_markers = set(gene_comp.keys()) - set(markers)
                if non_markers:
                    non_marker_expr = np.mean([gene_comp.get(m, 0.0) for m in non_markers])
                    specificity = score / (score + non_marker_expr + 0.01)
                else:
                    specificity = 1.0

                scores[cell_type] = {
                    'score': score,
                    'specificity': specificity,
                    'final': score * specificity,
                    'threshold': threshold
                }

            # Pick best cell type
            best_cell_type = max(scores.keys(), key=lambda k: scores[k]['final'])
            best_score = scores[best_cell_type]

            # Check if it passes threshold
            if best_score['final'] >= best_score['threshold']:
                assigned_type = best_cell_type
                confidence = min(1.0, best_score['final'])
            else:
                assigned_type = "Unknown / Mixed"
                confidence = best_score['final']

            # Compute region geometry
            spots_in_region = [
                s for s, c in zip(spots, cluster_labels) if c == cluster_id
            ]
            xs = [s.x for s in spots_in_region]
            ys = [s.y for s in spots_in_region]

            region = Region(
                region_id=int(cluster_id),
                label=f"Region {cluster_id}",
                cell_type=assigned_type,
                confidence=confidence,
                color=self._get_color(assigned_type),
                spots=spots_in_region,
                boundary=self._compute_boundary(xs, ys),
                area_um2=self._estimate_area(xs, ys),
                centroid=(np.mean(xs), np.mean(ys)),
                gene_composition=gene_comp,
                marker_score={k: v['final'] for k, v in scores.items()}
            )

            regions.append(region)

        return regions

    def _get_color(self, cell_type: str) -> str:
        """Assign consistent colors to cell types."""
        color_map = {
            "Luminal epithelial": "#FF6B9D",
            "Basal epithelial": "#C44569",
            "HER2+ epithelial": "#F8B500",
            "Proliferating epithelial": "#FF4757",
            "Fibroblast": "#2E86DE",
            "Endothelial": "#54A0FF",
            "Pericyte": "#5F27CD",
            "T cell": "#1DD1A1",
            "B cell": "#10AC84",
            "Plasma cell": "#00D2D3",
            "Macrophage": "#FF9F43",
            "NK cell": "#EE5A24",
            "Adipocyte": "#FECA57",
            "Nerve": "#FF6348",
            "Unknown / Mixed": "#8395A7"
        }
        return color_map.get(cell_type, "#8395A7")

    def _compute_boundary(self, xs: List[float], ys: List[float]) -> List[Tuple[float, float]]:
        """Compute convex hull or alpha shape boundary."""
        from scipy.spatial import ConvexHull

        if len(xs) < 3:
            return list(zip(xs, ys))

        points = np.array(list(zip(xs, ys)))
        try:
            hull = ConvexHull(points)
            boundary = [(points[v, 0], points[v, 1]) for v in hull.vertices]
            return boundary
        except:
            return list(zip(xs, ys))

    def _estimate_area(self, xs: List[float], ys: List[float]) -> float:
        """Estimate region area from spot spread."""
        if len(xs) < 3:
            return 0.0

        # Use convex hull area
        from scipy.spatial import ConvexHull
        points = np.array(list(zip(xs, ys)))
        try:
            hull = ConvexHull(points)
            return hull.volume  # In 2D, volume = area
        except:
            return 0.0


# ═══════════════════════════════════════════════════════════════════════════════
#  4. Region Size Adjustment Agent
# ═══════════════════════════════════════════════════════════════════════════════

class RegionSizeAdjustmentAgent:
    """
    Agent that adjusts region sizes based on cluster composition.

    Problem: Points2Regions produces regions of varying sizes.
    Small regions (< few cells) may be noise.
    Large regions may contain multiple cell types.

    This agent:
    1. Merges tiny regions into nearest large neighbor
    2. Splits large heterogeneous regions using sub-clustering
    3. Adjusts boundaries based on gene expression gradients
    """

    def __init__(self, 
                 min_region_area_um2: float = 500.0,
                 max_region_area_um2: float = 50000.0,
                 heterogeneity_threshold: float = 0.3):
        """
        Args:
            min_region_area_um2: Regions smaller than this are merged
            max_region_area_um2: Regions larger than this are checked for splitting
            heterogeneity_threshold: CV of gene expression above which to split
        """
        self.min_area = min_region_area_um2
        self.max_area = max_region_area_um2
        self.hetero_threshold = heterogeneity_threshold

    def adjust(self, regions: List[Region]) -> List[Region]:
        """
        Run the full adjustment pipeline.
        """
        # Step 1: Merge tiny regions
        regions = self._merge_tiny_regions(regions)

        # Step 2: Split large heterogeneous regions
        regions = self._split_large_regions(regions)

        # Step 3: Smooth boundaries
        regions = self._smooth_boundaries(regions)

        return regions

    def _merge_tiny_regions(self, regions: List[Region]) -> List[Region]:
        """Merge regions below min_area into nearest neighbor."""
        tiny = [r for r in regions if r.area_um2 < self.min_area]
        large = [r for r in regions if r.area_um2 >= self.min_area]

        for small in tiny:
            # Find nearest large region by centroid distance
            nearest = min(large, 
                         key=lambda r: np.hypot(
                             r.centroid[0] - small.centroid[0],
                             r.centroid[1] - small.centroid[1]
                         ))
            # Merge spots
            nearest.spots.extend(small.spots)
            nearest.area_um2 += small.area_um2
            # Recompute centroid
            xs = [s.x for s in nearest.spots]
            ys = [s.y for s in nearest.spots]
            nearest.centroid = (np.mean(xs), np.mean(ys))

        return large

    def _split_large_regions(self, regions: List[Region]) -> List[Region]:
        """Split large regions that contain multiple cell types."""
        result = []

        for region in regions:
            if region.area_um2 < self.max_area:
                result.append(region)
                continue

            # Check heterogeneity: coefficient of variation of gene expression
            exprs = [s.expr for s in region.spots]
            cv = np.std(exprs) / (np.mean(exprs) + 1e-8)

            if cv < self.hetero_threshold:
                result.append(region)
                continue

            # Split using k-means on spatial coordinates
            from sklearn.cluster import KMeans

            coords = np.array([[s.x, s.y] for s in region.spots])
            kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
            sub_labels = kmeans.fit_predict(coords)

            for sub_id in range(2):
                sub_spots = [s for s, l in zip(region.spots, sub_labels) if l == sub_id]
                if len(sub_spots) < 3:
                    continue

                xs = [s.x for s in sub_spots]
                ys = [s.y for s in sub_spots]

                sub_region = Region(
                    region_id=len(result),
                    label=f"{region.label}-sub{sub_id}",
                    cell_type=region.cell_type,  # Inherit, re-annotate later
                    confidence=region.confidence * 0.9,
                    color=region.color,
                    spots=sub_spots,
                    boundary=self._compute_boundary(xs, ys),
                    area_um2=self._estimate_area(xs, ys),
                    centroid=(np.mean(xs), np.mean(ys)),
                    gene_composition={},  # Recompute
                    marker_score={}
                )
                result.append(sub_region)

        return result

    def _smooth_boundaries(self, regions: List[Region]) -> List[Region]:
        """Apply morphological smoothing to region boundaries."""
        # Simplified: just return as-is for now
        # In production: use shapely buffer/ simplify operations
        return regions

    def _compute_boundary(self, xs, ys):
        from scipy.spatial import ConvexHull
        if len(xs) < 3:
            return list(zip(xs, ys))
        points = np.array(list(zip(xs, ys)))
        try:
            hull = ConvexHull(points)
            return [(points[v, 0], points[v, 1]) for v in hull.vertices]
        except:
            return list(zip(xs, ys))

    def _estimate_area(self, xs, ys):
        from scipy.spatial import ConvexHull
        if len(xs) < 3:
            return 0.0
        points = np.array(list(zip(xs, ys)))
        try:
            return ConvexHull(points).volume
        except:
            return 0.0


# ═══════════════════════════════════════════════════════════════════════════════
#  5. Full Pipeline Integration
# ═══════════════════════════════════════════════════════════════════════════════

class SequoiaPoints2RegionsPipeline:
    """
    End-to-end pipeline:
    SEQUOIA spots → Points2Regions clustering → Cell type annotation → 
    Region size adjustment → Final regions for visualization
    """

    def __init__(self, 
                 pixel_width: float = 1.0,
                 num_clusters: int = 10,
                 pixel_smoothing: float = 5.0,
                 expr_threshold: float = 0.5):
        self.converter = SequoiaToPoints2Regions(expr_threshold=expr_threshold)
        self.clusterer = Points2RegionsWrapper(pixel_width=pixel_width)
        self.annotator = CellTypeAnnotationAgent()
        self.adjuster = RegionSizeAdjustmentAgent()

        self.num_clusters = num_clusters
        self.pixel_smoothing = pixel_smoothing

    def run(self, sequoia_spots: List[SpatialSpot]) -> List[Region]:
        """Run the full pipeline."""

        # Step 1: Convert SEQUOIA spots to Points2Regions format
        print("Step 1: Converting SEQUOIA spots...")
        df = self.converter.convert(sequoia_spots)

        # Step 2: Cluster with Points2Regions
        print("Step 2: Running Points2Regions clustering...")
        self.clusterer.fit(
            df, 
            num_clusters=self.num_clusters,
            pixel_smoothing=self.pixel_smoothing
        )
        cluster_labels = self.clusterer.get_cluster_per_marker()

        # Step 3: Annotate clusters with cell types
        print("Step 3: Annotating cell types...")
        regions = self.annotator.annotate(sequoia_spots, cluster_labels)

        # Step 4: Adjust region sizes
        print("Step 4: Adjusting region sizes...")
        regions = self.adjuster.adjust(regions)

        # Step 5: Re-annotate after splitting
        print("Step 5: Final annotation...")
        # (Optional: re-run annotator on adjusted regions)

        print(f"✓ Pipeline complete: {len(regions)} regions")
        for r in regions:
            print(f"  Region {r.region_id}: {r.cell_type} (conf={r.confidence:.2f}, area={r.area_um2:.0f}µm², n_spots={len(r.spots)})")

        return regions


# ═══════════════════════════════════════════════════════════════════════════════
#  6. Modal.com Integration (Serverless)
# ═══════════════════════════════════════════════════════════════════════════════

"""
# Add this to your pathvision_modal.py:

@app.function(gpu="T4", timeout=300, memory=8192)
def sequoia_points2regions(sequoia_result_json: str, 
                           num_clusters: int = 10,
                           pixel_smoothing: float = 5.0):
    """
    Takes SEQUOIA output JSON, runs Points2Regions + cell type annotation.
    Returns: JSON string of annotated regions.
    """
    import json

    # Parse SEQUOIA spots
    data = json.loads(sequoia_result_json)
    spots = [SpatialSpot(x=s['cx'], y=s['cy'], gene=s['gene'], expr=s['expr']) 
             for s in data['spots']]

    # Run pipeline
    pipeline = SequoiaPoints2RegionsPipeline(
        num_clusters=num_clusters,
        pixel_smoothing=pixel_smoothing
    )
    regions = pipeline.run(spots)

    # Serialize for frontend
    result = []
    for r in regions:
        result.append({
            'region_id': r.region_id,
            'cell_type': r.cell_type,
            'confidence': r.confidence,
            'color': r.color,
            'centroid': r.centroid,
            'area_um2': r.area_um2,
            'n_spots': len(r.spots),
            'boundary': r.boundary,
            'gene_composition': r.gene_composition,
            'top_markers': sorted(r.marker_score.items(), key=lambda x: -x[1])[:5]
        })

    return json.dumps(result)
"""


# ═══════════════════════════════════════════════════════════════════════════════
#  7. Frontend Layer Integration
# ═══════════════════════════════════════════════════════════════════════════════

"""
# Add this layer to your frontend layer panel:

const LAYER_DEFS = {
  // ... existing layers ...

  p2r_regions: {
    id: "p2r_regions",
    name: "Points2Regions — Cell Types",
    icon: "🧩",
    color: "#FF6B9D",
    category: "celltypes",
    blend: "source-over"
  }
};

// In drawOverlay(), render regions as filled polygons with labels:
if (showP2R && p2rRegions) {
  p2rRegions.forEach(region => {
    ctx.save();
    ctx.globalAlpha = 0.6;
    ctx.fillStyle = region.color;
    ctx.strokeStyle = region.color;
    ctx.lineWidth = 2;

    ctx.beginPath();
    region.boundary.forEach(([x, y], i) => {
      const px = x/100 * W, py = y/100 * H;
      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    });
    ctx.closePath();
    ctx.fill();
    ctx.stroke();

    // Label
    ctx.fillStyle = "#fff";
    ctx.font = "12px sans-serif";
    ctx.fillText(
      `${region.cell_type} (${Math.round(region.confidence*100)}%)`,
      region.centroid[0]/100 * W,
      region.centroid[1]/100 * H
    );
    ctx.restore();
  });
}
"""


# ═══════════════════════════════════════════════════════════════════════════════
#  EXAMPLE USAGE
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Mock SEQUOIA output
    np.random.seed(42)

    # Create synthetic spots: 3 distinct regions
    spots = []

    # Region 1: Tumor (high ESR1, GATA3)
    for i in range(50):
        spots.append(SpatialSpot(
            x=np.random.normal(30, 5),
            y=np.random.normal(30, 5),
            gene=np.random.choice(["ESR1", "GATA3", "KRT8", "MKI67"]),
            expr=np.random.uniform(0.6, 1.0)
        ))

    # Region 2: Stroma (high COL1A1, VIM)
    for i in range(50):
        spots.append(SpatialSpot(
            x=np.random.normal(70, 5),
            y=np.random.normal(30, 5),
            gene=np.random.choice(["COL1A1", "VIM", "FAP", "PDGFRA"]),
            expr=np.random.uniform(0.6, 1.0)
        ))

    # Region 3: Immune infiltrate (high CD3D, CD68)
    for i in range(50):
        spots.append(SpatialSpot(
            x=np.random.normal(50, 5),
            y=np.random.normal(70, 5),
            gene=np.random.choice(["CD3D", "CD68", "IL7R", "LYZ"]),
            expr=np.random.uniform(0.6, 1.0)
        ))

    # Run pipeline
    pipeline = SequoiaPoints2RegionsPipeline(
        num_clusters=5,
        pixel_smoothing=8.0
    )
    regions = pipeline.run(spots)
