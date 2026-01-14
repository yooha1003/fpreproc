#!/usr/bin/env python3
"""
BrainNet-like 3D effective connectivity visualization.

Creates an interactive HTML (Plotly) with:
  - Translucent MNI brain surface ("glass brain", optional)
  - ROI nodes (atlas cut coordinates)
  - Directed edges (lines) + arrowheads for EC (directed) matrices

Matrix convention (matches this repo):
  matrix[target, source] = influence of source -> target
"""

import sys
import logging
import os
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.helpers import load_config, save_metadata  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _rgba_str(rgba: Tuple[float, float, float, float]) -> str:
    r, g, b, a = rgba
    return f"rgba({int(r * 255)},{int(g * 255)},{int(b * 255)},{float(a):.3f})"


def _resolve_nilearn_data_dir(override: Optional[str] = None) -> Optional[str]:
    """
    Resolve a reasonable nilearn dataset directory.

    If data has already been downloaded, this helps avoid re-downloading when network access
    is restricted. The directory is used for nilearn.datasets.fetch_* functions.
    """
    if override:
        return str(Path(override).expanduser())

    env = os.environ.get("NILEARN_DATA")
    if env:
        return str(Path(env).expanduser())

    return None


def _unit_sphere_mesh(n_theta: int = 14, n_phi: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a low-poly unit sphere triangle mesh.

    Returns
    -------
    verts : (n_verts, 3) float array
        Vertex coordinates on a unit sphere.
    faces : (n_faces, 3) int array
        Triangle indices into `verts`.
    """
    n_theta = int(n_theta)
    n_phi = int(n_phi)
    if n_theta < 6:
        raise ValueError("n_theta must be >= 6")
    if n_phi < 6:
        raise ValueError("n_phi must be >= 6")

    verts: List[List[float]] = [[0.0, 0.0, 1.0]]  # north pole

    thetas = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    phis = np.linspace(0.0, np.pi, n_phi)
    ring_phis = phis[1:-1]

    for phi in ring_phis:
        sin_phi = float(np.sin(phi))
        cos_phi = float(np.cos(phi))
        for theta in thetas:
            verts.append(
                [
                    sin_phi * float(np.cos(theta)),
                    sin_phi * float(np.sin(theta)),
                    cos_phi,
                ]
            )

    verts.append([0.0, 0.0, -1.0])  # south pole
    verts_arr = np.asarray(verts, dtype=float)

    faces: List[List[int]] = []
    north = 0
    south = len(verts) - 1
    n_rings = len(ring_phis)

    def ring_index(ring: int, t: int) -> int:
        return 1 + ring * n_theta + (t % n_theta)

    # North cap
    if n_rings >= 1:
        for t in range(n_theta):
            faces.append([north, ring_index(0, t), ring_index(0, t + 1)])

    # Middle quads (two triangles per cell)
    for ring in range(n_rings - 1):
        for t in range(n_theta):
            v00 = ring_index(ring, t)
            v01 = ring_index(ring, t + 1)
            v10 = ring_index(ring + 1, t)
            v11 = ring_index(ring + 1, t + 1)
            faces.append([v00, v10, v11])
            faces.append([v00, v11, v01])

    # South cap
    if n_rings >= 1:
        last = n_rings - 1
        for t in range(n_theta):
            faces.append([ring_index(last, t), south, ring_index(last, t + 1)])

    faces_arr = np.asarray(faces, dtype=int)
    return verts_arr, faces_arr


class EffectiveConnectivityGlassBrain3DViz:
    """Render directed EC as a BrainNet-like 3D connectome (HTML)."""

    def __init__(self, config: Optional[dict] = None):
        if config is None:
            config = load_config()
        self.config = config

        self._brain_mesh_cache: Optional[Tuple[np.ndarray, np.ndarray]] = None

    def _get_atlas(
        self,
        atlas_name: Optional[str],
        *,
        nilearn_data_dir: Optional[str] = None,
    ) -> Tuple[str, Optional[List[str]]]:
        from nilearn import datasets

        if atlas_name is None:
            atlas_name = self.config.get("atlas", {}).get("default", "AAL")

        data_dir = _resolve_nilearn_data_dir(nilearn_data_dir)

        atlas_upper = atlas_name.upper()
        if atlas_upper == "AAL":
            atlas = datasets.fetch_atlas_aal(data_dir=data_dir)
            return atlas["maps"], atlas.get("labels")

        if atlas_upper == "SCHAEFER":
            atlas = datasets.fetch_atlas_schaefer_2018(
                n_rois=400,
                resolution_mm=2,
                data_dir=data_dir,
            )
            return atlas["maps"], atlas.get("labels")

        if atlas_upper == "HARVARDOXFORD":
            atlas = datasets.fetch_atlas_harvard_oxford(
                "cort-maxprob-thr25-2mm",
                data_dir=data_dir,
            )
            return atlas["maps"], atlas.get("labels")

        raise ValueError(f"Unknown atlas: {atlas_name}")

    def _get_roi_coordinates(self, atlas_img: str) -> np.ndarray:
        from nilearn.plotting import find_parcellation_cut_coords

        coords = find_parcellation_cut_coords(atlas_img)
        coords = np.asarray(coords, dtype=float)
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError(f"Unexpected coordinates shape: {coords.shape}")
        return coords

    def _normalize_labels(self, labels: Optional[List[str]], n_rois: int) -> List[str]:
        if not labels:
            return [f"ROI {i + 1:03d}" for i in range(n_rois)]

        normalized = [str(l).strip() for l in labels if str(l).strip()]
        if not normalized:
            return [f"ROI {i + 1:03d}" for i in range(n_rois)]

        if len(normalized) == n_rois + 1 and normalized[0].lower() in {"background", "bg"}:
            normalized = normalized[1:]

        if len(normalized) != n_rois:
            logger.warning(
                f"Atlas label count ({len(normalized)}) != n_rois ({n_rois}); trimming/padding to match."
            )
            if len(normalized) > n_rois:
                normalized = normalized[:n_rois]
            else:
                normalized = normalized + [f"ROI {i + 1:03d}" for i in range(len(normalized), n_rois)]

        return normalized

    def _load_roi_labels_file(self, labels_file: Optional[str], n_rois: int) -> Optional[List[str]]:
        if not labels_file:
            return None

        path = Path(labels_file)
        if not path.exists():
            raise FileNotFoundError(f"ROI labels file not found: {path}")

        labels: List[str] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                label = line.strip()
                if label:
                    labels.append(label)

        if not labels:
            return None

        if len(labels) != n_rois:
            logger.warning(
                f"ROI label file count ({len(labels)}) != n_rois ({n_rois}); trimming/padding to match."
            )
            if len(labels) > n_rois:
                labels = labels[:n_rois]
            else:
                labels = labels + [f"ROI {i + 1:03d}" for i in range(len(labels), n_rois)]

        return labels

    def _select_edges(
        self,
        matrix: np.ndarray,
        *,
        top_k: int = 200,
        min_weight: Optional[float] = None,
    ) -> List[Tuple[int, int, float]]:
        """
        Return list of directed edges (src, tgt, weight).

        Notes:
          matrix[tgt, src] = influence of src -> tgt.
        """
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError(f"EC matrix must be square, got shape={matrix.shape}")

        cleaned = np.array(matrix, dtype=float, copy=True)
        cleaned[~np.isfinite(cleaned)] = 0.0
        np.fill_diagonal(cleaned, 0.0)

        n = cleaned.shape[0]
        edges: List[Tuple[int, int, float]] = []

        for tgt in range(n):
            for src in range(n):
                if src == tgt:
                    continue
                w = float(cleaned[tgt, src])
                if w == 0.0:
                    continue
                if min_weight is not None and abs(w) < float(min_weight):
                    continue
                edges.append((src, tgt, w))

        edges.sort(key=lambda e: abs(e[2]), reverse=True)
        if top_k is not None:
            top_k = int(top_k)
            if top_k <= 0:
                raise ValueError("--top-k must be a positive integer")
            edges = edges[:top_k]

        return edges

    def _matrix_from_edges(self, n: int, edges: List[Tuple[int, int, float]]) -> np.ndarray:
        m = np.zeros((n, n), dtype=float)
        for src, tgt, w in edges:
            m[tgt, src] = w
        return m

    def _compute_net_flow(self, matrix: np.ndarray) -> np.ndarray:
        """
        Compute net-flow (out - in) under matrix[target, source] convention.
        """
        out_strength = matrix.sum(axis=0)
        in_strength = matrix.sum(axis=1)
        return out_strength - in_strength

    def _get_brain_mesh(self, step_size: int = 2) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if self._brain_mesh_cache is not None:
            return self._brain_mesh_cache

        try:
            from nilearn import datasets
            from skimage.measure import marching_cubes
            from nibabel.affines import apply_affine
        except ImportError as e:
            logger.warning(f"Brain surface mesh dependencies missing; skipping glass brain mesh: {e}")
            return None

        mask_img = datasets.load_mni152_brain_mask()
        data = np.asarray(mask_img.get_fdata(), dtype=np.float32)

        verts, faces, _normals, _values = marching_cubes(data, level=0.5, step_size=int(step_size))
        verts_world = apply_affine(mask_img.affine, verts)

        self._brain_mesh_cache = (verts_world, faces.astype(int))
        return self._brain_mesh_cache

    def _camera_presets(self, points: np.ndarray) -> Dict[str, Dict[str, Any]]:
        # NOTE: Plotly camera eye coordinates are unitless (not mm); values around 1-3
        # give stable "fit-to-view" behavior. Using data-range-scaled values makes the
        # initial view/zoom extremely sensitive.
        dist = 2.2
        dist_side = 2.6
        dist_top = 2.2

        return {
            "isometric": dict(
                eye=dict(x=dist, y=dist, z=dist * 0.85),
                up=dict(x=0, y=0, z=1),
            ),
            "left": dict(
                eye=dict(x=-dist_side, y=0, z=0.1),
                up=dict(x=0, y=0, z=1),
            ),
            "right": dict(
                eye=dict(x=dist_side, y=0, z=0.1),
                up=dict(x=0, y=0, z=1),
            ),
            "anterior": dict(
                eye=dict(x=0.1, y=dist_side, z=0.1),
                up=dict(x=0, y=0, z=1),
            ),
            "posterior": dict(
                eye=dict(x=0.1, y=-dist_side, z=0.1),
                up=dict(x=0, y=0, z=1),
            ),
            "superior": dict(
                eye=dict(x=0, y=0.1, z=dist_top),
                up=dict(x=0, y=1, z=0),
            ),
            "inferior": dict(
                eye=dict(x=0, y=0.1, z=-dist_top),
                up=dict(x=0, y=1, z=0),
            ),
        }

    def plot(
        self,
        matrix: np.ndarray,
        coords: np.ndarray,
        labels: List[str],
        edges: List[Tuple[int, int, float]],
        output_path: Path,
        *,
        title: str,
        show_labels: bool,
        node_size: int,
        node_style: str,
        arrow_size: float,
        brain_opacity: float,
        brain_step_size: int,
        style: str,
        camera: Optional[str],
        camera_buttons: bool,
        export_png: bool,
    ) -> str:
        try:
            import plotly.graph_objects as go
        except ImportError as e:
            raise ImportError("plotly is required for 3D EC visualization") from e

        fig = go.Figure()

        # Glass brain surface (optional)
        if brain_opacity and brain_opacity > 0:
            brain_mesh = self._get_brain_mesh(step_size=brain_step_size)
            if brain_mesh is not None:
                verts, faces = brain_mesh
                fig.add_trace(
                    go.Mesh3d(
                        x=verts[:, 0],
                        y=verts[:, 1],
                        z=verts[:, 2],
                        i=faces[:, 0],
                        j=faces[:, 1],
                        k=faces[:, 2],
                        color="lightgray",
                        opacity=float(brain_opacity),
                        name="MNI152 brain",
                        hoverinfo="skip",
                        lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0, roughness=1.0, fresnel=0.0),
                    )
                )

        # Edge styling
        weights = np.array([w for _s, _t, w in edges], dtype=float)
        abs_weights = np.abs(weights)

        if abs_weights.size:
            vmax = float(np.percentile(abs_weights, 95))
            vmax = vmax if vmax > 0 else float(np.max(abs_weights))
            vmax = vmax if vmax > 0 else 1.0
        else:
            vmax = 1.0

        has_negative = bool(weights.size and np.any(weights < 0))
        edge_cmap = plt.get_cmap("coolwarm" if has_negative else "viridis")

        def edge_color(w: float) -> str:
            # Fixed red hue with 60% opacity.
            return "rgba(200,30,30,0.6)"

        def edge_width(w: float) -> float:
            return float(2.0 * (1.0 + 5.0 * (min(1.0, abs(w) / vmax) ** 0.7)))

        # Edges (lines)
        for src, tgt, w in edges:
            fig.add_trace(
                go.Scatter3d(
                    x=[coords[src, 0], coords[tgt, 0]],
                    y=[coords[src, 1], coords[tgt, 1]],
                    z=[coords[src, 2], coords[tgt, 2]],
                    mode="lines",
                    line=dict(color=edge_color(w), width=edge_width(w)),
                    name="EC edges",
                    showlegend=False,
                    hovertemplate=f"{labels[src]} → {labels[tgt]}<br>weight={w:.4g}<extra></extra>",
                )
            )

        # Arrowheads (cones at targets)
        if edges:
            xs: List[float] = []
            ys: List[float] = []
            zs: List[float] = []
            us: List[float] = []
            vs: List[float] = []
            ws: List[float] = []

            for src, tgt, _w in edges:
                start = coords[src]
                end = coords[tgt]
                vec = end - start
                norm = float(np.linalg.norm(vec))
                if norm == 0:
                    continue
                direction = vec / norm
                arrow_vec = direction * float(arrow_size) * 1.0

                xs.append(float(end[0]))
                ys.append(float(end[1]))
                zs.append(float(end[2]))
                us.append(float(arrow_vec[0]))
                vs.append(float(arrow_vec[1]))
                ws.append(float(arrow_vec[2]))

            if xs:
                fig.add_trace(
                    go.Cone(
                        x=xs,
                        y=ys,
                        z=zs,
                        u=us,
                        v=vs,
                        w=ws,
                        anchor="tip",
                        sizemode="raw",
                        sizeref=1.2,
                        showscale=False,
                        colorscale=[
                            [0.0, "rgba(0,120,255,0.95)"],
                            [1.0, "rgba(0,120,255,0.95)"],
                        ],
                        name="Direction",
                        hoverinfo="skip",
                    )
                )

        # Nodes: color by net-flow computed from displayed edges
        filtered = self._matrix_from_edges(matrix.shape[0], edges)
        net_flow = self._compute_net_flow(filtered)
        node_vmax = float(np.percentile(np.abs(net_flow), 95)) if net_flow.size else 1.0
        node_vmax = node_vmax if node_vmax > 0 else 1.0
        normalized_style = (style or "brainnet").strip().lower()
        normalized_node_style = (node_style or "marker").strip().lower()
        if normalized_node_style not in {"marker", "sphere"}:
            raise ValueError("node_style must be 'marker' or 'sphere'")

        node_text = labels
        use_sphere_nodes = normalized_node_style == "sphere"

        node_colors = ["rgba(40,180,60,0.95)" for _ in net_flow]

        if use_sphere_nodes:
            # Scatter3d markers use pixel sizes (screen-space), which makes nodes look too big/small
            # depending on zoom. Render nodes as true 3D spheres (data-space) with lighting.
            n_rois = int(coords.shape[0])
            n_theta = 18 if n_rois <= 200 else 12
            n_phi = 14 if n_rois <= 200 else 10
            sphere_verts, sphere_faces = _unit_sphere_mesh(n_theta=n_theta, n_phi=n_phi)

            # Approximate radius in mm from the existing node_size parameter.
            radius = float(node_size) * 0.5
            if n_rois > 0:
                radius *= float((116 / n_rois) ** 0.25)
            radius = float(max(1.0, min(5.0, radius)))

            base_verts = sphere_verts * radius
            base_faces = sphere_faces.astype(int)

            verts_all: List[np.ndarray] = []
            faces_all: List[np.ndarray] = []
            vertex_colors: List[str] = []

            v_count = base_verts.shape[0]
            for idx, center in enumerate(coords):
                verts_all.append(base_verts + center)
                faces_all.append(base_faces + idx * v_count)
                vertex_colors.extend([node_colors[idx]] * v_count)

            verts_concat = np.vstack(verts_all)
            faces_concat = np.vstack(faces_all)

            fig.add_trace(
                go.Mesh3d(
                    x=verts_concat[:, 0],
                    y=verts_concat[:, 1],
                    z=verts_concat[:, 2],
                    i=faces_concat[:, 0],
                    j=faces_concat[:, 1],
                    k=faces_concat[:, 2],
                    vertexcolor=vertex_colors,
                    name="ROIs",
                    hoverinfo="skip",
                    showscale=False,
                    lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0, roughness=1.0, fresnel=0.0),
                    flatshading=True,
                )
            )

            # Add a lightweight scatter trace for hover + optional labels.
            mode = "markers+text" if show_labels else "markers"
            fig.add_trace(
                go.Scatter3d(
                    x=coords[:, 0],
                    y=coords[:, 1],
                    z=coords[:, 2],
                    mode=mode,
                    text=node_text,
                    textposition="top center",
                    hovertemplate="%{text}<br>x=%{x:.1f}, y=%{y:.1f}, z=%{z:.1f}<extra></extra>",
                    textfont=dict(size=9, color="#222222"),
                    marker=dict(size=2, color="rgba(0,0,0,0.01)"),
                    name="ROI labels",
                    showlegend=False,
                )
            )
        else:
            mode = "markers+text" if show_labels else "markers"
            fig.add_trace(
                go.Scatter3d(
                    x=coords[:, 0],
                    y=coords[:, 1],
                    z=coords[:, 2],
                    mode=mode,
                    text=node_text,
                    textposition="top center",
                    hovertemplate="%{text}<br>x=%{x:.1f}, y=%{y:.1f}, z=%{z:.1f}<extra></extra>",
                    textfont=dict(size=9, color="#222222"),
                    marker=dict(size=int(node_size), color=node_colors, line=dict(color="white", width=0.5)),
                    name="ROIs",
                )
            )

        fig.update_layout(
            title=title,
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
                aspectmode="data",
            ),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            margin=dict(l=0, r=0, t=50, b=0),
        )

        if normalized_style == "netplotbrain":
            fig.update_layout(
                template="plotly_white",
                paper_bgcolor="white",
                font=dict(color="#111111"),
            )

        # Camera presets (netplotbrain-figures-like view switches)
        presets_points: List[np.ndarray] = [coords]
        brain_mesh = None
        if brain_opacity and brain_opacity > 0:
            brain_mesh = self._get_brain_mesh(step_size=brain_step_size)
        if brain_mesh is not None:
            verts, _faces = brain_mesh
            presets_points.append(verts)

        all_points = np.vstack(presets_points) if presets_points else coords
        presets = self._camera_presets(all_points)
        projection_type = "orthographic" if normalized_style == "netplotbrain" else "perspective"

        if camera:
            camera_key = camera.strip().lower()
            if camera_key not in presets:
                raise ValueError(
                    f"Unknown camera preset '{camera}'. "
                    f"Valid: {', '.join(sorted(presets.keys()))}"
                )
            cam = dict(presets[camera_key])
            cam["projection"] = dict(type=projection_type)
            fig.update_layout(scene_camera=cam)

        if camera_buttons:
            buttons = [
                dict(
                    label=name,
                    method="relayout",
                    args=[{"scene.camera": {**cam, "projection": dict(type=projection_type)}}],
                )
                for name, cam in presets.items()
            ]
            fig.update_layout(
                updatemenus=[
                    dict(
                        type="buttons",
                        direction="right",
                        x=0.0,
                        y=1.08,
                        xanchor="left",
                        yanchor="top",
                        showactive=True,
                        buttons=buttons,
                    )
                ]
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_path))
        logger.info(f"3D directed EC connectome saved: {output_path}")

        png_path: Optional[Path] = None
        if export_png:
            png_path = output_path.with_suffix(".png")
            try:
                fig.write_image(str(png_path), scale=2)
                logger.info(f"3D directed EC connectome PNG saved: {png_path}")
            except Exception as e:
                logger.warning(
                    "PNG export failed (requires plotly+kaleido). "
                    f"Continuing with HTML only: {e}"
                )

        return str(output_path)

    def run(
        self,
        matrix_file: str,
        output_dir: str,
        *,
        subject_id: str = "sub-001",
        method: str = "granger",
        atlas: Optional[str] = None,
        nilearn_data_dir: Optional[str] = None,
        top_k: int = 200,
        min_weight: Optional[float] = None,
        show_labels: bool = False,
        roi_labels: Optional[str] = None,
        use_atlas_labels: bool = False,
        node_size: int = 6,
        node_style: Optional[str] = None,
        arrow_size: float = 8.0,
        brain_opacity: float = 0.18,
        brain_step_size: int = 2,
        style: str = "brainnet",
        camera: Optional[str] = None,
        camera_buttons: Optional[bool] = None,
        export_png: bool = False,
    ) -> dict:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        matrix_file_path = Path(matrix_file)
        if not matrix_file_path.exists():
            raise FileNotFoundError(f"EC matrix file not found: {matrix_file_path}")

        matrix = np.load(matrix_file_path)
        logger.info(f"Loaded EC matrix: {matrix.shape}")

        atlas_img, atlas_labels = self._get_atlas(atlas, nilearn_data_dir=nilearn_data_dir)
        coords = self._get_roi_coordinates(atlas_img)

        # Adjust if sizes don't match
        if coords.shape[0] != matrix.shape[0]:
            logger.warning(
                f"Coordinate count ({coords.shape[0]}) doesn't match matrix size ({matrix.shape[0]}); "
                "trimming both to the minimum."
            )
            min_size = min(coords.shape[0], matrix.shape[0])
            coords = coords[:min_size]
            matrix = matrix[:min_size, :min_size]
            n_rois = min_size
        else:
            n_rois = matrix.shape[0]

        labels_list = self._load_roi_labels_file(roi_labels, n_rois)
        if labels_list is None:
            labels_list = self._normalize_labels(atlas_labels if use_atlas_labels else None, n_rois)

        edges = self._select_edges(matrix, top_k=top_k, min_weight=min_weight)

        output_path = output_dir / f"{subject_id}_ec_{method}_connectome_3d_directed.html"
        title = f"Directed EC 3D Connectome ({method}) - {subject_id}"

        normalized_style = (style or "brainnet").strip().lower()
        if camera_buttons is None:
            camera_buttons = normalized_style == "netplotbrain"

        if node_style is None:
            node_style = "sphere" if normalized_style == "netplotbrain" else "marker"

        if camera is None:
            camera = "isometric"

        html_path = self.plot(
            matrix,
            coords,
            labels_list,
            edges,
            output_path,
            title=title,
            show_labels=show_labels,
            node_size=node_size,
            node_style=node_style,
            arrow_size=arrow_size,
            brain_opacity=brain_opacity,
            brain_step_size=brain_step_size,
            style=style,
            camera=camera,
            camera_buttons=bool(camera_buttons),
            export_png=export_png,
        )

        metadata = {
            "subject_id": subject_id,
            "method": method,
            "matrix_file": str(matrix_file),
            "atlas": atlas or self.config.get("atlas", {}).get("default", "AAL"),
            "n_rois": int(n_rois),
            "n_edges_drawn": int(len(edges)),
            "top_k": int(top_k),
            "min_weight": min_weight,
            "show_labels": bool(show_labels),
            "roi_labels_file": roi_labels,
            "use_atlas_labels": bool(use_atlas_labels),
            "node_size": int(node_size),
            "node_style": str(node_style),
            "arrow_size": float(arrow_size),
            "brain_opacity": float(brain_opacity),
            "brain_step_size": int(brain_step_size),
            "style": style,
            "camera": camera,
            "camera_buttons": bool(camera_buttons),
            "export_png": bool(export_png),
            "nilearn_data_dir": _resolve_nilearn_data_dir(nilearn_data_dir),
            "figure": html_path,
        }

        metadata_file = output_dir / f"{subject_id}_ec_{method}_connectome_3d_directed_metadata.json"
        save_metadata(metadata, str(metadata_file))
        metadata["metadata_file"] = str(metadata_file)

        return metadata


def main() -> None:
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="BrainNet-like 3D directed effective connectivity visualization (HTML)."
    )
    parser.add_argument("matrix", help="EC matrix file (.npy)")
    parser.add_argument("output_dir", help="Directory to save outputs")
    parser.add_argument("--subject", default="sub-001", help="Subject ID")
    parser.add_argument(
        "--method",
        default="granger",
        help="Method name to show on the plot (e.g., granger, transfer_entropy)",
    )
    parser.add_argument("--atlas", help="Atlas name (AAL, Schaefer, HarvardOxford)")
    parser.add_argument(
        "--nilearn-data-dir",
        dest="nilearn_data_dir",
        help="Directory for nilearn datasets (atlas templates). Defaults to NILEARN_DATA or nilearn default (usually ~/nilearn_data).",
    )
    parser.add_argument("--top-k", dest="top_k", type=int, default=200, help="Top-k edges to draw")
    parser.add_argument("--min-weight", dest="min_weight", type=float, default=None, help="Minimum |weight|")
    parser.add_argument("--show-labels", action="store_true", help="Show ROI labels on the plot")
    parser.add_argument("--roi-labels", help="Optional text file with one ROI label per line")
    parser.add_argument(
        "--use-atlas-labels",
        action="store_true",
        help="Use atlas-provided ROI labels when available",
    )
    parser.add_argument(
        "--node-size",
        dest="node_size",
        type=int,
        default=6,
        help="Node size (pixels for brainnet; ~radius in mm for netplotbrain sphere nodes).",
    )
    parser.add_argument(
        "--node-style",
        dest="node_style",
        choices=["marker", "sphere"],
        default=None,
        help="Node rendering style (defaults to sphere for netplotbrain, marker otherwise).",
    )
    parser.add_argument("--arrow-size", dest="arrow_size", type=float, default=8.0, help="Arrowhead length (mm)")
    parser.add_argument("--brain-opacity", dest="brain_opacity", type=float, default=0.18, help="Brain opacity (0-1)")
    parser.add_argument(
        "--brain-step-size",
        dest="brain_step_size",
        type=int,
        default=2,
        help="Marching cubes step size (higher = faster, lower detail)",
    )
    parser.add_argument(
        "--style",
        default="brainnet",
        choices=["brainnet", "netplotbrain"],
        help="Plot styling preset",
    )
    parser.add_argument(
        "--camera",
        help="Initial camera preset (isometric, left, right, anterior, posterior, superior, inferior)",
    )
    parser.add_argument("--camera-buttons", dest="camera_buttons", action="store_true", help="Show camera buttons")
    parser.add_argument("--no-camera-buttons", dest="camera_buttons", action="store_false", help="Hide camera buttons")
    parser.set_defaults(camera_buttons=None)
    parser.add_argument(
        "--export-png",
        dest="export_png",
        action="store_true",
        help="Also export a PNG snapshot (requires plotly+kaleido).",
    )
    parser.add_argument("--config", help="Optional pipeline config file")

    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else None

    viz = EffectiveConnectivityGlassBrain3DViz(cfg)
    results = viz.run(
        args.matrix,
        args.output_dir,
        subject_id=args.subject,
        method=args.method,
        atlas=args.atlas,
        nilearn_data_dir=args.nilearn_data_dir,
        top_k=args.top_k,
        min_weight=args.min_weight,
        show_labels=args.show_labels,
        roi_labels=args.roi_labels,
        use_atlas_labels=args.use_atlas_labels,
        node_size=args.node_size,
        node_style=args.node_style,
        arrow_size=args.arrow_size,
        brain_opacity=args.brain_opacity,
        brain_step_size=args.brain_step_size,
        style=args.style,
        camera=args.camera,
        camera_buttons=args.camera_buttons,
        export_png=args.export_png,
    )

    print("\n✓ 3D directed EC connectome created")
    print(f"Figure: {results['figure']}")


if __name__ == "__main__":
    main()
