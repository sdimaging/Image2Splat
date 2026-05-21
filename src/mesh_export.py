"""
Export a Pixal3D/TRELLIS.2 MeshWithVoxel to PLY (per-vertex PBR) and GLB
(vertex-colored, universally openable).

PLY:  preserves base_color + metallic + roughness + alpha as per-vertex floats.
      Best for Blender/Maya/MeshLab where PBR-per-vertex is meaningful.
GLB:  vertex-colored triangles. Most universal — opens in any browser viewer,
      Cinema 4D, Maxon, three.js / model-viewer, etc.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def export_mesh(mesh, out_root: Path) -> dict:
    """Write mesh.ply + mesh.glb under out_root. Returns paths actually written."""
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    verts = mesh.vertices.detach().cpu().numpy().astype(np.float32)
    faces = mesh.faces.detach().cpu().numpy().astype(np.int32)

    # Per-vertex PBR via the voxel-grid sampler — same channels render_utils uses:
    #   [0:3] = base_color, [3] = metallic, [4] = roughness, [5] = alpha
    if hasattr(mesh, "query_vertex_attrs"):
        with torch.no_grad():
            attrs = mesh.query_vertex_attrs().detach().cpu().numpy().astype(np.float32)
    else:
        attrs = None

    # --- PLY (preserves PBR per-vertex) ---
    try:
        # Pixal3D's helper expects uint8 base_color and metallic/roughness/alpha as np arrays.
        mesh_module = type(mesh).__module__
        if mesh_module.startswith("pixal3d") and attrs is not None:
            from pixal3d.utils.mesh_utils import write_pbr_ply
            ply_path = out_root / "mesh.ply"
            base_color = np.clip(attrs[..., 0:3] * 255.0, 0, 255).astype(np.uint8)
            metallic = np.clip(attrs[..., 3] * 255.0, 0, 255).astype(np.uint8)
            roughness = np.clip(attrs[..., 4] * 255.0, 0, 255).astype(np.uint8)
            alpha = np.clip(attrs[..., 5] * 255.0, 0, 255).astype(np.uint8)
            write_pbr_ply(str(ply_path), verts, faces, base_color, metallic, roughness, alpha)
            written["ply"] = ply_path
        elif mesh_module.startswith("trellis2") and attrs is not None:
            # TRELLIS.2 ships an equivalent — try its writer, else fall back.
            try:
                from trellis2.utils.mesh_utils import write_pbr_ply  # type: ignore
            except (ImportError, ModuleNotFoundError):
                from pixal3d.utils.mesh_utils import write_pbr_ply  # both write the same PLY layout
            ply_path = out_root / "mesh.ply"
            base_color = np.clip(attrs[..., 0:3] * 255.0, 0, 255).astype(np.uint8)
            metallic = np.clip(attrs[..., 3] * 255.0, 0, 255).astype(np.uint8)
            roughness = np.clip(attrs[..., 4] * 255.0, 0, 255).astype(np.uint8)
            alpha = np.clip(attrs[..., 5] * 255.0, 0, 255).astype(np.uint8)
            write_pbr_ply(str(ply_path), verts, faces, base_color, metallic, roughness, alpha)
            written["ply"] = ply_path
    except Exception as e:
        print(f"  WARN: PLY export skipped: {e}")

    # --- GLB (vertex-colored, universal) ---
    try:
        import trimesh
        vc = None
        if attrs is not None:
            # 4-channel RGBA for trimesh vertex_colors
            base_color = np.clip(attrs[..., 0:3] * 255.0, 0, 255).astype(np.uint8)
            alpha_u8 = np.clip(attrs[..., 5] * 255.0, 0, 255).astype(np.uint8)
            vc = np.concatenate([base_color, alpha_u8[..., None]], axis=-1)
        tm = trimesh.Trimesh(vertices=verts, faces=faces, vertex_colors=vc, process=False)
        glb_path = out_root / "mesh.glb"
        tm.export(str(glb_path))
        written["glb"] = glb_path
    except Exception as e:
        print(f"  WARN: GLB export skipped: {e}")

    return written
