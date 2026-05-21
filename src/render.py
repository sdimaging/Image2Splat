"""
Multi-view PBR render of a TRELLIS.2 mesh using custom camera poses.

TRELLIS.2 exposes `render_utils.render_frames(sample, extrinsics, intrinsics, ...)`
which is the per-view render entry point we need. Public `render_video` wraps this
with a fixed orbit; we bypass that and feed our own Fibonacci-sphere poses.

Conventions:
  - TRELLIS.2 ships its example assets oriented Z-up. We default camera trajectories
    to Z-up here so rendered images look right-side-up relative to the asset.
  - The renderer's `intrinsics` are NORMALIZED (cx=cy=0.5, focal = 0.5/tan(fov/2)),
    distinct from cameras.intrinsics_from_fov which returns pixel-space focal/center.
  - Resolution is passed separately via the renderer options.

Output of render() is a list of N RGB(A) PIL Images in the order of `w2c_matrices`.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image


def _ensure_repo_on_path(repo_dir: Path) -> None:
    p = str(Path(repo_dir).resolve())
    if p not in sys.path:
        sys.path.insert(0, p)


def _normalized_intrinsics_from_fov(fov_deg: float, device: str = "cuda") -> torch.Tensor:
    """Build TRELLIS.2-flavored normalized intrinsics (3, 3) for square images.

    TRELLIS.2's nvdiffrast renderer uses normalized device coordinates internally,
    so the intrinsics matrix uses cx = cy = 0.5 and focal = 0.5/tan(fov/2) — not
    pixel-space. See utils3d.torch.intrinsics_from_fov_xy upstream.
    """
    fov_rad = float(np.radians(fov_deg))
    focal = 0.5 / np.tan(fov_rad / 2.0)
    intr = torch.tensor(
        [[focal, 0.0, 0.5], [0.0, focal, 0.5], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
        device=device,
    )
    return intr


def render_multiview(
    mesh,
    w2c_matrices: np.ndarray,
    *,
    fov_deg: float = 40.0,
    resolution: int = 1024,
    bg_color: tuple[float, float, float] = (0.0, 0.0, 0.0),
    envmap=None,
    repo_dir: Path | str = Path.home() / "projects" / "TRELLIS.2",
    device: str = "cuda",
    verbose: bool = True,
) -> list[Image.Image]:
    """Render N views of `mesh` from the given OpenCV world-to-camera matrices.

    Args:
        mesh: a MeshWithVoxel as returned by Backbone.run() / pipeline.run()
        w2c_matrices: (N, 4, 4) OpenCV W2C matrices (numpy). Use cameras.generate_camera_trajectory.
        fov_deg: vertical FOV in degrees, applied uniformly.
        resolution: output image edge length in pixels.
        bg_color: solid background tuple (0..1). For alpha-masked output, the
                  caller can post-process the alpha channel; setting bg here just
                  controls what shows behind transparent pixels in the RGB output.
        envmap: optional EnvMap; passed through to TRELLIS.2's renderer for PBR
                shading. If None, the renderer uses its default.
        repo_dir: path to the TRELLIS.2 clone (for sys.path injection).
        device: torch device.
        verbose: print tqdm progress.

    Returns:
        list of N PIL Image objects (RGB or RGBA depending on renderer output).
    """
    _ensure_repo_on_path(Path(repo_dir))
    # Both TRELLIS.2 and Pixal3D ship a `render_utils` module under their own
    # top-level package, with their own MeshWithVoxel class. The renderer must
    # match the MESH'S source — if a Pixal3D mesh hits TRELLIS.2's renderer, it
    # raises "Unsupported sample type". Detect from the mesh's module path.
    mesh_module = type(mesh).__module__
    if mesh_module.startswith("pixal3d"):
        from pixal3d.utils import render_utils  # type: ignore
    else:
        from trellis2.utils import render_utils  # type: ignore

    w2c_matrices = np.asarray(w2c_matrices)
    if w2c_matrices.ndim != 3 or w2c_matrices.shape[1:] != (4, 4):
        raise ValueError(f"w2c_matrices must be (N, 4, 4), got {w2c_matrices.shape}")

    extrinsics = [
        torch.tensor(m, dtype=torch.float32, device=device) for m in w2c_matrices
    ]
    intrinsics = [_normalized_intrinsics_from_fov(fov_deg, device)] * len(w2c_matrices)

    options = {"resolution": resolution, "bg_color": bg_color}
    render_kwargs = {}
    if envmap is not None:
        render_kwargs["envmap"] = envmap

    result = render_utils.render_frames(
        mesh,
        extrinsics,
        intrinsics,
        options=options,
        verbose=verbose,
        **render_kwargs,
    )

    # render_frames returns a dict of {channel: list of (H, W, C) uint8 arrays}.
    # For PBR rendering the keys are: shaded, normal, base_color, metallic,
    # roughness, alpha. We want `shaded` (final PBR-lit RGB) + `alpha` (object mask)
    # to produce clean RGBA renders for splat training. The diagnostic 6-panel
    # `make_pbr_vis_frames` output is for the demo video, NOT splat-training data.
    if "shaded" in result:
        shaded = result["shaded"]
        alpha = result.get("alpha")
        if alpha is not None:
            frames = [
                Image.fromarray(np.concatenate(
                    [s, a[..., :1] if a.ndim == 3 else a[..., None]], axis=-1
                ))
                for s, a in zip(shaded, alpha)
            ]
        else:
            frames = [Image.fromarray(s) for s in shaded]
    else:
        # Fallback for non-PBR renderers — pick whatever RGB channel exists
        key = next((k for k in ("color", "albedo", "rgb") if k in result), None)
        if key is None:
            key = list(result.keys())[0]
        frames = [Image.fromarray(np.asarray(f)) for f in result[key]]

    return frames


def save_views(
    images: list[Image.Image],
    output_dir: Path | str,
    filename_template: str = "{:03d}.png",
) -> list[str]:
    """Write a list of PIL images to disk and return the filenames (not paths).

    Filenames are the {index}-formatted strings — what gets written into the
    COLMAP/Nerfstudio exports.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    names: list[str] = []
    for i, img in enumerate(images):
        name = filename_template.format(i)
        img.save(output_dir / name)
        names.append(name)
    return names
