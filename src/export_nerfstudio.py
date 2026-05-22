"""
Nerfstudio transforms.json writer.

Output format reference:
    https://docs.nerf.studio/quickstart/data_conventions.html

Per-frame `transform_matrix` is CAMERA-TO-WORLD in OpenGL convention
(+X right, +Y up, +Z out of screen), NOT the OpenCV W2C our cameras module
produces. This module performs the convention swap.

The top-level `camera_model` field selects the intrinsics distortion model
("OPENCV" = pinhole + optional radial). It is unrelated to the transform
convention, which is hardcoded to OpenGL.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np

from .cameras import opencv_w2c_to_opengl_c2w


def write_nerfstudio_dataset(
    output_path: str | Path,
    w2c_matrices: np.ndarray,
    image_names: list[str] | Iterable[str],
    intrinsics: dict | list[dict],
    image_subdir: str = "images",
    aabb_scale: int = 2,
) -> Path:
    """Write a Nerfstudio-format transforms.json.

    Args:
        output_path: destination path. If it ends in `.json`, written as-is;
                     otherwise treated as a directory and written to
                     `<output_path>/transforms.json`.
        w2c_matrices: (N, 4, 4) OpenCV world-to-camera matrices.
        image_names: filenames of N rendered images.
        intrinsics: dict (one shared camera for all frames — top-level fl_x/fl_y/cx/cy)
                    OR a list[dict] of length N (per-frame intrinsics — Nerfstudio
                    supports `fl_x`/`fl_y`/`cx`/`cy`/`w`/`h` INSIDE each frame entry,
                    which overrides top-level values).
        image_subdir: directory name to prepend to each image filename in `file_path`.
                      Defaults to "images" to match the standard Nerfstudio layout.
        aabb_scale: scene bounding box scale hint for Nerfstudio. 2.0 covers our
                    asset-normalized [-0.5, 0.5]³ space with comfortable margin.

    Returns the final path of the written .json file.
    """
    output_path = Path(output_path)
    image_names = list(image_names)
    w2c_matrices = np.asarray(w2c_matrices)
    if w2c_matrices.ndim != 3 or w2c_matrices.shape[1:] != (4, 4):
        raise ValueError(
            f"w2c_matrices must be shape (N, 4, 4), got {w2c_matrices.shape}"
        )
    if len(w2c_matrices) != len(image_names):
        raise ValueError(
            f"matrices ({len(w2c_matrices)}) and names ({len(image_names)}) length mismatch"
        )

    multi_cam = not isinstance(intrinsics, dict)
    if multi_cam:
        intr_list = list(intrinsics)
        if len(intr_list) != len(image_names):
            raise ValueError(
                f"intrinsics list length {len(intr_list)} != n_images {len(image_names)}"
            )
        # Top-level fields use first frame as a fallback for tools that don't
        # honor per-frame overrides (Nerfstudio's standard path DOES honor them).
        top_intr = intr_list[0]
    else:
        intr_list = None
        top_intr = intrinsics

    frames = []
    for idx, (w2c, name) in enumerate(zip(w2c_matrices, image_names)):
        c2w_opengl = opencv_w2c_to_opengl_c2w(w2c)
        frame = {
            "file_path": f"{image_subdir}/{name}" if image_subdir else name,
            "transform_matrix": c2w_opengl.tolist(),
        }
        if multi_cam:
            per = intr_list[idx]
            frame["fl_x"] = float(per["fx"])
            frame["fl_y"] = float(per["fy"])
            frame["cx"] = float(per["cx"])
            frame["cy"] = float(per["cy"])
            frame["w"] = int(per["width"])
            frame["h"] = int(per["height"])
        frames.append(frame)

    payload = {
        "camera_model": "OPENCV",
        "fl_x": float(top_intr["fx"]),
        "fl_y": float(top_intr["fy"]),
        "cx": float(top_intr["cx"]),
        "cy": float(top_intr["cy"]),
        "w": int(top_intr["width"]),
        "h": int(top_intr["height"]),
        "aabb_scale": int(aabb_scale),
        "frames": frames,
    }

    if output_path.suffix.lower() == ".json":
        final_path = output_path
        final_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        output_path.mkdir(parents=True, exist_ok=True)
        final_path = output_path / "transforms.json"

    with final_path.open("w") as f:
        json.dump(payload, f, indent=4)

    return final_path
