"""
COLMAP text-format dataset writer.

Writes the three-file sparse model that splat trainers consume:

    output_dir/sparse/0/
        ├── cameras.txt    # intrinsics — one shared PINHOLE camera for all images
        ├── images.txt     # per-image pose (qvec + tvec) + name
        └── points3D.txt   # empty placeholder (header only); trainer init points

Format references:
    https://colmap.github.io/format.html#sparse-reconstruction
    https://colmap.github.io/format.html#cameras-txt
    https://colmap.github.io/format.html#images-txt
    https://colmap.github.io/format.html#points3d-txt

Critical convention: COLMAP poses encode WORLD-TO-CAMERA in OpenCV convention.
Our cameras module produces W2C-OpenCV directly, so no convention swap here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from .cameras import w2c_to_colmap_qvec_tvec

CAMERA_ID = 1  # we use a single shared camera for the whole dataset


def _format_cameras_txt(intrinsics: dict) -> str:
    """Header + one line for the single shared PINHOLE camera."""
    header = (
        "# Camera list with one line of data per camera:\n"
        "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
        "# Number of cameras: 1\n"
    )
    line = (
        f"{CAMERA_ID} PINHOLE "
        f"{intrinsics['width']} {intrinsics['height']} "
        f"{intrinsics['fx']:.6f} {intrinsics['fy']:.6f} "
        f"{intrinsics['cx']:.6f} {intrinsics['cy']:.6f}\n"
    )
    return header + line


def _format_images_txt(
    w2c_matrices: np.ndarray,
    image_names: list[str],
    intrinsics: dict,
    n_points: int = 0,
) -> str:
    """Two lines per image: pose line + POINTS2D line.

    For consistency with points3D.txt's fake tracks, we distribute n_points
    evenly across all images via modulo assignment. Image (point_idx % n_imgs)+1
    "observes" each point at local index point_idx//n_imgs. Strict COLMAP
    parsers (PostShot) reject the dataset if points3D claims observations
    that images.txt doesn't echo — distribution makes them agree.

    The 2D coords are (cx, cy) for every entry — synthetic data has no SIFT
    matches, but the format requires *some* coord. Splat trainers ignore them.
    """
    if len(w2c_matrices) != len(image_names):
        raise ValueError(
            f"matrices ({len(w2c_matrices)}) and names ({len(image_names)}) length mismatch"
        )
    n_imgs = len(image_names)
    cx = intrinsics["cx"]
    cy = intrinsics["cy"]

    # Per-image POINTS2D: list of (point3d_id) where (point3d_id - 1) % n_imgs == image_idx
    # point3d_id is 1-indexed in COLMAP.
    points_per_image: list[list[int]] = [[] for _ in range(n_imgs)]
    for pid_1based in range(1, n_points + 1):
        target_image_idx = (pid_1based - 1) % n_imgs
        points_per_image[target_image_idx].append(pid_1based)

    mean_obs = n_points / n_imgs if n_imgs else 0
    header = (
        "# Image list with two lines of data per image:\n"
        "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n"
        "#   POINTS2D[] as (X, Y, POINT3D_ID)\n"
        f"# Number of images: {n_imgs}, mean observations per image: {mean_obs:.1f}\n"
    )
    lines = []
    for i, (w2c, name) in enumerate(zip(w2c_matrices, image_names), start=1):
        qvec, tvec = w2c_to_colmap_qvec_tvec(w2c)
        pose_line = (
            f"{i} "
            f"{qvec[0]:.9f} {qvec[1]:.9f} {qvec[2]:.9f} {qvec[3]:.9f} "
            f"{tvec[0]:.9f} {tvec[1]:.9f} {tvec[2]:.9f} "
            f"{CAMERA_ID} {name}\n"
        )
        lines.append(pose_line)
        # POINTS2D line: (x, y, point3d_id) triples space-separated; empty if no points.
        observed = points_per_image[i - 1]
        if observed:
            tokens = []
            for pid in observed:
                tokens.append(f"{cx:.3f} {cy:.3f} {pid}")
            lines.append(" ".join(tokens) + "\n")
        else:
            lines.append("\n")
    return header + "".join(lines)


def _format_points3d_txt(
    points: np.ndarray | None = None,
    colors: np.ndarray | None = None,
    n_images: int = 1,
) -> str:
    """Write a points3D file. Empty if points is None, populated otherwise.

    PostShot (and some other splat trainers) REQUIRES non-empty points3D for COLMAP
    import — empty header-only files get rejected with "Import contains no 3D points data".
    Seed with mesh vertex positions to give the trainer a real geometric init.

    Track distribution: each point (1-indexed id i) is assigned to image
    ((i-1) % n_images) + 1 at local point2d_idx = (i-1) // n_images. This MUST
    match _format_images_txt's POINTS2D distribution — PostShot v1.1 bounds-checks
    each track's point2d_idx against that image's POINTS2D length and throws
    "invalid vector subscript" on mismatch.

    Args:
        points: optional (N, 3) array of XYZ positions in world coords.
        colors: optional (N, 3) uint8 array of RGB colors. Defaults to gray.
        n_images: number of images in the dataset — needed for consistent track
                  distribution. Required if `points` is non-empty.
    """
    header = (
        "# 3D point list with one line of data per point:\n"
        "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n"
    )
    if points is None or len(points) == 0:
        return header + "# Number of points: 0, mean track length: 0\n"

    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must be (N, 3), got {points.shape}")
    if n_images < 1:
        raise ValueError(f"n_images must be >= 1, got {n_images}")
    if colors is None:
        colors = np.full((len(points), 3), 128, dtype=np.uint8)
    else:
        colors = np.asarray(colors, dtype=np.uint8)
        if colors.shape != (len(points), 3):
            raise ValueError(f"colors must be (N, 3) uint8, got {colors.shape}")

    header += f"# Number of points: {len(points)}, mean track length: 1.0\n"
    lines = []
    for i, (p, c) in enumerate(zip(points, colors)):
        image_id = (i % n_images) + 1
        point2d_idx = i // n_images
        lines.append(
            f"{i+1} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f} "
            f"{c[0]} {c[1]} {c[2]} 1.0 {image_id} {point2d_idx}\n"
        )
    return header + "".join(lines)


def write_colmap_dataset(
    output_dir: str | Path,
    w2c_matrices: np.ndarray,
    image_names: list[str] | Iterable[str],
    intrinsics: dict,
    points: np.ndarray | None = None,
    point_colors: np.ndarray | None = None,
) -> Path:
    """Write a COLMAP sparse model under <output_dir>/sparse/0/.

    Args:
        output_dir: dataset root. Final structure is `<output_dir>/sparse/0/{cameras,images,points3D}.txt`.
        w2c_matrices: (N, 4, 4) OpenCV world-to-camera matrices (as produced by cameras.look_at).
        image_names: filenames of the N rendered images (without directory prefix —
                     these get joined with `images/` by the splat trainer).
        intrinsics: dict with fx, fy, cx, cy, width, height (as produced by
                    cameras.intrinsics_from_fov).
        points: optional (M, 3) array of init point positions in world coords.
                PostShot rejects empty points3D ("Import contains no 3D points data"),
                so for PostShot/COLMAP-importing trainers, pass mesh vertices here.
        point_colors: optional (M, 3) uint8 RGB colors per point. Defaults to gray.

    Returns the sparse model directory path.
    """
    output_dir = Path(output_dir)
    image_names = list(image_names)
    w2c_matrices = np.asarray(w2c_matrices)
    if w2c_matrices.ndim != 3 or w2c_matrices.shape[1:] != (4, 4):
        raise ValueError(
            f"w2c_matrices must be shape (N, 4, 4), got {w2c_matrices.shape}"
        )

    sparse_dir = output_dir / "sparse" / "0"
    sparse_dir.mkdir(parents=True, exist_ok=True)

    n_points = 0 if points is None else len(points)
    n_images = len(image_names)
    (sparse_dir / "cameras.txt").write_text(_format_cameras_txt(intrinsics))
    (sparse_dir / "images.txt").write_text(
        _format_images_txt(w2c_matrices, image_names, intrinsics, n_points=n_points)
    )
    (sparse_dir / "points3D.txt").write_text(
        _format_points3d_txt(points, point_colors, n_images=n_images)
    )

    return sparse_dir
