#!/usr/bin/env python
"""
Patch existing COLMAP sparse/0/*.txt files to import cleanly into PostShot v1.1.

Problem:
    PostShot v1.1 bounds-checks each point3D's track against the target image's
    POINTS2D vector length. Our older writer put `image_id=1, point2d_idx=i`
    in every track but left images.txt POINTS2D lines empty → PostShot does
    image_1.points2D[i] on an empty vector and throws "invalid vector subscript".

Fix:
    Distribute the existing point IDs across all images via modulo assignment:
      point id i (1-indexed)  →  image  ((i-1) % n_images) + 1
                              at  point2d_idx (i-1) // n_images
    images.txt POINTS2D lines get the matching (cx, cy, point3d_id) entries
    so each track has a valid backref. No re-rendering required.

Usage:
    python scripts/fix_colmap_postshot11.py <root> [<root> ...]

    where each <root> is either:
      - a sparse/0/ directory containing cameras.txt/images.txt/points3D.txt
      - any ancestor directory — we recursively find sparse/0/ subdirs

Examples:
    # Patch every chess prod dataset
    python scripts/fix_colmap_postshot11.py ~/projects/image_to_splat/chess_prod_pixal3d
    # Patch one specific dataset
    python scripts/fix_colmap_postshot11.py ~/projects/image_to_splat/chess_prod_pixal3d/E6ZJMwk/USER_Three_Large_Tent_120v_2048px
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


def find_sparse_dirs(root: Path) -> list[Path]:
    """Return every sparse/0 subdirectory under root (including root itself)."""
    if (root / "cameras.txt").exists() and (root / "images.txt").exists():
        return [root]
    found = []
    for p in root.rglob("sparse/0"):
        if (p / "cameras.txt").exists() and (p / "images.txt").exists():
            found.append(p)
    return sorted(set(found))


def parse_cameras_txt(path: Path) -> dict:
    """Return intrinsics dict for the FIRST (and only) camera in cameras.txt."""
    for line in path.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        # CAMERA_ID MODEL WIDTH HEIGHT fx fy cx cy
        if parts[1] != "PINHOLE":
            raise ValueError(f"expected PINHOLE camera, got {parts[1]} in {path}")
        return {
            "camera_id": int(parts[0]),
            "width": int(parts[2]),
            "height": int(parts[3]),
            "fx": float(parts[4]),
            "fy": float(parts[5]),
            "cx": float(parts[6]),
            "cy": float(parts[7]),
        }
    raise ValueError(f"no camera entry in {path}")


def parse_images_txt(path: Path) -> list[tuple[int, list[str], str]]:
    """Return list of (image_id, [pose_tokens...], pose_line_raw).

    Pose tokens are: image_id, qw, qx, qy, qz, tx, ty, tz, camera_id, name.
    We DROP whatever POINTS2D content was on the second line per image and
    return the count of images for downstream replacement.
    """
    images: list[tuple[int, list[str], str]] = []
    raw_lines = path.read_text().splitlines()
    i = 0
    # Skip header (lines starting with #)
    while i < len(raw_lines) and raw_lines[i].startswith("#"):
        i += 1
    # Pose lines come in pairs: pose_line, points2d_line (often blank).
    while i < len(raw_lines):
        line = raw_lines[i].strip()
        if not line:
            i += 1
            continue
        tokens = line.split()
        if len(tokens) < 10:
            raise ValueError(f"malformed image pose line in {path}: {line!r}")
        image_id = int(tokens[0])
        images.append((image_id, tokens, raw_lines[i]))
        # Skip the POINTS2D line (which we're about to replace)
        i += 1
        if i < len(raw_lines) and not raw_lines[i].startswith("#"):
            i += 1
    return images


def parse_points3d_txt(path: Path) -> list[list[str]]:
    """Return list of token lists for each point line (first 8 tokens kept).

    We keep ID, X, Y, Z, R, G, B, ERROR and discard the (image_id, point2d_idx)
    track since we're going to recompute it.
    """
    points: list[list[str]] = []
    for line in path.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        tokens = line.split()
        if len(tokens) < 8:
            raise ValueError(f"short point line in {path}: {line!r}")
        points.append(tokens[:8])  # ID X Y Z R G B ERROR
    return points


def rewrite_sparse_dir(sparse_dir: Path) -> tuple[int, int]:
    """Patch images.txt + points3D.txt in place. Returns (n_images, n_points)."""
    cam = parse_cameras_txt(sparse_dir / "cameras.txt")
    images = parse_images_txt(sparse_dir / "images.txt")
    points = parse_points3d_txt(sparse_dir / "points3D.txt")

    n_images = len(images)
    n_points = len(points)
    cx = cam["cx"]
    cy = cam["cy"]

    # Modulo distribution: point id (1-indexed) i  →  image ((i-1) % n_imgs)+1
    points_per_image: list[list[int]] = [[] for _ in range(n_images)]
    for pid_1based in range(1, n_points + 1):
        target = (pid_1based - 1) % n_images
        points_per_image[target].append(pid_1based)

    # --- rewrite images.txt ---
    mean_obs = n_points / n_images if n_images else 0
    img_lines = [
        "# Image list with two lines of data per image:\n",
        "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n",
        "#   POINTS2D[] as (X, Y, POINT3D_ID)\n",
        f"# Number of images: {n_images}, mean observations per image: {mean_obs:.1f}\n",
    ]
    # Sort images by image_id ascending so output is canonical
    images_sorted = sorted(images, key=lambda x: x[0])
    for image_id, tokens, _raw in images_sorted:
        img_lines.append(_raw.rstrip() + "\n")
        observed = points_per_image[image_id - 1]
        if observed:
            triples = [f"{cx:.3f} {cy:.3f} {pid}" for pid in observed]
            img_lines.append(" ".join(triples) + "\n")
        else:
            img_lines.append("\n")
    (sparse_dir / "images.txt").write_text("".join(img_lines))

    # --- rewrite points3D.txt ---
    pt_lines = [
        "# 3D point list with one line of data per point:\n",
        "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n",
        f"# Number of points: {n_points}, mean track length: 1.0\n",
    ]
    for i, tokens in enumerate(points):
        image_id = (i % n_images) + 1
        point2d_idx = i // n_images
        pt_lines.append(" ".join(tokens) + f" {image_id} {point2d_idx}\n")
    (sparse_dir / "points3D.txt").write_text("".join(pt_lines))

    return n_images, n_points


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("roots", nargs="+", type=Path,
                   help="One or more dataset roots or sparse/0 directories.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    sparse_dirs: list[Path] = []
    for root in args.roots:
        if not root.exists():
            print(f"WARN: {root} does not exist, skipping")
            continue
        sparse_dirs.extend(find_sparse_dirs(root))

    if not sparse_dirs:
        print("No sparse/0 directories found under given roots.")
        return 1

    print(f"Found {len(sparse_dirs)} sparse model(s) to patch:")
    for sd in sparse_dirs:
        print(f"  {sd}")
    print()

    for sd in sparse_dirs:
        try:
            n_imgs, n_pts = rewrite_sparse_dir(sd)
            print(f"  ✓ {sd}  ({n_imgs} images, {n_pts} points)")
        except Exception as e:
            print(f"  ✗ {sd}  FAILED: {e}")
            return 2
    print(f"\nDone. {len(sparse_dirs)} sparse model(s) patched.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
