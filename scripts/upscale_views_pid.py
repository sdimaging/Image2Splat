#!/usr/bin/env python
"""
PiD 4× super-resolution wrapper for a rendered Pixal3D splat dataset.

Takes a Pixal3D-rendered dataset directory (200 views @ Npx + COLMAP/Nerfstudio
metadata) and produces a sibling dataset with each view upscaled 4× via NVIDIA's
PiD (Pixel Diffusion Decoder). Pipeline per view: image → VAE encode → PiD decode
at 4× → upscaled PNG.

This enables "Strategy A": render via the daemon at 1024px (much faster than
3000px native), then PiD-upscale all 200 views to 4096px. Net wall time is
typically lower AND output resolution is higher than the legacy 3000px-native
path — at the cost of view-consistency risk from PiD's diffusion prior (see
below).

WHEN TO USE
    After the daemon has finished rendering a dataset. Point this script at the
    dataset folder, get a sibling folder with `_pid4x` suffix containing the
    upscaled views + a COLMAP/Nerfstudio metadata block with 4×-scaled
    intrinsics. PostShot / gsplat / Nerfstudio can train directly from the
    upscaled folder.

PRE-REQS
    1. Clone PiD locally:
        git clone https://github.com/nv-tlabs/PiD.git ~/projects/PiD
    2. Download checkpoints into the repo:
        cd ~/projects/PiD && hf download nvidia/PiD --local-dir . --include "checkpoints/*"
    3. PiD env (separate conda env recommended):
        cd ~/projects/PiD && conda env create -f environment.yml && conda activate pid && pip install -e .

USAGE
    python scripts/upscale_views_pid.py \\
        <dataset_dir> \\
        [--pid-repo ~/projects/PiD] \\
        [--pid-python /path/to/pid/env/python] \\
        [--vae flux2] \\
        [--ckpt-type 2kto4k] \\
        [--scale 4] \\
        [--steps 4] \\
        [--cfg-scale 2.75] \\
        [--seed 42] \\
        [--dry-run]

    <dataset_dir> must contain:
        images/*.png
        sparse/0/{cameras,images,points3D}.txt
        transforms.json

VIEW-CONSISTENCY KNOBS
    --seed (default 42)   Fixed seed across all frames → same diffusion noise
                          per frame, maximizes consistency for splat training.
    --steps (default 4)   The distilled checkpoints work in 4 steps. Lower step
                          counts give MORE consistency (less room for variation)
                          but less fine detail.

This is an experimental upgrade path. Validate by training a splat from the
upscaled folder and comparing to a baseline trained from the native render.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


VALID_VAES = ("flux2", "flux", "sd3", "zimage", "zimage_turbo", "dinov2", "siglip")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("dataset_dir", type=Path,
                   help="Path to the source rendered dataset "
                        "(must contain images/ + transforms.json + sparse/0/).")
    p.add_argument("--pid-repo", type=Path,
                   default=Path.home() / "projects" / "PiD",
                   help="Path to the cloned PiD repo. Required to invoke "
                        "pid._src.inference.from_clean_<vae>.")
    p.add_argument("--pid-python", type=str, default=sys.executable,
                   help="Python interpreter that has PiD installed. Default: "
                        "current python (works if PiD is in this env).")
    p.add_argument("--vae", choices=VALID_VAES, default="flux2",
                   help="Backbone VAE. flux2 (128-ch BN VAE) has the highest "
                        "fidelity round-trip. flux/sd3 are 16-ch alternatives.")
    p.add_argument("--ckpt-type", choices=["2k", "2kto4k"], default="2kto4k",
                   help="2k = trained at 2048px (use for 512→2048). "
                        "2kto4k = multi-res cascade (use for 1024→4096).")
    p.add_argument("--scale", type=int, default=4,
                   help="Upscale factor. PiD checkpoints support 4× (or 8× for "
                        "siglip backbone). Affects output filename + intrinsics.")
    p.add_argument("--steps", type=int, default=4,
                   help="PiD inference steps. Distilled checkpoints work in 4. "
                        "Higher = more detail, less consistency.")
    p.add_argument("--cfg-scale", type=float, default=2.75,
                   help="PiD classifier-free guidance scale.")
    p.add_argument("--seed", type=int, default=42,
                   help="Fixed seed across all frames — maximizes view "
                        "consistency for splat training. Same diffusion noise "
                        "applied to every frame.")
    p.add_argument("--output-suffix", type=str, default="_pid{scale}x",
                   help="Suffix appended to dataset folder name for output. "
                        "{scale} is replaced with --scale.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the plan + the PiD CLI invocation, then exit "
                        "without running PiD or moving files.")
    return p.parse_args()


def validate_source(dataset_dir: Path) -> tuple[list[Path], dict]:
    """Confirm the source dataset has the structure we expect.

    Returns (sorted_image_paths, transforms_json_dict).
    """
    if not dataset_dir.is_dir():
        sys.exit(f"FAIL: not a directory: {dataset_dir}")
    images_dir = dataset_dir / "images"
    if not images_dir.is_dir():
        sys.exit(f"FAIL: no images/ under {dataset_dir}")
    images = sorted(p for p in images_dir.iterdir()
                    if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    if not images:
        sys.exit(f"FAIL: no PNG/JPG images in {images_dir}")
    cameras_txt = dataset_dir / "sparse" / "0" / "cameras.txt"
    if not cameras_txt.exists():
        sys.exit(f"FAIL: no cameras.txt at {cameras_txt}")
    tj_path = dataset_dir / "transforms.json"
    if not tj_path.exists():
        sys.exit(f"FAIL: no transforms.json at {tj_path}")
    transforms = json.loads(tj_path.read_text())
    return images, transforms


def build_manifest(images: list[Path], manifest_path: Path) -> None:
    """Write a JSONL manifest of {"image": <abs_path>} per line for PiD's CLI."""
    with manifest_path.open("w") as f:
        for img in images:
            f.write(json.dumps({"image": str(img.resolve())}) + "\n")


def find_pid_output(pid_output_dir: Path) -> Path:
    """Locate the PiD output subdir.

    PiD writes to <output_dir>/<experiment_tag>/sigma_0.0/<sample_id>.png — the
    experiment tag varies by checkpoint, so we glob for the matching pattern.
    """
    if not pid_output_dir.is_dir():
        sys.exit(f"FAIL: PiD output dir not created: {pid_output_dir}")
    candidates = list(pid_output_dir.glob("*/sigma_0.0"))
    candidates = [c for c in candidates if c.parent.name != "vae_decode"]
    if not candidates:
        sys.exit(f"FAIL: no PiD output subdir under {pid_output_dir} matching "
                 f"<tag>/sigma_0.0/. Inspect PiD logs for errors.")
    if len(candidates) > 1:
        # Multiple checkpoints in one run — shouldn't happen with our CLI args
        sys.exit(f"FAIL: multiple PiD output candidates: {candidates}")
    return candidates[0]


def rewrite_cameras_txt(src: Path, dst: Path, scale: int) -> None:
    """Copy cameras.txt scaling width/height/fx/fy/cx/cy by `scale`."""
    out_lines = []
    for line in src.read_text().splitlines():
        if not line or line.startswith("#"):
            out_lines.append(line)
            continue
        parts = line.split()
        # CAMERA_ID MODEL WIDTH HEIGHT fx fy cx cy
        cam_id, model = parts[0], parts[1]
        w, h = int(parts[2]) * scale, int(parts[3]) * scale
        fx, fy = float(parts[4]) * scale, float(parts[5]) * scale
        cx, cy = float(parts[6]) * scale, float(parts[7]) * scale
        out_lines.append(
            f"{cam_id} {model} {w} {h} {fx:.6f} {fy:.6f} {cx:.6f} {cy:.6f}"
        )
    dst.write_text("\n".join(out_lines) + "\n")


def rewrite_transforms_json(src: Path, dst: Path, scale: int) -> None:
    """Copy transforms.json scaling fl_x/fl_y/cx/cy/w/h by `scale`."""
    data = json.loads(src.read_text())
    for key in ("fl_x", "fl_y", "cx", "cy"):
        if key in data:
            data[key] = float(data[key]) * scale
    for key in ("w", "h"):
        if key in data:
            data[key] = int(data[key]) * scale
    dst.write_text(json.dumps(data, indent=4))


def main() -> int:
    args = parse_args()
    args.dataset_dir = args.dataset_dir.resolve()
    args.pid_repo = args.pid_repo.expanduser().resolve()

    # 1. Validate source
    images, _transforms = validate_source(args.dataset_dir)
    print(f"Source dataset: {args.dataset_dir}")
    print(f"  views found  : {len(images)}")
    print(f"  first image  : {images[0].name}")
    print(f"  last image   : {images[-1].name}")

    # 2. Plan output paths
    suffix = args.output_suffix.format(scale=args.scale)
    out_dataset = args.dataset_dir.parent / (args.dataset_dir.name + suffix)
    pid_output_dir = out_dataset / "_pid_raw"  # PiD's nested-subdir layout
    target_images_dir = out_dataset / "images"

    print(f"\nTarget dataset: {out_dataset}")
    print(f"  vae          : {args.vae}")
    print(f"  ckpt type    : {args.ckpt_type}")
    print(f"  scale        : {args.scale}×")
    print(f"  steps        : {args.steps}")
    print(f"  cfg          : {args.cfg_scale}")
    print(f"  seed         : {args.seed} (fixed across all views)")

    # 3. Validate PiD repo (skip in dry-run so plan-preview works without install)
    pid_module = f"pid._src.inference.from_clean_{args.vae}"
    if args.vae in ("zimage", "zimage_turbo"):
        # Per PiD README: from_clean_zimage* reuses flux's clean script
        pid_module = "pid._src.inference.from_clean_flux"
    if not args.dry_run and not args.pid_repo.is_dir():
        sys.exit(f"\nFAIL: --pid-repo not found at {args.pid_repo}\n"
                 f"  Clone: git clone https://github.com/nv-tlabs/PiD.git {args.pid_repo}")

    # 4. Build manifest
    if args.dry_run:
        manifest_path = out_dataset / "manifest.jsonl"
        print(f"\n[DRY RUN] would write manifest with {len(images)} entries to:")
        print(f"  {manifest_path}")
    else:
        out_dataset.mkdir(parents=True, exist_ok=True)
        manifest_path = out_dataset / "manifest.jsonl"
        build_manifest(images, manifest_path)
        print(f"\nManifest written: {manifest_path} ({len(images)} entries)")

    # 5. Build PiD CLI invocation
    cli = [
        args.pid_python, "-m", pid_module,
        "--manifest", str(manifest_path),
        "--output_dir", str(pid_output_dir),
        "--keep_input_size",
        "--scale", str(args.scale),
        "--pid_inference_steps", str(args.steps),
        "--cfg_scale", str(args.cfg_scale),
        "--pid_ckpt_type", args.ckpt_type,
        "--degrade_sigmas", "0.0",
        "--seed", str(args.seed),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{args.pid_repo}:{env.get('PYTHONPATH', '')}"

    print("\nPiD invocation:")
    print(f"  cwd = {args.pid_repo}")
    print(f"  PYTHONPATH = {args.pid_repo}:$PYTHONPATH")
    print(f"  cmd = {' '.join(cli)}")

    if args.dry_run:
        print("\n[DRY RUN] not running PiD; not rewriting metadata.")
        return 0

    # 6. Run PiD
    print(f"\nRunning PiD on {len(images)} views (this is the slow part)...")
    result = subprocess.run(cli, cwd=str(args.pid_repo), env=env)
    if result.returncode != 0:
        sys.exit(f"\nFAIL: PiD subprocess exited with code {result.returncode}")

    # 7. Locate PiD outputs + move into final images/ folder
    pid_results = find_pid_output(pid_output_dir)
    print(f"\nPiD wrote to: {pid_results}")
    target_images_dir.mkdir(parents=True, exist_ok=True)
    moved = 0
    for src_img in sorted(pid_results.glob("*.png")):
        # Sample IDs are the input image stems (e.g. "000") — preserve naming
        dst_img = target_images_dir / src_img.name
        shutil.move(str(src_img), str(dst_img))
        moved += 1
    print(f"  moved {moved} upscaled views → {target_images_dir}")
    if moved != len(images):
        print(f"  WARN: source had {len(images)} views, only {moved} upscaled — "
              "check PiD logs for per-frame failures")

    # 8. Rewrite COLMAP + Nerfstudio metadata with scaled intrinsics
    sparse_src = args.dataset_dir / "sparse" / "0"
    sparse_dst = out_dataset / "sparse" / "0"
    sparse_dst.mkdir(parents=True, exist_ok=True)
    rewrite_cameras_txt(sparse_src / "cameras.txt", sparse_dst / "cameras.txt",
                        args.scale)
    # images.txt + points3D.txt: poses + 3D points don't change with FOV scaling,
    # only intrinsics do. Direct copy.
    shutil.copy2(sparse_src / "images.txt", sparse_dst / "images.txt")
    shutil.copy2(sparse_src / "points3D.txt", sparse_dst / "points3D.txt")
    rewrite_transforms_json(args.dataset_dir / "transforms.json",
                            out_dataset / "transforms.json", args.scale)
    print(f"  rewrote metadata → {sparse_dst} + {out_dataset / 'transforms.json'}")

    # 9. Clean up the PiD nested output (VAE baseline, input dumps, etc.)
    # Keep them if user wants to compare; comment out the rmtree to retain.
    shutil.rmtree(pid_output_dir, ignore_errors=True)
    manifest_path.unlink(missing_ok=True)

    print(f"\n✓ DONE. Upscaled dataset at: {out_dataset}")
    print(f"  Train via PostShot/gsplat pointing at: {out_dataset}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
