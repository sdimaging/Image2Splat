#!/usr/bin/env python
"""Channel-isolation diagnostic for the metal-faceting artifact.

Runs ONE image through Pixal3D at a given tier+seed, renders ONE view, and
saves every intermediate channel the PBR renderer produces as separate PNGs:

    shaded.png       final composite (what the probe normally saves)
    base_color.png   albedo from the texture SLat decode      ← suspect #1
    roughness.png    roughness channel                        ← suspect #2
    metallic.png     metallic channel
    normal.png       camera-space shading normals (post smooth-normals patch)
    clay.png         (1 - SSAO) — pure geometry occlusion     ← suspect #3
    mask.png         coverage

Whichever channel shows the polygonal/patchy plateaus is the artifact source:
  - base_color/roughness → texture decoder content → fixable via attr-smooth
    (3D bilateral on voxel attrs) without retraining
  - clay → SSAO over chorded geometry → tune/disable SSAO intensity
  - normal → shading normals still faceted → smooth-normals patch not active

USAGE (needs the GPU — don't run while the daemon is mid-batch):
    ~/miniconda3/envs/trellis2/bin/python scripts/diagnose_facets.py \
        <input_image> [--tier 5] [--seed 222] [--view 129] [--resolution 1024] \
        [--out-dir /tmp/facet_diag]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# Normally set inside the backbone init; needed on the --mesh-cache fast
# path too (HDRI EXR loading via OpenCV).
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import numpy as np
from PIL import Image

from src.backbone import Backbone
from src.cameras import generate_camera_trajectory

SAMPLER_TIERS = {
    "1": ("Default",  12, 7.5, 1.1),
    "2": ("Subtle",   13, 7.6, 1.2),
    "3": ("Balanced", 14, 7.7, 1.25),
    "4": ("Refined",  15, 7.6, 1.2),
    "5": ("Sculpted", 15, 8.0, 1.5),
}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input_image", type=Path)
    p.add_argument("--tier", choices=list(SAMPLER_TIERS), default="5",
                   help="Sampler tier (pick the tier of a KNOWN-BAD probe cell).")
    p.add_argument("--seed", type=int, default=222,
                   help="Seed of the known-bad cell.")
    p.add_argument("--view", type=int, default=129)
    p.add_argument("--resolution", type=int, default=1024)
    p.add_argument("--max-faces", type=int, default=16_000_000)
    p.add_argument("--out-dir", type=Path, default=Path("/tmp/facet_diag"))
    p.add_argument("--ssao-compare", action="store_true",
                   help="After normal render (PIXAL3D_SSAO_INTENSITY or 0.8 default), "
                        "re-render the same mesh at SSAO=1.5 for direct A/B comparison. "
                        "Saves old-ssao/ subdir alongside the main output.")
    p.add_argument("--geo-compare", action="store_true",
                   help="Render the mesh as-is, then apply plateau_band_stop geometry "
                        "smoothing (mesh_smooth.py) and re-render — direct A/B of the "
                        "geometry fix on one inference run. Saves geo_smooth/ subdir + "
                        "GEO_AB_comparison.png composite + plateau metrics.")
    p.add_argument("--geo-alpha", type=float, default=1.0)
    p.add_argument("--geo-passes", type=int, default=1)
    p.add_argument("--mesh-cache", type=Path, default=None,
                   help="torch.save/load the generated mesh here. If the file "
                        "exists, inference is SKIPPED (render-only iteration "
                        "~30s instead of ~9min). Delete the file to regenerate.")
    args = p.parse_args()

    if not args.input_image.is_file():
        print(f"FAIL: input not found: {args.input_image}")
        return 1

    name, steps, shape_g, tex_g = SAMPLER_TIERS[args.tier]
    print(f"tier {args.tier} ({name})  steps={steps} shape={shape_g} tex={tex_g}  seed={args.seed}")

    import torch
    device = "cuda"
    if args.mesh_cache and args.mesh_cache.exists():
        print(f"loading cached mesh from {args.mesh_cache} (skipping inference)...")
        repo_dir = Path.home() / "projects/Pixal3D"
        sys.path.insert(0, str(repo_dir))   # MeshWithVoxel class for unpickle
        mesh = torch.load(args.mesh_cache, map_location=device, weights_only=False)
        print(f"mesh: {mesh.vertices.shape[0]:,}v / {mesh.faces.shape[0]:,}f (cached)")
    else:
        t0 = time.time()
        print("loading backbone...")
        bb = Backbone(
            name="pixal3d",
            pixal3d_pipeline_type="1536_cascade",
            max_num_tokens=131072,
            sparse_structure_sampler_params={"steps": steps, "guidance_strength": float(shape_g)},
            shape_slat_sampler_params={"steps": steps, "guidance_strength": float(shape_g)},
            tex_slat_sampler_params={"steps": steps, "guidance_strength": float(tex_g)},
        ).load()
        print(f"backbone loaded ({time.time()-t0:.0f}s)")

        image = Image.open(args.input_image)
        print("running inference...")
        t1 = time.time()
        mesh = bb.run(image, seed=args.seed)
        n_faces = int(mesh.faces.shape[0])
        print(f"mesh: {mesh.vertices.shape[0]:,}v / {n_faces:,}f  ({time.time()-t1:.0f}s)")
        if n_faces > args.max_faces:
            mesh.simplify(target=args.max_faces, verbose=False)
            print(f"decimated: {mesh.vertices.shape[0]:,}v / {mesh.faces.shape[0]:,}f")
        repo_dir = bb.repo_dir
        if args.mesh_cache:
            torch.save(mesh, args.mesh_cache)
            print(f"mesh cached to {args.mesh_cache}")

    # Camera: same Fibonacci rig as the daemon, single view
    up = np.array([0, 0, 1.0])
    w2c = generate_camera_trajectory(200, radius=2.0, up=up)[args.view:args.view + 1]

    # Render via the pixal3d renderer directly so we get ALL channels
    sys.path.insert(0, str(repo_dir))
    from pixal3d.utils import render_utils  # type: ignore

    extrinsics = [torch.tensor(m, dtype=torch.float32, device=device) for m in w2c]
    # normalized intrinsics for fov=40 (matches daemon default pre-adaptive)
    fov_rad = float(np.radians(40.0))
    focal = 0.5 / np.tan(fov_rad / 2.0)
    intr = torch.tensor([[focal, 0, 0.5], [0, focal, 0.5], [0, 0, 1]],
                        dtype=torch.float32, device=device)
    intrinsics = [intr]

    print("rendering all channels...")
    # No envmap arg → renderer still produces base_color/roughness/normal/clay.
    # With envmap we get the production look; load it like the daemon does.
    from src.hdri_io import read_exr_rgb
    try:
        from pixal3d.renderers import EnvMap  # type: ignore
        hdri = Path.home() / "projects/TRELLIS.2/assets/hdri/USER_Alt2.exr"
        env_img = read_exr_rgb(hdri) * 2.5
        envmap = EnvMap(torch.tensor(env_img, dtype=torch.float32, device=device))
        kwargs = {"envmap": envmap}
    except Exception as e:
        print(f"(envmap load failed, rendering without: {e})")
        kwargs = {}

    def _render_and_save(out_dir: Path, ssao_override: float = None):
        if ssao_override is not None:
            os.environ["PIXAL3D_SSAO_INTENSITY"] = str(ssao_override)
        result = render_utils.render_frames(
            mesh, extrinsics, intrinsics,
            {"resolution": args.resolution, "bg_color": (0.1, 0.1, 0.1)},
            verbose=False, **kwargs,
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        saved = []
        for key, frames in result.items():
            arr = frames[0]
            Image.fromarray(arr).save(out_dir / f"{key}.png")
            saved.append(key)
        if ssao_override is not None:
            del os.environ["PIXAL3D_SSAO_INTENSITY"]
        return saved

    print("rendering all channels (new SSAO)...")
    saved = _render_and_save(args.out_dir)
    print(f"\nsaved channels: {saved}")
    print(f"output dir: {args.out_dir}")

    if args.ssao_compare:
        print("\nrendering comparison at SSAO=1.5 (old default)...")
        old_dir = args.out_dir / "old_ssao"
        _render_and_save(old_dir, ssao_override=1.5)
        print(f"old-SSAO output: {old_dir}")
        # Side-by-side composite
        _make_ssao_comparison(args.out_dir, old_dir, args.out_dir / "AB_comparison.png")

    if args.geo_compare:
        from mesh_smooth import plateau_band_stop
        print(f"\napplying plateau_band_stop (alpha={args.geo_alpha}, "
              f"passes={args.geo_passes})...")
        t2 = time.time()
        orig_verts = mesh.vertices.clone()
        new_verts, gstats = plateau_band_stop(
            mesh.vertices, mesh.faces,
            alpha=args.geo_alpha, passes=args.geo_passes)
        try:
            mesh.vertices = new_verts
        except (AttributeError, TypeError):
            mesh.vertices.copy_(new_verts)
        # Anchor voxel-attr texture sampling to the pre-displacement
        # surface (renderer reads mesh.attr_vertices if present).
        mesh.attr_vertices = orig_verts
        print(f"smoothed in {time.time()-t2:.1f}s  stats: {gstats}")
        geo_dir = args.out_dir / "geo_smooth"
        _render_and_save(geo_dir)
        print(f"geo-smoothed output: {geo_dir}")
        _make_geo_comparison(args.out_dir, geo_dir,
                             args.out_dir / "GEO_AB_comparison.png")

    print("\nLook for the polygonal patches in each channel:")
    print("  base_color / roughness → texture decode content (attr-smooth fixable)")
    print("  clay                   → SSAO over chorded geometry")
    print("  normal                 → shading normals (patch not active?)")
    return 0


def _plateau_metrics(d: Path) -> dict:
    """Bright/dark plateau contrast on the foreground of a channel dump."""
    shaded = np.array(Image.open(d / "shaded.png").convert("RGB")).astype(float) / 255.0
    mask = np.array(Image.open(d / "mask.png").convert("L")).astype(float) / 255.0 > 0.5
    lum = 0.2126 * shaded[..., 0] + 0.7152 * shaded[..., 1] + 0.0722 * shaded[..., 2]
    fg = lum[mask]
    return {
        "p90": float(np.percentile(fg, 90)),
        "p10": float(np.percentile(fg, 10)),
        "ratio": float(np.percentile(fg, 90) / (np.percentile(fg, 10) + 1e-6)),
        "std": float(fg.std()),
    }


def _make_geo_comparison(orig_dir: Path, geo_dir: Path, out_path: Path):
    from PIL import ImageDraw
    o_shaded = Image.open(orig_dir / "shaded.png").convert("RGB")
    g_shaded = Image.open(geo_dir / "shaded.png").convert("RGB")
    o_norm = Image.open(orig_dir / "normal.png").convert("RGB")
    g_norm = Image.open(geo_dir / "normal.png").convert("RGB")

    def label(img, text):
        out = img.copy()
        draw = ImageDraw.Draw(out)
        draw.rectangle([0, 0, img.width, 44], fill=(0, 0, 0))
        draw.text((6, 6), text, fill=(255, 255, 255))
        return out

    W, H = o_shaded.size
    combined = Image.new("RGB", (W * 2, H * 2))
    combined.paste(label(o_shaded, "SHADED  original geometry"), (0, 0))
    combined.paste(label(g_shaded, "SHADED  plateau band-stop"), (W, 0))
    combined.paste(label(o_norm, "NORMAL  original geometry"), (0, H))
    combined.paste(label(g_norm, "NORMAL  plateau band-stop"), (W, H))
    combined.save(out_path)
    print(f"GEO A/B composite: {out_path}  ({combined.width}x{combined.height})")

    m_o = _plateau_metrics(orig_dir)
    m_g = _plateau_metrics(geo_dir)
    print(f"  original: lum p90/p10 ratio={m_o['ratio']:.2f}  std={m_o['std']:.3f}")
    print(f"  smoothed: lum p90/p10 ratio={m_g['ratio']:.2f}  std={m_g['std']:.3f}")


def _make_ssao_comparison(new_dir: Path, old_dir: Path, out_path: Path):
    from PIL import ImageDraw
    new_shaded = Image.open(new_dir / "shaded.png").convert("RGB")
    old_shaded = Image.open(old_dir / "shaded.png").convert("RGB")
    new_clay   = Image.open(new_dir / "clay.png").convert("L").convert("RGB")
    old_clay   = Image.open(old_dir / "clay.png").convert("L").convert("RGB")

    def label(img, text):
        out = img.copy()
        draw = ImageDraw.Draw(out)
        draw.rectangle([0, 0, img.width, 44], fill=(0, 0, 0))
        draw.text((6, 6), text, fill=(255, 255, 255))
        return out

    row1 = [label(old_shaded, "SHADED  SSAO=1.5 (OLD)"),
            label(new_shaded, "SHADED  SSAO=0.8 (NEW)")]
    row2 = [label(old_clay,   "CLAY  SSAO=1.5 (OLD)"),
            label(new_clay,   "CLAY  SSAO=0.8 (NEW)")]

    W, H = new_shaded.size
    combined = Image.new("RGB", (W * 2, H * 2))
    for i, im in enumerate(row1): combined.paste(im, (i * W, 0))
    for i, im in enumerate(row2): combined.paste(im, (i * W, H))
    combined.save(out_path)
    print(f"A/B composite: {out_path}  ({combined.width}x{combined.height})")

    # Numeric summary
    mask = np.array(Image.open(new_dir / "mask.png").convert("L")).astype(float) / 255.0 > 0.5
    for tag, clay_path in [("SSAO=1.5", old_dir / "clay.png"),
                             ("SSAO=0.8", new_dir / "clay.png")]:
        clay = np.array(Image.open(clay_path).convert("L")).astype(float) / 255.0
        print(f"  {tag}: clay_fg={clay[mask].mean():.3f}  (higher=less AO darkening)")


if __name__ == "__main__":
    raise SystemExit(main())
