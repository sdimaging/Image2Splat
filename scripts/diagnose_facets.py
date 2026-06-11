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
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

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
    args = p.parse_args()

    if not args.input_image.is_file():
        print(f"FAIL: input not found: {args.input_image}")
        return 1

    name, steps, shape_g, tex_g = SAMPLER_TIERS[args.tier]
    print(f"tier {args.tier} ({name})  steps={steps} shape={shape_g} tex={tex_g}  seed={args.seed}")

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

    # Camera: same Fibonacci rig as the daemon, single view
    import torch
    up = np.array([0, 0, 1.0])
    w2c = generate_camera_trajectory(200, radius=2.0, up=up)[args.view:args.view + 1]

    # Render via the pixal3d renderer directly so we get ALL channels
    sys.path.insert(0, str(bb.repo_dir))
    from pixal3d.utils import render_utils  # type: ignore

    extrinsics = [torch.tensor(m, dtype=torch.float32, device=bb.device) for m in w2c]
    # normalized intrinsics for fov=40 (matches daemon default pre-adaptive)
    fov_rad = float(np.radians(40.0))
    focal = 0.5 / np.tan(fov_rad / 2.0)
    intr = torch.tensor([[focal, 0, 0.5], [0, focal, 0.5], [0, 0, 1]],
                        dtype=torch.float32, device=bb.device)
    intrinsics = [intr]

    print("rendering all channels...")
    # No envmap arg → renderer still produces base_color/roughness/normal/clay.
    # With envmap we get the production look; load it like the daemon does.
    from src.hdri_io import read_exr_rgb
    try:
        from pixal3d.renderers import EnvMap  # type: ignore
        hdri = Path.home() / "projects/TRELLIS.2/assets/hdri/USER_Alt2.exr"
        env_img = read_exr_rgb(hdri) * 2.5
        envmap = EnvMap(torch.tensor(env_img, dtype=torch.float32, device=bb.device))
        kwargs = {"envmap": envmap}
    except Exception as e:
        print(f"(envmap load failed, rendering without: {e})")
        kwargs = {}

    result = render_utils.render_frames(
        mesh, extrinsics, intrinsics,
        {"resolution": args.resolution, "bg_color": (0.1, 0.1, 0.1)},
        verbose=False, **kwargs,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for key, frames in result.items():
        arr = frames[0]
        out = args.out_dir / f"{key}.png"
        Image.fromarray(arr).save(out)
        saved.append(key)
    print(f"\nsaved channels: {saved}")
    print(f"output dir: {args.out_dir}")
    print("\nLook for the polygonal patches in each channel:")
    print("  base_color / roughness → texture decode content (attr-smooth fixable)")
    print("  clay                   → SSAO over chorded geometry")
    print("  normal                 → shading normals (patch not active?)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
