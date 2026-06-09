#!/usr/bin/env python
"""One-shot TripoSplat quick-splat for a single input image.

Designed to be invoked as a SUBPROCESS by the hot-folder daemon (so its
~7.5 GB VRAM footprint is fully released when it exits, never coexisting
with Pixal3D's render peaks). Also usable standalone.

Produces a fast feed-forward Gaussian splat (~8s on a 5090) alongside the
daemon's full COLMAP dataset — a browser-viewable preview / secondary-asset
version of every processed image.

USAGE
    python scripts/quick_splat.py <input_image> <output_dir>
        [--triposplat-repo ~/projects/TripoSplat]
        [--num-gaussians 262144]
        [--seed 42]

OUTPUT
    <output_dir>/
        quick_splat_<N>.ply      # INRIA 3DGS layout (PostShot/gsplat-ingestible)
        quick_splat_<N>.splat    # antimatter15 format (SuperSplat / browser drag-drop)
        quick_splat_input.webp   # the matted image TripoSplat actually saw

Exit codes: 0 ok, 1 bad args/missing repo, 2 inference failure.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input_image", type=Path)
    p.add_argument("output_dir", type=Path)
    p.add_argument("--triposplat-repo", type=Path,
                   default=Path.home() / "projects" / "TripoSplat")
    p.add_argument("--num-gaussians", type=int, default=262144,
                   help="Gaussian budget (32768-262144, rounded to multiple of 32).")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if not args.input_image.is_file():
        print(f"quick_splat FAIL: input not found: {args.input_image}")
        return 1
    repo = args.triposplat_repo.expanduser()
    ckpts = repo / "ckpts"
    if not (ckpts / "diffusion_models" / "triposplat_fp16.safetensors").exists():
        print(f"quick_splat FAIL: TripoSplat weights not found under {ckpts}")
        return 1

    sys.path.insert(0, str(repo))
    try:
        from triposplat import TripoSplatPipeline  # noqa: E402
    except ImportError as e:
        print(f"quick_splat FAIL: cannot import TripoSplat from {repo}: {e}")
        return 1

    t0 = time.time()
    try:
        pipe = TripoSplatPipeline(
            ckpt_path              = str(ckpts / "diffusion_models/triposplat_fp16.safetensors"),
            decoder_path           = str(ckpts / "vae/triposplat_vae_decoder_fp16.safetensors"),
            dinov3_path            = str(ckpts / "clip_vision/dino_v3_vit_h.safetensors"),
            flux2_vae_encoder_path = str(ckpts / "vae/flux2-vae.safetensors"),
            rmbg_path              = str(ckpts / "background_removal/birefnet.safetensors"),
            device                 = "cuda",
        )
        gaussian, prepared = pipe.run(str(args.input_image), seed=args.seed,
                                      num_gaussians=args.num_gaussians,
                                      show_progress=False)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        n = args.num_gaussians
        prepared.save(str(args.output_dir / "quick_splat_input.webp"))
        gaussian.save_ply(str(args.output_dir / f"quick_splat_{n}.ply"))
        gaussian.save_splat(str(args.output_dir / f"quick_splat_{n}.splat"))
        print(f"quick_splat OK: {args.input_image.name} → {args.output_dir} "
              f"({time.time() - t0:.1f}s incl. model load)")
        return 0
    except Exception as e:
        print(f"quick_splat FAIL: {type(e).__name__}: {e}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
