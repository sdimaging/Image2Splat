#!/usr/bin/env python3
"""Final tier validation: 5 inputs × 5 tier candidates × 1 view = 25 frames.

Tier candidates being validated:
  T0  Baseline (locked)        12 / 7.5 / 1.1
  T1A Medium midpoint (rec)    14 / 7.7 / 1.25
  T1B Medium close-to-base     13 / 7.6 / 1.2
  T1C Medium high-steps        15 / 7.6 / 1.2
  T2  High (locked)            15 / 8.0 / 1.5

Pixal3D's preprocess_image() automatically handles BG removal via BiRefNet
when input has no alpha channel.

Output:
  probe_compare/tier_validation/<stem>/<stem>_129_<TIER>.png
"""
from __future__ import annotations
import gc
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.backbone import Backbone
from src.cameras import generate_camera_trajectory, intrinsics_from_fov
from src.polish import polish_rgba

HDRI = Path.home() / "projects/TRELLIS.2/assets/hdri/USER_Alt2.exr"
HDRI_EXPOSURE = 2.5
PIPELINE_TYPE = "1536_cascade"
MAX_NUM_TOKENS = 131072
TOTAL_VIEWS = 200
RESOLUTION = 3000
FOV = 40.0
SEED = 222
RENDER_VIEW = 129
MAX_FACES = 16_777_216

# (label, steps, shape_guidance, tex_guidance)
TIERS = [
    ("T0_baseline",      12, 7.5, 1.1),
    ("T1A_mid_14-7.7-1.25", 14, 7.7, 1.25),
    ("T1B_mid_13-7.6-1.2",  13, 7.6, 1.2),
    ("T1C_mid_15-7.6-1.2",  15, 7.6, 1.2),
    ("T2_high",          15, 8.0, 1.5),
]

POLISH_SHARPEN_PERCENT = 40
POLISH_SHARPEN_RADIUS = 1.5
POLISH_SHARPEN_THRESHOLD = 3
POLISH_GRAIN_STRENGTH = 3.0

import os as _os
_HOTFOLDER = Path(_os.environ.get("IMAGE2SPLAT_HOTFOLDER",
                                  str(Path.home() / "image2splat"))).expanduser()
PROCESSING_DIR = _HOTFOLDER / "processing"
# Edit INPUT_NAMES to point at the images you want to validate tiers against.
# All files must exist under PROCESSING_DIR (or you can override per-script).
INPUT_NAMES: list[str] = [
    # "MyImage1.png",
    # "MyImage2.png",
]
OUT_ROOT = _HOTFOLDER / "probe_compare" / "tier_validation"


def short_stem(name: str) -> str:
    stem = Path(name).stem
    return stem if len(stem) <= 32 else f"hash_{stem[:12]}"


class TeeLog:
    def __init__(self, p):
        p.parent.mkdir(parents=True, exist_ok=True)
        self.fh = open(p, "w", buffering=1)
    def __call__(self, m):
        s = f"[{time.strftime('%H:%M:%S')}] {m}"
        print(s, flush=True); self.fh.write(s + "\n")
    def close(self): self.fh.close()


def main() -> int:
    inputs = [PROCESSING_DIR / name for name in INPUT_NAMES]
    for p in inputs:
        if not p.exists():
            print(f"FAIL: input missing: {p}", file=sys.stderr); return 1
    if not HDRI.exists():
        print(f"FAIL: HDRI missing: {HDRI}", file=sys.stderr); return 1

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    log = TeeLog(OUT_ROOT / "tier_validation.log")
    try:
        log("=== tier validation: 5 inputs × 5 tiers ===")
        log(f"inputs   : {INPUT_NAMES}")
        log(f"tiers    : {[t[0] for t in TIERS]}")
        log(f"view     : {RENDER_VIEW}")
        log(f"polish   : sharpen={POLISH_SHARPEN_PERCENT}%  grain={POLISH_GRAIN_STRENGTH}")
        log(f"render   : {RESOLUTION}px fov={FOV} Z-up  seed={SEED}")
        n_total = len(inputs) * len(TIERS)
        log(f"total    : {len(inputs)} × {len(TIERS)} = {n_total} frames")

        up = np.array([0, 0, 1.0])
        full_w2c = generate_camera_trajectory(TOTAL_VIEWS, radius=2.0, up=up)
        w2c_subset = full_w2c[[RENDER_VIEW]]

        log("loading Pixal3D backbone...")
        t0 = time.time()
        bb = Backbone(
            name="pixal3d",
            max_num_tokens=MAX_NUM_TOKENS,
            pixal3d_pipeline_type=PIPELINE_TYPE,
        ).load()
        log(f"  loaded in {time.time()-t0:.1f}s")

        from src.render import render_multiview
        from src.hdri_io import read_exr_rgb
        try:
            from trellis2.renderers import EnvMap
        except ModuleNotFoundError:
            from pixal3d.renderers import EnvMap
        hdri = read_exr_rgb(HDRI) * float(HDRI_EXPOSURE)
        envmap = EnvMap(torch.tensor(hdri, dtype=torch.float32, device="cuda"))

        global_t0 = time.time()
        run_idx = 0
        for input_path in inputs:
            stem = short_stem(input_path.name)
            image = Image.open(input_path)
            out_dir = OUT_ROOT / stem
            out_dir.mkdir(parents=True, exist_ok=True)
            log(f"--- {stem}  mode={image.mode} ---")

            for tier_label, steps, shape_g, tex_g in TIERS:
                run_idx += 1
                out_path = out_dir / f"{stem}_{RENDER_VIEW:03d}_{tier_label}.png"
                if out_path.exists():
                    log(f"[{run_idx}/{n_total}] {stem} | {tier_label}  SKIP (exists)")
                    continue
                log(f"[{run_idx}/{n_total}] {stem} | {tier_label}: steps={steps} shape={shape_g} tex={tex_g}")
                t_run = time.time()
                bb.sparse_structure_sampler_params = {"steps": steps, "guidance_strength": shape_g}
                bb.shape_slat_sampler_params       = {"steps": steps, "guidance_strength": shape_g}
                bb.tex_slat_sampler_params         = {"steps": steps, "guidance_strength": tex_g}

                try:
                    t_mesh = time.time()
                    mesh = bb.run(image, seed=SEED)
                    raw_faces = mesh.faces.shape[0]
                    log(f"  raw mesh: {mesh.vertices.shape[0]:,}v/{raw_faces:,}f ({time.time()-t_mesh:.1f}s)")
                    if raw_faces > MAX_FACES:
                        mesh.simplify(target=MAX_FACES, verbose=False)
                        log(f"  decim: {mesh.vertices.shape[0]:,}v/{mesh.faces.shape[0]:,}f")

                    frames = render_multiview(
                        mesh, w2c_subset, fov_deg=FOV,
                        resolution=RESOLUTION, envmap=envmap,
                        repo_dir=bb.repo_dir, device=bb.device, verbose=False,
                    )
                    rgba = np.array(frames[0].convert("RGBA"))
                    rgba = polish_rgba(
                        rgba, image_index=RENDER_VIEW,
                        sharpen_percent=POLISH_SHARPEN_PERCENT,
                        sharpen_radius=POLISH_SHARPEN_RADIUS,
                        sharpen_threshold=POLISH_SHARPEN_THRESHOLD,
                        grain_strength=POLISH_GRAIN_STRENGTH,
                        grain_seed_base=SEED,
                        grain_monochrome=False,
                    )
                    Image.fromarray(rgba).save(out_path)

                    del frames, mesh
                    gc.collect()
                    torch.cuda.empty_cache()
                    elapsed = time.time() - t_run
                    avg = (time.time() - global_t0) / run_idx
                    eta_min = (n_total - run_idx) * avg / 60
                    log(f"  ✓ {out_path.name} ({elapsed:.1f}s · ETA ~{eta_min:.0f} min)")
                except Exception as e:
                    log(f"  FAIL: {e}")
                    import traceback
                    log(traceback.format_exc())

        log(f"=== DONE in {(time.time()-global_t0)/60:.1f} min ===")
        log(f"Outputs: {OUT_ROOT}")
        log("For visual A/B per asset: open <stem>/*.png as PS layers, flip between tiers")
        return 0

    except Exception as e:
        import traceback
        log(f"FATAL: {e}"); log(traceback.format_exc())
        return 2
    finally:
        log.close()


if __name__ == "__main__":
    sys.exit(main())
