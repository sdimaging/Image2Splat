#!/usr/bin/env python
"""
Image-to-splat hot folder daemon.

Drop an image into the inbox folder; the daemon picks it up, runs Pixal3D
inference, renders N views at the configured resolution, writes a COLMAP +
Nerfstudio dataset, then moves the source image to completed/ (or failed/ on
error). The backbone is loaded ONCE at startup and reused across images.

Layout (auto-created on first run):
    <hotfolder>/
        inbox/        # drop images here
        processing/   # in-flight (image moves here while running)
        completed/    # successful source images
        failed/       # failed sources + companion .error.log
        datasets/     # per-image COLMAP/Nerfstudio outputs
            <slug>/<hdri-stem>_<n_views>v_<res>px/
        daemon.log    # rolling log

Image extensions accepted: .jpg, .jpeg, .png, .webp, .bmp
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import shutil
import signal
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np
import torch
from PIL import Image

from src.backbone import Backbone
from src.cameras import generate_camera_trajectory, intrinsics_from_fov
from src.export_colmap import write_colmap_dataset
from src.export_nerfstudio import write_nerfstudio_dataset
from src.mesh_export import export_mesh
from src.polish import polish_rgba
from src.render import render_multiview


VALID_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
# Override via $IMAGE2SPLAT_HOTFOLDER env var or --hotfolder CLI arg. On Windows-WSL,
# point this to a Desktop folder so the BAT can drop files into it from Windows:
#   export IMAGE2SPLAT_HOTFOLDER='/mnt/c/Users/<you>/Desktop/image2splat'
DEFAULT_HOTFOLDER = Path(
    os.environ.get("IMAGE2SPLAT_HOTFOLDER", str(Path.home() / "image2splat"))
).expanduser()
HDRI_DIR = Path.home() / "projects/TRELLIS.2/assets/hdri"
DEFAULT_HDRI = HDRI_DIR / "USER_Alt2.exr"  # production-locked HDRI
DEFAULT_SEED = 222                          # production-locked seed
DEFAULT_HDRI_EXPOSURE = 2.5                 # production-locked exposure (linear ×)
DEFAULT_PIPELINE_TYPE = "1536_cascade"      # 2048 tested 2026-05-15 — model OOM/asserts in HR.
DEFAULT_MAX_NUM_TOKENS = 131072             # 128k — production-locked, validated on chess+gargoyle.
# Sampler params left EMPTY → Pixal3D uses its own tuned per-sampler defaults from
# pipeline.json. ACTUAL VALUES (verified 2026-05-20 from model config):
#   sparse_structure: 12 steps, guidance_strength=7.5, guidance_interval=[0.6,1.0], rescale_t=5.0
#   shape_slat:       12 steps, guidance_strength=7.5, guidance_interval=[0.6,1.0], rescale_t=3.0
#   tex_slat:         12 steps, guidance_strength=1.0, guidance_interval=[0.6,0.9], rescale_t=3.0
# The shape stages use STRONG image adherence (7.5); texture lets the prior dominate (1.0).
# DO NOT override with a single value across all three — that destroys per-stage tuning,
# producing smoother shape + over-fit texture. If experimenting, override per-sampler.
# Bumping these past defaults made textures *worse* in 2026-05-15 testing — high guidance
# over-amplifies conditioning, more steps integrate along a distorted trajectory.
DEFAULT_SPARSE_STRUCTURE_SAMPLER_PARAMS: dict = {}
DEFAULT_SHAPE_SLAT_SAMPLER_PARAMS: dict = {}
DEFAULT_TEX_SLAT_SAMPLER_PARAMS: dict = {}

# Sampler tier system — validated 2026-05-21 via shape + steps + tex sweeps on
# Gargoyle2 + Knight + Column2 + T + c3d-chest. Each tier is a single (steps,
# shape_guidance, tex_guidance) triple applied uniformly to all 3 samplers
# (sparse_structure, shape_slat) for shape, and tex_slat for tex. Tier 6 is
# special — runs ALL 5 tiers at view PROBE_VIEW for asset-fit comparison.
SAMPLER_TIERS = {
    "1": ("Default",  12, 7.5, 1.1),
    "2": ("Subtle",   13, 7.6, 1.2),
    "3": ("Balanced", 14, 7.7, 1.25),
    "4": ("Refined",  15, 7.6, 1.2),
    "5": ("Sculpted", 15, 8.0, 1.5),
}
PROBE_TIER_KEY = "6"
PROBE_VIEW = 129  # which Fibonacci-orbit view to render in probe mode


def _tier_sampler_params(tier_key: str) -> tuple[dict, dict, dict]:
    """Return (sparse, shape, tex) sampler-param dicts for the given tier key."""
    _, steps, shape_g, tex_g = SAMPLER_TIERS[tier_key]
    sparse = {"steps": steps, "guidance_strength": float(shape_g)}
    shape = {"steps": steps, "guidance_strength": float(shape_g)}
    tex = {"steps": steps, "guidance_strength": float(tex_g)}
    return sparse, shape, tex


def _apply_tier_to_backbone(bb, tier_key: str) -> None:
    """Mutate backbone's per-sampler param dicts to match the given tier."""
    sparse, shape, tex = _tier_sampler_params(tier_key)
    bb.sparse_structure_sampler_params = dict(sparse)
    bb.shape_slat_sampler_params = dict(shape)
    bb.tex_slat_sampler_params = dict(tex)


def interactive_select_tier(default: str = "1") -> str:
    """Prompt for sampler tier (1-5) or probe mode (6). Returns the key."""
    print("Sampler tier:")
    for k, (name, steps, sg, tg) in SAMPLER_TIERS.items():
        tag = "  [default]" if k == default else ""
        print(f"  {k}) {name:<9}  steps={steps}  shape={sg}  tex={tg}{tag}")
    print(f"  {PROBE_TIER_KEY}) Probe      runs ALL 5 tiers at view {PROBE_VIEW} (multi-tier comparison)")
    while True:
        s = input(f"Select [1-{PROBE_TIER_KEY}, default {default}]: ").strip()
        if not s:
            return default
        if s in SAMPLER_TIERS or s == PROBE_TIER_KEY:
            return s
        print(f"  invalid, pick 1-{PROBE_TIER_KEY}")


def interactive_select_probe_seed_count(default: int = 1, max_seeds: int = 8) -> int:
    """Prompt for # of seeds in probe mode. Returns int 1..max_seeds."""
    print(f"Probe seeds (1 = just {DEFAULT_SEED}, N = {DEFAULT_SEED} + N-1 randoms):")
    while True:
        s = input(f"Select [1-{max_seeds}, default {default}]: ").strip()
        if not s:
            return default
        try:
            n = int(s)
            if 1 <= n <= max_seeds:
                return n
        except ValueError:
            pass
        print(f"  invalid, pick 1-{max_seeds}")


def build_probe_seed_list(seed_count: int, anchor_seed: int = DEFAULT_SEED) -> list[int]:
    """Anchor seed + (seed_count - 1) random seeds, no duplicates."""
    seeds = [anchor_seed]
    rng = random.Random()  # nondeterministic — we LOG the seeds for reproducibility
    while len(seeds) < seed_count:
        s = rng.randint(1, 99999)
        if s not in seeds:
            seeds.append(s)
    return seeds


def interactive_select_force_rembg(default: bool) -> bool:
    """Y/N: should we re-run BiRefNet bg removal even on PNGs that already have alpha?

    Default N — trust the input. Pixal3D's pipeline already auto-skips BiRefNet
    when the input is RGBA with non-fully-opaque alpha (i.e. you've masked it).
    Saying Y here strips alpha before feeding the pipeline, forcing BiRefNet
    to run regardless. Useful only if you suspect your mask is bad.
    """
    dflt = "y" if default else "N"
    while True:
        s = input(f"Force background removal even on already-masked PNGs? [y/{dflt}]: ").strip().lower()
        if not s:
            return default
        if s in ("y", "yes"):
            return True
        if s in ("n", "no"):
            return False
        print("  please answer y or n")


def interactive_select_seed(default: int) -> int:
    """Prompt for seed, accepting blank → default."""
    while True:
        s = input(f"Seed [default {default}]: ").strip()
        if not s:
            return default
        try:
            return int(s)
        except ValueError:
            print(f"  not a number, try again")


def interactive_select_backbone(default: str) -> str:
    """Prompt for backbone: pixal3d (1) or trellis2 (2)."""
    print("Backbone:")
    print(f"  1) Pixal3D{'  [default]' if default == 'pixal3d' else ''}")
    print(f"  2) TRELLIS.2{'  [default]' if default == 'trellis2' else ''}")
    default_idx = 1 if default == "pixal3d" else 2
    while True:
        s = input(f"Select [1-2, default {default_idx}]: ").strip()
        if not s:
            return default
        if s == "1": return "pixal3d"
        if s == "2": return "trellis2"
        print("  invalid")


def interactive_select_mode(default: str) -> str:
    """Prompt for output mode: splat / mesh / both."""
    options = [("splat", "Splat dataset only (COLMAP + Nerfstudio + rendered views)"),
               ("mesh",  "Mesh only (PLY + GLB, no rendering)"),
               ("both",  "Both — mesh AND splat dataset")]
    print("Output:")
    for i, (name, desc) in enumerate(options, start=1):
        tag = "  [default]" if name == default else ""
        print(f"  {i}) {desc}{tag}")
    default_idx = next(i for i, (n, _) in enumerate(options, start=1) if n == default)
    while True:
        s = input(f"Select [1-3, default {default_idx}]: ").strip()
        if not s:
            return default
        try:
            idx = int(s)
            if 1 <= idx <= 3:
                return options[idx - 1][0]
        except ValueError:
            pass
        print("  invalid")


def interactive_select_hdri(default: Path) -> Path:
    """Show numbered list of .exr files in HDRI_DIR, return the chosen path."""
    exrs = sorted(HDRI_DIR.glob("*.exr"))
    if not exrs:
        print(f"  no .exr files in {HDRI_DIR}, using default {default.name}")
        return default
    default_idx = 0
    for i, p in enumerate(exrs):
        if p.resolve() == default.resolve():
            default_idx = i
            break
    print("HDRI:")
    for i, p in enumerate(exrs, start=1):
        tag = " [default]" if (i - 1) == default_idx else ""
        print(f"  {i:>2}) {p.name}{tag}")
    while True:
        s = input(f"Select [1-{len(exrs)}, default {default_idx + 1}]: ").strip()
        if not s:
            return exrs[default_idx]
        try:
            idx = int(s) - 1
            if 0 <= idx < len(exrs):
                return exrs[idx]
        except ValueError:
            pass
        print(f"  invalid selection, try again")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--hotfolder", type=Path, default=DEFAULT_HOTFOLDER,
                   help="Hot folder root. Inbox/processing/completed/failed/datasets created beneath.")
    p.add_argument("--backbone", choices=["pixal3d", "trellis2"], default="pixal3d")
    p.add_argument("--mode", choices=["splat", "mesh", "both"], default="splat",
                   help="Output mode. splat=render multiview + COLMAP/Nerfstudio. "
                        "mesh=PLY+GLB only, no render. both=both.")
    p.add_argument("--hdri", type=Path, default=DEFAULT_HDRI)
    p.add_argument("--hdri-exposure", type=float, default=DEFAULT_HDRI_EXPOSURE,
                   help="Linear multiplier on HDRI radiance — equivalent to an "
                        "exposure-stop on the lighting. 1.0=unchanged, 2.0=+1 stop, "
                        "2.5=+1.3 stops (production-locked).")
    p.add_argument("--num-views", type=int, default=200)
    p.add_argument("--resolution", type=int, default=3000)
    p.add_argument("--chunk-size", type=int, default=4,
                   help="Render chunk size. 4 is safe at 3K, 8 at 2K, 16 at 1K.")
    p.add_argument("--fov", type=float, default=40.0)
    p.add_argument("--radius", type=float, default=2.0)
    p.add_argument("--up-axis", choices=["y", "z"], default="z")
    p.add_argument("--num-init-points", type=int, default=100_000)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--pipeline-type",
                   choices=["1024_cascade", "1536_cascade", "2048_cascade"],
                   default=DEFAULT_PIPELINE_TYPE,
                   help="Pixal3D cascade mode. 1024 = original, 1536 = +50%% grid axis, "
                        "2048 = +100%% grid axis (EXPERIMENTAL, OOD for the 1024-trained "
                        "models, may produce noise on some inputs).")
    p.add_argument("--max-num-tokens", type=int, default=DEFAULT_MAX_NUM_TOKENS,
                   help="Voxel-token budget. Pipeline auto-downgrades HR resolution "
                        "by 128 until tokens < budget. Higher = preserves 1536 longer.")
    p.add_argument("--max-faces", type=int, default=16_000_000,
                   help="If the generated mesh exceeds this face count, simplify "
                        "before rendering. nvdiffrast's hard cap is 2^24 = 16.78M; "
                        "1536_cascade routinely overshoots, so default 16M.")
    p.add_argument("--sampler-steps", type=int, default=50,
                   help="Diffusion steps for shape+texture SLat samplers. Higher = "
                        "finer detail, longer inference. Pixal3D default 12, we use 50.")
    p.add_argument("--cfg-tex", type=float, default=5.0,
                   help="guidance_strength for the texture SLat sampler.")
    p.add_argument("--cfg-shape", type=float, default=4.0,
                   help="guidance_strength for the shape SLat sampler.")
    # Note: sparse_structure uses a non-CFG sampler — no guidance param.
    p.add_argument("--poll-interval", type=float, default=5.0,
                   help="Seconds between inbox scans when idle.")
    p.add_argument("--no-prompt", action="store_true",
                   help="Skip the interactive seed/HDRI prompts at startup. Use the "
                        "values from --seed / --hdri (or their defaults).")
    # --- Auto-polish (sharpen + grain) applied to every saved frame ---
    p.add_argument("--polish-sharpen-percent", type=int, default=40,
                   help="Unsharp mask strength %% (0 = OFF). Default 40 = subtle.")
    p.add_argument("--polish-sharpen-radius", type=float, default=1.5)
    p.add_argument("--polish-sharpen-threshold", type=int, default=3)
    p.add_argument("--polish-grain-strength", type=float, default=3.0,
                   help="Gaussian grain σ in 0-255 units (0 = OFF). Default 3.")
    p.add_argument("--polish-grain-monochrome", action="store_true",
                   help="If set, single grain channel applied to all RGB.")
    p.add_argument("--force-rembg", action="store_true",
                   help="Force BiRefNet bg removal even on RGBA inputs whose alpha "
                        "channel is already set. Default behavior: trust the input.")
    p.add_argument("--tier", choices=list(SAMPLER_TIERS.keys()) + [PROBE_TIER_KEY],
                   default="1",
                   help=f"Sampler tier 1-5 (production) or {PROBE_TIER_KEY} (probe mode "
                        "= all 5 tiers at view {PROBE_VIEW}).")
    p.add_argument("--probe-seed-count", type=int, default=1,
                   help="In probe mode, number of seeds to test (1=just DEFAULT_SEED, "
                        "N=anchor + N-1 randoms). Ignored when --tier != 6.")
    return p.parse_args()


def slug_for(path: Path) -> str:
    """Sanitize image filename stem for use as a folder name."""
    stem = path.stem
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in stem)
    safe = safe[:60].strip("_")
    return safe or "unnamed"


class Daemon:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.hotfolder = args.hotfolder
        self.inbox = self.hotfolder / "inbox"
        self.processing = self.hotfolder / "processing"
        self.completed = self.hotfolder / "completed"
        self.failed = self.hotfolder / "failed"
        self.datasets = self.hotfolder / "datasets"
        self.daemon_log = self.hotfolder / "daemon.log"
        self.stopping = False

    def setup_dirs(self) -> None:
        for d in (self.inbox, self.processing, self.completed,
                  self.failed, self.datasets):
            d.mkdir(parents=True, exist_ok=True)

    def log(self, msg: str) -> None:
        line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
        print(line, flush=True)
        try:
            with self.daemon_log.open("a") as f:
                f.write(line + "\n")
        except OSError:
            pass

    def find_next_image(self) -> Path | None:
        """Find the oldest pending image whose size has stabilized."""
        candidates: list[tuple[float, Path]] = []
        try:
            entries = list(self.inbox.iterdir())
        except FileNotFoundError:
            return None
        for p in entries:
            if not p.is_file():
                continue
            if p.suffix.lower() not in VALID_EXT:
                continue
            if p.name.endswith(":Zone.Identifier") or p.name.startswith("."):
                continue
            try:
                s1 = p.stat().st_size
                if s1 == 0:
                    continue
                time.sleep(0.3)
                s2 = p.stat().st_size
                if s1 != s2:
                    continue
                candidates.append((p.stat().st_mtime, p))
            except FileNotFoundError:
                continue
        candidates.sort()
        return candidates[0][1] if candidates else None

    def load_envmap(self, bb: Backbone):
        from src.hdri_io import read_exr_rgb
        sys.path.insert(0, str(bb.repo_dir))
        try:
            from trellis2.renderers import EnvMap  # type: ignore
        except ModuleNotFoundError:
            from pixal3d.renderers import EnvMap  # type: ignore
        img = read_exr_rgb(self.args.hdri) * float(self.args.hdri_exposure)
        return EnvMap(torch.tensor(img, dtype=torch.float32, device=bb.device))

    def process_one(self, bb: Backbone, envmap, w2c, intr,
                    image_path: Path) -> Path:
        slug = slug_for(image_path)
        slug_root = self.datasets / slug
        mode = self.args.mode

        # Skip-existing logic — different markers per mode
        if mode == "mesh":
            mesh_marker = slug_root / "mesh.glb"
            if mesh_marker.exists():
                self.log(f"  SKIP {slug}: mesh already exists at {mesh_marker}")
                return slug_root
        else:
            hdri_stem = self.args.hdri.stem
            splat_dir = (slug_root /
                         f"{hdri_stem}_{self.args.num_views}v_{self.args.resolution}px")
            if (splat_dir / "transforms.json").exists():
                self.log(f"  SKIP {slug}: dataset already exists at {splat_dir}")
                return slug_root

        self.log(f"  INFER {image_path.name} → slug={slug}  mode={mode}")
        image = Image.open(image_path)
        if self.args.force_rembg and image.mode == "RGBA":
            # Strip alpha → Pixal3D's pipeline sees RGB-only input, runs BiRefNet
            image = image.convert("RGB")
            self.log(f"  force-rembg: stripped alpha, BiRefNet will run")
        mesh = bb.run(image, seed=self.args.seed)
        n_faces = int(mesh.faces.shape[0])
        self.log(f"  mesh: {mesh.vertices.shape[0]:,} verts, {n_faces:,} faces")
        if n_faces > self.args.max_faces:
            self.log(f"  mesh > {self.args.max_faces:,} faces — simplifying...")
            mesh.simplify(target=self.args.max_faces, verbose=False)
            self.log(f"  simplified: {mesh.vertices.shape[0]:,} verts, "
                     f"{mesh.faces.shape[0]:,} faces")

        # --- Mesh export (mode in {mesh, both}) ---
        if mode in ("mesh", "both"):
            self.log(f"  exporting mesh...")
            written = export_mesh(mesh, slug_root)
            for kind, p in written.items():
                self.log(f"    {kind}: {p}")

        # --- Splat dataset (mode in {splat, both}) ---
        if mode in ("splat", "both"):
            out_dir = (slug_root /
                       f"{self.args.hdri.stem}_{self.args.num_views}v_{self.args.resolution}px")
            images_dir = out_dir / "images"
            images_dir.mkdir(parents=True, exist_ok=True)
            names: list[str] = []
            n_views = self.args.num_views
            for ck_start in range(0, n_views, self.args.chunk_size):
                ck_end = min(ck_start + self.args.chunk_size, n_views)
                self.log(f"  chunk {ck_start:>3}-{ck_end-1:<3} "
                         f"({ck_end - ck_start} views)")
                frames = render_multiview(
                    mesh, w2c[ck_start:ck_end], fov_deg=self.args.fov,
                    resolution=self.args.resolution, envmap=envmap,
                    repo_dir=bb.repo_dir, device=bb.device, verbose=False,
                )
                for off, img in enumerate(frames):
                    idx = ck_start + off
                    name = f"{idx:03d}.png"
                    if (self.args.polish_sharpen_percent > 0
                            or self.args.polish_grain_strength > 0):
                        rgba = np.array(img.convert("RGBA"))
                        rgba = polish_rgba(
                            rgba, image_index=idx,
                            sharpen_percent=self.args.polish_sharpen_percent,
                            sharpen_radius=self.args.polish_sharpen_radius,
                            sharpen_threshold=self.args.polish_sharpen_threshold,
                            grain_strength=self.args.polish_grain_strength,
                            grain_seed_base=self.args.seed,
                            grain_monochrome=self.args.polish_grain_monochrome,
                        )
                        Image.fromarray(rgba).save(images_dir / name)
                    else:
                        img.save(images_dir / name)
                    names.append(name)
                del frames
                gc.collect()
                torch.cuda.empty_cache()

            mesh_verts = mesh.vertices.detach().cpu().numpy()
            n_init = min(self.args.num_init_points, len(mesh_verts))
            rng = np.random.default_rng(self.args.seed)
            init_pts = mesh_verts[rng.choice(len(mesh_verts), size=n_init, replace=False)]
            write_colmap_dataset(out_dir, w2c, names, intr, points=init_pts)
            write_nerfstudio_dataset(out_dir, w2c, names, intr)
            self.log(f"  ✓ {len(names)} images + COLMAP + Nerfstudio → {out_dir}")

        return slug_root

    def process_one_probe(self, bb, envmap, w2c, image_path: Path) -> Path:
        """Probe mode: render PROBE_VIEW at all 5 tiers × N seeds, save to <slug>/probe/.

        Unlike normal processing, this does NOT generate a 200v COLMAP dataset.
        It produces a small comparison grid for asset-fit decisions:
            5 tiers × N seeds = 5N PNG files per asset
        Plus probe_meta.json documenting the tier configs + seed list used.
        """
        slug = slug_for(image_path)
        slug_root = self.datasets / slug
        probe_dir = slug_root / "probe"
        probe_dir.mkdir(parents=True, exist_ok=True)

        self.log(f"  PROBE {image_path.name} → slug={slug}")
        image = Image.open(image_path)
        if self.args.force_rembg and image.mode == "RGBA":
            image = image.convert("RGB")
            self.log(f"  force-rembg: stripped alpha, BiRefNet will run")

        view_w2c = w2c[PROBE_VIEW:PROBE_VIEW + 1]
        n_total = len(SAMPLER_TIERS) * len(self.probe_seeds)
        run_idx = 0

        meta = {
            "image": image_path.name,
            "slug": slug,
            "render_view": PROBE_VIEW,
            "seeds": self.probe_seeds,
            "tiers": {k: {"name": v[0], "steps": v[1], "shape": v[2], "tex": v[3]}
                       for k, v in SAMPLER_TIERS.items()},
            "hdri": self.args.hdri.name,
            "hdri_exposure": self.args.hdri_exposure,
            "pipeline_type": self.args.pipeline_type,
            "max_num_tokens": self.args.max_num_tokens,
            "resolution": self.args.resolution,
            "fov": self.args.fov,
            "polish": {
                "sharpen_percent": self.args.polish_sharpen_percent,
                "sharpen_radius": self.args.polish_sharpen_radius,
                "sharpen_threshold": self.args.polish_sharpen_threshold,
                "grain_strength": self.args.polish_grain_strength,
                "grain_monochrome": self.args.polish_grain_monochrome,
            },
            "generated_at": datetime.now().isoformat(),
        }

        for tier_key, (tier_name, _, _, _) in SAMPLER_TIERS.items():
            _apply_tier_to_backbone(bb, tier_key)
            for seed in self.probe_seeds:
                run_idx += 1
                out_path = probe_dir / f"{PROBE_VIEW:03d}_T{tier_key}_{tier_name.lower()}_seed{seed}.png"
                if out_path.exists():
                    self.log(f"  [{run_idx}/{n_total}] tier {tier_key} ({tier_name}) seed={seed}  SKIP (exists)")
                    continue
                self.log(f"  [{run_idx}/{n_total}] tier {tier_key} ({tier_name}) seed={seed}")
                t_run = time.time()
                try:
                    mesh = bb.run(image, seed=seed)
                    raw_verts = int(mesh.vertices.shape[0])
                    raw_faces = int(mesh.faces.shape[0])
                    if raw_faces > self.args.max_faces:
                        mesh.simplify(target=self.args.max_faces, verbose=False)
                        self.log(f"    mesh: {raw_verts:,}v / {raw_faces:,}f "
                                 f"→ DECIM → {mesh.vertices.shape[0]:,}v / {mesh.faces.shape[0]:,}f")
                    else:
                        self.log(f"    mesh: {raw_verts:,}v / {raw_faces:,}f (under {self.args.max_faces:,} cap, no decim)")
                    frames = render_multiview(
                        mesh, view_w2c, fov_deg=self.args.fov,
                        resolution=self.args.resolution, envmap=envmap,
                        repo_dir=bb.repo_dir, device=bb.device, verbose=False,
                    )
                    frame = frames[0]
                    if (self.args.polish_sharpen_percent > 0
                            or self.args.polish_grain_strength > 0):
                        rgba = np.array(frame.convert("RGBA"))
                        rgba = polish_rgba(
                            rgba, image_index=PROBE_VIEW,
                            sharpen_percent=self.args.polish_sharpen_percent,
                            sharpen_radius=self.args.polish_sharpen_radius,
                            sharpen_threshold=self.args.polish_sharpen_threshold,
                            grain_strength=self.args.polish_grain_strength,
                            grain_seed_base=seed,
                            grain_monochrome=self.args.polish_grain_monochrome,
                        )
                        Image.fromarray(rgba).save(out_path)
                    else:
                        frame.save(out_path)
                    self.log(f"    ✓ {out_path.name} ({time.time()-t_run:.1f}s)")
                    del frames, mesh
                    gc.collect()
                    torch.cuda.empty_cache()
                except Exception as e:
                    self.log(f"    FAIL: {e}")

        try:
            (probe_dir / "probe_meta.json").write_text(json.dumps(meta, indent=2))
        except OSError:
            pass

        self.log(f"  ✓ probe complete → {probe_dir} ({n_total} cells)")
        return slug_root

    def shutdown(self, *_):
        if not self.stopping:
            self.stopping = True
            self.log("Shutdown requested — finishing current item then exiting.")

    def run(self) -> int:
        self.setup_dirs()

        # Interactive seed/HDRI picker — opt-out with --no-prompt
        if not self.args.no_prompt:
            print()
            print("=" * 60)
            print("  IMAGE TO SPLAT - Hot Folder Daemon")
            print("=" * 60)
            # Mesh-output is disabled for now — always splat. (Mode prompt removed;
            # mesh code path still works via --mode {mesh,both} CLI override.)
            self.args.mode = "splat"
            try:
                self.args.seed = interactive_select_seed(self.args.seed)
                self.args.hdri = interactive_select_hdri(self.args.hdri)
                self.args.force_rembg = interactive_select_force_rembg(self.args.force_rembg)
                self.args.tier = interactive_select_tier(self.args.tier)
                if self.args.tier == PROBE_TIER_KEY:
                    self.args.probe_seed_count = interactive_select_probe_seed_count(
                        self.args.probe_seed_count
                    )
            except EOFError:
                # Non-interactive stdin (e.g. piped) → fall through with defaults
                print("(stdin not interactive — using defaults)")

        # Build the probe seed list once if probe mode
        self.probe_seeds: list[int] = []
        if self.args.tier == PROBE_TIER_KEY:
            self.probe_seeds = build_probe_seed_list(
                self.args.probe_seed_count, anchor_seed=self.args.seed
            )

        self.log("=" * 60)
        self.log("Hot-folder daemon starting")
        self.log(f"  hotfolder : {self.hotfolder}")
        self.log(f"  backbone  : {self.args.backbone}")
        self.log(f"  mode      : {self.args.mode}")
        if self.args.mode != "mesh":
            self.log(f"  hdri      : {self.args.hdri.name}")
            self.log(f"  views     : {self.args.num_views}")
            self.log(f"  resolution: {self.args.resolution}")
            self.log(f"  chunk     : {self.args.chunk_size}")
        self.log(f"  seed      : {self.args.seed}")
        self.log(f"  exposure  : x{self.args.hdri_exposure}")
        self.log(f"  pipeline  : {self.args.pipeline_type}")
        self.log(f"  max_tokens: {self.args.max_num_tokens:,}")
        if self.args.tier == PROBE_TIER_KEY:
            self.log(f"  sampler   : PROBE mode — all 5 tiers at view {PROBE_VIEW}")
            self.log(f"  seeds     : {self.probe_seeds}")
        else:
            t_name, t_steps, t_sg, t_tg = SAMPLER_TIERS[self.args.tier]
            self.log(f"  sampler   : tier {self.args.tier} ({t_name})  "
                     f"steps={t_steps}  shape={t_sg}  tex={t_tg}")
        self.log(f"  rembg     : {'force BiRefNet' if self.args.force_rembg else 'auto (use input alpha if present)'}")
        self.log("=" * 60)

        # Initial sampler params: apply tier 1-5 (probe mode mutates per-iteration)
        if self.args.tier in SAMPLER_TIERS:
            init_sparse, init_shape, init_tex = _tier_sampler_params(self.args.tier)
        else:
            init_sparse, init_shape, init_tex = {}, {}, {}

        self.log("Loading backbone (one-time)...")
        bb = Backbone(
            name=self.args.backbone,
            pixal3d_pipeline_type=self.args.pipeline_type,
            max_num_tokens=self.args.max_num_tokens,
            sparse_structure_sampler_params=init_sparse,
            shape_slat_sampler_params=init_shape,
            tex_slat_sampler_params=init_tex,
        ).load()

        envmap = None
        w2c = None
        intr = None
        if self.args.mode != "mesh":
            self.log("Loading envmap...")
            envmap = self.load_envmap(bb)
            self.log("Building shared camera trajectory...")
            up = (np.array([0, 0, 1.0]) if self.args.up_axis == "z"
                  else np.array([0, 1.0, 0]))
            w2c = generate_camera_trajectory(
                self.args.num_views, radius=self.args.radius, up=up,
            )
            intr = intrinsics_from_fov(self.args.fov, self.args.resolution)
        self.log(f"Ready. Polling {self.inbox} every "
                 f"{self.args.poll_interval}s. Drop images to process.")

        signal.signal(signal.SIGINT, self.shutdown)
        signal.signal(signal.SIGTERM, self.shutdown)

        while not self.stopping:
            try:
                img = self.find_next_image()
                if img is None:
                    time.sleep(self.args.poll_interval)
                    continue

                proc_path = self.processing / img.name
                try:
                    shutil.move(str(img), str(proc_path))
                except OSError as e:
                    self.log(f"FAIL claim {img.name}: {e}")
                    time.sleep(self.args.poll_interval)
                    continue

                self.log(f"START {proc_path.name}")
                t0 = time.time()
                try:
                    if self.args.tier == PROBE_TIER_KEY:
                        self.process_one_probe(bb, envmap, w2c, proc_path)
                    else:
                        self.process_one(bb, envmap, w2c, intr, proc_path)
                    shutil.move(str(proc_path),
                                str(self.completed / proc_path.name))
                    self.log(f"DONE  {proc_path.name} in "
                             f"{time.time() - t0:.1f}s")
                except Exception as e:
                    tb = traceback.format_exc()
                    self.log(f"FAIL  {proc_path.name}: {e}")
                    fail_path = self.failed / proc_path.name
                    try:
                        shutil.move(str(proc_path), str(fail_path))
                    except OSError:
                        pass
                    err_log = self.failed / f"{proc_path.stem}.error.log"
                    try:
                        err_log.write_text(tb)
                    except OSError:
                        pass
                    torch.cuda.empty_cache()
            except KeyboardInterrupt:
                break
            except Exception as e:
                self.log(f"DAEMON ERROR: {e}\n{traceback.format_exc()}")
                time.sleep(self.args.poll_interval)

        self.log("Daemon stopped.")
        return 0


def main() -> int:
    args = parse_args()
    return Daemon(args).run()


if __name__ == "__main__":
    sys.exit(main())
