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
import re
import shutil
import signal
import subprocess
import sys
import time
import traceback
from collections import deque
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
from src.cameras import (
    compute_adaptive_fov,
    generate_camera_trajectory,
    intrinsics_from_fov,
)
from src.export_colmap import write_colmap_dataset
from src.export_nerfstudio import write_nerfstudio_dataset
from src.mesh_export import export_mesh
from src.polish import polish_rgba
from src.render import render_multiview


VALID_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
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


# Error signatures that mean the CUDA context is dead and no further GPU work
# can succeed in this process. Triggers: Windows TDR resets, driver hiccups,
# allocator corruption. Once seen, only a process restart recovers.
CUDA_FATAL_PATTERNS = (
    "CUDA driver error",
    "device not ready",
    "CUDACachingAllocator",
    "CUDA error",
    "INTERNAL ASSERT FAILED",
    "device-side assert",
)


def _is_cuda_fatal(exc: BaseException) -> bool:
    msg = str(exc)
    return any(p in msg for p in CUDA_FATAL_PATTERNS)


TIER_SEED_RE = re.compile(r"^T(\d+)_(\d+)$")


def parse_tier_folder(folder: Path) -> tuple[str, int] | None:
    """Parse `T<tier_key>_<seed>` from a folder name. Returns (tier_key, seed) or None.

    tier_key must be in SAMPLER_TIERS (1-5). Probe tier (6) is intentionally
    NOT supported in batch mode — probes are a per-asset diagnostic, not a
    bulk production target.
    """
    m = TIER_SEED_RE.match(folder.name)
    if not m:
        return None
    tier_key, seed_str = m.group(1), m.group(2)
    if tier_key not in SAMPLER_TIERS:
        return None
    try:
        seed = int(seed_str)
    except ValueError:
        return None
    return tier_key, seed


def interactive_select_batch_mode(inbox: Path) -> Path | None:
    """Ask whether to enter batch mode. Returns parent path or None for normal mode.

    Default parent is the daemon's inbox — drop T<N>_<seed>/ folders straight
    into inbox to batch-run them. User can type a different path at the prompt
    if they keep tier folders staged elsewhere.
    """
    while True:
        s = input("Batch mode? Walk T<N>_<seed>/ subfolders and apply tier+seed per "
                  "folder [y/N]: ").strip().lower()
        if s in ("", "n", "no"):
            return None
        if s in ("y", "yes"):
            break
        print("  invalid, try again")
    while True:
        s = input(f"Batch parent folder [{inbox}]: ").strip()
        candidate = Path(s).expanduser() if s else inbox
        if not candidate.is_dir():
            print(f"  not a directory: {candidate} — try again or Ctrl-C to bail")
            continue
        # Validate at least one T<N>_<seed>/ subfolder is present
        valid = any(parse_tier_folder(p) is not None
                    for p in candidate.iterdir() if p.is_dir())
        if not valid:
            print(f"  no T<N>_<seed>/ subfolders detected under {candidate}")
            again = input("  use it anyway? [y/N]: ").strip().lower()
            if again not in ("y", "yes"):
                continue
        return candidate


def build_probe_seed_list(seed_count: int, anchor_seed: int = DEFAULT_SEED) -> list[int]:
    """Anchor seed + (seed_count - 1) random seeds, no duplicates."""
    seeds = [anchor_seed]
    rng = random.Random()  # nondeterministic — we LOG the seeds for reproducibility
    while len(seeds) < seed_count:
        s = rng.randint(1, 99999)
        if s not in seeds:
            seeds.append(s)
    return seeds


def _build_probe_summary_md(meta: dict) -> str:
    """Human-readable companion to probe_meta.json. One table per asset."""
    slug = meta.get("slug", "?")
    batch = meta.get("batch", {})
    started = batch.get("started_at", "?")
    finished = batch.get("finished_at", "?")
    total_min = (batch.get("total_wall_time_s") or 0) / 60.0
    n_run = batch.get("n_cells_run", 0)
    n_skip = batch.get("n_cells_skipped", 0)
    n_total = batch.get("n_cells_total", 0)
    img = meta.get("input_image") or {}

    lines: list[str] = []
    lines.append(f"# Probe Summary — {slug}")
    lines.append("")
    lines.append(f"**Started:** {started}  ")
    lines.append(f"**Finished:** {finished}  ")
    lines.append(f"**Wall time:** {total_min:.1f} min  ")
    lines.append(f"**Cells:** {n_run} run, {n_skip} skipped, {n_total} total")
    lines.append("")
    lines.append("## Input")
    lines.append("")
    lines.append(f"- File: `{meta.get('image', '?')}`")
    lines.append(f"- Size: {img.get('width', '?')}×{img.get('height', '?')} ({img.get('mode', '?')})")
    lines.append(f"- HDRI: `{meta.get('hdri', '?')}` @ exposure {meta.get('hdri_exposure', '?')}")
    lines.append(f"- Pipeline: `{meta.get('pipeline_type', '?')}` · max_tokens={meta.get('max_num_tokens', '?')}")
    lines.append(f"- Render: {meta.get('resolution', '?')}px @ FOV {meta.get('fov', '?')}° · view {meta.get('render_view', '?')}")
    lines.append("")
    lines.append("## Tier configuration")
    lines.append("")
    lines.append("| Tier | Name | Steps | Shape | Tex |")
    lines.append("|---|---|---|---|---|")
    for k, t in (meta.get("tiers") or {}).items():
        lines.append(f"| {k} | {t.get('name','?')} | {t.get('steps','?')} | {t.get('shape','?')} | {t.get('tex','?')} |")
    lines.append("")
    lines.append("## Cells")
    lines.append("")
    lines.append("| # | Tier | Seed | Raw mesh (v / f) | Post-decim (v / f) | Decim? | FOV | Total | Mesh | Render | Save | GPU peak | PNG |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for i, c in enumerate(meta.get("cells") or [], start=1):
        if "error" in c:
            lines.append(f"| {i} | {c.get('tier_key','?')} ({c.get('tier_name','?')}) | {c.get('seed','?')} | — | — | — | — | — | — | — | — | — | FAIL: `{c['error'][:60]}` |")
            continue
        m = c.get("mesh", {})
        t = c.get("timing_s", {})
        raw = f"{m.get('raw_verts',0):,} / {m.get('raw_faces',0):,}"
        post = f"{m.get('post_decim_verts',0):,} / {m.get('post_decim_faces',0):,}"
        decim = "YES" if m.get("decimated") else "no"
        fov_val = c.get("effective_fov_deg", "?")
        lines.append(
            f"| {i} | {c.get('tier_key','?')} ({c.get('tier_name','?')}) | {c.get('seed','?')} | "
            f"{raw} | {post} | {decim} | "
            f"{fov_val}° | "
            f"{t.get('total','?')}s | {t.get('mesh_gen','?')}s | {t.get('render','?')}s | {t.get('polish_save','?')}s | "
            f"{c.get('gpu_peak_gb','?')} GB | {c.get('output_size_mb','?')} MB |"
        )
    lines.append("")

    # Batch stats — average over successful cells
    ok = [c for c in (meta.get("cells") or []) if "error" not in c]
    if ok:
        avg_total = sum((c.get("timing_s") or {}).get("total", 0) for c in ok) / len(ok)
        avg_mesh = sum((c.get("timing_s") or {}).get("mesh_gen", 0) for c in ok) / len(ok)
        avg_render = sum((c.get("timing_s") or {}).get("render", 0) for c in ok) / len(ok)
        peak_gpu = max(c.get("gpu_peak_gb", 0) for c in ok)
        decim_count = sum(1 for c in ok if (c.get("mesh") or {}).get("decimated"))
        lines.append("## Batch stats")
        lines.append("")
        lines.append(f"- Successful cells: {len(ok)} / {n_total}")
        lines.append(f"- Avg total / cell: {avg_total:.1f}s")
        lines.append(f"- Avg mesh gen: {avg_mesh:.1f}s · Avg render: {avg_render:.1f}s")
        lines.append(f"- Peak GPU across batch: {peak_gpu:.1f} GB")
        lines.append(f"- Cells that hit decim cap: {decim_count} / {len(ok)}")
        lines.append("")

    return "\n".join(lines)


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
    p.add_argument("--adaptive-fov", action=argparse.BooleanOptionalAction, default=True,
                   help="Per-asset adaptive FOV: tighten FOV to fit mesh silhouette "
                        "(worst-view-fits-all across all cameras). On by default — "
                        "produces a single PINHOLE camera per dataset (COLMAP-schema "
                        "identical to legacy fixed-FOV output) but uses the tightest "
                        "FOV that contains the asset from every view. Pass "
                        "--no-adaptive-fov to fall back to fixed --fov (legacy mode).")
    p.add_argument("--adaptive-fov-margin", type=float, default=0.10,
                   help="Fractional padding on adaptive FOV (0.10 = 10%% extra margin "
                        "around the tight silhouette fit). Higher = safer, lower = "
                        "tighter zoom = more subject pixels per view.")
    p.add_argument("--adaptive-fov-min", type=float, default=12.0,
                   help="Lower clamp for adaptive FOV (deg). Floor prevents pathological "
                        "zoom on degenerate / empty meshes.")
    p.add_argument("--adaptive-fov-max", type=float, default=55.0,
                   help="Upper clamp for adaptive FOV (deg). Ceiling caps how loose "
                        "the auto-FOV can go; if your default --fov is higher than "
                        "this, --no-adaptive-fov uses --fov as-is.")
    # NOTE: geometry plateau smoothing was REMOVED from the daemon
    # (2026-06-11, Spenser's call): it traded plateau facets for
    # high-frequency feature-edge noise. Seed selection is the production
    # answer — see scripts/rank_seeds.py and FACETING_FINDINGS.md.
    # The experiment survives offline in mesh_smooth.py +
    # diagnose_facets.py --geo-compare.
    p.add_argument("--seed-sweep", type=int, default=0, metavar="N",
                   help="SEED SWEEP: render PROBE_VIEW for N seeds (anchor "
                        "--seed + N-1 randoms) at ONE tier (--tier 1-5, "
                        "default production tier). Cells land in "
                        "datasets/<slug>/probe/ in the standard naming, then "
                        "rank_seeds.py auto-ranks them (seed_ranking.md). "
                        "~60-65s per seed. Skips already-rendered cells, so "
                        "re-running with a bigger N resumes.")
    # --- Batch mode: walk pre-organized T<N>_<seed>/ folders, one tier per folder ---
    p.add_argument("--batch-tiered", type=Path, default=None,
                   help="BATCH MODE: parent directory containing T<N>_<seed>/ "
                        "subfolders (matches the daemon's tier dict). Default expected "
                        "location: drop tier folders into <hotfolder>/inbox and answer "
                        "'y' to the batch-mode prompt at startup. Daemon walks each "
                        "subfolder, applies the matching tier+seed, runs process_one "
                        "on every image inside. Bypasses single-tier hot-folder loop. "
                        "Outputs land in datasets/<slug>/T<N>_seed<seed>_<...>/ so "
                        "re-running the same asset at different (tier, seed) doesn't "
                        "collide.")
    p.add_argument("--batch-order", choices=["largest-first", "alphabetical", "random"],
                   default="largest-first",
                   help="Order to process batch subfolders. largest-first puts the "
                        "long-pole jobs first so overnight runs maximize useful work "
                        "even if interrupted late.")
    p.add_argument("--batch-failure-policy",
                   choices=["skip", "abort-folder", "abort-all"],
                   default="skip",
                   help="On per-image failure in batch mode: skip=continue to next image "
                        "(default, robust for overnight runs); abort-folder=skip remaining "
                        "images in the current tier folder, move to next folder; "
                        "abort-all=stop the entire batch immediately.")
    # --- TripoSplat quick-splat companion output ---
    p.add_argument("--quick-splat", action=argparse.BooleanOptionalAction, default=True,
                   help="Also produce a fast feed-forward TripoSplat Gaussian splat "
                        "(~25s/asset incl. model load, runs as an isolated subprocess "
                        "before mesh generation so VRAM never overlaps with rendering). "
                        "Output lands in datasets/<slug>/quick_splat/ — a browser-viewable "
                        ".splat + PostShot-ingestible .ply preview alongside the full "
                        "COLMAP dataset. Requires ~/projects/TripoSplat with weights. "
                        "Failure is non-fatal (logged, pipeline continues).")
    p.add_argument("--triposplat-repo", type=Path,
                   default=Path.home() / "projects" / "TripoSplat",
                   help="Path to the TripoSplat repo + ckpts/ for --quick-splat.")
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
        # Batch mode sets this per-folder so process_one prefixes the dataset
        # subdir name with T<tier>_seed<seed>_ — prevents collisions when the
        # same asset runs at multiple (tier, seed) combos.
        self._batch_dataset_prefix: str = ""

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

    def _resolve_adaptive_fov(
        self, mesh_verts_np: np.ndarray, w2c: np.ndarray, intr: dict,
    ) -> tuple[float, dict]:
        """Resolve the FOV + intrinsics for this asset.

        Returns (fov_deg, intrinsics_dict). When adaptive FOV is enabled (default),
        computes the worst-view-fits-all FOV from the actual mesh + camera rig,
        then derives matching intrinsics. When disabled, returns the daemon's
        fixed --fov and the pre-built intr unchanged.
        """
        if not self.args.adaptive_fov:
            return float(self.args.fov), intr
        fov = compute_adaptive_fov(
            mesh_verts_np, w2c,
            margin=self.args.adaptive_fov_margin,
            fov_min=self.args.adaptive_fov_min,
            fov_max=self.args.adaptive_fov_max,
        )
        new_intr = intrinsics_from_fov(fov, self.args.resolution)
        self.log(f"  adaptive FOV: {self.args.fov:.1f}° → {fov:.2f}° "
                 f"(margin {self.args.adaptive_fov_margin:.2f})")
        return float(fov), new_intr

    def run_quick_splat(self, image_path: Path, slug_root: Path) -> None:
        """Produce a TripoSplat quick-splat preview as an isolated subprocess.

        Runs BEFORE mesh generation while the GPU is at idle baseline — the
        subprocess holds ~7.5 GB only for its ~25s lifetime and fully releases
        on exit, so it never overlaps with Pixal3D's 20 GB render peaks.
        Non-fatal on any failure: a missing TripoSplat install or a bad input
        just logs a warning and the main pipeline continues.
        """
        out_dir = slug_root / "quick_splat"
        if any(out_dir.glob("quick_splat_*.splat")):
            self.log(f"  quick-splat: already exists, skipping")
            return
        script = Path(__file__).resolve().parent / "quick_splat.py"
        if not script.exists():
            self.log(f"  quick-splat WARN: {script} not found, skipping")
            return
        t0 = time.time()
        try:
            result = subprocess.run(
                [sys.executable, str(script), str(image_path), str(out_dir),
                 "--triposplat-repo", str(self.args.triposplat_repo),
                 "--seed", str(self.args.seed)],
                capture_output=True, text=True, timeout=300,
            )
            tail = (result.stdout or "").strip().splitlines()
            msg = tail[-1] if tail else "(no output)"
            if result.returncode == 0:
                self.log(f"  quick-splat ✓ ({time.time() - t0:.1f}s) → {out_dir}")
            else:
                self.log(f"  quick-splat WARN (rc={result.returncode}): {msg}")
        except subprocess.TimeoutExpired:
            self.log(f"  quick-splat WARN: timed out after 300s, continuing")
        except OSError as e:
            self.log(f"  quick-splat WARN: {e}")

    def process_one(self, bb: Backbone, envmap, w2c, intr,
                    image_path: Path) -> Path:
        slug = slug_for(image_path)
        slug_root = self.datasets / slug
        mode = self.args.mode

        # TripoSplat quick-splat companion (isolated subprocess, GPU-idle window)
        if self.args.quick_splat:
            self.run_quick_splat(image_path, slug_root)

        # Dataset subdir name (per-config); batch mode prefixes with T<N>_seed<S>_ to
        # keep multi-(tier, seed) runs of the same asset side-by-side without colliding.
        hdri_stem = self.args.hdri.stem
        ds_subdir = (f"{self._batch_dataset_prefix}"
                     f"{hdri_stem}_{self.args.num_views}v_{self.args.resolution}px")

        # Skip-existing logic — different markers per mode
        if mode == "mesh":
            mesh_marker = slug_root / "mesh.glb"
            if mesh_marker.exists():
                self.log(f"  SKIP {slug}: mesh already exists at {mesh_marker}")
                return slug_root
        else:
            splat_dir = slug_root / ds_subdir
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
            out_dir = slug_root / ds_subdir
            images_dir = out_dir / "images"
            images_dir.mkdir(parents=True, exist_ok=True)
            names: list[str] = []
            n_views = self.args.num_views

            # --- Adaptive FOV (per-asset, recomputed for each mesh) ---
            mesh_verts_np = mesh.vertices.detach().cpu().numpy()
            effective_fov, render_intr = self._resolve_adaptive_fov(
                mesh_verts_np, w2c, intr,
            )

            for ck_start in range(0, n_views, self.args.chunk_size):
                ck_end = min(ck_start + self.args.chunk_size, n_views)
                self.log(f"  chunk {ck_start:>3}-{ck_end-1:<3} "
                         f"({ck_end - ck_start} views)")
                frames = render_multiview(
                    mesh, w2c[ck_start:ck_end], fov_deg=effective_fov,
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

            n_init = min(self.args.num_init_points, len(mesh_verts_np))
            rng = np.random.default_rng(self.args.seed)
            init_pts = mesh_verts_np[rng.choice(len(mesh_verts_np), size=n_init, replace=False)]
            write_colmap_dataset(out_dir, w2c, names, render_intr, points=init_pts)
            write_nerfstudio_dataset(out_dir, w2c, names, render_intr)
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

        # TripoSplat quick-splat companion — once per image regardless of how
        # many tier/seed cells this probe runs (run_quick_splat self-skips if
        # datasets/<slug>/quick_splat/ already has output from any prior run).
        if self.args.quick_splat:
            self.run_quick_splat(image_path, slug_root)

        self.log(f"  PROBE {image_path.name} → slug={slug}")
        image = Image.open(image_path)
        if self.args.force_rembg and image.mode == "RGBA":
            image = image.convert("RGB")
            self.log(f"  force-rembg: stripped alpha, BiRefNet will run")

        view_w2c = w2c[PROBE_VIEW:PROBE_VIEW + 1]
        # Seed sweep = one tier x many seeds; classic probe = all 5 tiers.
        if self.args.seed_sweep > 0:
            tier_items = [(self.args.tier, SAMPLER_TIERS[self.args.tier])]
        else:
            tier_items = list(SAMPLER_TIERS.items())
        n_total = len(tier_items) * len(self.probe_seeds)
        run_idx = 0

        # Per-cell production data accumulated here
        cells: list[dict] = []
        n_skipped = 0
        cuda_fatal = False
        batch_start = time.time()
        started_at = datetime.now().isoformat()
        input_w, input_h = image.size
        input_mode = image.mode

        for tier_key, (tier_name, _, _, _) in tier_items:
            _apply_tier_to_backbone(bb, tier_key)
            eff_sparse = dict(bb.sparse_structure_sampler_params)
            eff_shape = dict(bb.shape_slat_sampler_params)
            eff_tex = dict(bb.tex_slat_sampler_params)
            for seed in self.probe_seeds:
                run_idx += 1
                out_path = probe_dir / f"{PROBE_VIEW:03d}_T{tier_key}_{tier_name.lower()}_seed{seed}.png"
                if out_path.exists():
                    self.log(f"  [{run_idx}/{n_total}] tier {tier_key} ({tier_name}) seed={seed}  SKIP (exists)")
                    n_skipped += 1
                    continue
                self.log(f"  [{run_idx}/{n_total}] tier {tier_key} ({tier_name}) seed={seed}")
                t_cell = time.time()
                torch.cuda.reset_peak_memory_stats()
                try:
                    t_mesh = time.time()
                    mesh = bb.run(image, seed=seed)
                    mesh_gen_s = time.time() - t_mesh
                    raw_verts = int(mesh.vertices.shape[0])
                    raw_faces = int(mesh.faces.shape[0])

                    decimated = False
                    post_verts = raw_verts
                    post_faces = raw_faces
                    if raw_faces > self.args.max_faces:
                        mesh.simplify(target=self.args.max_faces, verbose=False)
                        decimated = True
                        post_verts = int(mesh.vertices.shape[0])
                        post_faces = int(mesh.faces.shape[0])
                        self.log(f"    mesh: {raw_verts:,}v / {raw_faces:,}f "
                                 f"→ DECIM → {post_verts:,}v / {post_faces:,}f")
                    else:
                        self.log(f"    mesh: {raw_verts:,}v / {raw_faces:,}f (under {self.args.max_faces:,} cap, no decim)")
            
                    # Adaptive FOV for the probe view (per-cell — mesh changes per tier/seed).
                    if not self.args.adaptive_fov:
                        cell_fov = float(self.args.fov)
                    else:
                        cell_fov = float(compute_adaptive_fov(
                            mesh.vertices.detach().cpu().numpy(), view_w2c,
                            margin=self.args.adaptive_fov_margin,
                            fov_min=self.args.adaptive_fov_min,
                            fov_max=self.args.adaptive_fov_max,
                        ))
                        self.log(f"    adaptive FOV: {self.args.fov:.1f}° → {cell_fov:.2f}°")

                    t_render = time.time()
                    frames = render_multiview(
                        mesh, view_w2c, fov_deg=cell_fov,
                        resolution=self.args.resolution, envmap=envmap,
                        repo_dir=bb.repo_dir, device=bb.device, verbose=False,
                    )
                    render_s = time.time() - t_render

                    t_save = time.time()
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
                    polish_save_s = time.time() - t_save

                    total_s = time.time() - t_cell
                    gpu_peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
                    out_size_mb = out_path.stat().st_size / (1024 ** 2)

                    cells.append({
                        "tier_key": tier_key,
                        "tier_name": tier_name,
                        "seed": seed,
                        "out_file": out_path.name,
                        "mesh": {
                            "raw_verts": raw_verts,
                            "raw_faces": raw_faces,
                            "decimated": decimated,
                            "post_decim_verts": post_verts,
                            "post_decim_faces": post_faces,
                        },
                        "timing_s": {
                            "total": round(total_s, 2),
                            "mesh_gen": round(mesh_gen_s, 2),
                            "render": round(render_s, 2),
                            "polish_save": round(polish_save_s, 2),
                        },
                        "output_size_mb": round(out_size_mb, 2),
                        "gpu_peak_gb": round(gpu_peak_gb, 2),
                        "effective_fov_deg": round(cell_fov, 2),
                        "effective_sampler_params": {
                            "sparse": eff_sparse,
                            "shape": eff_shape,
                            "tex": eff_tex,
                        },
                    })

                    self.log(f"    ✓ {out_path.name} ({total_s:.1f}s, gpu peak {gpu_peak_gb:.1f}GB)")
                    del frames, mesh
                    gc.collect()
                    torch.cuda.empty_cache()
                except Exception as e:
                    self.log(f"    FAIL: {e}")
                    cells.append({
                        "tier_key": tier_key,
                        "tier_name": tier_name,
                        "seed": seed,
                        "error": str(e),
                    })
                    if _is_cuda_fatal(e):
                        # The CUDA context is poisoned — every subsequent cell
                        # would fail in seconds (lived this 2026-06-02: one
                        # "device not ready" → 19 cascade FAILs + zombie daemon).
                        # Abort the asset AND stop the daemon; a fresh process
                        # is the only reliable recovery.
                        self.log("  ✗ CUDA-FATAL: context unrecoverable — aborting "
                                 "asset and stopping daemon. Restart to continue "
                                 "(skip-existing will resume).")
                        cuda_fatal = True
                        self.stopping = True
                        break
            if self.stopping:
                break

        total_wall = time.time() - batch_start
        meta = {
            "image": image_path.name,
            "slug": slug,
            "input_image": {
                "width": input_w,
                "height": input_h,
                "mode": input_mode,
            },
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
            "adaptive_fov": {
                "enabled": bool(self.args.adaptive_fov),
                "margin": self.args.adaptive_fov_margin,
                "min_deg": self.args.adaptive_fov_min,
                "max_deg": self.args.adaptive_fov_max,
            },
            "polish": {
                "sharpen_percent": self.args.polish_sharpen_percent,
                "sharpen_radius": self.args.polish_sharpen_radius,
                "sharpen_threshold": self.args.polish_sharpen_threshold,
                "grain_strength": self.args.polish_grain_strength,
                "grain_monochrome": self.args.polish_grain_monochrome,
            },
            "batch": {
                "started_at": started_at,
                "finished_at": datetime.now().isoformat(),
                "total_wall_time_s": round(total_wall, 2),
                "n_cells_total": n_total,
                "n_cells_run": len(cells),
                "n_cells_skipped": n_skipped,
            },
            "cells": cells,
        }

        try:
            (probe_dir / "probe_meta.json").write_text(json.dumps(meta, indent=2))
        except OSError:
            pass
        try:
            (probe_dir / "probe_summary.md").write_text(_build_probe_summary_md(meta))
        except OSError:
            pass

        # Auto-rank the cells by plateau-faceting score (seed selection).
        try:
            from rank_seeds import rank_probe_dir
            ranked = rank_probe_dir(probe_dir)
            if ranked:
                top = ranked[:3]
                self.log("  seed ranking (lower = smoother, full table in "
                         "probe/seed_ranking.md):")
                for i, c in enumerate(top, 1):
                    self.log(f"    {i}. T{c['tier']} seed={c['seed']}  "
                             f"score={c['score']:.2f}")
        except Exception as e:  # noqa: BLE001 — ranking is advisory
            self.log(f"  seed-ranking WARN: {e}")

        if cuda_fatal:
            # Meta is written for diagnosis, but the asset must NOT be marked
            # complete — raise so the outer loop moves the source to failed/.
            raise RuntimeError(
                f"CUDA-fatal abort during probe ({len(cells)}/{n_total} cells attempted)"
            )

        self.log(f"  ✓ probe complete → {probe_dir} ({len(cells)} cells in {total_wall/60:.1f} min)")
        return slug_root

    def shutdown(self, *_):
        if not self.stopping:
            self.stopping = True
            self.log("Shutdown requested — finishing current item then exiting.")

    def run_batch(self, batch_parent: Path, bb, envmap, w2c, intr) -> int:
        """Walk T<N>_<seed>/ subfolders under batch_parent and process each.

        Each subfolder's tier+seed is parsed from its name; tier params + seed
        are applied to the backbone before that folder's images run. Outputs
        get a T<N>_seed<S>_ prefix on the dataset subdir name to avoid collisions
        when the same asset runs under multiple (tier, seed) combos.

        Returns shell-style exit code: 0=clean, 1=partial fail, 2=hard fail.
        """
        if not batch_parent.is_dir():
            self.log(f"BATCH FAIL: parent not a directory: {batch_parent}")
            return 2

        # Discover + parse
        jobs: list[tuple[str, int, Path, list[Path]]] = []
        for sub in sorted(batch_parent.iterdir()):
            if not sub.is_dir():
                continue
            parsed = parse_tier_folder(sub)
            if parsed is None:
                continue
            tier_key, seed = parsed
            images = sorted(
                p for p in sub.iterdir()
                if p.is_file() and p.suffix.lower() in VALID_EXT
                and not p.name.startswith(".") and not p.name.endswith(":Zone.Identifier")
            )
            if not images:
                continue
            jobs.append((tier_key, seed, sub, images))

        if not jobs:
            self.log(f"BATCH FAIL: no T<N>_<seed>/ folders with images under {batch_parent}")
            return 2

        # Order
        if self.args.batch_order == "largest-first":
            jobs.sort(key=lambda j: -len(j[3]))
        elif self.args.batch_order == "random":
            random.shuffle(jobs)
        # alphabetical = leave the sorted() order from discovery

        total_imgs = sum(len(j[3]) for j in jobs)
        self.log("=" * 60)
        self.log(f"BATCH MODE: {len(jobs)} tier-folders, {total_imgs} total images")
        self.log(f"  parent: {batch_parent}")
        self.log(f"  order : {self.args.batch_order}")
        self.log(f"  on-fail: {self.args.batch_failure_policy}")
        for tier_key, seed, sub, imgs in jobs:
            tier_name = SAMPLER_TIERS[tier_key][0]
            self.log(f"  [queued] T{tier_key} ({tier_name}) seed={seed:<5}  "
                     f"{len(imgs):>3} images  ({sub.name})")
        self.log("=" * 60)

        n_done = 0
        n_failed = 0
        per_image_times: deque[float] = deque(maxlen=20)
        batch_start = time.time()
        signal.signal(signal.SIGINT, self.shutdown)
        signal.signal(signal.SIGTERM, self.shutdown)

        for job_idx, (tier_key, seed, sub, images) in enumerate(jobs, start=1):
            if self.stopping:
                break
            tier_name = SAMPLER_TIERS[tier_key][0]
            self.log("")
            self.log("=" * 60)
            self.log(f"[batch folder {job_idx}/{len(jobs)}] "
                     f"T{tier_key} ({tier_name}) seed={seed}  "
                     f"{len(images)} images  ({sub.name})")
            self.log("=" * 60)

            # Apply tier + seed for this folder. _apply_tier mutates the backbone's
            # per-sampler dicts; self.args.seed mutates so process_one and the
            # grain seed pick up the new value.
            _apply_tier_to_backbone(bb, tier_key)
            self.args.seed = seed
            self._batch_dataset_prefix = f"T{tier_key}_seed{seed}_"

            folder_aborted = False
            for img_idx, img_path in enumerate(images, start=1):
                if self.stopping:
                    break
                self.log(f"  [{job_idx}.{img_idx}/{len(images)}] {img_path.name}")
                t0 = time.time()
                try:
                    self.process_one(bb, envmap, w2c, intr, img_path)
                    dt = time.time() - t0
                    per_image_times.append(dt)
                    n_done += 1
                    avg = sum(per_image_times) / len(per_image_times)
                    remaining = total_imgs - (n_done + n_failed)
                    eta_h = remaining * avg / 3600
                    elapsed_h = (time.time() - batch_start) / 3600
                    self.log(f"  ✓ {img_path.name} in {dt:.1f}s  "
                             f"(batch: {n_done}/{total_imgs} done, {n_failed} failed, "
                             f"{elapsed_h:.1f}h elapsed, ~{eta_h:.1f}h to go)")
                except Exception as e:
                    n_failed += 1
                    self.log(f"  ✗ FAIL {img_path.name}: {e}")
                    self.log(traceback.format_exc())
                    if _is_cuda_fatal(e):
                        # Poisoned CUDA context — overrides failure policy.
                        # No further GPU work can succeed in this process.
                        self.log("  ✗ CUDA-FATAL: context unrecoverable — stopping "
                                 "batch. Restart daemon to resume (skip-existing "
                                 "picks up where this left off).")
                        self._batch_dataset_prefix = ""
                        return 2
                    policy = self.args.batch_failure_policy
                    if policy == "abort-all":
                        self.log(f"  abort-all: stopping batch "
                                 f"({n_done} done, {n_failed} failed)")
                        self._batch_dataset_prefix = ""
                        return 2
                    if policy == "abort-folder":
                        self.log(f"  abort-folder: skipping remaining "
                                 f"{len(images) - img_idx} images in {sub.name}")
                        folder_aborted = True
                        break

            if folder_aborted:
                continue

        self._batch_dataset_prefix = ""
        total_h = (time.time() - batch_start) / 3600
        self.log("")
        self.log("=" * 60)
        self.log(f"BATCH COMPLETE: {n_done} done, {n_failed} failed, "
                 f"{total_h:.2f}h total wall time")
        self.log("=" * 60)
        if n_failed == 0:
            return 0
        if n_done == 0:
            return 2
        return 1

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
                # Batch mode first — if yes, skip tier/seed prompts entirely
                if self.args.batch_tiered is None:
                    batch_parent = interactive_select_batch_mode(self.inbox)
                    if batch_parent is not None:
                        self.args.batch_tiered = batch_parent
                # Common prompts (apply to both modes)
                self.args.hdri = interactive_select_hdri(self.args.hdri)
                self.args.force_rembg = interactive_select_force_rembg(self.args.force_rembg)
                # Tier/seed prompts only apply to single-tier mode
                if self.args.batch_tiered is None:
                    self.args.seed = interactive_select_seed(self.args.seed)
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
        if self.args.seed_sweep > 0:
            # Seed sweep: one tier × N seeds through the probe pipeline.
            if self.args.tier == PROBE_TIER_KEY:
                self.args.tier = "5"        # sweep needs a concrete tier
            self.probe_seeds = build_probe_seed_list(
                self.args.seed_sweep, anchor_seed=self.args.seed
            )
        elif self.args.tier == PROBE_TIER_KEY:
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
        if self.args.batch_tiered is None:
            self.log(f"  seed      : {self.args.seed}")
        self.log(f"  exposure  : x{self.args.hdri_exposure}")
        self.log(f"  pipeline  : {self.args.pipeline_type}")
        self.log(f"  max_tokens: {self.args.max_num_tokens:,}")
        if self.args.batch_tiered is not None:
            self.log(f"  batch     : {self.args.batch_tiered}")
            self.log(f"  order     : {self.args.batch_order}")
            self.log(f"  on-fail   : {self.args.batch_failure_policy}")
        elif self.args.seed_sweep > 0:
            t_name = SAMPLER_TIERS[self.args.tier][0]
            self.log(f"  sampler   : SEED SWEEP — tier {self.args.tier} ({t_name}) "
                     f"× {len(self.probe_seeds)} seeds at view {PROBE_VIEW} "
                     f"(~{len(self.probe_seeds)} min/asset, auto-ranked)")
            self.log(f"  seeds     : {self.probe_seeds[:10]}"
                     f"{' ...' if len(self.probe_seeds) > 10 else ''}")
        elif self.args.tier == PROBE_TIER_KEY:
            self.log(f"  sampler   : PROBE mode — all 5 tiers at view {PROBE_VIEW}")
            self.log(f"  seeds     : {self.probe_seeds}")
        else:
            t_name, t_steps, t_sg, t_tg = SAMPLER_TIERS[self.args.tier]
            self.log(f"  sampler   : tier {self.args.tier} ({t_name})  "
                     f"steps={t_steps}  shape={t_sg}  tex={t_tg}")
        self.log(f"  rembg     : {'force BiRefNet' if self.args.force_rembg else 'auto (use input alpha if present)'}")
        if self.args.adaptive_fov:
            self.log(f"  adapt-FOV : ON  per-asset uniform  "
                     f"(margin {self.args.adaptive_fov_margin:.2f}, "
                     f"clamp {self.args.adaptive_fov_min:.0f}-{self.args.adaptive_fov_max:.0f}°)")
        else:
            self.log(f"  adapt-FOV : OFF — fixed {self.args.fov}°")
        self.log(f"  quick-splat: {'ON (TripoSplat preview per asset)' if self.args.quick_splat else 'off'}")
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

        # --- Batch mode short-circuit: walk T<N>_<seed>/ subfolders and exit ---
        if self.args.batch_tiered is not None:
            if self.args.mode == "mesh":
                self.log("Batch mode requires splat output (mesh-only currently "
                         "unsupported in batch). Set --mode splat or both.")
                return 2
            return self.run_batch(self.args.batch_tiered, bb, envmap, w2c, intr)

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
                    if self.args.tier == PROBE_TIER_KEY or self.args.seed_sweep > 0:
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
                    if _is_cuda_fatal(e):
                        self.log("✗ CUDA-FATAL: context unrecoverable — daemon "
                                 "exiting. Restart to continue processing.")
                        self.stopping = True
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
