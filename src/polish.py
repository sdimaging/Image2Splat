"""
Lightweight image-polish primitives — unsharp mask, film grain, color correction.

All ops:
  * accept an RGBA uint8 (H, W, 4) numpy array
  * preserve alpha exactly (only RGB is touched)
  * are deterministic given a fixed seed

Used by both:
  * `scripts/polish_dataset.py` for retroactive polishing of existing datasets
  * `scripts/hotfolder_daemon.py` to auto-polish every rendered frame before saving
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


# ---------- Sharpening ----------

def unsharp_rgb(rgb: np.ndarray, percent: int, radius: float, threshold: int) -> np.ndarray:
    """PIL UnsharpMask on RGB uint8 array. No-op if percent <= 0."""
    if percent <= 0:
        return rgb
    img = Image.fromarray(rgb).filter(
        ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=threshold)
    )
    return np.array(img)


# ---------- Film grain ----------

def add_grain(rgb: np.ndarray, sigma: float, seed: int, monochrome: bool = False) -> np.ndarray:
    """Add Gaussian luminance grain (σ in 0-255 units). Deterministic via seed.

    For multi-view splat training, use a different seed per view (e.g. base + view_idx)
    so views don't have identical patterns but are reproducible.
    """
    if sigma <= 0:
        return rgb
    rng = np.random.default_rng(seed)
    H, W, _ = rgb.shape
    if monochrome:
        noise = rng.normal(0.0, sigma, size=(H, W, 1)).repeat(3, axis=-1)
    else:
        noise = rng.normal(0.0, sigma, size=(H, W, 3))
    return np.clip(rgb.astype(np.float32) + noise, 0, 255).astype(np.uint8)


# ---------- Manual color adjustments (PIL.ImageEnhance + gamma LUT) ----------

def adjust_color_pil(rgb_pil: Image.Image, brightness: float = 1.0,
                     contrast: float = 1.0, saturation: float = 1.0,
                     gamma: float = 1.0) -> Image.Image:
    if brightness != 1.0:
        rgb_pil = ImageEnhance.Brightness(rgb_pil).enhance(brightness)
    if contrast != 1.0:
        rgb_pil = ImageEnhance.Contrast(rgb_pil).enhance(contrast)
    if saturation != 1.0:
        rgb_pil = ImageEnhance.Color(rgb_pil).enhance(saturation)
    if gamma != 1.0:
        inv = 1.0 / gamma
        lut = np.array([((v / 255.0) ** inv) * 255.0 for v in range(256)], dtype=np.uint8)
        rgb_pil = Image.fromarray(lut[np.array(rgb_pil)])
    return rgb_pil


# ---------- Histogram matching (single-image reference, per-channel) ----------

def hist_match_rgb(src_rgb: np.ndarray, ref_rgb: np.ndarray) -> np.ndarray:
    """Per-channel histogram match. uint8 in/out."""
    out = np.empty_like(src_rgb)
    for c in range(3):
        s_vals, s_counts = np.unique(src_rgb[..., c].ravel(), return_counts=True)
        r_vals, r_counts = np.unique(ref_rgb[..., c].ravel(), return_counts=True)
        s_cdf = np.cumsum(s_counts).astype(np.float64) / s_counts.sum()
        r_cdf = np.cumsum(r_counts).astype(np.float64) / r_counts.sum()
        lut = np.zeros(256, dtype=np.uint8)
        for v in range(256):
            cdf_v = np.interp(v, s_vals, s_cdf, left=0.0, right=1.0)
            r_idx = np.searchsorted(r_cdf, cdf_v)
            r_idx = min(max(r_idx, 0), len(r_vals) - 1)
            lut[v] = r_vals[r_idx]
        out[..., c] = lut[src_rgb[..., c]]
    return out


# ---------- 3D LUT — extract from a single before/after pair, then apply ----------

def build_3d_lut(orig_rgb: np.ndarray, corrected_rgb: np.ndarray,
                 mask: np.ndarray | None = None,
                 lut_size: int = 33, max_samples: int = 500_000) -> np.ndarray:
    """Build a (N, N, N, 3) float32 LUT in [0, 1] from a single before/after image pair.

    Each LUT bin averages the corrected colors of all source pixels whose original
    color falls in that bin. Empty bins fall back to identity (no change).

    Args:
        orig_rgb / corrected_rgb: (H, W, 3) uint8 arrays.
        mask: optional (H, W) bool array — only pixels where mask is True
            contribute to the LUT. Use this to exclude background pixels in
            RGBA datasets (Photoshop's editor compositing on the canvas color
            corrupts (0,0,0)-under-alpha regions and poisons the LUT).
        lut_size: bins per axis. 33³ is standard; raise to 65 for strong corrections.
        max_samples: random subsample cap (3000² = 9M pixels otherwise).
    """
    if orig_rgb.shape != corrected_rgb.shape:
        raise ValueError(f"shape mismatch: {orig_rgb.shape} vs {corrected_rgb.shape}")

    src = orig_rgb.reshape(-1, 3).astype(np.float32) / 255.0
    dst = corrected_rgb.reshape(-1, 3).astype(np.float32) / 255.0
    if mask is not None:
        flat_mask = mask.reshape(-1).astype(bool)
        if flat_mask.shape[0] != src.shape[0]:
            raise ValueError(f"mask size {flat_mask.shape} mismatch with image")
        src = src[flat_mask]
        dst = dst[flat_mask]
        if src.shape[0] == 0:
            raise ValueError("mask is all False — no samples to build LUT from")

    if len(src) > max_samples:
        idx = np.random.default_rng(0).choice(len(src), max_samples, replace=False)
        src, dst = src[idx], dst[idx]

    N = lut_size
    lut = np.zeros((N, N, N, 3), dtype=np.float64)
    counts = np.zeros((N, N, N), dtype=np.int64)

    bin_idx = np.clip((src * (N - 1) + 0.5).astype(int), 0, N - 1)
    np.add.at(lut, (bin_idx[:, 0], bin_idx[:, 1], bin_idx[:, 2]), dst)
    np.add.at(counts, (bin_idx[:, 0], bin_idx[:, 1], bin_idx[:, 2]), 1)

    nonempty = counts > 0
    lut[nonempty] /= counts[nonempty, None]

    # Fill empty bins with identity — pixels with no source-coverage shouldn't shift
    grid = np.indices((N, N, N), dtype=np.float32) / (N - 1)  # (3, N, N, N)
    grid = grid.transpose(1, 2, 3, 0)  # (N, N, N, 3)
    lut[~nonempty] = grid[~nonempty]

    coverage = nonempty.sum() / (N ** 3)
    return lut.astype(np.float32), coverage


def apply_3d_lut(rgb: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """Apply a (N, N, N, 3) LUT to an RGB uint8 image via trilinear interpolation."""
    N = lut.shape[0]
    src = rgb.astype(np.float32) / 255.0 * (N - 1)
    i0 = np.floor(src).astype(np.int32).clip(0, N - 2)
    f = src - i0  # (H, W, 3) fractional part in [0, 1)
    # Trilinear over the 8 corners
    out = np.zeros((*rgb.shape[:2], 3), dtype=np.float32)
    for di in (0, 1):
        for dj in (0, 1):
            for dk in (0, 1):
                wi = (1 - f[..., 0]) if di == 0 else f[..., 0]
                wj = (1 - f[..., 1]) if dj == 0 else f[..., 1]
                wk = (1 - f[..., 2]) if dk == 0 else f[..., 2]
                w = (wi * wj * wk)[..., None]
                ii = (i0[..., 0] + di).clip(0, N - 1)
                jj = (i0[..., 1] + dj).clip(0, N - 1)
                kk = (i0[..., 2] + dk).clip(0, N - 1)
                out += w * lut[ii, jj, kk]
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


# ---------- Convenience wrapper: polish an RGBA image with default settings ----------

def polish_rgba(rgba: np.ndarray, *, image_index: int,
                sharpen_percent: int = 40, sharpen_radius: float = 1.5,
                sharpen_threshold: int = 3,
                grain_strength: float = 3.0, grain_seed_base: int = 42,
                grain_monochrome: bool = False) -> np.ndarray:
    """Apply default polish (sharpen + grain) and return RGBA uint8.

    Alpha is preserved exactly. Used inline by the hot-folder daemon's save step.
    """
    rgb = rgba[..., :3]
    alpha = rgba[..., 3:]
    rgb = unsharp_rgb(rgb, percent=sharpen_percent,
                      radius=sharpen_radius, threshold=sharpen_threshold)
    rgb = add_grain(rgb, sigma=grain_strength,
                    seed=grain_seed_base + image_index,
                    monochrome=grain_monochrome)
    return np.concatenate([rgb, alpha], axis=-1)
