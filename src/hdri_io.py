"""
HDRI (EXR) reader that handles Photoshop's multi-layer EXR exports.

OpenCV's `cv2.imread` only understands single-layer R/G/B EXRs and returns
None on multi-layer files (channel names like `Background.R/G/B` +
`Layer 1.R/G/B`). For those, we fall back to the OpenEXR python binding and
pick the canonical RGB triple, in preference order:
    1. bare "R/G/B" (single-layer)
    2. "Background.R/G/B" (PS "Background" = the visible flattened result)
    3. first layer with all three RGB channels
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def read_exr_rgb(hdri_path: Path | str) -> np.ndarray:
    """Read an EXR as (H, W, 3) float32 RGB (linear, HDR). Handles multi-layer EXRs."""
    import cv2
    hdri_path = Path(hdri_path)
    img = cv2.imread(str(hdri_path), cv2.IMREAD_UNCHANGED)
    if img is not None:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)

    # OpenCV failed → multi-layer EXR fallback
    import OpenEXR
    exr = OpenEXR.InputFile(str(hdri_path))
    header = exr.header()
    chans = list(header["channels"].keys())
    dw = header["displayWindow"]
    W = dw.max.x - dw.min.x + 1
    H = dw.max.y - dw.min.y + 1

    candidates: list[str] = []
    for layer_prefix in ("", "Background."):
        if all(f"{layer_prefix}{c}" in chans for c in "RGB"):
            candidates = [f"{layer_prefix}R", f"{layer_prefix}G", f"{layer_prefix}B"]
            break
    if not candidates:
        layers = {c.rsplit(".", 1)[0] for c in chans if "." in c}
        for layer in sorted(layers):
            if all(f"{layer}.{c}" in chans for c in "RGB"):
                candidates = [f"{layer}.R", f"{layer}.G", f"{layer}.B"]
                break
    if not candidates:
        raise ValueError(f"{hdri_path.name}: no RGB triple in channels {chans}")

    bufs = exr.channels(candidates)
    rgb = np.stack([np.frombuffer(b, dtype=np.float32).reshape(H, W) for b in bufs],
                   axis=-1).copy()
    return rgb
