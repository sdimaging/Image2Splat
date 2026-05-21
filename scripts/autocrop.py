"""Autocrop foundational tool — pixel-statistic bbox detection + tight crop.

Walks a folder of input images, identifies the bounding box of NON-background
pixels (background = near-white OR solid-black OR transparent), expands the
bbox by a fixed pixel margin on each side, and writes the crop IN PLACE
(overwriting the original).

Designed as a pre-pass before AuraSR / Image2Splat daemon. The goal is to
maximize subject pixel density at Pixal3D's 1024-pixel image conditioning by
eliminating wasted negative-space pixels at the input stage.

Background detection logic (any one match = background):
  - RGBA alpha < ALPHA_THRESH (i.e., transparent)
  - All RGB channels > WHITE_THRESH (i.e., near-white)
  - All RGB channels < BLACK_THRESH (i.e., near-black / solid black)

Anything not matching is treated as subject. The bbox of subject pixels is
computed, then expanded by --margin on all 4 sides (clipped to image bounds).

Usage:
  python autocrop.py --input /path/to/folder [--margin 40] [--dry-run]
  python autocrop.py --input /path/to/folder --no-backup   # disable safety backup
"""
from __future__ import annotations
import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

VALID_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
DEFAULT_MARGIN = 40
WHITE_THRESH = 240   # all RGB channels > this = near-white background
BLACK_THRESH = 15    # all RGB channels < this = near-black background
ALPHA_THRESH = 10    # alpha < this = transparent background


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", "--src", dest="input", type=Path, required=True,
                   help="Folder containing images to autocrop in place.")
    p.add_argument("--margin", type=int, default=DEFAULT_MARGIN,
                   help=f"Pixel margin to add on each side of the subject bbox (default {DEFAULT_MARGIN}).")
    p.add_argument("--white-thresh", type=int, default=WHITE_THRESH,
                   help=f"Near-white RGB threshold; pixel is bg if all channels > this (default {WHITE_THRESH}).")
    p.add_argument("--black-thresh", type=int, default=BLACK_THRESH,
                   help=f"Solid-black RGB threshold; pixel is bg if all channels < this (default {BLACK_THRESH}).")
    p.add_argument("--alpha-thresh", type=int, default=ALPHA_THRESH,
                   help=f"Transparent alpha threshold; pixel is bg if alpha < this (default {ALPHA_THRESH}).")
    p.add_argument("--dry-run", action="store_true",
                   help="Print intended crops but don't write any files.")
    p.add_argument("--no-backup", action="store_true",
                   help="Disable the .original_backups/ safety mirror (default: enabled).")
    return p.parse_args()


def detect_subject_bbox(im: Image.Image, *, white_thresh: int, black_thresh: int,
                        alpha_thresh: int) -> tuple[int, int, int, int] | None:
    """Return (left, top, right, bottom) bbox of subject pixels, or None if no subject."""
    arr = np.array(im)
    h, w = arr.shape[:2]

    if im.mode == "RGBA":
        alpha = arr[..., 3]
        rgb = arr[..., :3]
        # subject = NOT (transparent) AND NOT (near-white) AND NOT (near-black)
        is_transparent = alpha < alpha_thresh
        is_white = (rgb > white_thresh).all(axis=2)
        is_black = (rgb < black_thresh).all(axis=2)
        is_bg = is_transparent | is_white | is_black
    else:
        if im.mode != "RGB":
            im = im.convert("RGB")
            arr = np.array(im)
        is_white = (arr > white_thresh).all(axis=2)
        is_black = (arr < black_thresh).all(axis=2)
        is_bg = is_white | is_black

    is_subject = ~is_bg
    rows = np.any(is_subject, axis=1)
    cols = np.any(is_subject, axis=0)
    if not rows.any() or not cols.any():
        return None  # no subject pixels detected

    top, bottom = int(np.argmax(rows)), int(h - 1 - np.argmax(rows[::-1]))
    left, right = int(np.argmax(cols)), int(w - 1 - np.argmax(cols[::-1]))
    return left, top, right + 1, bottom + 1  # right/bottom exclusive (PIL convention)


def expand_bbox(bbox: tuple[int, int, int, int], margin: int,
                w: int, h: int) -> tuple[int, int, int, int]:
    """Expand bbox by margin on all sides, clipped to image bounds."""
    l, t, r, b = bbox
    return (max(0, l - margin), max(0, t - margin),
            min(w, r + margin), min(h, b + margin))


def main() -> int:
    args = parse_args()
    if not args.input.exists() or not args.input.is_dir():
        print(f"FAIL: input dir missing or not a directory: {args.input}", file=sys.stderr)
        return 1

    inputs = sorted([p for p in args.input.iterdir()
                     if p.is_file() and p.suffix.lower() in VALID_EXT])
    if not inputs:
        print(f"no images in {args.input}")
        return 0

    backup_dir = args.input / ".original_backups"
    if not args.no_backup and not args.dry_run:
        backup_dir.mkdir(exist_ok=True)

    print(f"=== Image2Splat AutoCrop ===", flush=True)
    print(f"input   : {args.input}", flush=True)
    print(f"images  : {len(inputs)}", flush=True)
    print(f"margin  : {args.margin}px", flush=True)
    print(f"thresh  : white > {args.white_thresh}, black < {args.black_thresh}, alpha < {args.alpha_thresh}", flush=True)
    print(f"dry-run : {args.dry_run}", flush=True)
    if not args.no_backup and not args.dry_run:
        print(f"backup  : {backup_dir}", flush=True)
    print()

    t0 = time.time()
    cropped = skipped = no_subject = 0

    for i, src in enumerate(inputs, 1):
        if src.parent == backup_dir or src.name.startswith(".") or src.name == ".original_backups":
            continue
        try:
            im = Image.open(src)
            orig_size = im.size  # (w, h)
            bbox = detect_subject_bbox(
                im,
                white_thresh=args.white_thresh,
                black_thresh=args.black_thresh,
                alpha_thresh=args.alpha_thresh,
            )
            if bbox is None:
                print(f"[{i}/{len(inputs)}] {src.name}  NO SUBJECT (all bg) — skipped", flush=True)
                no_subject += 1
                continue

            l, t, r, b = bbox
            new_box = expand_bbox(bbox, args.margin, orig_size[0], orig_size[1])
            nl, nt, nr, nb = new_box
            new_w, new_h = nr - nl, nb - nt
            subj_w, subj_h = r - l, b - t

            if new_w == orig_size[0] and new_h == orig_size[1]:
                # Already at edges — no actual cropping happens
                print(f"[{i}/{len(inputs)}] {src.name}  {orig_size[0]}x{orig_size[1]}  "
                      f"subj@({l},{t},{r},{b})={subj_w}x{subj_h}  → no crop needed", flush=True)
                skipped += 1
                continue

            if args.dry_run:
                print(f"[{i}/{len(inputs)}] {src.name}  {orig_size[0]}x{orig_size[1]}  "
                      f"subj={subj_w}x{subj_h}  → CROP to {new_w}x{new_h} @ ({nl},{nt})", flush=True)
                cropped += 1
                continue

            # Backup original
            if not args.no_backup:
                bk = backup_dir / src.name
                if not bk.exists():
                    shutil.copy2(src, bk)

            # Crop + save in place
            cropped_im = im.crop(new_box)
            # Preserve original format
            if src.suffix.lower() in (".jpg", ".jpeg"):
                cropped_im.convert("RGB").save(src, quality=95)
            else:
                cropped_im.save(src)

            print(f"[{i}/{len(inputs)}] {src.name}  {orig_size[0]}x{orig_size[1]}  "
                  f"subj={subj_w}x{subj_h}  → cropped to {new_w}x{new_h}", flush=True)
            cropped += 1
        except Exception as e:
            print(f"[{i}/{len(inputs)}] {src.name}  FAIL: {e}", flush=True)

    print(f"\n=== DONE in {time.time()-t0:.1f}s ===", flush=True)
    print(f"  cropped:    {cropped}", flush=True)
    print(f"  skipped:    {skipped}  (already at edges)", flush=True)
    print(f"  no subject: {no_subject}", flush=True)
    if not args.no_backup and not args.dry_run and cropped > 0:
        print(f"  backups at: {backup_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
