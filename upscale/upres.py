"""AuraSR 4x generative upscale — folder-to-folder.

Reads images from --input, writes 4x upscaled PNGs to --output.
Preserves filename stems. Preserves RGBA alpha via separate LANCZOS upscale.
Skips files that already exist in --output.

AuraSR is GAN-based — adds plausible micro-detail (vs pure SR which only
sharpens what's already there). Best for inputs that look slightly soft,
low-res, or compressed.

Usage:
  python upres.py --input /path/to/in --output /path/to/out
"""
from __future__ import annotations
import argparse
import gc
import sys
import time
from pathlib import Path

from PIL import Image
import torch
from aura_sr import AuraSR

VALID_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", "--src", dest="input", type=Path, required=True,
                   help="Input folder containing images to upscale.")
    p.add_argument("--output", "--out", dest="output", type=Path, required=True,
                   help="Output folder for upscaled PNGs.")
    p.add_argument("--model", type=str, default="fal/AuraSR-v2",
                   help="HuggingFace AuraSR model id.")
    p.add_argument("--quiet", action="store_true",
                   help="Less verbose per-file logging.")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if not args.input.exists():
        print(f"FAIL: input dir missing: {args.input}", file=sys.stderr)
        return 1
    args.output.mkdir(parents=True, exist_ok=True)

    inputs = sorted([p for p in args.input.iterdir()
                     if p.is_file() and p.suffix.lower() in VALID_EXT])
    if not inputs:
        print(f"no images found in {args.input}")
        return 0

    print(f"=== AuraSR 4x generative upscale ===", flush=True)
    print(f"src:    {args.input}", flush=True)
    print(f"out:    {args.output}", flush=True)
    print(f"count:  {len(inputs)}", flush=True)

    print(f"\nLoading AuraSR ({args.model})...", flush=True)
    t0 = time.time()
    aura = AuraSR.from_pretrained(args.model)
    print(f"  loaded in {time.time()-t0:.1f}s", flush=True)

    global_t0 = time.time()
    done = skipped = failed = 0

    for i, src in enumerate(inputs, 1):
        out_path = args.output / f"{src.stem}.png"
        if out_path.exists():
            if not args.quiet:
                print(f"[{i}/{len(inputs)}] {src.name}  SKIP (exists)", flush=True)
            skipped += 1
            continue

        t_run = time.time()
        try:
            im = Image.open(src)
            mode = im.mode
            if mode == "RGBA":
                rgb = im.convert("RGB")
                alpha = im.getchannel("A")
                up_rgb = aura.upscale_4x_overlapped(rgb)
                up_alpha = alpha.resize(up_rgb.size, Image.LANCZOS)
                up = Image.merge("RGBA", (*up_rgb.split(), up_alpha))
            else:
                rgb = im.convert("RGB")
                up = aura.upscale_4x_overlapped(rgb)

            up.save(out_path, optimize=True)
            dt = time.time() - t_run
            avg = (time.time() - global_t0) / max(1, (i - skipped))
            eta_min = (len(inputs) - i) * avg / 60
            in_size = f"{im.size[0]}x{im.size[1]}"
            out_size = f"{up.size[0]}x{up.size[1]}"
            print(f"[{i}/{len(inputs)}] {src.name}  {in_size} -> {out_size}  "
                  f"mode={mode}  ({dt:.1f}s · ETA ~{eta_min:.0f} min)", flush=True)
            done += 1
            del up, im
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"[{i}/{len(inputs)}] {src.name}  FAIL: {e}", flush=True)
            failed += 1

    print(f"\n=== DONE in {(time.time()-global_t0)/60:.1f} min ===", flush=True)
    print(f"  done:    {done}", flush=True)
    print(f"  skipped: {skipped}", flush=True)
    print(f"  failed:  {failed}", flush=True)
    print(f"  out:     {args.output}", flush=True)
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
