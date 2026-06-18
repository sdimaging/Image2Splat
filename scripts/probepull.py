#!/usr/bin/env python
"""probepull — stage probe-selected inputs back into inbox batch folders.

For every dataset folder directly under <hotfolder>/datasets/ that has a
`probe/` subdir, read the probe cells you kept (your "selects"), find that
asset's source image in `completed/`, and copy it into the matching
`inbox/T<tier>_<seed>/` batch folder — ready for a `--batch-tiered` run.

The "has a probe/ subdir" rule is deliberate: real dataset folders have one,
grouping folders like `Pre`/`Post` do NOT, so they're skipped automatically
and never recursed into. No folder name call-outs needed.

Multi-cell selects (an asset you kept at 2-3 tier/seed combos) are copied once
per cell, renamed `_1`, `_2`, ... so each lands in its own batch folder with a
unique slug. Single-cell selects keep their original name. Re-running is safe:
already-staged copies are skipped.

USAGE:
    python scripts/probepull.py [--hotfolder DIR] [--dry-run]

Stdlib only — no GPU, no conda env needed.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from pathlib import Path

VALID_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
CELL_RE = re.compile(r"^(\d+)_T(\d+)_([a-z]+)_seed(\d+)\.png$", re.IGNORECASE)


def slug_for(stem: str) -> str:
    """Match the daemon's slug rule exactly."""
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in stem)
    return safe[:60].strip("_") or "unnamed"


def main() -> int:
    default_hf = os.environ.get(
        "IMAGE2SPLAT_HOTFOLDER", str(Path.home() / "image2splat")
    )
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hotfolder", type=Path, default=Path(default_hf).expanduser())
    p.add_argument("--dry-run", action="store_true",
                   help="Show what would be staged without copying anything.")
    args = p.parse_args()

    hf = args.hotfolder
    datasets = hf / "datasets"
    completed = hf / "completed"
    inbox = hf / "inbox"
    if not datasets.is_dir():
        print(f"FAIL: no datasets/ under {hf}")
        return 1

    # Index source images in completed/ by slug.
    sources: dict[str, Path] = {}
    if completed.is_dir():
        for f in completed.iterdir():
            if f.is_file() and f.suffix.lower() in VALID_EXT:
                sources.setdefault(slug_for(f.stem), f)

    n_datasets = n_staged = n_already = n_skipped_grouping = 0
    missing_source: list[str] = []
    multi_report: list[tuple[str, list[str]]] = []

    for d in sorted(p for p in datasets.iterdir() if p.is_dir()):
        probe = d / "probe"
        if not probe.is_dir():
            n_skipped_grouping += 1            # Pre/Post/etc — not a dataset
            continue
        cells = sorted({(int(m.group(2)), int(m.group(4)))
                        for png in probe.glob("*.png")
                        if (m := CELL_RE.match(png.name))})
        if not cells:
            continue
        n_datasets += 1
        src = sources.get(d.name)
        if src is None:
            missing_source.append(d.name)
            continue
        multi = len(cells) > 1
        placed = []
        for i, (tier, seed) in enumerate(cells, 1):
            folder = inbox / f"T{tier}_{seed}"
            name = f"{src.stem}_{i}{src.suffix}" if multi else f"{src.stem}{src.suffix}"
            dest = folder / name
            if dest.exists():
                n_already += 1
                continue
            if args.dry_run:
                print(f"  [dry] {folder.name}/{name}")
            else:
                folder.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dest)
            n_staged += 1
            placed.append(f"{folder.name}/{name}")
        if multi and placed:
            multi_report.append((d.name, placed))

    verb = "would stage" if args.dry_run else "staged"
    print(f"\n{'DRY RUN — ' if args.dry_run else ''}probepull summary")
    print(f"  hotfolder            : {hf}")
    print(f"  datasets with selects: {n_datasets}")
    print(f"  grouping dirs skipped: {n_skipped_grouping} (no probe/ — e.g. Pre/Post)")
    print(f"  {verb:<20} : {n_staged} placement(s) into inbox batch folders")
    print(f"  already present      : {n_already}")
    if multi_report:
        print(f"  multi-cell selects (renamed _1/_2/...):")
        for name, placed in multi_report:
            print(f"    {name[:46]} -> {', '.join(placed)}")
    if missing_source:
        print(f"  NO source in completed/ ({len(missing_source)}) — not staged:")
        for s in missing_source:
            print(f"    {s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
