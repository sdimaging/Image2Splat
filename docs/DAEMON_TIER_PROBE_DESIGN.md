# Daemon Tier + Probe Design — 2026-05-21

Draft for review before editing `hotfolder_daemon.py`.

## Tier system

5 named tiers + 1 probe mode, locked from probe data:

```python
SAMPLER_TIERS = {
    "1": ("Default",  {"steps":12, "guidance_strength":7.5},
                       {"steps":12, "guidance_strength":7.5},
                       {"steps":12, "guidance_strength":1.1}),
    "2": ("Subtle",   {"steps":13, "guidance_strength":7.6},
                       {"steps":13, "guidance_strength":7.6},
                       {"steps":13, "guidance_strength":1.2}),
    "3": ("Balanced", {"steps":14, "guidance_strength":7.7},
                       {"steps":14, "guidance_strength":7.7},
                       {"steps":14, "guidance_strength":1.25}),
    "4": ("Refined",  {"steps":15, "guidance_strength":7.6},
                       {"steps":15, "guidance_strength":7.6},
                       {"steps":15, "guidance_strength":1.2}),
    "5": ("Sculpted", {"steps":15, "guidance_strength":8.0},
                       {"steps":15, "guidance_strength":8.0},
                       {"steps":15, "guidance_strength":1.5}),
}
PROBE_TIER_KEY = "6"  # not a sampler tier — special probe mode
PROBE_VIEW = 129       # view index used for probe-mode comparison
```

## Interactive prompts (new, after the existing seed/HDRI/rembg)

```
Sampler tier?
  1) Default     12 / 7.5 / 1.1
  2) Subtle      13 / 7.6 / 1.2
  3) Balanced    14 / 7.7 / 1.25
  4) Refined     15 / 7.6 / 1.2
  5) Sculpted    15 / 8.0 / 1.5
  6) Probe       runs 1–5 at view 129 (multi-tier × multi-seed grid)
Select [1-6, default 1]: 

# Only if "6" picked:
How many seeds for probe? [1=just 222, 2=222+1 random, ... up to 8] (1):
> N
  → seed list: [222, <rand1>, <rand2>, ..., <rand_N-1>]
  → ETA: ~M min per asset (5 tiers × N seeds × ~60s)
Continue? [Y/n]:
```

## New code structure (in hotfolder_daemon.py)

### 1. Constants section
- `SAMPLER_TIERS` dict (above)
- `PROBE_TIER_KEY = "6"`, `PROBE_VIEW = 129`

### 2. New helper functions
- `interactive_select_tier(default: str = "1") -> str`
- `interactive_select_probe_seed_count(default: int = 1) -> int`
- `_apply_tier_to_backbone(bb, tier_key: str) -> None` (mutates bb's 3 sampler dicts)

### 3. CLI args (parse_args)
- `--tier`, choices=["1","2","3","4","5","6"], default="1"
- `--probe-seed-count`, type=int, default=1

### 4. run() changes
- Call `interactive_select_tier()` after force_rembg prompt
- If tier == "6": call `interactive_select_probe_seed_count()`, generate `self.probe_seed_list`
- Apply tier to backbone via `_apply_tier_to_backbone(bb, args.tier)` for tiers 1-5
- Log line shows selected tier + actual sampler params (no more hardcoded misleading text)

### 5. New method: `process_one_probe()`
Branches off `process_one()`:
- Output dir: `datasets/<slug>/probe/`
- Iterate 5 tiers × N seeds = 5N runs per asset
- Each run: apply tier sampler params, `bb.run(seed=this_seed)`, render single view (PROBE_VIEW), polish, save as `<view>_T<key>_<name>_seed<S>.png`
- Write `probe_meta.json` alongside outputs documenting tier dict + seed list for traceability

### 6. Main poll loop branch
```python
if self.args.tier == PROBE_TIER_KEY:
    self.process_one_probe(bb, envmap, w2c, intr, image_path)
else:
    self.process_one(bb, envmap, w2c, intr, image_path)
```

## Output file naming

### Production runs (tier 1-5)
Unchanged from today:
```
datasets/<slug>/USER_Alt2_200v_3000px/images/000.png ... 199.png
```
Tier choice logged in daemon log AND in `transforms.json` (add `"sampler_tier"` field).

### Probe runs (tier 6)
```
datasets/<slug>/probe/
  129_T1_default_seed222.png
  129_T1_default_seed8742.png   (if seed_count > 1)
  129_T2_subtle_seed222.png
  ...
  129_T5_sculpted_seed222.png
  probe_meta.json
```

`probe_meta.json` includes:
- The full tier dict
- The seed list (so the user can reproduce a specific cell later via `--tier 3 --seed 8742`)
- Timestamp + daemon version

## Backwards compatibility

- `--no-prompt` path: defaults to `--tier 1` (Default).
- Existing BAT scripts that pipe input or use defaults: no change in behavior except tier 1 now has tex=1.1 instead of Pixal3D's tex=1.0. **This is a deliberate small bump** the user wants as the new "production-default."
- `--tier 0` if needed in future: could reserve for "true Pixal3D defaults (empty dicts)" — not implemented now but easy to add.

## Memory entry (after rollout)

Update `reference_pixal3d_sampler_defaults.md` with:
- The 5 locked tier values (verified 2026-05-21)
- The probe mode UX
- Symptom-to-tier mapping based on the tier_validation pass

## Estimated code diff

- ~30 lines new constants
- ~50 lines new interactive helpers
- ~80 lines new `process_one_probe()` method
- ~15 lines `run()` mods
- ~10 lines CLI args
- **Total: ~185 new lines, ~10 lines modified**

Backwards compatible. Production users see one extra prompt (tier picker). Probe users get the multi-tier UX.

---

**Status:** design only. No code changes yet. Awaiting user confirmation to apply.
