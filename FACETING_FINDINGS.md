# Metal faceting investigation — findings (2026-06-11)

## Symptom
~90% of metal probe cells (helms/swords/armor) show large angular bright/dark
plateaus on smooth surfaces. 2-3 seeds out of 20 come out clean.

## Root cause (numerically confirmed)
**The generated GEOMETRY itself undulates in plateau-scale tilted patches.**
Channel-isolation diagnostic on a worst-case breastplate (T5 seed222, view 129):

| channel                  | bright vs dark plateau delta |
|--------------------------|------------------------------|
| shaded luminance         | 1.80× ratio (the artifact)   |
| base_color               | +1.5% (clean)                |
| roughness                | -3.5% (clean)                |
| metallic                 | +4.9% (clean)                |
| AO (clay)                | +14.9% (secondary contributor)|
| **surface normal angle** | **19.2° between plateaus**   |

Metal + HDRI turns 19° orientation differences into 1.8× brightness jumps.
Seed-dependence: some shape-SLat samples decode smooth panels, most decode
"hand-forged bumpy" panels. NOT a texture problem, NOT flat shading.

## What was tried
1. **Smooth Phong vertex normals** (pbr_mesh_renderer.py patch) — correct fix
   for flat shading, verified working (normal channel smooth), but plateaus
   are real geometry so faithful normals still show them. KEEP (it helps
   decimation-related faceting and costs nothing).
2. **Crease-aware normal-field diffusion** (25 iters, 35° crease,
   env PIXAL3D_NORMAL_SMOOTH_ITERS / _CREASE in the same patch) —
   measured 1.80→1.76 ratio, 19.2°→15.5°. INSUFFICIENT at feasible
   iteration counts (diffusion radius too small vs plateau scale).

## Candidate next steps (untested)
1. **Geometry-space smoothing** post-decim: Taubin lambda-mu on vertices,
   curvature-thresholded to protect flutes/edges. Also fixes the +15% AO term.
2. **SDF/voxel-grid smoothing** before mesh extraction (attacks root).
3. **SSAO intensity reduction** (hardcoded 1.5 in pbr_mesh_renderer.py render();
   ~0.8 would halve the AO contribution). One-line, cheap partial win.
4. **Seed auto-selection**: score probe cells by plateau metric (large-scale
   luminance std on metal masks) and auto-rank — automates the current
   manual 2-3-good-out-of-20 workflow.
5. Longer term: SLat-PiD H100 training (see Desktop/Image2Splat_Research/).

## Tooling
- `scripts/diagnose_facets.py` — renders ONE asset cell, dumps ALL renderer
  channels (shaded/base_color/roughness/metallic/normal/clay) for channel
  isolation. Plus the numerical plateau measurement used above (in git log
  of this file / session notes).
- `patches/pixal3d_smooth_normals.patch` — current full local Pixal3D diff
  (Phong normals + optional diffusion + "(proj)" labels).
