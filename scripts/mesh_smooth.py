"""Plateau-band geometry smoothing for generated meshes.

Pixal3D/TRELLIS shape decodes undulate in plateau-scale tilted patches
(~19 deg between neighbors, wavelength ~5-15% of object size). On glossy
metal under an HDRI these read as hard polygonal facets. The plateaus are
REAL GEOMETRY — render-side fixes (Phong normals, SSAO) only attenuate the
symptom.

Approaches that FAILED (kept here so nobody retries them):
- Direct Laplacian/Taubin on the full mesh: plateau wavelength is 100-400
  edge-lengths at ~7.5M verts; unreachable at any iteration count.
- Explicit Taubin on a vertex-cluster graph: the plateau frequency sits at
  the filter's passband knee; ~0 attenuation.
- 3D-grid Gaussian band-stop with normal-binned splatting: normal bins
  separate OPPOSITE-facing surfaces but not SAME-facing surfaces stacked
  within the blur radius (mouth roof over tongue, teeth rows, lid over
  planks). Verified to shred a Mimic-chest asset.

What this module does instead — surface-aware implicit band-stop:

1. Cluster vertices on a coarse grid (cell ~1.6% of bbox), key including a
   6-way normal bin so opposite faces of thin parts never merge. Cluster
   centroid = mid-scale surface estimate; sub-cluster detail cancels.
2. Build the cluster GRAPH from mesh edges. Surface connectivity: stacked
   surfaces that are close in space but far along the surface never mix.
3. Low-pass the centroids by solving (I + tL)^2 x = c with conjugate
   gradient (L = graph Laplacian). The implicit solve reaches plateau
   wavelengths in ONE solve — no explicit-iteration frequency wall — and
   t calibrates the cutoff directly.
4. band = centroid - x  (the plateau-scale undulation).
5. Soft amplitude gate: plateaus are inherently LOW-AMPLITUDE
   (tilt ~10 deg x wavelength ~10% -> amplitude ~0.3-0.5% of bbox), while
   legitimate mid-band shape (teeth, straps, ridges) is high-amplitude.
   d <- s * tanh(d / s) with s ~0.5% bbox corrects plateaus fully and
   leaves real features nearly untouched.
6. Subtract from member vertices along their own normals.

All ops are pure torch on GPU; ~7.5M verts runs in ~1-2 s.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_vertex_normals(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """Area-weighted vertex normals (same accumulation as the PBR renderer)."""
    fi = faces.long()
    v0 = vertices[fi[:, 0]]
    v1 = vertices[fi[:, 1]]
    v2 = vertices[fi[:, 2]]
    face_n = torch.cross(v1 - v0, v2 - v0, dim=1)  # magnitude = 2*area
    vn = torch.zeros_like(vertices)
    vn.index_add_(0, fi.reshape(-1), face_n.repeat_interleave(3, dim=0))
    return F.normalize(vn, dim=1)


def _laplacian_matvec(x: torch.Tensor, src: torch.Tensor, dst: torch.Tensor,
                      deg: torch.Tensor) -> torch.Tensor:
    """L x = deg*x - sum_neighbors(x)  (unnormalized graph Laplacian)."""
    acc = torch.zeros_like(x)
    acc.index_add_(0, dst, x[src])
    return deg * x - acc


def _cg_solve(matvec, b: torch.Tensor, iters: int = 200,
              tol: float = 1e-7) -> torch.Tensor:
    """Conjugate gradient for SPD systems, vectorized over columns of b."""
    x = b.clone()
    r = b - matvec(x)
    p = r.clone()
    rs = (r * r).sum(dim=0, keepdim=True)
    b_norm = (b * b).sum().sqrt() + 1e-30
    for _ in range(iters):
        Ap = matvec(p)
        alpha = rs / ((p * Ap).sum(dim=0, keepdim=True) + 1e-30)
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = (r * r).sum(dim=0, keepdim=True)
        if rs_new.sum().sqrt() / b_norm < tol:
            break
        p = r + (rs_new / (rs + 1e-30)) * p
        rs = rs_new
    return x


@torch.no_grad()
def plateau_smooth(vertices: torch.Tensor, faces: torch.Tensor,
                   cluster_frac: float = 0.016, t_scale: float = 4.0,
                   alpha: float = 1.0, gate_frac: float = 0.005,
                   cg_iters: int = 200) -> tuple[torch.Tensor, dict]:
    """Remove plateau-band undulation from vertex positions.

    Args:
        vertices: [V, 3] float tensor (cuda)
        faces: [F, 3] int tensor
        cluster_frac: clustering cell size as fraction of the longest bbox
            axis (~1.6%; well below the 5-15% plateau wavelength, above
            detail scale so engraving/rivets cancel inside clusters).
        t_scale: implicit-diffusion time of EACH of the two (I+tL) solves.
            t=4 puts ~94% attenuation at the plateau band (lambda~1 for a
            6-8-cluster wavelength on a deg~6 graph) while leaving
            >4x-wavelength shape mostly intact.
        alpha: fraction of the isolated band to remove.
        gate_frac: soft amplitude gate (tanh knee) as fraction of bbox
            diagonal. Plateau amplitudes (~0.3-0.5%) pass through almost
            linearly; high-amplitude real features saturate and are
            preserved.
        cg_iters: max conjugate-gradient iterations per solve.

    Returns:
        (new_vertices, stats_dict)
    """
    device = vertices.device
    dtype = vertices.dtype
    fi = faces.long()
    lo = vertices.min(dim=0).values
    span = (vertices.max(dim=0).values - lo).clamp(min=1e-6)
    diag = float(span.norm())
    cell = float(span.max()) * cluster_frac

    normals = compute_vertex_normals(vertices, fi)

    # --- vertex-graph-smoothed normals -------------------------------------
    # Raw per-vertex normals carry the mesh's vertex-scale roughness. Every
    # downstream use (triplanar weights, projection, displacement direction)
    # must use a SMOOTH field, or the roughness modulates the displacement
    # and injects high-frequency noise into the surface (measured +50-77%
    # normal-channel HF energy when raw normals leak in anywhere).
    v_src = torch.cat([fi[:, 0], fi[:, 1], fi[:, 2], fi[:, 1], fi[:, 2], fi[:, 0]])
    v_dst = torch.cat([fi[:, 1], fi[:, 2], fi[:, 0], fi[:, 0], fi[:, 1], fi[:, 2]])
    vdeg_n = torch.zeros((vertices.shape[0], 1), device=device, dtype=dtype)
    vdeg_n.index_add_(0, v_dst, torch.ones((v_dst.shape[0], 1), device=device, dtype=dtype))
    vdeg_n = vdeg_n.clamp(min=1.0)
    n_filt = normals
    for _ in range(8):
        acc = torch.zeros_like(n_filt)
        acc.index_add_(0, v_dst, n_filt[v_src])
        n_filt = F.normalize(n_filt + acc / vdeg_n, dim=1)

    # --- cluster key: grid cell + 6-way normal bin ------------------------
    gi = ((vertices - lo) / cell).long()
    dom = n_filt.abs().argmax(dim=1)
    sign_neg = (torch.gather(n_filt, 1, dom.unsqueeze(1)).squeeze(1) < 0).long()
    nbin = dom * 2 + sign_neg
    G = int(gi.max().item()) + 1
    key = ((gi[:, 0] * G + gi[:, 1]) * G + gi[:, 2]) * 6 + nbin
    uk, inv = torch.unique(key, return_inverse=True)
    C = int(inv.max().item()) + 1

    counts = torch.zeros(C, device=device, dtype=dtype)
    counts.index_add_(0, inv, torch.ones_like(inv, dtype=dtype))
    centroid = torch.zeros((C, 3), device=device, dtype=dtype)
    centroid.index_add_(0, inv, vertices)
    centroid = centroid / counts.unsqueeze(1)

    # --- cluster graph from mesh edges (surface connectivity) -------------
    e_src = torch.cat([fi[:, 0], fi[:, 1], fi[:, 2]])
    e_dst = torch.cat([fi[:, 1], fi[:, 2], fi[:, 0]])
    cs, cd = inv[e_src], inv[e_dst]
    keep = cs != cd
    cs, cd = cs[keep], cd[keep]
    ekey = torch.unique(torch.cat([cs * C + cd, cd * C + cs]))
    src, dst = ekey // C, ekey % C
    deg = torch.zeros((C, 1), device=device, dtype=dtype)
    deg.index_add_(0, dst, torch.ones((dst.shape[0], 1), device=device, dtype=dtype))

    # --- implicit low-pass: solve (I + tL)^2 x = centroid ------------------
    def matvec(x):
        return x + t_scale * _laplacian_matvec(x, src, dst, deg)

    x = _cg_solve(matvec, centroid, iters=cg_iters)
    x = _cg_solve(matvec, x, iters=cg_iters)

    band = centroid - x
    # The band still leaks ~16% of large-scale shape (diffusion rolloff is
    # gentle). Remove the band's own low-frequency content: squares the
    # stopband leak (16% -> ~3%) while keeping ~83% of the plateau signal.
    band_lf = _cg_solve(matvec, band, iters=cg_iters)
    band_lf = _cg_solve(matvec, band_lf, iters=cg_iters)
    band = band - band_lf

    # --- normal-coherence gate: only correct SMOOTH surfaces --------------
    # Plateau facets live on smooth panels (helmet domes, breastplates).
    # Busy geometry (teeth, straps, engraving ridges) has incoherent
    # member normals; correcting it does harm and no good. R = mean
    # resultant length of cluster normals: 1 on smooth panels, low on
    # detail. Full correction above R=0.95, none below 0.7.
    nsum = torch.zeros((C, 3), device=device, dtype=dtype)
    nsum.index_add_(0, inv, normals)
    R = nsum.norm(dim=1) / counts
    coherence = ((R - 0.85) / 0.12).clamp(0.0, 1.0)

    # --- interpolate the band to vertices (kills cluster-border steps) ----
    # The band is per-cluster; applied as-is it is piecewise-constant over
    # ~50-edge tiles -> visible micro-steps ("speckle") on glossy surfaces.
    # Clusters are grid cells, so write (band, coherence, weight) into a
    # dense per-bin grid and TRILINEARLY sample it at each vertex: the
    # field becomes piecewise-linear and the steps vanish. A vertex only
    # reads its own normal-bin's grid, so stacked opposite/other-facing
    # surfaces still never mix.
    rest = uk // 6
    c_bin = uk % 6
    c_iz = rest % G
    c_iy = (rest // G) % G
    c_ix = rest // (G * G)
    cluster_n = F.normalize(nsum, dim=1)                 # smoothed normal field
    dense = torch.zeros((6, 11, G + 1, G + 1, G + 1), device=device, dtype=dtype)
    dense[c_bin, 0, c_ix, c_iy, c_iz] = band[:, 0]
    dense[c_bin, 1, c_ix, c_iy, c_iz] = band[:, 1]
    dense[c_bin, 2, c_ix, c_iy, c_iz] = band[:, 2]
    dense[c_bin, 3, c_ix, c_iy, c_iz] = coherence
    dense[c_bin, 4, c_ix, c_iy, c_iz] = 1.0
    dense[c_bin, 5, c_ix, c_iy, c_iz] = centroid[:, 0]   # for shell-thickness
    dense[c_bin, 6, c_ix, c_iy, c_iz] = centroid[:, 1]
    dense[c_bin, 7, c_ix, c_iy, c_iz] = centroid[:, 2]
    dense[c_bin, 8, c_ix, c_iy, c_iz] = cluster_n[:, 0]  # smooth displacement dir
    dense[c_bin, 9, c_ix, c_iy, c_iz] = cluster_n[:, 1]
    dense[c_bin, 10, c_ix, c_iy, c_iz] = cluster_n[:, 2]

    # cluster value sits at the cell CENTER -> sample at (pos/cell - 0.5).
    # TRIPLANAR bin blending: hard per-vertex bin assignment makes adjacent
    # vertices on curved rims read different bin grids -> displacement
    # jumps -> ragged silhouettes (verified on breastplate armholes).
    # Blend all 6 bin grids weighted by (n . axis)^2 instead — weights sum
    # to 1, the field is smooth everywhere, opposite-facing surfaces still
    # contribute zero to each other's bins.
    p = ((vertices - lo) / cell - 0.5).clamp(min=0.0)
    f = p.floor()
    w = p - f
    f = f.long().clamp(max=G - 1)
    wpos = n_filt.clamp(min=0.0) ** 2                     # +x +y +z
    wneg = (-n_filt).clamp(min=0.0) ** 2                  # -x -y -z
    tri_w = torch.cat([
        wpos[:, 0:1], wneg[:, 0:1],
        wpos[:, 1:2], wneg[:, 1:2],
        wpos[:, 2:3], wneg[:, 2:3]], dim=1)               # [V, 6] bin order = axis*2+sign
    # blended channels: band(3) + coherence + weight + smooth normal(3)
    ch = torch.tensor([0, 1, 2, 3, 4, 8, 9, 10], device=device)
    samp = torch.zeros((8, vertices.shape[0]), device=device, dtype=dtype)
    flat_grid = dense.view(6, 11, -1)
    S = (G + 1)
    corner_cache = []
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                ci = f + torch.tensor([dx, dy, dz], device=device)
                idx = (ci[:, 0] * S + ci[:, 1]) * S + ci[:, 2]
                off = torch.tensor([dx, dy, dz], device=device, dtype=dtype)
                wt = (w * off + (1 - w) * (1 - off)).prod(dim=1)
                corner_cache.append((idx, wt))
    for b in range(6):
        wb = tri_w[:, b]
        sel = wb > 0.05
        if not sel.any():
            continue
        idxs = sel.nonzero(as_tuple=True)[0]
        sb = torch.zeros((8, idxs.shape[0]), device=device, dtype=dtype)
        gb = flat_grid[b][ch]
        for idx, wt in corner_cache:
            sb += gb[:, idx[idxs]] * wt[idxs].unsqueeze(0)
        samp[:, idxs] += sb * wb[idxs].unsqueeze(0)

    # --- local shell thickness from the OPPOSITE normal bin ---------------
    # Generated armor/props are thin shells (often ~0.2% of bbox). A
    # displacement larger than the local shell semi-thickness pushes the
    # front surface through the back -> z-fighting speckle. The opposing
    # surface lives in the opposite bin (b XOR 1) at the same location;
    # sample its position to bound the displacement.
    opp = torch.zeros((4, vertices.shape[0]), device=device, dtype=dtype)
    b_opp = (nbin ^ 1)
    for idx, wt in corner_cache:
        # advanced indices (b_opp, idx) broadcast to the leading dim ->
        # result is [V, 3] / [V]; transpose into our [C, V] accumulators
        opp[:3] += flat_grid[b_opp, 5:8, idx].T * wt.unsqueeze(0)
        opp[3] += flat_grid[b_opp, 4, idx] * wt
    has_opp = opp[3] > 1e-6
    opp_pos = (opp[:3] / opp[3].clamp(min=1e-8)).T
    wsum = samp[4].clamp(min=1e-8)
    band_v = (samp[:3] / wsum).T
    coh_v = samp[3] / wsum
    # Smooth displacement DIRECTION: raw per-vertex normals carry the
    # mesh's vertex-scale roughness; displacing along them injects
    # high-frequency noise everywhere the correction is active (measured
    # +53-77% normal-channel HF energy). The interpolated cluster-mean
    # normal field is smooth at the 1.6% cluster scale.
    n_smooth = F.normalize((samp[5:8] / wsum).T, dim=1)
    empty = samp[4] < 1e-6
    band_v[empty] = band[inv][empty]      # fallback: own cluster's band
    coh_v[empty] = coherence[inv][empty]
    n_smooth[empty] = n_filt[empty]
    # Interpolation pulls rim/detail coherence UP toward neighboring flat
    # panels — exactly where suppression matters most (curling rims notch
    # the silhouette because the displacement DIRECTION rotates with the
    # normal). The vertex's own cluster coherence is a hard cap.
    coh_v = torch.minimum(coh_v, coherence[inv])

    # --- open-boundary protection ------------------------------------------
    # Open rims (armholes, plate borders) have one-sided cluster
    # neighborhoods -> biased band -> ragged/eroded silhouettes if moved.
    # Freeze boundary vertices + 2 rings around them.
    eu = torch.cat([
        torch.stack([fi[:, 0], fi[:, 1]], dim=1),
        torch.stack([fi[:, 1], fi[:, 2]], dim=1),
        torch.stack([fi[:, 2], fi[:, 0]], dim=1)], dim=0)
    eu = torch.sort(eu, dim=1).values
    ek, ecnt = torch.unique(eu[:, 0] * vertices.shape[0] + eu[:, 1],
                            return_counts=True)
    bk = ek[ecnt == 1]                              # boundary edges
    bflag = torch.zeros(vertices.shape[0], device=device, dtype=dtype)
    bflag[bk // vertices.shape[0]] = 1.0
    bflag[bk % vertices.shape[0]] = 1.0
    if bflag.any():
        v_src = torch.cat([fi[:, 0], fi[:, 1], fi[:, 2],
                           fi[:, 1], fi[:, 2], fi[:, 0]])
        v_dst = torch.cat([fi[:, 1], fi[:, 2], fi[:, 0],
                           fi[:, 0], fi[:, 1], fi[:, 2]])
        for _ in range(2):                          # dilate 2 rings
            nb = torch.zeros_like(bflag)
            nb.index_reduce_(0, v_dst, bflag[v_src], reduce="amax")
            bflag = torch.maximum(bflag, nb)
        # fade the freeze over a few more rings (a hard 0/1 edge would
        # itself create a displacement step)
        vdeg = torch.zeros_like(bflag)
        vdeg.index_add_(0, v_dst, torch.ones_like(bflag[v_src]))
        vdeg = vdeg.clamp(min=1.0)
        for _ in range(4):
            acc = torch.zeros_like(bflag)
            acc.index_add_(0, v_dst, bflag[v_src])
            bflag = torch.maximum(bflag * 0.5 + 0.5 * acc / vdeg, bflag * 0.0)
            bflag = bflag.clamp(0.0, 1.0)

    # --- apply: smooth-normal projection + amplitude DISCRIMINATION -------
    # Scale alone cannot separate plateau noise from real mid-band features
    # (flutes/straps live at plateau wavelength). Amplitude can: plateaus
    # are SHALLOW (<~gate), features are DEEP. A saturating gate (tanh)
    # still applies gate-sized displacement at feature edges and chews
    # them (measured: added HF noise traces every flute/strap/rivet).
    # Derivative-of-Gaussian gate instead: ~identity for |band| <= gate,
    # decays to ZERO for |band| >> gate — features are left untouched.
    thickness = ((vertices - opp_pos) * n_smooth).sum(dim=1).abs()
    d_n = (band_v * n_smooth).sum(dim=1, keepdim=True)
    gate = diag * gate_frac
    gated_frac = float((d_n.abs() > gate).float().mean())
    sigma = 1.5 * gate
    d_n = d_n * torch.exp(-0.5 * (d_n / sigma) ** 2)
    d_n = d_n * coh_v.clamp(0.0, 1.0).unsqueeze(1)
    d_n = d_n * (1.0 - bflag).unsqueeze(1)
    # shell-thickness cap: each side may use 35% of the local thickness
    # (the opposing surface gets its own 35%, leaving a 30% gap)
    t_cap = torch.where(has_opp, 0.35 * thickness,
                        torch.full_like(thickness, float("inf"))).unsqueeze(1)
    d_n = torch.clamp(d_n, -t_cap, t_cap)

    disp = alpha * d_n * n_smooth
    new_vertices = vertices - disp

    mag = disp.norm(dim=1)
    stats = {
        "clusters": C,
        "cluster_edges": int(src.shape[0]),
        "mean_disp": float(mag.mean()),
        "p95_disp": float(mag.quantile(0.95)),
        "max_disp": float(mag.max()),
        "bbox_diag": diag,
        "beyond_gate_frac": gated_frac,
    }
    return new_vertices, stats


# Back-compat alias used by diagnose_facets.py
def plateau_band_stop(vertices, faces, alpha: float = 1.0,
                      passes: int = 1, **kw):
    v, stats = vertices, {}
    for _ in range(max(1, passes)):
        v, stats = plateau_smooth(v, faces, alpha=alpha, **kw)
    return v, stats


if __name__ == "__main__":
    import time
    torch.manual_seed(0)
    n = 512
    ax = torch.linspace(0, 1, n)
    gy, gx = torch.meshgrid(ax, ax, indexing="ij")
    idx = torch.arange(n * n).view(n, n)
    f1 = torch.stack([idx[:-1, :-1], idx[1:, :-1], idx[:-1, 1:]], dim=-1).reshape(-1, 3)
    f2 = torch.stack([idx[1:, :-1], idx[1:, 1:], idx[:-1, 1:]], dim=-1).reshape(-1, 3)
    plane_faces = torch.cat([f1, f2]).int().cuda()

    # --- Test 1: realistic plateau (19 deg) + detail + large-scale shape --
    plateau = 0.004 * torch.sin(gx * 2 * torch.pi * 8)     # wavelength 12.5%
    detail = 0.0008 * torch.sin(gx * 2 * torch.pi * 80)    # wavelength 1.25%
    shape = 0.05 * torch.sin(gx * 2 * torch.pi * 1.5)      # wavelength 66%
    z = plateau + detail + shape
    verts = torch.stack([gx.flatten(), gy.flatten(), z.flatten()], dim=1).cuda()
    t0 = time.time()
    out, stats = plateau_smooth(verts, plane_faces)
    dt = time.time() - t0
    z2 = out[:, 2].view(n, n).cpu()
    c = slice(n // 4, 3 * n // 4)
    ideal = (detail + shape).view(n, n)[c, c]
    resid = (z2[c, c] - ideal).std()
    plateau_in = plateau.view(n, n)[c, c].std()
    shape_err = (z2[c, c] - ideal).abs().max()
    print(f"T1 plateau: residual={resid:.5f}/{plateau_in:.5f} "
          f"(reduction {(1 - resid / plateau_in) * 100:.0f}%)  "
          f"max_dev_from_ideal={shape_err:.5f}  time={dt:.2f}s")
    print(f"   stats: {stats}")

    # --- Test 2: stacked same-facing surfaces 5% apart must NOT mix -------
    v_a = torch.stack([gx.flatten(), gy.flatten(),
                       torch.zeros_like(gx).flatten()], dim=1)
    v_b = torch.stack([gx.flatten(), gy.flatten(),
                       torch.full_like(gx, 0.05).flatten()], dim=1)
    verts2 = torch.cat([v_a, v_b]).cuda()
    faces2 = torch.cat([plane_faces, plane_faces + n * n]).int().cuda()
    out2, stats2 = plateau_smooth(verts2, faces2)
    moved = (out2 - verts2).norm(dim=1).max()
    print(f"T2 stacked planes: max displacement={moved:.6f} (want ~0)")

    # --- Test 3: high-amplitude ridge (tooth/strap) must survive ----------
    ridge = 0.05 * torch.exp(-((gx - 0.5) / 0.015) ** 2)   # amp 5%, width 3%
    z3 = ridge
    verts3 = torch.stack([gx.flatten(), gy.flatten(), z3.flatten()], dim=1).cuda()
    out3, _ = plateau_smooth(verts3, plane_faces)
    z3o = out3[:, 2].view(n, n).cpu()
    peak_in = float(ridge.max())
    peak_out = float(z3o[:, n // 2 - 2:n // 2 + 2].max())
    print(f"T3 ridge survival: peak {peak_in:.4f} -> {peak_out:.4f} "
          f"(kept {peak_out / peak_in * 100:.0f}%, want >85%)")
