"""
Camera math for the image_to_splat pipeline.

We render N synthetic views of a TRELLIS.2-produced mesh, then export those
views to two formats that downstream splat trainers expect:
  - COLMAP: world-to-camera in OpenCV convention (+Z forward, +Y down)
  - Nerfstudio: camera-to-world in OpenGL convention (+Z out of screen, +Y up)

This module owns:
  - The camera trajectory generator (Fibonacci sphere — near-uniform coverage)
  - The look_at builder (returns OpenCV world-to-camera 4x4)
  - Intrinsics from vertical FOV
  - Convention conversions (OpenCV W2C <-> OpenGL C2W)
  - The quaternion reorder that COLMAP demands (scipy returns x,y,z,w; COLMAP wants w,x,y,z)

Test coverage lives in tests/test_cameras.py. Wrong handedness here = mirror-image splats,
so this module is the highest-risk math in the whole pipeline.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def fibonacci_sphere(n: int, radius: float = 2.0) -> np.ndarray:
    """Return (n, 3) camera positions on a sphere of given radius.

    Uses the golden-angle spiral, which gives near-uniform angular coverage with no
    pole clustering (unlike naive lat/long sampling). 250 views gives ~14° average
    angular separation, enough for splat training to converge.
    """
    if n < 2:
        raise ValueError(f"n must be >= 2, got {n}")
    if radius <= 0:
        raise ValueError(f"radius must be positive, got {radius}")
    phi = np.pi * (3.0 - np.sqrt(5.0))  # golden angle
    indices = np.arange(n, dtype=np.float64)
    y = 1.0 - (indices / (n - 1)) * 2.0  # y in [-1, 1] inclusive at both ends
    r = np.sqrt(1.0 - y * y)
    theta = phi * indices
    x = np.cos(theta) * r
    z = np.sin(theta) * r
    return np.stack([x, y, z], axis=1) * radius


def look_at(
    eye: np.ndarray,
    target: np.ndarray = np.zeros(3),
    up: np.ndarray = np.array([0.0, 1.0, 0.0]),
) -> np.ndarray:
    """Return a 4x4 world-to-camera matrix in OpenCV convention.

    OpenCV camera convention:
        +X right, +Y down, +Z forward (into scene)
    so the camera "looks down +Z" and a point in front of the camera has positive Z.

    The returned matrix transforms world-space homogeneous points into the camera frame.
    """
    eye = np.asarray(eye, dtype=np.float64).reshape(3)
    target = np.asarray(target, dtype=np.float64).reshape(3)
    up = np.asarray(up, dtype=np.float64).reshape(3)

    forward = target - eye
    fnorm = np.linalg.norm(forward)
    if fnorm < 1e-12:
        raise ValueError("eye and target are coincident; cannot construct camera frame")
    forward /= fnorm

    right = np.cross(forward, up)
    rnorm = np.linalg.norm(right)
    if rnorm < 1e-9:
        raise ValueError(
            "look direction is parallel to up vector; pick a different up vector"
        )
    right /= rnorm

    true_up = np.cross(right, forward)  # already unit since right⊥forward and both unit

    # OpenCV: camera axes are [right, -up, forward] in world frame.
    # Row-stack to build R that takes world → camera.
    R = np.stack([right, -true_up, forward], axis=0)
    t = -R @ eye

    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = R
    mat[:3, 3] = t
    return mat


def intrinsics_from_fov(fov_vertical_deg: float, image_size: int) -> dict:
    """Pinhole intrinsics from vertical FOV, assuming a square image.

    Returns a dict with fx, fy, cx, cy, width, height (all float for json-friendliness).
    """
    if fov_vertical_deg <= 0 or fov_vertical_deg >= 180:
        raise ValueError(f"fov must be in (0, 180), got {fov_vertical_deg}")
    if image_size <= 0:
        raise ValueError(f"image_size must be positive, got {image_size}")
    focal = (image_size / 2.0) / np.tan(np.radians(fov_vertical_deg / 2.0))
    return {
        "fx": float(focal),
        "fy": float(focal),
        "cx": float(image_size / 2.0),
        "cy": float(image_size / 2.0),
        "width": int(image_size),
        "height": int(image_size),
    }


def compute_adaptive_fov(
    vertices: np.ndarray,
    w2c_matrices: np.ndarray,
    *,
    margin: float = 0.10,
    fov_min: float = 12.0,
    fov_max: float = 55.0,
    per_view: bool = False,
) -> np.ndarray | float:
    """Compute tight vertical FOV(s) by projecting mesh vertices through camera poses.

    For each view, projects all front-facing vertices into camera space and finds
    the max angular extent: max(atan|x|/z, atan|y|/z). 2× this gives the FOV that
    JUST contains the silhouette; multiply by (1+margin) for padding, then clamp
    to [fov_min, fov_max] to avoid pathological zoom on degenerate geometry.

    Args:
        vertices: (V, 3) world-space mesh vertices.
        w2c_matrices: (N, 4, 4) OpenCV world-to-camera matrices — must be the
            EXACT matrices that will be passed to render_multiview (i.e. include
            any mesh-rotation baking).
        margin: fractional padding on the tight FOV (0.10 = 10% padding).
        fov_min: lower clamp in degrees. Below this, the projection is so tight
            that nvdiffrast precision starts to matter; 12° is a safe floor.
        fov_max: upper clamp. Above this we'd be looking at empty space anyway —
            55° is wider than the daemon's fixed 40° default so it never enlarges.
        per_view: if True, returns (N,) array of per-view FOVs. If False (default),
            returns a single scalar — the max across all views (so a uniform FOV
            still fits every view's silhouette).

    Returns:
        Scalar float (per_view=False) or (N,) np.ndarray (per_view=True).
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    w2c_matrices = np.asarray(w2c_matrices, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"vertices must be (V, 3), got {vertices.shape}")
    if w2c_matrices.ndim != 3 or w2c_matrices.shape[1:] != (4, 4):
        raise ValueError(f"w2c_matrices must be (N, 4, 4), got {w2c_matrices.shape}")
    V = vertices.shape[0]
    if V == 0:
        # Degenerate mesh — fall back to fov_max so we definitely don't clip.
        if per_view:
            return np.full(len(w2c_matrices), fov_max)
        return float(fov_max)

    # Homogeneous projection: (N, 4, 4) @ (V, 4).T → (N, 4, V)
    verts_h = np.concatenate([vertices, np.ones((V, 1))], axis=1)  # (V, 4)
    # einsum is clearer than reshape gymnastics: (N, V, 4)
    verts_cam = np.einsum("nij,vj->nvi", w2c_matrices, verts_h)
    x = verts_cam[..., 0]
    y = verts_cam[..., 1]
    z = verts_cam[..., 2]

    # Only consider verts in front of the camera (z > 0 in OpenCV convention)
    in_front = z > 1e-3
    z_safe = np.where(in_front, z, 1.0)  # avoid div-by-zero; we'll mask after
    theta_h = np.arctan(np.abs(x) / z_safe)
    theta_v = np.arctan(np.abs(y) / z_safe)
    theta = np.maximum(theta_h, theta_v)
    theta = np.where(in_front, theta, 0.0)

    # Per-view max angular extent (V axis)
    max_theta_per_view = theta.max(axis=1)  # (N,)

    if per_view:
        fov_rad = 2.0 * max_theta_per_view * (1.0 + margin)
        fov_deg = np.degrees(fov_rad)
        fov_deg = np.clip(fov_deg, fov_min, fov_max)
        return fov_deg

    # Uniform: take the worst view, then pad
    worst = float(max_theta_per_view.max())
    fov_deg = float(np.degrees(2.0 * worst * (1.0 + margin)))
    return float(np.clip(fov_deg, fov_min, fov_max))


def opencv_w2c_to_opengl_c2w(w2c: np.ndarray) -> np.ndarray:
    """Convert an OpenCV world-to-camera matrix to OpenGL camera-to-world.

    OpenCV convention:  +X right, +Y down, +Z forward (into scene)
    OpenGL convention:  +X right, +Y up,   +Z out of screen (toward viewer)

    Steps:
      1. Invert W2C to get C2W (camera-to-world) in OpenCV convention
      2. Right-multiply by diag(1, -1, -1, 1) to flip Y and Z axes of the
         camera frame (OpenCV cam → OpenGL cam, applied to columns of C2W)

    This is what Nerfstudio's transforms.json expects in each frame's transform_matrix.
    """
    w2c = np.asarray(w2c, dtype=np.float64)
    if w2c.shape != (4, 4):
        raise ValueError(f"w2c must be 4x4, got {w2c.shape}")
    c2w_opencv = np.linalg.inv(w2c)
    flip_yz = np.diag([1.0, -1.0, -1.0, 1.0])
    return c2w_opencv @ flip_yz


def w2c_to_colmap_qvec_tvec(w2c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Decompose an OpenCV W2C matrix into COLMAP's (qvec, tvec).

    COLMAP's images.txt format wants:
      - qvec = (qw, qx, qy, qz)  — quaternion of the world-to-camera rotation
      - tvec = (tx, ty, tz)      — world-to-camera translation

    scipy.spatial.transform.Rotation.as_quat() returns (qx, qy, qz, qw), so we
    reorder. This single line is the most common bug in COLMAP exporters.
    """
    w2c = np.asarray(w2c, dtype=np.float64)
    if w2c.shape != (4, 4):
        raise ValueError(f"w2c must be 4x4, got {w2c.shape}")
    R = w2c[:3, :3]
    t = w2c[:3, 3]
    quat_xyzw = Rotation.from_matrix(R).as_quat()  # scipy returns (x, y, z, w)
    quat_wxyz = np.array(
        [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float64
    )
    return quat_wxyz, t.astype(np.float64)


# Empirically determined via calibrate_orientation.py on the mimic image (2026-05-12):
#
#   TRELLIS.2's mimic frame: front = -Y, up = +Z, right = +X
#   Pixal3D's  mimic frame:  front = +Z, up = +Y, right = +X
#
# To align Pixal3D's mesh into TRELLIS.2's frame: rotate +90° around the X axis,
# which maps Pixal3D's (+Z, +Y) front/up onto TRELLIS.2's (-Y, +Z).
#
# This is a PURE rotation (det = +1), distinct from Pixal3D's hardcoded GLB
# rotation matrix in their inference.py (which includes a reflection — det = -1
# — and is for a different alignment purpose entirely).
#
# Applied to the cameras (not the mesh) so the voxel-coord/attr structure stays
# intact. The HDRI envmap gets the same rotation applied in direction-space.
PIXAL3D_TO_TRELLIS2_ROTATION = np.array(
    [
        [1.0,  0.0,  0.0, 0.0],
        [0.0,  0.0, -1.0, 0.0],
        [0.0,  1.0,  0.0, 0.0],
        [0.0,  0.0,  0.0, 1.0],
    ],
    dtype=np.float64,
)

# Kept as alias for backward compat with existing tests/imports.
PIXAL3D_MESH_ROTATION = PIXAL3D_TO_TRELLIS2_ROTATION


def apply_mesh_rotation_to_trajectory(
    w2c_matrices: np.ndarray, rotation_4x4: np.ndarray
) -> np.ndarray:
    """Apply a 4x4 mesh-frame rotation equivalently to a camera trajectory.

    If we want the cameras to see the mesh AS IF the mesh had been rotated by R,
    the effective world-to-camera matrix is `new_W2C = old_W2C @ R`. This bakes
    the rotation into the camera's view of an un-rotated mesh.

    Derivation:
      camera sees a point p via: cam_p = W2C @ p_world.
      If mesh is rotated: p_world = R @ p_native.
      So: cam_p = (W2C @ R) @ p_native — define effective W2C = W2C @ R.

    Returns a new array with the same shape as `w2c_matrices`.
    """
    if rotation_4x4.shape != (4, 4):
        raise ValueError(f"rotation_4x4 must be (4, 4), got {rotation_4x4.shape}")
    # (N, 4, 4) @ (4, 4) → (N, 4, 4)
    return w2c_matrices @ rotation_4x4


def apply_mesh_rotation_to_points(
    points: np.ndarray, rotation_4x4: np.ndarray
) -> np.ndarray:
    """Rotate (M, 3) world-space points by the rotation part of a 4x4 transform.

    Use this on points sampled from the mesh that get written to points3D.txt,
    so they land in the same world frame the (camera-rotated) views describe.
    """
    R3 = rotation_4x4[:3, :3]
    # new_p = R @ p, applied per-row: points @ R^T
    return points @ R3.T


def generate_camera_trajectory(
    num_views: int,
    radius: float = 2.0,
    target: np.ndarray = np.zeros(3),
    up: np.ndarray = np.array([0.0, 1.0, 0.0]),
) -> np.ndarray:
    """Generate (num_views, 4, 4) OpenCV world-to-camera matrices on a Fibonacci sphere.

    All cameras look toward `target` from positions on the sphere of given radius.
    Returns a stack of 4x4 matrices in the order Fibonacci sampling produces.
    """
    positions = fibonacci_sphere(num_views, radius=radius)
    target = np.asarray(target, dtype=np.float64).reshape(3)
    up = np.asarray(up, dtype=np.float64).reshape(3)

    matrices = np.empty((num_views, 4, 4), dtype=np.float64)
    for i, eye in enumerate(positions):
        # Avoid the rare case where eye is exactly on the up axis (cross product → 0)
        # by perturbing up if the cross with forward is nearly zero
        forward = target - eye
        forward_unit = forward / np.linalg.norm(forward)
        if abs(np.dot(forward_unit, up)) > 0.999:
            local_up = np.array([0.0, 0.0, 1.0]) if abs(up[1]) > 0.5 else up
        else:
            local_up = up
        matrices[i] = look_at(eye, target, local_up)
    return matrices
