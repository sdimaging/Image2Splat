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
