import re
import numpy as np
import cv2 as cv
import open3d as o3d
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List

#  Calibración 
@dataclass
class Calib:
    fx: float
    fy: float
    cx: float
    cy: float
    baseline_m: float
    width: int
    height: int

def calib_from_obj(calibration) -> Calib:
    # Extrae campos del objeto Calibration de tu pipeline
    return Calib(
        fx=float(calibration.fx),
        fy=float(calibration.fy),
        cx=float(calibration.cx0),
        cy=float(calibration.cy),
        baseline_m=float(calibration.baseline_meters),
        width=int(calibration.width),
        height=int(calibration.height),
    )

#  IO y utilidades de dataset 
def list_indices(disp_dir: Path) -> List[int]:
    idxs = []
    for p in sorted(disp_dir.glob("disp_*.npz")):
        m = re.search(r"disp_(\d+)\.npz$", p.name)
        if m:
            idxs.append(int(m.group(1)))
    return sorted(idxs)

def path_left(left_dir: Path, i: int) -> Optional[Path]:
    for ext in ("jpg", "JPG", "png", "PNG"):
        cand = list(left_dir.rglob(f"left_{i}.{ext}"))
        if cand:
            cand.sort(key=lambda p: len(str(p)))
            return cand[0]
    return None

def load_left(left_dir: Path, i: int) -> Optional[np.ndarray]:
    p = path_left(left_dir, i)
    if p is None:
        return None
    return cv.imread(str(p), cv.IMREAD_COLOR)

def load_disp_npz(disp_dir: Path, idx: int, target_shape: Optional[Tuple[int,int]]=None) -> np.ndarray:
    z = np.load(str(disp_dir / f"disp_{idx}.npz"))
    key = next((k for k in ["disparity","disp","arr_0","data"] if k in z), z.files[0])
    D = z[key].astype(np.float32)

    # Normaliza si quedó x16
    finite = np.isfinite(D) & (D > 0)
    if finite.any() and float(np.nanmedian(D[finite])) > 300:
        D = D / 16.0

    if target_shape and D.shape != target_shape:
        h_src, w_src = D.shape
        H_tgt, W_tgt = target_shape
        D = cv.resize(D, (W_tgt, H_tgt), interpolation=cv.INTER_NEAREST)
        sx = W_tgt / float(w_src)  # disparidad horizontal
        D *= sx

    D[D <= 0] = np.nan
    return D

#  Geometría
def compute_depth(disparity: np.ndarray, fx: float, baseline_m: float, default=np.nan) -> np.ndarray:
    Z = np.full_like(disparity, default, dtype=np.float32)
    mask = np.isfinite(disparity) & (disparity > 0)
    Z[mask] = (fx * baseline_m) / disparity[mask]
    return Z

def depth_to_pcd(Z: np.ndarray, img_bgr: np.ndarray, fx: float, cx: float, cy: float,
                 z_min: float, z_max: Optional[float]) -> o3d.geometry.PointCloud:
    H, W = Z.shape
    m = np.isfinite(Z) & (Z > z_min)
    if z_max is not None:
        m &= (Z < z_max)
    if not m.any():
        return o3d.geometry.PointCloud()

    uu, vv = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    Zm = Z[m]
    X = (uu[m] - cx) * Zm / fx
    Y = (vv[m] - cy) * Zm / fx
    pts = np.stack([X, Y, Zm], 1).astype(np.float32)

    col = img_bgr[m][:, ::-1].astype(np.float32) / 255.0  # RGB
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(pts)
    pc.colors = o3d.utility.Vector3dVector(col)
    return pc

#  ChArUco / pose
def build_K(calib: Calib) -> np.ndarray:
    return np.array([[calib.fx, 0, calib.cx],
                     [0, calib.fy, calib.cy],
                     [0, 0, 1]], dtype=np.float32)

def charuco_pose_on_left(img_bgr: np.ndarray,
                         board,
                         K: np.ndarray,
                         detect_charuco_markers,
                         estimate_camera_pose_with_homography) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
    det = detect_charuco_markers(img_bgr, board)
    if det is None:
        return False, None, None
    ret = estimate_camera_pose_with_homography(
        img_bgr, board, det, calibration=(K, np.zeros(5, np.float32)), undistort=False
    )
    if ret is None:
        return False, None, None
    ok, rvec, tvec = ret
    R, _ = cv.Rodrigues(rvec)
    return True, R.astype(np.float64), tvec.reshape(3).astype(np.float64)

def crop_with_obb(pcd: o3d.geometry.PointCloud, R: np.ndarray, t: np.ndarray,
                  extent_xyz: Tuple[float,float,float], lift_m: float = 0.02) -> o3d.geometry.PointCloud:
    center = t + lift_m * R[:, 2]
    box = o3d.geometry.OrientedBoundingBox(center=center, R=R, extent=np.asarray(extent_xyz))
    idx_in = box.get_point_indices_within_bounding_box(pcd.points)
    if len(idx_in) > 1000:
        return pcd.select_by_index(idx_in)
    return pcd

def denoise_statistical(pcd: o3d.geometry.PointCloud, nb: int, std_ratio: float) -> o3d.geometry.PointCloud:
    if len(pcd.points) == 0:
        return pcd
    out, _ = pcd.remove_statistical_outlier(nb_neighbors=nb, std_ratio=std_ratio)
    return out
