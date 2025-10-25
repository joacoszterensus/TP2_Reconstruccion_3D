from pathlib import Path
import numpy as np
import cv2 as cv
import pickle, re
from typing import Tuple, List

#  Calibración desde maps.pkl (K' y baseline en metros) 
def load_fx_baseline(maps_pkl: Path) -> Tuple[float, float, np.ndarray, np.ndarray]:
    with open(maps_pkl, "rb") as f:
        maps = pickle.load(f)
    P1 = np.asarray(maps["P1"], dtype=np.float32)
    P2 = np.asarray(maps["P2"], dtype=np.float32)
    Kp = P1[:, :3]
    fx = float(Kp[0, 0])
    if "T" in maps:
        B_m = float(np.linalg.norm(np.asarray(maps["T"], np.float32))) / 1000.0
    else:
        fxP2 = float(P2[0, 0])
        Tx_mm = float(-P2[0, 3] / fxP2)
        B_m = Tx_mm / 1000.0
    return fx, B_m, P1, P2

#  Empareja left_i con disp_i 
def pair_left_disp(left_root: Path, disp_dir: Path) -> Tuple[List[Path], List[Path], List[int]]:
    L = list(left_root.rglob("left_*.jpg")) + list(left_root.rglob("left_*.JPG"))
    rxL = re.compile(r"^left_(\d+)\.jpg$", re.IGNORECASE)
    Lmap = {int(rxL.match(p.name).group(1)): p for p in L if rxL.match(p.name)}
    Dmap = {}
    for p in disp_dir.glob("disp_*.npz"):
        m = re.match(r"^disp_(\d+)\.npz$", p.name, re.IGNORECASE)
        if m: Dmap[int(m.group(1))] = p
    idxs = sorted(set(Lmap) & set(Dmap))
    return [Lmap[i] for i in idxs], [Dmap[i] for i in idxs], idxs

#  Carga disparidad desde .npz (clave robusta) 
def load_disp_npz_anykey(npz_path: Path) -> np.ndarray:
    z = np.load(str(npz_path))
    for k in ["disp", "disparity", "disparity_pixels", "D", "data", "arr_0"]:
        if k in z:
            D = z[k].astype(np.float32)
            break
    else:
        k0 = z.files[0]
        D = z[k0].astype(np.float32)
    if D.ndim == 3 and D.shape[2] == 1:
        D = D[..., 0]
    finite = np.isfinite(D) & (D != 0)
    if finite.any():
        med = float(np.nanmedian(D[finite]))
        samp = D[finite]
        samp = samp[np.random.choice(samp.size, min(samp.size, 5000), replace=False)]
        mul16 = np.mean((np.mod(samp, 16) == 0).astype(np.float32))
        if med > 300 or mul16 > 0.9:
            D = D / 16.0
    D[D <= 0] = np.nan
    return D

#  Preview de profundidad (m) 
def save_depth_preview(depth_m: np.ndarray, path_png: Path, zmax=None) -> None:
    d = depth_m.copy()
    d[~np.isfinite(d) | (d <= 0)] = 0
    if zmax is None:
        zmax = float(np.percentile(d[d > 0], 99)) if np.any(d > 0) else 1.0
    d = np.clip(d, 0, zmax) / (zmax if zmax > 0 else 1.0)
    d8 = (d * 255).astype(np.uint8)
    col = cv.applyColorMap(255 - d8, cv.COLORMAP_TURBO)
    cv.imwrite(str(path_png), col)
