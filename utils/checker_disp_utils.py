import re, csv
import numpy as np
import cv2 as cv
from pathlib import Path
from typing import Optional, Tuple, List

#  Pares L/R 
def list_pair_indices(left_dir: Path, right_dir: Path) -> List[Tuple[Path, Path, int]]:
    L, R = {}, {}
    for p in left_dir.rglob("left_*.jpg"):
        L[int(re.search(r"(\d+)", p.stem).group(1))] = p
    for p in left_dir.rglob("left_*.JPG"):
        L.setdefault(int(re.search(r"(\d+)", p.stem).group(1)), p)
    for p in right_dir.rglob("right_*.jpg"):
        R[int(re.search(r"(\d+)", p.stem).group(1))] = p
    for p in right_dir.rglob("right_*.JPG"):
        R.setdefault(int(re.search(r"(\d+)", p.stem).group(1)), p)
    idxs = sorted(set(L) & set(R))
    return [(L[i], R[i], i) for i in idxs]

#  Disparidad (.npy/.npz) 
def load_disparity(disp_dir: Path, idx: int, target_shape: Optional[Tuple[int,int]]=None) -> np.ndarray:
    cands = [
        disp_dir / f"disp_{idx}.npy",
        disp_dir / f"disp_{idx}.npz",
        disp_dir / f"left_{idx}.npy",
        disp_dir / f"left_{idx}.npz",
        disp_dir / f"disparity_{idx}.npy",
        disp_dir / f"disparity_{idx}.npz",
    ]
    if not any(p.exists() for p in cands):
        wild = sorted(list(disp_dir.glob(f"*_{idx}*.npy")) + list(disp_dir.glob(f"*_{idx}*.npz")))
        cands = wild + cands

    def _load_npz_any_key(p: Path):
        z = np.load(str(p))
        for k in ["disparity","disp","disparity_pixels","D","data","arr_0"]:
            if k in z: return z[k]
        for k in z.files:
            a = z[k]
            if isinstance(a, np.ndarray) and a.ndim >= 2: return a
        raise KeyError(f"NPZ sin claves útiles: {p.name} (claves: {list(z.files)})")

    for p in cands:
        if not p.exists(): continue
        D = np.load(str(p)) if p.suffix == ".npy" else _load_npz_any_key(p)
        D = np.asarray(D, dtype=np.float32)
        if D.ndim == 3 and D.shape[2] == 1: D = D[...,0]

        finite = np.isfinite(D) & (D != 0)
        if finite.any():
            med = float(np.nanmedian(D[finite]))
            sample = D[finite]
            sample = sample[np.random.choice(sample.size, min(sample.size, 5000), replace=False)]
            mul16 = np.mean((np.mod(sample, 16) == 0).astype(np.float32))
            if med > 300 or mul16 > 0.9:  # SGBM *16
                D = D / 16.0

        D[D <= 0] = np.nan
        if target_shape is not None and D.shape != tuple(target_shape):
            h, w = target_shape
            D = cv.resize(D, (w, h), interpolation=cv.INTER_NEAREST)
        return D

    raise FileNotFoundError(f"No encontré disparidad idx={idx} en {disp_dir}")

# Checker robusto
def _norm_none(gray): return gray
def _norm_clahe(gray):
    clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)); return clahe.apply(gray)
def _norm_eq(gray): return cv.equalizeHist(gray)
def _norm_adapt(gray): return cv.equalizeHist(cv.GaussianBlur(gray, (5,5), 0))

_NORMS  = [_norm_none, _norm_clahe, _norm_eq, _norm_adapt]
_SCALES = [1.0, 0.75, 0.5]

def _detect_checker_once(gray, cols, rows):
    flags = cv.CALIB_CB_EXHAUSTIVE | cv.CALIB_CB_ACCURACY
    ok, corners = cv.findChessboardCornersSB(gray, (cols, rows), flags=flags)
    if ok: return True, corners.reshape(-1,2)
    ok, corners = cv.findChessboardCorners(
        gray, (cols, rows),
        flags=cv.CALIB_CB_ADAPTIVE_THRESH | cv.CALIB_CB_NORMALIZE_IMAGE
    )
    if ok:
        cv.cornerSubPix(gray, corners, (11,11), (-1,-1),
                        (cv.TERM_CRITERIA_EPS+cv.TERM_CRITERIA_MAX_ITER, 40, 1e-3))
        return True, corners.reshape(-1,2)
    return False, None

def detect_checker_robusto(gray, cols=10, rows=7):
    H, W = gray.shape[:2]
    for (c, r) in [(cols, rows), (rows, cols)]:
        for scale in _SCALES:
            g = gray if scale == 1.0 else cv.resize(gray, (int(W*scale), int(H*scale)), interpolation=cv.INTER_AREA)
            for norm in _NORMS:
                gn = norm(g)
                ok, corners = _detect_checker_once(gn, c, r)
                if ok:
                    if scale != 1.0: corners = corners * (1.0/scale)
                    return True, corners, (c, r)
    return False, None, (cols, rows)

# Fallback ArUco/ChArUco → homografía
_ARUCO_DICT = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_6X6_250)
_CHAR_SX, _CHAR_SY = 10, 7
_CHAR_SQ, _CHAR_MK = 1.0, 0.7
_CHAR_BOARD = cv.aruco.CharucoBoard((_CHAR_SX, _CHAR_SY), _CHAR_SQ, _CHAR_MK, _ARUCO_DICT)

def _collect_aruco_2d_3d(image_bgr, aruco_dict=_ARUCO_DICT, board=_CHAR_BOARD):
    gray = cv.cvtColor(image_bgr, cv.COLOR_BGR2GRAY)
    detector = cv.aruco.ArucoDetector(aruco_dict, cv.aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None or len(ids) == 0: return None, None
    ids = ids.flatten().tolist()
    ids_board = board.getIds().flatten().tolist()
    obj_list  = board.getObjPoints()
    lut = {int(i): np.asarray(obj, np.float32)[:,:2] for i, obj in zip(ids_board, obj_list)}
    pts_img, pts_board = [], []
    for c_img, id_ in zip(corners, ids):
        if id_ not in lut: continue
        pts_img.append(c_img[0].astype(np.float32))
        pts_board.append(lut[id_])
    if not pts_img: return None, None
    return np.concatenate(pts_img, axis=0), np.concatenate(pts_board, axis=0)

def _synthesize_checker_inner_corners(cols, rows, square_len=1.0):
    xs, ys = np.meshgrid(np.arange(cols), np.arange(rows))
    return np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float32) * square_len

def detect_checker_by_charuco_homography(left_bgr, cols=10, rows=7, aruco_dict=_ARUCO_DICT, board=_CHAR_BOARD):
    pts_img, pts_board = _collect_aruco_2d_3d(left_bgr, aruco_dict, board)
    if pts_img is None: return False, None
    H, _ = cv.findHomography(pts_board, pts_img, method=cv.RANSAC, ransacReprojThreshold=2.0)
    if H is None: return False, None
    inner = _synthesize_checker_inner_corners(cols, rows, square_len=board.getSquareLength())
    inner_1 = np.concatenate([inner, np.ones((inner.shape[0],1), np.float32)], axis=1)
    proj_h = (H @ inner_1.T).T
    proj = proj_h[:, :2] / proj_h[:, 2:3]
    return True, proj

#  Muestreo y LK 
def bilinear_sample(img: np.ndarray, x: float, y: float) -> float:
    h, w = img.shape
    if x < 0 or y < 0 or x > w-1 or y > h-1: return np.nan
    x0, y0 = int(np.floor(x)), int(np.floor(y))
    x1, y1 = min(x0+1, w-1), min(y0+1, h-1)
    fx, fy = x - x0, y - y0
    Q11, Q21 = img[y0, x0], img[y0, x1]
    Q12, Q22 = img[y1, x0], img[y1, x1]
    if not np.isfinite([Q11, Q21, Q12, Q22]).any(): return np.nan
    q = np.array([Q11, Q21, Q12, Q22], dtype=np.float32)
    if np.isnan(q).any():
        m = np.isfinite(q)
        if not m.any(): return np.nan
        q[~m] = np.mean(q[m])
        Q11, Q21, Q12, Q22 = q
    return float(Q11*(1-fx)*(1-fy) + Q21*fx*(1-fy) + Q12*(1-fx)*fy + Q22*fx*fy)

def disparity_lk(left_gray: np.ndarray, right_gray: np.ndarray, corners_L: np.ndarray) -> np.ndarray:
    p0 = corners_L.reshape(-1,1,2).astype(np.float32)
    lk_params = dict(winSize=(21,21), maxLevel=3,
                     criteria=(cv.TERM_CRITERIA_EPS|cv.TERM_CRITERIA_COUNT, 30, 0.01))
    p1, st, _ = cv.calcOpticalFlowPyrLK(left_gray, right_gray, p0, None, **lk_params)
    disp = np.full((len(corners_L),), np.nan, dtype=np.float32)
    if p1 is not None:
        for i, ok in enumerate(st.reshape(-1)):
            if ok:
                disp[i] = p0[i,0,0] - p1[i,0,0]
    return disp

#  Procesamiento por par 
def process_pair(lpath: Path, rpath: Path, idx: int, disp_dir: Path,
                 out_dir: Path, cb_cols=10, cb_rows=7, do_lk=True) -> None:
    left  = cv.imread(str(lpath), cv.IMREAD_COLOR)
    right = cv.imread(str(rpath), cv.IMREAD_COLOR)
    if left is None or right is None:
        print(f"[{idx}] no pude leer imágenes"); return

    h, w = left.shape[:2]
    disp = load_disparity(disp_dir, idx, target_shape=(h, w))
    grayL = cv.cvtColor(left, cv.COLOR_BGR2GRAY)
    grayR = cv.cvtColor(right, cv.COLOR_BGR2GRAY)

    ok, corners, used = detect_checker_robusto(grayL, cb_cols, cb_rows)
    if not ok:
        ok, corners_char = detect_checker_by_charuco_homography(left, cols=cb_cols, rows=cb_rows)
        if ok: corners, used = corners_char, (cb_cols, cb_rows)
        else:
            ok, corners_char = detect_checker_by_charuco_homography(left, cols=cb_rows, rows=cb_cols)
            if ok: corners, used = corners_char, (cb_rows, cb_cols)

    if not ok:
        print(f"[{idx}] no se detectó checker (multi-intento + fallback ArUco)")
        return

    d_map = np.array([bilinear_sample(disp, u, v) for (u, v) in corners], dtype=np.float32)
    valid_map = np.isfinite(d_map) & (d_map > 0)

    if do_lk:
        d_lk = disparity_lk(grayL, grayR, corners)
        valid_lk = np.isfinite(d_lk) & (d_lk > 0)
    else:
        d_lk = np.full_like(d_map, np.nan)
        valid_lk = np.zeros_like(valid_map, dtype=bool)

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"corners_disp_pair_{idx}.csv"
    with open(csv_path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["corner_id","uL","vL","disp_map_px","disp_lk_px","valid_map","valid_lk"])
        for k, (uv, dm, dl, vm, vl) in enumerate(zip(corners, d_map, d_lk, valid_map, valid_lk)):
            wr.writerow([k, float(uv[0]), float(uv[1]),
                         float(dm) if np.isfinite(dm) else -1.0,
                         float(dl) if np.isfinite(dl) else -1.0,
                         int(vm), int(vl)])

    vis = left.copy()
    for (u,v), dm, dl, vm, vl in zip(corners, d_map, d_lk, valid_map, valid_lk):
        color = (0,255,0) if vm else (0,0,255)
        cv.circle(vis, (int(round(u)), int(round(v))), 3, color, -1)
        txt = f"dmap={0 if not vm else dm:.1f}"
        if np.isfinite(dl): txt += f"  dLK={0 if not vl else dl:.1f}"
        cv.putText(vis, txt, (int(u)+5, int(v)-5), cv.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,0), 1, cv.LINE_AA)
    cv.imwrite(str(out_dir / f"corners_disp_pair_{idx}.png"), vis)

    m = valid_map
    mlk = valid_lk & np.isfinite(d_map)
    msg = f"[{idx}] corners={len(corners)}  valid_map={m.sum()}  valid_lk={valid_lk.sum()}"
    if mlk.any():
        diff = np.abs(d_map[mlk] - d_lk[mlk])
        msg += f"  |  |d_map - d_LK| mean={diff.mean():.2f}px"
    print(msg, "→", csv_path.name)
