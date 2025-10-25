from pathlib import Path
import re
import numpy as np
import cv2 as cv
from utils.disparity.methods import InputPair, Config
from utils.disparity.method_cre_stereo import CREStereo


# Empareja left_#.jpg con right_#.jpg (recursivo)
def pair_rectified(left_root: Path, right_root: Path):
    L = list(left_root.rglob("left_*.jpg"))  + list(left_root.rglob("left_*.JPG"))
    R = list(right_root.rglob("right_*.jpg"))+ list(right_root.rglob("right_*.JPG"))
    rxL = re.compile(r"^left_(\d+)\.jpg$", re.IGNORECASE)
    rxR = re.compile(r"^right_(\d+)\.jpg$", re.IGNORECASE)
    Lmap = {int(rxL.match(p.name).group(1)): p for p in L if rxL.match(p.name)}
    Rmap = {int(rxR.match(p.name).group(1)): p for p in R if rxR.match(p.name)}
    idxs = sorted(set(Lmap) & set(Rmap))
    return [Lmap[i] for i in idxs], [Rmap[i] for i in idxs], idxs


# Instancia y configura CREStereo una sola vez
def make_cre(models_path: Path, shape: str = "1280x720", mode: str | None = None) -> CREStereo:
    cfg = Config(models_path=models_path)
    cre = CREStereo(cfg)
    cre.parameters["Shape"].set_value(shape)
    if mode is not None:
        cre.parameters["Mode"].set_value(mode)
    return cre


# PNG color para “preview” de disparidad
def save_disp_preview(disp_px: np.ndarray, path_png: Path, vmax=None) -> None:
    d = disp_px.copy()
    d[~np.isfinite(d)] = 0
    if vmax is None:
        vmax = np.nanpercentile(d, 99.0) if np.any(d > 0) else 1.0
        if vmax <= 0:
            vmax = 1.0
    d = np.clip(d, 0, vmax) / vmax
    d8 = (d * 255).astype(np.uint8)
    col = cv.applyColorMap(d8, cv.COLORMAP_TURBO)
    cv.imwrite(str(path_png), col)


# Ejecuta CREStereo para un par y guarda .npz (+ preview)
def run_cre_on_pair(cre: CREStereo,
                    calibration,
                    left_path: Path,
                    right_path: Path,
                    out_dir: Path,
                    idx: int,
                    preview: bool = True) -> dict:
    left_img  = cv.imread(str(left_path),  cv.IMREAD_COLOR)
    right_img = cv.imread(str(right_path), cv.IMREAD_COLOR)
    if left_img is None or right_img is None:
        return {"idx": idx, "ok": False, "msg": "imread None"}

    pair = InputPair(left_img, right_img, calibration)
    result = cre.compute_disparity(pair)
    disp = np.asarray(result.disparity_pixels, dtype=np.float32)
    disp[~np.isfinite(disp) | (disp <= 0)] = np.nan

    out_npz = out_dir / f"disp_{idx}.npz"
    np.savez_compressed(out_npz, disp=disp)

    if preview:
        save_disp_preview(disp, out_dir / f"disp_{idx}.png")

    valid = np.isfinite(disp)
    cov = 100.0 * valid.sum() / disp.size
    mn = float(np.nanmin(disp)) if valid.any() else np.nan
    mx = float(np.nanmax(disp)) if valid.any() else np.nan
    mean = float(np.nanmean(disp[valid])) if valid.any() else np.nan

    return {
        "idx": idx, "ok": True, "npz": str(out_npz),
        "cover_pct": cov, "min": mn, "mean": mean, "max": mx
    }
