# icp_refine.py
from __future__ import annotations
import json, re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import open3d as o3d

# Configs y helpers de paths

@dataclass
class ICPConfig:
    """Parámetros del pipeline ICP/fusión/mesh."""
    icp_levels: Tuple[float, ...] = (0.02, 0.01, 0.005)  # dmax por nivel (m)
    icp_iters:  Tuple[int,  ...] = (60,   80,   100)     # iteraciones por nivel
    voxel_seed: float = 0.006                             # downsample previo (m)
    voxel_fuse: float = 0.003                             # voxel final de fusión (m)
    nb_neighbors: int = 20                                # limpieza estadística
    std_ratio:   float = 1.5
    min_points:  int = 2000                               # mínimo de puntos para aceptar nube


@dataclass
class Paths:
    """Rutas del dataset/entradas/salidas."""
    base: Path
    ply_dir: Path
    pose_dir: Path
    out_ply: Path
    out_mesh: Path

    @classmethod
    def from_base(cls, base: Path) -> "Paths":
        ply_dir  = base / "pcd_charuco_boxed_calib"
        pose_dir = base / "out" / "out" / "per_pair"
        out_ply  = base / "buda_refined_icp_vox.ply"
        out_mesh = base / "buda_refined_mesh_bpa.ply"
        return cls(base, ply_dir, pose_dir, out_ply, out_mesh)


# Utilidades

def list_indices(pose_dir: Path) -> List[int]:
    """Lista los índices 'pose_#.json' presentes en pose_dir."""
    idxs: List[int] = []
    for p in sorted(pose_dir.glob("pose_*.json")):
        m = re.search(r"pose_(\d+)\.json$", p.name)
        if m:
            idxs.append(int(m.group(1)))
    return sorted(idxs)


def load_pose(pose_dir: Path, i: int) -> np.ndarray:
    """
    Carga R,t (tablero→cámara) desde pose_{i}.json y devuelve T (4x4) cámara→tablero.
    t en milímetros; se convierte a metros.
    """
    jpath = pose_dir / f"pose_{i}.json"
    J = json.loads(jpath.read_text(encoding="utf-8"))
    R = np.array(J["R"], dtype=np.float64)              # 3x3
    t = np.array(J["tvec"], dtype=np.float64).reshape(3) * 1e-3  # mm→m

    Rt = R.T
    T = np.eye(4)
    T[:3, :3] = Rt
    T[:3,  3] = -Rt @ t
    return T


def prep(pc: o3d.geometry.PointCloud, voxel: Optional[float] = None) -> o3d.geometry.PointCloud:
    """
    Downsample opcional + estimación de normales.
    Retorna la MISMA instancia modificada (como hace Open3D), para eficiencia.
    """
    if voxel:
        pc = pc.voxel_down_sample(voxel)
    # Radio proporcional al voxel (o 5mm) * 8, max_nn conservador
    radius = (voxel or 0.005) * 8
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))
    return pc


# Carga y preprocesado

def load_and_prepare_clouds(paths: Paths, cfg: ICPConfig) -> List[o3d.geometry.PointCloud]:
    """
    Carga las nubes PLY, las transforma a marco 'tablero' con T (cam→tablero),
    aplica un voxel_seed y estima normales. Filtra por tamaño.
    """
    idxs = list_indices(paths.pose_dir)
    if not idxs:
        raise FileNotFoundError(f"No encontré pose_*.json en {paths.pose_dir}")

    acc_list: List[o3d.geometry.PointCloud] = []
    for i in idxs:
        ply = paths.ply_dir / f"buda_charuco_box_calib_{i}.ply"
        if not ply.exists():
            print(f"[{i}] ⚠️ falta {ply.name} → salto")
            continue

        pc = o3d.io.read_point_cloud(str(ply))
        if len(pc.points) < cfg.min_points:
            print(f"[{i}] ⚠️ nube chica ({len(pc.points)}) → salto")
            continue

        T_cb = load_pose(paths.pose_dir, i)  # cámara→tablero
        pc.transform(T_cb)
        prep(pc, cfg.voxel_seed)
        acc_list.append(pc)
        print(f"[{i}] ok ({len(pc.points)} pts)")

    if not acc_list:
        raise RuntimeError("No hay nubes válidas tras cargar/transformar.")
    return acc_list


# ICP multi-escala

def multiscale_icp_merge(
    pcs: List[o3d.geometry.PointCloud],
    cfg: ICPConfig
) -> o3d.geometry.PointCloud:
    """
    Refina/merguea TODAS las nubes contra un modelo acumulado con ICP point-to-plane multi-escala.
    Devuelve la nube merged (sin fusión fina todavía).
    """
    model = o3d.geometry.PointCloud(pcs[0])  # seed
    for k in range(1, len(pcs)):
        src = pcs[k]
        transform = np.eye(4)

        # 3 niveles coarse→fine
        for dmax, iters in zip(cfg.icp_levels, cfg.icp_iters):
            # En cada nivel se usa el target downsampleado a voxel_seed para velocidad/estabilidad
            model_k = prep(o3d.geometry.PointCloud(model), cfg.voxel_seed)
            reg = o3d.pipelines.registration.registration_icp(
                src, model_k, dmax, transform,
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=iters)
            )
            transform = reg.transformation

        src.transform(transform)
        model += src
        print(f"fusión {k}/{len(pcs)-1} → total {len(model.points):,}")
    return model


# Fusión fina + limpieza

def fuse_and_clean(
    model: o3d.geometry.PointCloud,
    cfg: ICPConfig
) -> o3d.geometry.PointCloud:
    """
    Voxel downsample fino + limpieza estadística.
    """
    fused = model.voxel_down_sample(cfg.voxel_fuse)
    fused, _ = fused.remove_statistical_outlier(
        nb_neighbors=cfg.nb_neighbors,
        std_ratio=cfg.std_ratio
    )
    return fused


# Estimación de mesh

def estimate_mean_spacing(
    pc: o3d.geometry.PointCloud,
    sample_cap: int = 5000
) -> float:
    """
    Estima un spacing medio entre vecinos por muestreo.
    """
    kdt = o3d.geometry.KDTreeFlann(pc)
    pts_np = np.asarray(pc.points)
    if len(pts_np) == 0:
        return 0.0

    step = max(len(pts_np) // sample_cap, 1)
    spacings = []
    for i in range(0, len(pts_np), step):
        _, idx, _ = kdt.search_knn_vector_3d(pc.points[i], 8)
        nn = pts_np[idx]
        d = np.linalg.norm(nn - pts_np[i], axis=1)
        med = np.median(d[d > 0])
        if np.isfinite(med) and med > 0:
            spacings.append(med)

    return float(np.median(spacings)) if spacings else 0.0


def mesh_ball_pivoting(
    pc: o3d.geometry.PointCloud,
    mean_nn: Optional[float] = None
) -> o3d.geometry.TriangleMesh:
    """
    Construye una malla con Ball Pivoting usando radios derivados del spacing medio.
    """
    if mean_nn is None or mean_nn <= 0:
        mean_nn = estimate_mean_spacing(pc) or 0.003  # fallback 

    r1, r2 = 1.5 * mean_nn, 2.2 * mean_nn

    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=8*mean_nn, max_nn=30))
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pc, o3d.utility.DoubleVector([r1, r2])
    )
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.compute_vertex_normals()
    return mesh


# Orquestador

def run_pipeline(
    paths: Paths,
    cfg: ICPConfig,
    make_mesh: bool = True,
    visualize: bool = False
) -> Tuple[o3d.geometry.PointCloud, Optional[o3d.geometry.TriangleMesh]]:
    """
    Ejecuta: carga→ICP→fusión fina→(opcional) BPA mesh.
    Escribe OUT_PLY/OUT_MESH en disco y retorna (nube, malla o None).
    """
    pcs = load_and_prepare_clouds(paths, cfg)
    merged = multiscale_icp_merge(pcs, cfg)
    fused  = fuse_and_clean(merged, cfg)

    o3d.io.write_point_cloud(str(paths.out_ply), fused, write_ascii=False)
    print(f"OK nube → {paths.out_ply} ({len(fused.points):,} pts)")

    mesh = None
    if make_mesh:
        mean_nn = estimate_mean_spacing(fused) or cfg.voxel_fuse
        mesh = mesh_ball_pivoting(fused, mean_nn)
        o3d.io.write_triangle_mesh(str(paths.out_mesh), mesh)
        print(f"OK malla → {paths.out_mesh}")

    if visualize:
        o3d.visualization.draw_geometries([fused], window_name="Nube refinada")

    return fused, mesh
