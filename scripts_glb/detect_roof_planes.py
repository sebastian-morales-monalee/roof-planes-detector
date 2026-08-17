#!/usr/bin/env python3
"""Detect exterior planar roof regions in a GLB and export JSON/debug GLBs.

Run from scripts_glb with regular Python. The script relaunches itself inside
Blender so that it can read and write GLB scenes.
"""

from __future__ import annotations

import argparse
import colorsys
from copy import deepcopy
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import traceback
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

try:
    import bpy
    import numpy as np
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree
    from mathutils.geometry import tessellate_polygon
except ModuleNotFoundError:
    bpy = None
    np = None
    Vector = None
    BVHTree = None
    tessellate_polygon = None


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = SCRIPT_DIR / "input_glbs"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "outputs_glb"
VENDOR_DIR = SCRIPT_DIR / ".vendor"

if bpy is not None:
    if VENDOR_DIR.is_dir() and str(VENDOR_DIR) not in sys.path:
        sys.path.append(str(VENDOR_DIR))
    try:
        from shapely import coverage_union_all, make_valid, unary_union
        from shapely.geometry import MultiPolygon, Polygon
        from shapely.geometry.polygon import orient as orient_polygon
    except ModuleNotFoundError:
        coverage_union_all = None
        make_valid = None
        unary_union = None
        MultiPolygon = None
        Polygon = None
        orient_polygon = None
else:
    coverage_union_all = None
    make_valid = None
    unary_union = None
    MultiPolygon = None
    Polygon = None
    orient_polygon = None
VIEW_FROM_GLTF = {
    "positive_x": (1.0, 0.0, 0.0),
    "negative_x": (-1.0, 0.0, 0.0),
    "positive_y": (0.0, 1.0, 0.0),
    "negative_y": (0.0, -1.0, 0.0),
    "positive_z": (0.0, 0.0, 1.0),
    "negative_z": (0.0, 0.0, -1.0),
}
REGULARIZED_ROTATIONS = (
    "x_positive_90",
    "x_negative_90",
    "y_positive_90",
    "y_negative_90",
    "z_positive_90",
    "z_negative_90",
)


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, first: int, second: int) -> None:
        root_first = self.find(first)
        root_second = self.find(second)
        if root_first == root_second:
            return
        if self.rank[root_first] < self.rank[root_second]:
            root_first, root_second = root_second, root_first
        self.parent[root_second] = root_first
        if self.rank[root_first] == self.rank[root_second]:
            self.rank[root_first] += 1


def user_arguments() -> list[str]:
    if bpy is not None and "--" in sys.argv:
        return sys.argv[sys.argv.index("--") + 1 :]
    return sys.argv[1:]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Detecta y describe planos exteriores de techo en un GLB, y genera "
            "un JSON y GLB de validación."
        )
    )
    parser.add_argument("input", help="Archivo GLB que se analizará.")
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Carpeta raíz de resultados (predeterminado: outputs_glb).",
    )
    parser.add_argument(
        "--roof-up",
        choices=tuple(VIEW_FROM_GLTF),
        default="negative_x",
        help=(
            "Dirección de referencia para clasificar superficies de techo "
            "(predeterminado: negative_x)."
        ),
    )
    parser.add_argument(
        "--roof-ups",
        nargs="+",
        choices=(*VIEW_FROM_GLTF.keys(), "all"),
        default=None,
        help=(
            "Compara varias referencias de techo; usa --roof-ups all para generar "
            "los seis resultados y una imagen comparativa."
        ),
    )
    parser.add_argument(
        "--normal-direction",
        choices=("two-sided", "signed"),
        default="signed",
        help=(
            "two-sided tolera normales invertidas; signed distingue los sentidos "
            "positivo y negativo (predeterminado: signed)."
        ),
    )
    parser.add_argument(
        "--visibility",
        choices=("none", "directional"),
        default="none",
        help=(
            "none analiza todo el modelo sin filtrar por una vista; directional "
            "aplica oclusión desde --view-from (predeterminado: none)."
        ),
    )
    parser.add_argument(
        "--view-from",
        "--preview-from",
        dest="view_from",
        choices=tuple(VIEW_FROM_GLTF),
        default="negative_z",
        help=(
            "Lado de la cámara de validación; con --visibility directional también "
            "controla la dirección de oclusión (predeterminado: negative_z)."
        ),
    )
    parser.add_argument(
        "--views",
        nargs="+",
        choices=(*VIEW_FROM_GLTF.keys(), "all"),
        default=None,
        help=(
            "Genera una calibración para varias vistas; usa --views all para los "
            "seis lados del modelo."
        ),
    )
    parser.add_argument(
        "--visible-part",
        choices=("only", "full-plane"),
        default="only",
        help=(
            "only conserva únicamente los triángulos visibles; full-plane conserva "
            "el plano completo cuando es visible (predeterminado: only)."
        ),
    )
    parser.add_argument(
        "--max-pitch",
        type=float,
        default=75.0,
        help="Pendiente máxima admitida para un plano de techo, en grados.",
    )
    parser.add_argument(
        "--angle-tolerance",
        type=float,
        default=3.0,
        help=(
            "Diferencia angular máxima entre triángulos vecinos, en grados "
            "(predeterminado: 3)."
        ),
    )
    parser.add_argument(
        "--min-area",
        type=float,
        default=0.08,
        help="Área mínima de un plano en unidades cuadradas del modelo.",
    )
    parser.add_argument(
        "--min-faces",
        type=int,
        default=20,
        help="Cantidad mínima de triángulos de un plano.",
    )
    parser.add_argument(
        "--max-plane-rms",
        type=float,
        default=0.035,
        help="Error RMS máximo del ajuste plano, en unidades del modelo.",
    )
    parser.add_argument(
        "--min-view-visibility",
        "--min-sky-visibility",
        dest="min_view_visibility",
        type=float,
        default=0.25,
        help="Fracción mínima de muestras visibles desde el lado seleccionado.",
    )
    parser.add_argument(
        "--boundary-simplify",
        type=float,
        default=0.02,
        help="Tolerancia para simplificar contornos, en unidades del modelo.",
    )
    parser.add_argument(
        "--geometry-output",
        choices=("exact", "simplified", "both"),
        default="both",
        help=(
            "Genera geometría original, vectorizada o ambas "
            "(predeterminado: both)."
        ),
    )
    parser.add_argument(
        "--merge-angle",
        type=float,
        default=5.0,
        help="Ángulo máximo para fusionar regiones coplanares, en grados.",
    )
    parser.add_argument(
        "--merge-plane-distance",
        type=float,
        default=0.05,
        help="Separación máxima entre planos que se pueden fusionar.",
    )
    parser.add_argument(
        "--merge-max-residual",
        type=float,
        default=0.06,
        help="Error máximo admitido después del ajuste plano conjunto.",
    )
    parser.add_argument(
        "--merge-residual-percentile",
        type=float,
        default=95.0,
        help=(
            "Percentil de residuos que debe quedar dentro de "
            "--merge-max-residual en la validación robusta."
        ),
    )
    parser.add_argument(
        "--merge-min-inlier-ratio",
        type=float,
        default=0.95,
        help=(
            "Fracción mínima de vértices dentro de --merge-max-residual "
            "para aceptar unos pocos valores atípicos."
        ),
    )
    parser.add_argument(
        "--merge-robust-max-residual",
        type=float,
        default=0.10,
        help="Límite absoluto de seguridad para una fusión robusta.",
    )
    parser.add_argument(
        "--merge-min-boundary-outlier-ratio",
        type=float,
        default=0.30,
        help=(
            "Fracción mínima de valores atípicos que debe estar en el borde "
            "de la región para aceptar la fusión robusta."
        ),
    )
    parser.add_argument(
        "--merge-gap",
        type=float,
        default=0.08,
        help="Separación espacial máxima que se cerrará entre parches coplanares.",
    )
    parser.add_argument(
        "--simplified-boundary-tolerance",
        type=float,
        default=0.04,
        help="Tolerancia para reducir vértices del contorno vectorizado.",
    )
    parser.add_argument(
        "--boundary-regularization",
        choices=("none", "lines"),
        default="lines",
        help=(
            "Regulariza los dientes del contorno mediante segmentos dominantes "
            "y conserva la versión simplificada para comparación."
        ),
    )
    parser.add_argument(
        "--regularization-tolerance",
        type=float,
        default=0.15,
        help="Distancia máxima para reemplazar dientes por una línea recta.",
    )
    parser.add_argument(
        "--regularization-min-iou",
        type=float,
        default=0.95,
        help="IoU mínimo entre el contorno simplificado y el regularizado.",
    )
    parser.add_argument(
        "--regularization-max-area-change",
        type=float,
        default=0.05,
        help="Cambio relativo máximo de área permitido al regularizar.",
    )
    parser.add_argument(
        "--regularized-rotations",
        nargs="+",
        choices=("none", "all", *REGULARIZED_ROTATIONS),
        default=["none"],
        help=(
            "Genera copias rígidas de regularized-only rotadas 90 grados en "
            "ejes glTF; usa all para las seis o indica una o varias rotaciones."
        ),
    )
    parser.add_argument(
        "--min-hole-area",
        type=float,
        default=0.01,
        help="Área mínima de un hueco que se conservará en la versión simplificada.",
    )
    parser.add_argument(
        "--timezone",
        default="America/Bogota",
        help="Zona horaria usada para identificar la ejecución.",
    )
    parser.add_argument(
        "--blender",
        default=os.environ.get("BLENDER_EXECUTABLE", "blender"),
        help="Ejecutable de Blender cuando se lanza con Python.",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if not 0.0 <= args.max_pitch < 90.0:
        raise ValueError("--max-pitch debe estar entre 0 y menos de 90 grados.")
    if not 0.0 < args.angle_tolerance < 90.0:
        raise ValueError("--angle-tolerance debe estar entre 0 y 90 grados.")
    if args.min_area <= 0 or args.min_faces < 1 or args.max_plane_rms <= 0:
        raise ValueError("Las tolerancias de área, caras y RMS deben ser positivas.")
    if not 0.0 <= args.min_view_visibility <= 1.0:
        raise ValueError("--min-view-visibility debe estar entre 0 y 1.")
    if args.boundary_simplify < 0:
        raise ValueError("--boundary-simplify no puede ser negativo.")
    if not 0.0 < args.merge_angle < 90.0:
        raise ValueError("--merge-angle debe estar entre 0 y 90 grados.")
    if not 0.0 < args.merge_residual_percentile <= 100.0:
        raise ValueError("--merge-residual-percentile debe estar entre 0 y 100.")
    if not 0.0 <= args.merge_min_inlier_ratio <= 1.0:
        raise ValueError("--merge-min-inlier-ratio debe estar entre 0 y 1.")
    if not 0.0 <= args.merge_min_boundary_outlier_ratio <= 1.0:
        raise ValueError(
            "--merge-min-boundary-outlier-ratio debe estar entre 0 y 1."
        )
    if args.regularization_tolerance < 0:
        raise ValueError("--regularization-tolerance no puede ser negativa.")
    if not 0.0 <= args.regularization_min_iou <= 1.0:
        raise ValueError("--regularization-min-iou debe estar entre 0 y 1.")
    if not 0.0 <= args.regularization_max_area_change <= 1.0:
        raise ValueError(
            "--regularization-max-area-change debe estar entre 0 y 1."
        )
    if "none" in args.regularized_rotations and len(args.regularized_rotations) != 1:
        raise ValueError("'none' no se puede combinar con otras rotaciones.")
    if "all" in args.regularized_rotations and len(args.regularized_rotations) != 1:
        raise ValueError("'all' no se puede combinar con otras rotaciones.")
    if args.regularized_rotations != ["none"] and args.geometry_output == "exact":
        raise ValueError(
            "Las rotaciones regularizadas requieren --geometry-output simplified o both."
        )
    if min(
        args.merge_plane_distance,
        args.merge_max_residual,
        args.merge_robust_max_residual,
        args.merge_gap,
        args.simplified_boundary_tolerance,
        args.min_hole_area,
    ) < 0:
        raise ValueError("Las tolerancias de fusión y simplificación no pueden ser negativas.")
    if args.views and "all" in args.views and len(args.views) != 1:
        raise ValueError("'all' no se puede combinar con otras vistas.")
    if args.roof_ups and "all" in args.roof_ups and len(args.roof_ups) != 1:
        raise ValueError("'all' no se puede combinar con otros valores de --roof-ups.")
    if args.views and args.roof_ups:
        raise ValueError("--views y --roof-ups son modalidades diferentes y no se combinan.")
    if args.regularized_rotations != ["none"] and (args.views or args.roof_ups):
        raise ValueError(
            "--regularized-rotations solo se admite en la detección principal, "
            "sin --views ni --roof-ups."
        )
    if args.views and args.visible_part != "full-plane":
        raise ValueError("Las ejecuciones multivista requieren --visible-part full-plane.")


def resolved_views(args: argparse.Namespace) -> tuple[str, ...]:
    if not args.views:
        return (args.view_from,)
    if args.views == ["all"]:
        return tuple(VIEW_FROM_GLTF)
    return tuple(dict.fromkeys(args.views))


def resolved_roof_ups(args: argparse.Namespace) -> tuple[str, ...]:
    if not args.roof_ups:
        return ()
    if args.roof_ups == ["all"]:
        return tuple(VIEW_FROM_GLTF)
    return tuple(dict.fromkeys(args.roof_ups))


def resolved_regularized_rotations(args: argparse.Namespace) -> tuple[str, ...]:
    if args.regularized_rotations == ["none"]:
        return ()
    if args.regularized_rotations == ["all"]:
        return REGULARIZED_ROTATIONS
    return tuple(dict.fromkeys(args.regularized_rotations))


def resolve_input(value: str) -> Path:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute() and not candidate.exists():
        candidate = DEFAULT_INPUT_DIR / candidate
    candidate = candidate.resolve()
    if not candidate.is_file() or candidate.suffix.lower() != ".glb":
        raise FileNotFoundError(f"No existe un GLB de entrada válido: {candidate}")
    return candidate


def launch_in_blender(args: argparse.Namespace, raw_args: list[str]) -> int:
    executable = shutil.which(args.blender)
    if executable is None:
        raise RuntimeError(f"No se encontró Blender: {args.blender}")
    command = [
        executable,
        "--background",
        "--python",
        str(Path(__file__).resolve()),
        "--",
        *raw_args,
    ]
    return subprocess.run(command, check=False).returncode


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def blender_to_gltf_array(points: Any) -> Any:
    points = np.asarray(points, dtype=np.float64)
    converted = np.empty_like(points)
    converted[..., 0] = points[..., 0]
    converted[..., 1] = points[..., 2]
    converted[..., 2] = -points[..., 1]
    return converted


def gltf_to_blender_array(points: Any) -> Any:
    points = np.asarray(points, dtype=np.float64)
    converted = np.empty_like(points)
    converted[..., 0] = points[..., 0]
    converted[..., 1] = -points[..., 2]
    converted[..., 2] = points[..., 1]
    return converted


def prepare_roof_candidates(
    data: dict[str, Any],
    roof_up_name: str,
    max_pitch: float,
    normal_direction: str,
) -> tuple[Any, Any, Any, dict[str, Any]]:
    """Orient normals consistently and select faces relative to a roof axis.

    Imported meshes can contain the same geometric surface with either winding.
    In two-sided mode, absolute alignment tolerates inverted winding. Signed mode
    preserves the original normal sense so opposite roof-up directions differ.
    """
    roof_up_gltf = np.asarray(VIEW_FROM_GLTF[roof_up_name], dtype=np.float64)
    roof_up_blender = gltf_to_blender_array(roof_up_gltf)
    roof_up_blender /= np.linalg.norm(roof_up_blender)
    prepared_data = dict(data)
    prepared_data["normals"] = data["normals"].copy()
    alignment = prepared_data["normals"] @ roof_up_blender
    flipped_normals = np.zeros(len(alignment), dtype=bool)
    if normal_direction == "two-sided":
        flipped_normals = alignment < 0.0
        prepared_data["normals"][flipped_normals] *= -1.0
    threshold = math.cos(math.radians(max_pitch))
    candidates = np.flatnonzero(
        prepared_data["valid"]
        & ((prepared_data["normals"] @ roof_up_blender) >= threshold)
    )
    return candidates, flipped_normals, roof_up_blender, prepared_data


def clean_number(value: float) -> float:
    value = float(value)
    return 0.0 if abs(value) < 5e-10 else round(value, 9)


def clean_vector(values: Any) -> list[float]:
    return [clean_number(value) for value in values]


def regularized_rotation_matrix_gltf(rotation_name: str) -> Any:
    sign = 1.0 if "_positive_" in rotation_name else -1.0
    axis = rotation_name[0]
    if axis == "x":
        return np.asarray(
            ((1.0, 0.0, 0.0), (0.0, 0.0, -sign), (0.0, sign, 0.0)),
            dtype=np.float64,
        )
    if axis == "y":
        return np.asarray(
            ((0.0, 0.0, sign), (0.0, 1.0, 0.0), (-sign, 0.0, 0.0)),
            dtype=np.float64,
        )
    return np.asarray(
        ((0.0, -sign, 0.0), (sign, 0.0, 0.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )


def rotate_gltf_points(points: Any, matrix: Any, pivot: Any) -> Any:
    values = np.asarray(points, dtype=np.float64)
    return (values - pivot) @ matrix.T + pivot


def rotate_boundary_components(components: list[dict[str, Any]], matrix: Any, pivot: Any) -> list[dict[str, Any]]:
    rotated = []
    for component in components:
        rotated.append(
            {
                "outer": [
                    clean_vector(point)
                    for point in rotate_gltf_points(component["outer"], matrix, pivot)
                ],
                "holes": [
                    [
                        clean_vector(point)
                        for point in rotate_gltf_points(hole, matrix, pivot)
                    ]
                    for hole in component["holes"]
                ],
            }
        )
    return rotated


def build_regularized_rotation_results(
    planes: list[dict[str, Any]],
    rotation_names: tuple[str, ...],
    roof_up_name: str,
) -> tuple[Any, dict[str, dict[str, Any]]]:
    all_vertices_gltf = np.concatenate(
        [blender_to_gltf_array(plane["regularized_vertices"]) for plane in planes]
    )
    pivot = (all_vertices_gltf.min(axis=0) + all_vertices_gltf.max(axis=0)) * 0.5
    roof_up = np.asarray(VIEW_FROM_GLTF[roof_up_name], dtype=np.float64)
    roof_up /= np.linalg.norm(roof_up)
    results: dict[str, dict[str, Any]] = {}

    for rotation_name in rotation_names:
        matrix = regularized_rotation_matrix_gltf(rotation_name)
        axis = rotation_name[0]
        angle_degrees = 90.0 if "_positive_" in rotation_name else -90.0
        translation = pivot - matrix @ pivot
        matrix_4x4 = np.eye(4, dtype=np.float64)
        matrix_4x4[:3, :3] = matrix
        matrix_4x4[:3, 3] = translation
        rotated_planes = []
        rotated_json_planes = []
        rotated_vertices_for_bounds = []

        for plane in planes:
            vertices_gltf = blender_to_gltf_array(plane["regularized_vertices"])
            rotated_vertices_gltf = rotate_gltf_points(vertices_gltf, matrix, pivot)
            rotated_vertices_blender = gltf_to_blender_array(rotated_vertices_gltf)
            centroid_gltf = blender_to_gltf_array(plane["centroid"])
            rotated_centroid_gltf = rotate_gltf_points(centroid_gltf, matrix, pivot)
            normal_gltf = blender_to_gltf_array(plane["normal"])
            normal_gltf /= np.linalg.norm(normal_gltf)
            rotated_normal_gltf = matrix @ normal_gltf
            rotated_normal_gltf /= np.linalg.norm(rotated_normal_gltf)
            rotated_normal_blender = gltf_to_blender_array(rotated_normal_gltf)
            pitch = math.degrees(
                math.acos(
                    max(-1.0, min(1.0, float(np.dot(rotated_normal_gltf, roof_up))))
                )
            )
            plane_d = -float(np.dot(rotated_normal_gltf, rotated_centroid_gltf))

            plane_json = deepcopy(plane["regularized_json"])
            plane_json["centroid"] = clean_vector(rotated_centroid_gltf)
            plane_json["normal"] = clean_vector(rotated_normal_gltf)
            plane_json["pitch_degrees"] = clean_number(pitch)
            plane_json["plane_equation"] = {
                "a": clean_number(rotated_normal_gltf[0]),
                "b": clean_number(rotated_normal_gltf[1]),
                "c": clean_number(rotated_normal_gltf[2]),
                "d": clean_number(plane_d),
                "form": "a*x + b*y + c*z + d = 0",
            }
            plane_json["boundary_components"] = rotate_boundary_components(
                plane["regularized_boundaries"], matrix, pivot
            )
            plane_json["mesh"] = {
                "vertices": [clean_vector(point) for point in rotated_vertices_gltf],
                "triangles": [list(face) for face in plane["regularized_faces"]],
            }
            plane_json["rotation"] = {
                "id": rotation_name,
                "axis_gltf": axis.upper(),
                "angle_degrees": clean_number(angle_degrees),
            }
            rotated_json_planes.append(plane_json)
            rotated_vertices_for_bounds.append(rotated_vertices_gltf)
            rotated_planes.append(
                {
                    "id": plane_json["id"],
                    "vertices": rotated_vertices_blender,
                    "faces": plane["regularized_faces"],
                    "normal": rotated_normal_blender,
                    "source_plane_ids": list(plane["source_plane_ids"]),
                    "source_simplified_plane_id": plane["id"],
                    "area": float(plane["regularized_area"]),
                    "pitch_degrees": pitch,
                    "confidence": float(plane["confidence"]),
                    "color_index": int(plane["color_index"]),
                    "regularization_accepted": bool(
                        plane["regularization"]["accepted"]
                    ),
                }
            )

        rotated_bounds = np.concatenate(rotated_vertices_for_bounds)
        results[rotation_name] = {
            "planes": rotated_planes,
            "json": {
                "id": rotation_name,
                "axis_gltf": axis.upper(),
                "angle_degrees": clean_number(angle_degrees),
                "angle_convention": "right-hand rule",
                "pivot_gltf": clean_vector(pivot),
                "rotation_matrix_4x4_gltf": [
                    clean_vector(row) for row in matrix_4x4
                ],
                "bounds_gltf": {
                    "min": clean_vector(rotated_bounds.min(axis=0)),
                    "max": clean_vector(rotated_bounds.max(axis=0)),
                },
                "roof_plane_count": len(rotated_planes),
                "regularized_roof_planes": rotated_json_planes,
            },
        }
    return pivot, results


def import_glb(path: Path) -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    result = bpy.ops.import_scene.gltf(filepath=str(path))
    if "FINISHED" not in result:
        raise RuntimeError(f"Blender no pudo importar {path}")
    if not any(obj.type == "MESH" for obj in bpy.context.scene.objects):
        raise ValueError("El GLB no contiene mallas.")


def extract_world_triangles() -> dict[str, Any]:
    vertices: list[list[float]] = []
    triangles: list[tuple[int, int, int]] = []
    triangle_objects: list[str] = []
    object_summaries = []

    for obj in sorted(
        (item for item in bpy.context.scene.objects if item.type == "MESH"),
        key=lambda item: item.name,
    ):
        mesh = obj.data
        mesh.calc_loop_triangles()
        offset = len(vertices)
        object_vertices = [obj.matrix_world @ vertex.co for vertex in mesh.vertices]
        vertices.extend([list(point) for point in object_vertices])
        for triangle in mesh.loop_triangles:
            triangles.append(tuple(offset + index for index in triangle.vertices))
            triangle_objects.append(obj.name)
        object_summaries.append(
            {
                "name": obj.name,
                "vertices": len(mesh.vertices),
                "triangles": len(mesh.loop_triangles),
            }
        )

    vertex_array = np.asarray(vertices, dtype=np.float64)
    triangle_array = np.asarray(triangles, dtype=np.int64)
    first = vertex_array[triangle_array[:, 0]]
    second = vertex_array[triangle_array[:, 1]]
    third = vertex_array[triangle_array[:, 2]]
    cross = np.cross(second - first, third - first)
    double_area = np.linalg.norm(cross, axis=1)
    valid = double_area > 1e-12
    normals = np.zeros_like(cross)
    normals[valid] = cross[valid] / double_area[valid, None]
    areas = double_area * 0.5
    centers = (first + second + third) / 3.0
    return {
        "vertices": vertex_array,
        "triangles": triangle_array,
        "normals": normals,
        "areas": areas,
        "centers": centers,
        "valid": valid,
        "triangle_objects": triangle_objects,
        "objects": object_summaries,
    }


def group_candidate_faces(
    triangles: Any,
    normals: Any,
    candidate_indices: Any,
    angle_tolerance: float,
) -> list[list[int]]:
    union_find = UnionFind(len(triangles))
    edge_faces: dict[tuple[int, int], list[int]] = defaultdict(list)
    candidate_set = set(int(index) for index in candidate_indices)
    for face_index in candidate_indices:
        a, b, c = (int(value) for value in triangles[face_index])
        edge_faces[tuple(sorted((a, b)))].append(int(face_index))
        edge_faces[tuple(sorted((b, c)))].append(int(face_index))
        edge_faces[tuple(sorted((c, a)))].append(int(face_index))

    cosine_threshold = math.cos(math.radians(angle_tolerance))
    for linked_faces in edge_faces.values():
        if len(linked_faces) < 2:
            continue
        first = linked_faces[0]
        for second in linked_faces[1:]:
            if second in candidate_set and np.dot(normals[first], normals[second]) >= cosine_threshold:
                union_find.union(first, second)

    groups: dict[int, list[int]] = defaultdict(list)
    for face_index in candidate_indices:
        groups[union_find.find(int(face_index))].append(int(face_index))
    return list(groups.values())


def fit_plane(
    group: list[int], data: dict[str, Any], normal_reference: Any
) -> dict[str, Any]:
    face_indices = np.asarray(group, dtype=np.int64)
    areas = data["areas"][face_indices]
    normals = data["normals"][face_indices]
    centers = data["centers"][face_indices]
    total_area = float(areas.sum())
    centroid = np.average(centers, axis=0, weights=areas)
    normal = np.sum(normals * areas[:, None], axis=0)
    normal_length = float(np.linalg.norm(normal))
    if normal_length <= 1e-12:
        normal = np.asarray(normal_reference, dtype=np.float64).copy()
    else:
        normal /= normal_length
    if float(np.dot(normal, normal_reference)) < 0.0:
        normal *= -1

    vertex_ids = np.unique(data["triangles"][face_indices].reshape(-1))
    points = data["vertices"][vertex_ids]
    residuals = np.abs((points - centroid) @ normal)
    rms = float(math.sqrt(float(np.mean(residuals**2))))
    maximum = float(residuals.max(initial=0.0))
    return {
        "face_indices": group,
        "vertex_ids": vertex_ids,
        "area": total_area,
        "centroid": centroid,
        "normal": normal,
        "rms": rms,
        "max_residual": maximum,
        "residuals": residuals,
        "residual_p95": float(np.percentile(residuals, 95.0)),
        "residual_p99": float(np.percentile(residuals, 99.0)),
    }


def boundary_vertex_ids(group: list[int], data: dict[str, Any]) -> set[int]:
    edge_counts: Counter[tuple[int, int]] = Counter()
    for face_index in group:
        triangle = [int(value) for value in data["triangles"][face_index]]
        for edge_index in range(3):
            first = triangle[edge_index]
            second = triangle[(edge_index + 1) % 3]
            edge_counts[tuple(sorted((first, second)))] += 1
    return {
        vertex_id
        for edge, count in edge_counts.items()
        if count == 1
        for vertex_id in edge
    }


def robust_merge_diagnostics(
    plane: dict[str, Any], data: dict[str, Any], args: argparse.Namespace
) -> dict[str, Any]:
    residuals = plane["residuals"]
    tolerance = args.merge_max_residual
    inlier_mask = residuals <= tolerance
    inlier_ratio = float(np.mean(inlier_mask)) if len(residuals) else 1.0
    percentile_residual = float(
        np.percentile(residuals, args.merge_residual_percentile)
    )
    outlier_vertex_ids = {
        int(vertex_id)
        for vertex_id, is_inlier in zip(plane["vertex_ids"], inlier_mask)
        if not bool(is_inlier)
    }
    boundary_ids = boundary_vertex_ids(plane["face_indices"], data)
    boundary_outlier_count = len(outlier_vertex_ids & boundary_ids)
    boundary_outlier_ratio = (
        boundary_outlier_count / len(outlier_vertex_ids)
        if outlier_vertex_ids
        else 1.0
    )
    strict = plane["max_residual"] <= tolerance
    robust = (
        percentile_residual <= tolerance
        and inlier_ratio >= args.merge_min_inlier_ratio
        and plane["max_residual"] <= args.merge_robust_max_residual
        and boundary_outlier_ratio >= args.merge_min_boundary_outlier_ratio
    )
    return {
        "accepted": bool(strict or robust),
        "mode": "strict" if strict else ("robust" if robust else "rejected"),
        "residual_percentile": float(args.merge_residual_percentile),
        "percentile_error": percentile_residual,
        "inlier_ratio": inlier_ratio,
        "outlier_count": len(outlier_vertex_ids),
        "boundary_outlier_count": boundary_outlier_count,
        "boundary_outlier_ratio": float(boundary_outlier_ratio),
    }


def plane_tangent_radius(plane: dict[str, Any], data: dict[str, Any]) -> float:
    points = data["vertices"][plane["vertex_ids"]]
    offsets = points - plane["centroid"]
    tangent = offsets - np.outer(offsets @ plane["normal"], plane["normal"])
    return float(np.linalg.norm(tangent, axis=1).max(initial=0.0))


def coplanar_candidates(
    first: dict[str, Any],
    second: dict[str, Any],
    data: dict[str, Any],
    angle_tolerance: float,
    plane_distance: float,
    spatial_gap: float,
) -> bool:
    cosine_threshold = math.cos(math.radians(angle_tolerance))
    if float(np.dot(first["normal"], second["normal"])) < cosine_threshold:
        return False
    center_delta = second["centroid"] - first["centroid"]
    if min(
        abs(float(np.dot(first["normal"], center_delta))),
        abs(float(np.dot(second["normal"], center_delta))),
    ) > plane_distance:
        return False
    tangent_delta = center_delta - first["normal"] * float(
        np.dot(first["normal"], center_delta)
    )
    return float(np.linalg.norm(tangent_delta)) <= (
        plane_tangent_radius(first, data)
        + plane_tangent_radius(second, data)
        + spatial_gap
    )


def merge_coplanar_planes(
    planes: list[dict[str, Any]],
    data: dict[str, Any],
    normal_reference: Any,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    clusters = []
    for plane in planes:
        copy = dict(plane)
        copy["source_plane_ids"] = [plane["id"]]
        clusters.append(copy)

    changed = True
    while changed:
        changed = False
        for first_index in range(len(clusters)):
            if changed:
                break
            for second_index in range(first_index + 1, len(clusters)):
                first = clusters[first_index]
                second = clusters[second_index]
                if not coplanar_candidates(
                    first,
                    second,
                    data,
                    args.merge_angle,
                    args.merge_plane_distance,
                    args.merge_gap,
                ):
                    continue
                combined_faces = sorted(
                    set(first["face_indices"]) | set(second["face_indices"])
                )
                combined = fit_plane(combined_faces, data, normal_reference)
                diagnostics = robust_merge_diagnostics(combined, data, args)
                if combined["rms"] > args.max_plane_rms or not diagnostics["accepted"]:
                    continue
                combined["view_visibility"] = None
                combined["merge_validation"] = diagnostics
                combined["merge_history"] = (
                    list(first.get("merge_history", []))
                    + list(second.get("merge_history", []))
                    + [diagnostics]
                )
                combined["source_plane_ids"] = sorted(
                    set(first["source_plane_ids"]) | set(second["source_plane_ids"])
                )
                clusters[first_index] = combined
                clusters.pop(second_index)
                changed = True
                break

    clusters.sort(key=lambda plane: plane["area"], reverse=True)
    return clusters


def polygon_parts(geometry: Any) -> list[Any]:
    if geometry is None or geometry.is_empty:
        return []
    if geometry.geom_type == "Polygon":
        return [geometry]
    if hasattr(geometry, "geoms"):
        parts = []
        for item in geometry.geoms:
            parts.extend(polygon_parts(item))
        return parts
    return []


def remove_small_holes(geometry: Any, minimum_area: float) -> Any:
    cleaned = []
    for polygon in polygon_parts(geometry):
        holes = [
            ring.coords[:]
            for ring in polygon.interiors
            if Polygon(ring).area >= minimum_area
        ]
        cleaned.append(Polygon(polygon.exterior.coords[:], holes))
    return unary_union(cleaned) if cleaned else geometry


def vectorize_plane(
    plane: dict[str, Any], data: dict[str, Any], args: argparse.Namespace
) -> dict[str, Any]:
    if Polygon is None or coverage_union_all is None:
        raise RuntimeError(
            "Shapely no está disponible. Instálelo en scripts_glb/.vendor para "
            "generar geometría simplificada."
        )
    first_axis, second_axis = plane_basis(plane["normal"])
    triangle_polygons = []
    for face_index in plane["face_indices"]:
        points = data["vertices"][data["triangles"][face_index]]
        offsets = points - plane["centroid"]
        local = np.column_stack((offsets @ first_axis, offsets @ second_axis))
        polygon = Polygon(local)
        if polygon.area > 1e-14:
            triangle_polygons.append(polygon)
    if not triangle_polygons:
        raise RuntimeError("No fue posible proyectar los triángulos del plano.")
    try:
        geometry = coverage_union_all(triangle_polygons)
        if geometry.is_empty or not geometry.is_valid:
            geometry = unary_union(triangle_polygons)
    except Exception:
        geometry = unary_union(triangle_polygons)
    geometry = make_valid(geometry)
    if args.merge_gap > 0:
        half_gap = args.merge_gap * 0.5
        geometry = geometry.buffer(half_gap, join_style=2).buffer(
            -half_gap, join_style=2
        )
    geometry = remove_small_holes(make_valid(geometry), args.min_hole_area)
    if args.simplified_boundary_tolerance > 0:
        geometry = geometry.simplify(
            args.simplified_boundary_tolerance, preserve_topology=True
        )
    geometry = remove_small_holes(make_valid(geometry), args.min_hole_area)
    polygons = [
        orient_polygon(polygon, sign=1.0)
        for polygon in polygon_parts(geometry)
        if polygon.area >= args.min_area
    ]
    if not polygons:
        raise RuntimeError("La simplificación eliminó toda la geometría del plano.")

    coordinates: list[list[float]] = []
    faces: list[tuple[int, int, int]] = []
    coordinate_index: dict[tuple[float, float], int] = {}
    boundaries = []

    def coordinate_id(x: float, y: float) -> int:
        key = (round(float(x), 10), round(float(y), 10))
        if key not in coordinate_index:
            point = plane["centroid"] + first_axis * key[0] + second_axis * key[1]
            coordinate_index[key] = len(coordinates)
            coordinates.append([float(value) for value in point])
        return coordinate_index[key]

    for polygon in polygons:
        exterior = list(polygon.exterior.coords)[:-1]
        holes = [list(ring.coords)[:-1] for ring in polygon.interiors]
        loops = [
            [Vector((float(x), float(y), 0.0)) for x, y in ring]
            for ring in [exterior, *holes]
            if len(ring) >= 3
        ]
        flattened_loop_coordinates = [
            (vertex.x, vertex.y) for loop in loops for vertex in loop
        ]
        for triangle in tessellate_polygon(loops):
            indices = [
                coordinate_id(*flattened_loop_coordinates[int(vertex_index)])
                for vertex_index in triangle
            ]
            first, second, third = (np.asarray(coordinates[index]) for index in indices)
            if float(np.dot(np.cross(second - first, third - first), plane["normal"])) < 0:
                indices[1], indices[2] = indices[2], indices[1]
            faces.append(tuple(indices))
        boundaries.append(
            {
                "outer": [
                    clean_vector(
                        blender_to_gltf_array(
                            plane["centroid"] + first_axis * x + second_axis * y
                        )
                    )
                    for x, y in exterior
                ],
                "holes": [
                    [
                        clean_vector(
                            blender_to_gltf_array(
                                plane["centroid"] + first_axis * x + second_axis * y
                            )
                        )
                        for x, y in hole
                    ]
                    for hole in holes
                ],
            }
        )
    if not faces:
        raise RuntimeError("Blender no pudo triangular el contorno simplificado.")
    plane["simplified_vertices"] = np.asarray(coordinates, dtype=np.float64)
    plane["simplified_faces"] = faces
    plane["simplified_boundaries"] = boundaries
    plane["simplified_area"] = float(sum(polygon.area for polygon in polygons))
    plane["_simplified_geometry"] = unary_union(polygons)
    plane["_first_axis"] = first_axis
    plane["_second_axis"] = second_axis
    return plane


def geometry_vertex_count(geometry: Any) -> int:
    count = 0
    for polygon in polygon_parts(geometry):
        count += max(0, len(polygon.exterior.coords) - 1)
        count += sum(max(0, len(ring.coords) - 1) for ring in polygon.interiors)
    return count


def regularize_plane(
    plane: dict[str, Any], args: argparse.Namespace
) -> dict[str, Any]:
    source_geometry = plane["_simplified_geometry"]
    candidate = source_geometry
    if args.boundary_regularization == "lines" and args.regularization_tolerance > 0:
        candidate = source_geometry.simplify(
            args.regularization_tolerance, preserve_topology=True
        )
        candidate = remove_small_holes(make_valid(candidate), args.min_hole_area)
        candidate_parts = [
            orient_polygon(polygon, sign=1.0)
            for polygon in polygon_parts(candidate)
            if polygon.area >= args.min_area
        ]
        candidate = unary_union(candidate_parts) if candidate_parts else source_geometry

    source_area = float(source_geometry.area)
    candidate_area = float(candidate.area)
    union_area = float(source_geometry.union(candidate).area)
    intersection_area = float(source_geometry.intersection(candidate).area)
    iou = intersection_area / union_area if union_area > 1e-14 else 1.0
    area_change = (
        abs(candidate_area - source_area) / source_area if source_area > 1e-14 else 0.0
    )
    source_vertex_count = geometry_vertex_count(source_geometry)
    candidate_vertex_count = geometry_vertex_count(candidate)
    accepted = (
        args.boundary_regularization == "lines"
        and candidate_vertex_count < source_vertex_count
        and iou >= args.regularization_min_iou
        and area_change <= args.regularization_max_area_change
    )
    geometry = candidate if accepted else source_geometry
    first_axis = plane["_first_axis"]
    second_axis = plane["_second_axis"]
    polygons = [
        orient_polygon(polygon, sign=1.0)
        for polygon in polygon_parts(geometry)
        if polygon.area >= args.min_area
    ]

    coordinates: list[list[float]] = []
    faces: list[tuple[int, int, int]] = []
    coordinate_index: dict[tuple[float, float], int] = {}
    boundaries = []

    def coordinate_id(x: float, y: float) -> int:
        key = (round(float(x), 10), round(float(y), 10))
        if key not in coordinate_index:
            point = plane["centroid"] + first_axis * key[0] + second_axis * key[1]
            coordinate_index[key] = len(coordinates)
            coordinates.append([float(value) for value in point])
        return coordinate_index[key]

    for polygon in polygons:
        exterior = list(polygon.exterior.coords)[:-1]
        holes = [list(ring.coords)[:-1] for ring in polygon.interiors]
        loops = [
            [Vector((float(x), float(y), 0.0)) for x, y in ring]
            for ring in [exterior, *holes]
            if len(ring) >= 3
        ]
        flattened = [(vertex.x, vertex.y) for loop in loops for vertex in loop]
        for triangle in tessellate_polygon(loops):
            indices = [coordinate_id(*flattened[int(vertex_index)]) for vertex_index in triangle]
            first, second, third = (np.asarray(coordinates[index]) for index in indices)
            if float(np.dot(np.cross(second - first, third - first), plane["normal"])) < 0:
                indices[1], indices[2] = indices[2], indices[1]
            faces.append(tuple(indices))
        boundaries.append(
            {
                "outer": [
                    clean_vector(
                        blender_to_gltf_array(
                            plane["centroid"] + first_axis * x + second_axis * y
                        )
                    )
                    for x, y in exterior
                ],
                "holes": [
                    [
                        clean_vector(
                            blender_to_gltf_array(
                                plane["centroid"] + first_axis * x + second_axis * y
                            )
                        )
                        for x, y in hole
                    ]
                    for hole in holes
                ],
            }
        )
    if not faces:
        raise RuntimeError("No fue posible triangular el contorno regularizado.")

    plane["regularized_vertices"] = np.asarray(coordinates, dtype=np.float64)
    plane["regularized_faces"] = faces
    plane["regularized_boundaries"] = boundaries
    plane["regularized_area"] = float(sum(polygon.area for polygon in polygons))
    plane["regularization"] = {
        "accepted": bool(accepted),
        "method": "line_simplification" if accepted else "preserved_simplified",
        "tolerance": float(args.regularization_tolerance),
        "source_vertex_count": int(source_vertex_count),
        "candidate_vertex_count": int(candidate_vertex_count),
        "output_vertex_count": len(coordinates),
        "iou": float(iou),
        "area_change_ratio": float(area_change),
        "hausdorff_distance": float(source_geometry.hausdorff_distance(candidate)),
    }
    return plane


def directional_visibility(
    plane: dict[str, Any],
    data: dict[str, Any],
    bvh: Any,
    diagonal: float,
    view_from_blender: Any,
    sample_count: int = 24,
) -> float:
    if float(np.dot(plane["normal"], view_from_blender)) <= 1e-6:
        return 0.0
    group = plane["face_indices"]
    if len(group) <= sample_count:
        samples = group
    else:
        positions = np.linspace(0, len(group) - 1, sample_count, dtype=np.int64)
        samples = [group[int(position)] for position in positions]
    group_set = set(group)
    visible = 0
    ray_direction = -view_from_blender
    for face_index in samples:
        center = data["centers"][face_index]
        origin_point = center + view_from_blender * diagonal * 1.5
        origin = Vector(tuple(float(value) for value in origin_point))
        hit = bvh.ray_cast(
            origin,
            Vector(tuple(float(value) for value in ray_direction)),
            diagonal * 3.0,
        )
        hit_index = hit[2]
        if hit_index is not None and int(hit_index) in group_set:
            visible += 1
    return visible / max(len(samples), 1)


def visible_faces_from_direction(
    candidate_indices: Any,
    data: dict[str, Any],
    bvh: Any,
    diagonal: float,
    view_from_blender: Any,
) -> Any:
    ray_direction = -view_from_blender
    visible = []
    for face_index in candidate_indices:
        face_index = int(face_index)
        if float(np.dot(data["normals"][face_index], view_from_blender)) <= 1e-6:
            continue
        center = data["centers"][face_index]
        origin_point = center + view_from_blender * diagonal * 1.5
        hit = bvh.ray_cast(
            Vector(tuple(float(value) for value in origin_point)),
            Vector(tuple(float(value) for value in ray_direction)),
            diagonal * 3.0,
        )
        if hit[2] is not None and int(hit[2]) == face_index:
            visible.append(face_index)
    return np.asarray(visible, dtype=np.int64)


def plane_basis(normal: Any) -> tuple[Any, Any]:
    reference = np.asarray((0.0, 0.0, 1.0))
    if abs(float(np.dot(normal, reference))) > 0.9:
        reference = np.asarray((1.0, 0.0, 0.0))
    first = np.cross(reference, normal)
    first /= np.linalg.norm(first)
    second = np.cross(normal, first)
    second /= np.linalg.norm(second)
    return first, second


def point_segment_distance(point: Any, start: Any, end: Any) -> float:
    segment = end - start
    denominator = float(np.dot(segment, segment))
    if denominator <= 1e-18:
        return float(np.linalg.norm(point - start))
    factor = max(0.0, min(1.0, float(np.dot(point - start, segment) / denominator)))
    return float(np.linalg.norm(point - (start + factor * segment)))


def simplify_open(points: Any, tolerance: float) -> list[int]:
    if len(points) <= 2 or tolerance <= 0:
        return list(range(len(points)))
    keep = {0, len(points) - 1}
    stack = [(0, len(points) - 1)]
    while stack:
        start, end = stack.pop()
        best_index = None
        best_distance = -1.0
        for index in range(start + 1, end):
            distance = point_segment_distance(points[index], points[start], points[end])
            if distance > best_distance:
                best_distance = distance
                best_index = index
        if best_index is not None and best_distance > tolerance:
            keep.add(best_index)
            stack.append((start, best_index))
            stack.append((best_index, end))
    return sorted(keep)


def simplify_closed(points: Any, tolerance: float) -> list[int]:
    selected = list(range(len(points)))
    if tolerance <= 0:
        return selected
    changed = True
    while changed and len(selected) > 3:
        changed = False
        next_selected = []
        for position, index in enumerate(selected):
            previous_index = selected[position - 1]
            next_index = selected[(position + 1) % len(selected)]
            if (
                len(selected) - len(next_selected) > 3
                and point_segment_distance(
                    points[index], points[previous_index], points[next_index]
                )
                <= tolerance
            ):
                changed = True
                continue
            next_selected.append(index)
        if len(next_selected) < 3:
            break
        selected = next_selected
    return selected


def trace_boundary_components(
    plane: dict[str, Any], data: dict[str, Any]
) -> list[tuple[list[int], bool]]:
    edge_counts: dict[tuple[int, int], int] = defaultdict(int)
    for face_index in plane["face_indices"]:
        a, b, c = (int(value) for value in data["triangles"][face_index])
        for edge in ((a, b), (b, c), (c, a)):
            edge_counts[tuple(sorted(edge))] += 1
    unused = {edge for edge, count in edge_counts.items() if count == 1}
    adjacency: dict[int, set[int]] = defaultdict(set)
    for first, second in unused:
        adjacency[first].add(second)
        adjacency[second].add(first)

    components = []
    while unused:
        degree_one = [vertex for vertex, neighbors in adjacency.items() if len(neighbors) == 1 and any(tuple(sorted((vertex, other))) in unused for other in neighbors)]
        start = degree_one[0] if degree_one else next(iter(unused))[0]
        path = [start]
        previous = None
        current = start
        closed = False
        while True:
            candidates = [
                other
                for other in adjacency[current]
                if tuple(sorted((current, other))) in unused
            ]
            if not candidates:
                break
            if previous is not None and len(candidates) > 1:
                previous_direction = data["vertices"][current] - data["vertices"][previous]
                previous_length = np.linalg.norm(previous_direction)
                if previous_length > 0:
                    previous_direction /= previous_length
                candidates.sort(
                    key=lambda candidate: float(
                        np.dot(
                            previous_direction,
                            (data["vertices"][candidate] - data["vertices"][current])
                            / max(
                                np.linalg.norm(
                                    data["vertices"][candidate] - data["vertices"][current]
                                ),
                                1e-12,
                            ),
                        )
                    ),
                    reverse=True,
                )
            following = candidates[0]
            unused.remove(tuple(sorted((current, following))))
            previous, current = current, following
            if current == start:
                closed = True
                break
            path.append(current)
        if len(path) >= (3 if closed else 2):
            components.append((path, closed))
    return components


def build_boundaries(
    plane: dict[str, Any], data: dict[str, Any], tolerance: float
) -> dict[str, Any]:
    first_axis, second_axis = plane_basis(plane["normal"])
    components_json = []
    for vertex_ids, closed in trace_boundary_components(plane, data):
        raw = data["vertices"][vertex_ids]
        projected = raw - np.outer((raw - plane["centroid"]) @ plane["normal"], plane["normal"])
        local = np.column_stack(
            (
                (projected - plane["centroid"]) @ first_axis,
                (projected - plane["centroid"]) @ second_axis,
            )
        )
        selected = (
            simplify_closed(local, tolerance)
            if closed
            else simplify_open(local, tolerance)
        )
        local_selected = local[selected]
        signed_area = 0.0
        if closed and len(selected) >= 3:
            signed_area = 0.5 * float(
                np.sum(
                    local_selected[:, 0] * np.roll(local_selected[:, 1], -1)
                    - np.roll(local_selected[:, 0], -1) * local_selected[:, 1]
                )
            )
        components_json.append(
            {
                "closed": closed,
                "signed_area_2d": clean_number(signed_area),
                "raw_vertex_count": len(vertex_ids),
                "vertices": [
                    clean_vector(point)
                    for point in blender_to_gltf_array(projected[selected])
                ],
            }
        )
    components_json.sort(key=lambda component: abs(component["signed_area_2d"]), reverse=True)
    closed_components = [component for component in components_json if component["closed"]]
    return {
        "outer": closed_components[0]["vertices"] if closed_components else None,
        "holes": [component["vertices"] for component in closed_components[1:]],
        "components": components_json,
    }


def plane_to_json(
    plane: dict[str, Any],
    data: dict[str, Any],
    identifier: str,
    args: argparse.Namespace,
    roof_up_name: str,
    roof_up_gltf: Any,
) -> dict[str, Any]:
    normal_gltf = blender_to_gltf_array(plane["normal"])
    centroid_gltf = blender_to_gltf_array(plane["centroid"])
    normal_gltf /= np.linalg.norm(normal_gltf)
    plane_d = -float(np.dot(normal_gltf, centroid_gltf))
    roof_up_gltf = np.asarray(roof_up_gltf, dtype=np.float64)
    roof_up_gltf /= np.linalg.norm(roof_up_gltf)
    pitch = math.degrees(
        math.acos(max(-1.0, min(1.0, float(np.dot(normal_gltf, roof_up_gltf)))))
    )
    horizontal = math.hypot(float(normal_gltf[0]), float(normal_gltf[2]))
    azimuth = None
    if roof_up_name in ("positive_y", "negative_y") and horizontal > 1e-9:
        azimuth = math.degrees(math.atan2(float(normal_gltf[0]), float(normal_gltf[2]))) % 360.0

    vertex_ids = [int(value) for value in plane["vertex_ids"]]
    local_index = {vertex_id: index for index, vertex_id in enumerate(vertex_ids)}
    mesh_vertices = blender_to_gltf_array(data["vertices"][vertex_ids])
    mesh_triangles = [
        [local_index[int(value)] for value in data["triangles"][face_index]]
        for face_index in plane["face_indices"]
    ]
    planarity_score = max(0.0, 1.0 - plane["rms"] / args.max_plane_rms)
    area_score = min(1.0, plane["area"] / max(args.min_area * 5.0, 1e-12))
    if plane["view_visibility"] is None:
        confidence = 0.75 * planarity_score + 0.25 * area_score
    else:
        confidence = 0.55 * planarity_score + 0.30 * plane["view_visibility"] + 0.15 * area_score
    return {
        "id": identifier,
        "classification": "exterior_roof_plane",
        "confidence": clean_number(confidence),
        "area": clean_number(plane["area"]),
        "triangle_count": len(plane["face_indices"]),
        "vertex_count": len(vertex_ids),
        "centroid": clean_vector(centroid_gltf),
        "normal": clean_vector(normal_gltf),
        "roof_up_reference": clean_vector(roof_up_gltf),
        "plane_equation": {
            "a": clean_number(normal_gltf[0]),
            "b": clean_number(normal_gltf[1]),
            "c": clean_number(normal_gltf[2]),
            "d": clean_number(plane_d),
            "form": "a*x + b*y + c*z + d = 0",
        },
        "pitch_degrees": clean_number(pitch),
        "pitch_reference": roof_up_name,
        "downslope_azimuth_degrees": None if azimuth is None else clean_number(azimuth),
        "downslope_azimuth_convention": "0 = +Z, 90 = +X, clockwise in XZ plane",
        "planarity": {
            "rms_error": clean_number(plane["rms"]),
            "max_error": clean_number(plane["max_residual"]),
            "p95_error": clean_number(plane["residual_p95"]),
            "p99_error": clean_number(plane["residual_p99"]),
        },
        "view_visibility": (
            None
            if plane["view_visibility"] is None
            else clean_number(plane["view_visibility"])
        ),
        "boundary": build_boundaries(plane, data, args.boundary_simplify),
        "mesh": {
            "vertices": [clean_vector(point) for point in mesh_vertices],
            "triangles": mesh_triangles,
        },
    }


def simplified_plane_to_json(
    plane: dict[str, Any], roof_up_name: str, roof_up_gltf: Any
) -> dict[str, Any]:
    normal_gltf = blender_to_gltf_array(plane["normal"])
    normal_gltf /= np.linalg.norm(normal_gltf)
    centroid_gltf = blender_to_gltf_array(plane["centroid"])
    roof_up_gltf = np.asarray(roof_up_gltf, dtype=np.float64)
    roof_up_gltf /= np.linalg.norm(roof_up_gltf)
    pitch = math.degrees(
        math.acos(max(-1.0, min(1.0, float(np.dot(normal_gltf, roof_up_gltf)))))
    )
    plane_d = -float(np.dot(normal_gltf, centroid_gltf))
    return {
        "id": plane["id"],
        "classification": "simplified_exterior_roof_plane",
        "source_plane_ids": list(plane["source_plane_ids"]),
        "source_plane_count": len(plane["source_plane_ids"]),
        "area": clean_number(plane["simplified_area"]),
        "source_triangle_count": len(plane["face_indices"]),
        "vertex_count": len(plane["simplified_vertices"]),
        "triangle_count": len(plane["simplified_faces"]),
        "centroid": clean_vector(centroid_gltf),
        "normal": clean_vector(normal_gltf),
        "roof_up_reference": clean_vector(roof_up_gltf),
        "pitch_degrees": clean_number(pitch),
        "pitch_reference": roof_up_name,
        "plane_equation": {
            "a": clean_number(normal_gltf[0]),
            "b": clean_number(normal_gltf[1]),
            "c": clean_number(normal_gltf[2]),
            "d": clean_number(plane_d),
            "form": "a*x + b*y + c*z + d = 0",
        },
        "planarity": {
            "rms_error": clean_number(plane["rms"]),
            "max_error": clean_number(plane["max_residual"]),
            "p95_error": clean_number(plane["residual_p95"]),
            "p99_error": clean_number(plane["residual_p99"]),
        },
        "merge_validation": (
            None
            if "merge_validation" not in plane
            else {
                key: clean_number(value) if isinstance(value, float) else value
                for key, value in plane["merge_validation"].items()
            }
        ),
        "boundary_components": plane["simplified_boundaries"],
        "mesh": {
            "vertices": [
                clean_vector(point)
                for point in blender_to_gltf_array(plane["simplified_vertices"])
            ],
            "triangles": [list(face) for face in plane["simplified_faces"]],
        },
    }


def regularized_plane_to_json(
    plane: dict[str, Any], roof_up_name: str, roof_up_gltf: Any
) -> dict[str, Any]:
    document = simplified_plane_to_json(plane, roof_up_name, roof_up_gltf)
    document["id"] = plane["id"].replace(
        "simplified_roof_plane_", "regularized_roof_plane_"
    )
    document["classification"] = "regularized_exterior_roof_plane"
    document["source_simplified_plane_id"] = plane["id"]
    document["area"] = clean_number(plane["regularized_area"])
    document["vertex_count"] = len(plane["regularized_vertices"])
    document["triangle_count"] = len(plane["regularized_faces"])
    document["boundary_components"] = plane["regularized_boundaries"]
    document["regularization"] = {
        key: clean_number(value) if isinstance(value, float) else value
        for key, value in plane["regularization"].items()
    }
    document["mesh"] = {
        "vertices": [
            clean_vector(point)
            for point in blender_to_gltf_array(plane["regularized_vertices"])
        ],
        "triangles": [list(face) for face in plane["regularized_faces"]],
    }
    return document


def make_material(identifier: str, index: int) -> Any:
    hue = (index * 0.61803398875) % 1.0
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.72, 1.0)
    material = bpy.data.materials.new(identifier)
    material.diffuse_color = (red, green, blue, 1.0)
    material.use_backface_culling = False
    material.use_nodes = True
    principled = material.node_tree.nodes.get("Principled BSDF")
    principled.inputs["Base Color"].default_value = (red, green, blue, 1.0)
    principled.inputs["Roughness"].default_value = 0.55
    return material


def add_plane_objects(planes: list[dict[str, Any]], data: dict[str, Any], offset: float) -> None:
    for index, plane in enumerate(planes):
        identifier = plane["id"]
        vertex_ids = [int(value) for value in plane["vertex_ids"]]
        local_index = {vertex_id: position for position, vertex_id in enumerate(vertex_ids)}
        coordinates = data["vertices"][vertex_ids] + plane["normal"] * offset
        faces = [
            tuple(local_index[int(value)] for value in data["triangles"][face_index])
            for face_index in plane["face_indices"]
        ]
        mesh = bpy.data.meshes.new(identifier)
        mesh.from_pydata(coordinates.tolist(), [], faces)
        mesh.update()
        obj = bpy.data.objects.new(identifier, mesh)
        bpy.context.scene.collection.objects.link(obj)
        color_index = int(plane.get("color_index", index))
        obj.data.materials.append(make_material(identifier, color_index))
        obj["roof_plane_id"] = identifier
        obj["area"] = float(plane["area"])
        obj["pitch_degrees"] = float(plane["pitch_degrees"])
        obj["confidence"] = float(plane["confidence"])


def add_simplified_plane_objects(
    planes: list[dict[str, Any]], offset: float
) -> None:
    for index, plane in enumerate(planes):
        identifier = plane["id"]
        coordinates = plane["simplified_vertices"] + plane["normal"] * offset
        mesh = bpy.data.meshes.new(identifier)
        mesh.from_pydata(coordinates.tolist(), [], plane["simplified_faces"])
        mesh.update()
        obj = bpy.data.objects.new(identifier, mesh)
        bpy.context.scene.collection.objects.link(obj)
        color_index = int(plane.get("color_index", index))
        obj.data.materials.append(make_material(identifier, color_index))
        obj["roof_plane_id"] = identifier
        obj["source_plane_ids"] = list(plane["source_plane_ids"])
        obj["area"] = float(plane["simplified_area"])
        obj["pitch_degrees"] = float(plane["pitch_degrees"])
        obj["confidence"] = float(plane["confidence"])


def add_regularized_plane_objects(
    planes: list[dict[str, Any]], offset: float
) -> None:
    for index, plane in enumerate(planes):
        identifier = plane["id"].replace(
            "simplified_roof_plane_", "regularized_roof_plane_"
        )
        coordinates = plane["regularized_vertices"] + plane["normal"] * offset
        mesh = bpy.data.meshes.new(identifier)
        mesh.from_pydata(coordinates.tolist(), [], plane["regularized_faces"])
        mesh.update()
        obj = bpy.data.objects.new(identifier, mesh)
        bpy.context.scene.collection.objects.link(obj)
        color_index = int(plane.get("color_index", index))
        obj.data.materials.append(make_material(identifier, color_index))
        obj["roof_plane_id"] = identifier
        obj["source_simplified_plane_id"] = plane["id"]
        obj["source_plane_ids"] = list(plane["source_plane_ids"])
        obj["area"] = float(plane["regularized_area"])
        obj["pitch_degrees"] = float(plane["pitch_degrees"])
        obj["confidence"] = float(plane["confidence"])
        obj["regularization_accepted"] = bool(
            plane["regularization"]["accepted"]
        )


def add_rotated_regularized_plane_objects(
    rotation_name: str, rotated_planes: list[dict[str, Any]]
) -> None:
    for plane in rotated_planes:
        identifier = plane["id"]
        mesh = bpy.data.meshes.new(identifier)
        mesh.from_pydata(plane["vertices"].tolist(), [], plane["faces"])
        mesh.update()
        obj = bpy.data.objects.new(identifier, mesh)
        bpy.context.scene.collection.objects.link(obj)
        obj.data.materials.append(
            make_material(identifier, int(plane["color_index"]))
        )
        obj["roof_plane_id"] = identifier
        obj["source_simplified_plane_id"] = plane[
            "source_simplified_plane_id"
        ]
        obj["source_plane_ids"] = plane["source_plane_ids"]
        obj["area"] = plane["area"]
        obj["pitch_degrees"] = plane["pitch_degrees"]
        obj["confidence"] = plane["confidence"]
        obj["regularization_accepted"] = plane["regularization_accepted"]
        obj["rotation_id"] = rotation_name


def add_validation_camera(
    center: Any, diagonal: float, view_from_blender: Any, view_from_name: str
) -> None:
    bpy.ops.object.camera_add()
    camera = bpy.context.object
    camera.name = f"view_from_{view_from_name}"
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = diagonal * 0.9
    camera.location = Vector(
        tuple(float(value) for value in center + view_from_blender * diagonal * 1.5)
    )
    target = Vector(tuple(float(value) for value in center))
    camera.rotation_euler = (target - camera.location).to_track_quat("-Z", "Y").to_euler()
    camera["view_from"] = view_from_name
    bpy.context.scene.camera = camera


def add_comparison_camera(center: Any, diagonal: float) -> None:
    """Add the same oblique camera to every roof-up comparison asset."""
    camera_direction_gltf = np.asarray((-1.0, 0.7, -1.0), dtype=np.float64)
    camera_direction_gltf /= np.linalg.norm(camera_direction_gltf)
    camera_direction = gltf_to_blender_array(camera_direction_gltf)
    bpy.ops.object.camera_add()
    camera = bpy.context.object
    camera.name = "roof_up_comparison_isometric"
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = diagonal * 0.9
    camera.location = Vector(
        tuple(float(value) for value in center + camera_direction * diagonal * 1.5)
    )
    target = Vector(tuple(float(value) for value in center))
    camera.rotation_euler = (target - camera.location).to_track_quat("-Z", "Y").to_euler()
    camera["role"] = "visual_comparison_only"
    camera["direction_gltf"] = clean_vector(camera_direction_gltf)
    bpy.context.scene.camera = camera


def export_scene(path: Path) -> None:
    result = bpy.ops.export_scene.gltf(
        filepath=str(path),
        export_format="GLB",
        export_yup=True,
        export_apply=False,
        export_animations=False,
        export_cameras=True,
        export_lights=False,
        export_extras=True,
    )
    if "FINISHED" not in result or not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"No se pudo exportar {path}")


def export_validation_glbs(
    input_path: Path,
    output_directory: Path,
    planes: list[dict[str, Any]],
    data: dict[str, Any],
    diagonal: float,
    view_from_blender: Any,
    view_from_name: str,
) -> tuple[Path, Path]:
    center = (data["vertices"].min(axis=0) + data["vertices"].max(axis=0)) * 0.5
    bpy.ops.wm.read_factory_settings(use_empty=True)
    add_plane_objects(planes, data, offset=0.0)
    add_validation_camera(center, diagonal, view_from_blender, view_from_name)
    planes_only = output_directory / "roof_planes_only.glb"
    export_scene(planes_only)

    import_glb(input_path)
    add_plane_objects(planes, data, offset=diagonal * 0.0015)
    add_validation_camera(center, diagonal, view_from_blender, view_from_name)
    overlay = output_directory / "roof_planes_overlay.glb"
    export_scene(overlay)
    return planes_only, overlay


def export_exact_and_simplified_assets(
    input_path: Path,
    output_directory: Path,
    exact_planes: list[dict[str, Any]],
    simplified_planes: list[dict[str, Any]],
    data: dict[str, Any],
    diagonal: float,
    geometry_output: str,
) -> list[tuple[Path, str]]:
    center = (data["vertices"].min(axis=0) + data["vertices"].max(axis=0)) * 0.5
    preview_directory = output_directory / "previews"
    preview_directory.mkdir()
    assets: list[tuple[Path, str]] = []
    previews = []
    preview_by_kind: dict[str, Path] = {}

    if geometry_output in ("exact", "both"):
        bpy.ops.wm.read_factory_settings(use_empty=True)
        add_plane_objects(exact_planes, data, offset=0.0)
        add_comparison_camera(center, diagonal)
        exact_only = output_directory / "roof_planes_exact_only.glb"
        export_scene(exact_only)
        assets.append((exact_only, "exact_planes_only_glb"))

        import_glb(input_path)
        add_plane_objects(exact_planes, data, offset=diagonal * 0.0015)
        add_comparison_camera(center, diagonal)
        exact_overlay = output_directory / "roof_planes_exact_overlay.glb"
        export_scene(exact_overlay)
        assets.append((exact_overlay, "exact_overlay_glb"))
        exact_preview = preview_directory / "exact.png"
        render_preview(exact_preview, center, diagonal)
        previews.append(exact_preview)
        preview_by_kind["exact"] = exact_preview
        assets.append((exact_preview, "exact_preview"))

    if geometry_output in ("simplified", "both"):
        bpy.ops.wm.read_factory_settings(use_empty=True)
        add_simplified_plane_objects(simplified_planes, offset=0.0)
        add_comparison_camera(center, diagonal)
        simplified_only = output_directory / "roof_planes_simplified_only.glb"
        export_scene(simplified_only)
        assets.append((simplified_only, "simplified_planes_only_glb"))

        import_glb(input_path)
        add_simplified_plane_objects(
            simplified_planes, offset=diagonal * 0.0015
        )
        add_comparison_camera(center, diagonal)
        simplified_overlay = output_directory / "roof_planes_simplified_overlay.glb"
        export_scene(simplified_overlay)
        assets.append((simplified_overlay, "simplified_overlay_glb"))
        simplified_preview = preview_directory / "simplified.png"
        render_preview(simplified_preview, center, diagonal)
        previews.append(simplified_preview)
        preview_by_kind["simplified"] = simplified_preview
        assets.append((simplified_preview, "simplified_preview"))

        bpy.ops.wm.read_factory_settings(use_empty=True)
        add_regularized_plane_objects(simplified_planes, offset=0.0)
        add_comparison_camera(center, diagonal)
        regularized_only = output_directory / "roof_planes_regularized_only.glb"
        export_scene(regularized_only)
        assets.append((regularized_only, "regularized_planes_only_glb"))

        import_glb(input_path)
        add_regularized_plane_objects(
            simplified_planes, offset=diagonal * 0.0015
        )
        add_comparison_camera(center, diagonal)
        regularized_overlay = output_directory / "roof_planes_regularized_overlay.glb"
        export_scene(regularized_overlay)
        assets.append((regularized_overlay, "regularized_overlay_glb"))
        regularized_preview = preview_directory / "regularized.png"
        render_preview(regularized_preview, center, diagonal)
        previews.append(regularized_preview)
        preview_by_kind["regularized"] = regularized_preview
        assets.append((regularized_preview, "regularized_preview"))

    if "exact" in preview_by_kind and "simplified" in preview_by_kind:
        comparison = preview_directory / "comparison_exact_vs_simplified.png"
        create_horizontal_comparison(
            [preview_by_kind["exact"], preview_by_kind["simplified"]], comparison
        )
        assets.append((comparison, "exact_vs_simplified_preview"))
    if "simplified" in preview_by_kind and "regularized" in preview_by_kind:
        comparison = preview_directory / "comparison_simplified_vs_regularized.png"
        create_horizontal_comparison(
            [preview_by_kind["simplified"], preview_by_kind["regularized"]],
            comparison,
        )
        assets.append((comparison, "simplified_vs_regularized_preview"))
    if len(previews) == 3:
        comparison = preview_directory / "comparison_all_geometries.png"
        create_horizontal_comparison(previews, comparison)
        assets.append((comparison, "all_geometries_preview"))
    return assets


def export_regularized_rotation_assets(
    output_directory: Path,
    pivot_gltf: Any,
    rotation_results: dict[str, dict[str, Any]],
    diagonal: float,
) -> list[tuple[Path, str]]:
    if not rotation_results:
        return []
    rotation_directory = output_directory / "rotations"
    preview_directory = rotation_directory / "previews"
    rotation_directory.mkdir()
    preview_directory.mkdir()
    center_blender = gltf_to_blender_array(pivot_gltf)
    assets: list[tuple[Path, str]] = []
    previews = []

    for rotation_name, result in rotation_results.items():
        bpy.ops.wm.read_factory_settings(use_empty=True)
        add_rotated_regularized_plane_objects(rotation_name, result["planes"])
        add_comparison_camera(center_blender, diagonal)
        glb_path = rotation_directory / (
            f"roof_planes_regularized_rotated_{rotation_name}.glb"
        )
        export_scene(glb_path)
        assets.append((glb_path, f"regularized_rotation_{rotation_name}_glb"))

        preview_path = preview_directory / f"{rotation_name}.png"
        render_preview(preview_path, center_blender, diagonal)
        previews.append(preview_path)
        assets.append(
            (preview_path, f"regularized_rotation_{rotation_name}_preview")
        )

    comparison_path = preview_directory / "comparison.png"
    if len(previews) == 6:
        create_contact_sheet(previews, comparison_path)
    else:
        create_horizontal_comparison(previews, comparison_path)
    assets.append((comparison_path, "regularized_rotations_comparison"))
    return assets


def add_preview_lights(center: Any, diagonal: float) -> None:
    for offset, energy, size in (
        ((1.2, -1.4, 1.8), 1100, 5.0),
        ((-1.5, -0.5, 0.8), 700, 4.0),
        ((0.2, 1.5, 1.3), 850, 4.0),
    ):
        bpy.ops.object.light_add(type="AREA")
        light = bpy.context.object
        direction = Vector(offset).normalized()
        light.location = Vector(tuple(center)) + direction * diagonal * 1.4
        light.data.energy = energy
        light.data.size = size
        target = Vector(tuple(center))
        light.rotation_euler = (target - light.location).to_track_quat("-Z", "Y").to_euler()


def render_preview(path: Path, center: Any, diagonal: float) -> None:
    add_preview_lights(center, diagonal)
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.eevee.taa_render_samples = 8
    scene.render.resolution_x = 600
    scene.render.resolution_y = 600
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    if scene.world is None:
        scene.world = bpy.data.worlds.new("Preview World")
    scene.world.color = (0.035, 0.035, 0.035)
    scene.view_settings.look = "AgX - Medium High Contrast"
    scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)


def create_contact_sheet(previews: list[Path], output_path: Path) -> None:
    loaded = [bpy.data.images.load(str(path), check_existing=False) for path in previews]
    width, height = loaded[0].size
    canvas = np.zeros((height * 2, width * 3, 4), dtype=np.float32)
    canvas[..., :3] = 0.025
    canvas[..., 3] = 1.0
    for index, image in enumerate(loaded):
        pixels = np.asarray(image.pixels[:], dtype=np.float32).reshape(
            image.size[1], image.size[0], 4
        )
        row, column = divmod(index, 3)
        row = 1 - row  # Blender's pixel buffer starts at the bottom row.
        canvas[
            row * height : (row + 1) * height,
            column * width : (column + 1) * width,
        ] = pixels
    sheet = bpy.data.images.new(
        "multiview_comparison", width=width * 3, height=height * 2, alpha=True
    )
    sheet.pixels.foreach_set(canvas.reshape(-1))
    sheet.filepath_raw = str(output_path)
    sheet.file_format = "PNG"
    sheet.save()
    for image in loaded:
        bpy.data.images.remove(image)


def create_horizontal_comparison(previews: list[Path], output_path: Path) -> None:
    loaded = [bpy.data.images.load(str(path), check_existing=False) for path in previews]
    width, height = loaded[0].size
    canvas = np.zeros((height, width * len(loaded), 4), dtype=np.float32)
    canvas[..., :3] = 0.025
    canvas[..., 3] = 1.0
    for index, loaded_image in enumerate(loaded):
        pixels = np.asarray(loaded_image.pixels[:], dtype=np.float32).reshape(
            loaded_image.size[1], loaded_image.size[0], 4
        )
        canvas[:, index * width : (index + 1) * width] = pixels
    comparison = bpy.data.images.new(
        "exact_vs_simplified", width=width * len(loaded), height=height, alpha=True
    )
    comparison.pixels.foreach_set(canvas.reshape(-1))
    comparison.filepath_raw = str(output_path)
    comparison.file_format = "PNG"
    comparison.save()
    for loaded_image in loaded:
        bpy.data.images.remove(loaded_image)


def export_multiview_assets(
    input_path: Path,
    output_directory: Path,
    planes: list[dict[str, Any]],
    data: dict[str, Any],
    diagonal: float,
    views: tuple[str, ...],
    minimum_visibility: float,
) -> list[tuple[Path, str]]:
    center = (data["vertices"].min(axis=0) + data["vertices"].max(axis=0)) * 0.5
    assets: list[tuple[Path, str]] = []

    bpy.ops.wm.read_factory_settings(use_empty=True)
    add_plane_objects(planes, data, offset=0.0)
    for view_name in views:
        view_vector = gltf_to_blender_array(
            np.asarray(VIEW_FROM_GLTF[view_name], dtype=np.float64)
        )
        add_validation_camera(center, diagonal, view_vector, view_name)
    all_only = output_directory / "roof_planes_all_only.glb"
    export_scene(all_only)
    assets.append((all_only, "all_planes_glb"))

    preview_directory = output_directory / "previews"
    preview_directory.mkdir()
    preview_paths = []
    for view_name in views:
        selected = [
            plane
            for plane in planes
            if plane["visibility_by_view"].get(view_name, 0.0) >= minimum_visibility
        ]
        view_vector = gltf_to_blender_array(
            np.asarray(VIEW_FROM_GLTF[view_name], dtype=np.float64)
        )
        import_glb(input_path)
        add_plane_objects(selected, data, offset=diagonal * 0.0015)
        add_validation_camera(center, diagonal, view_vector, view_name)
        overlay = output_directory / f"roof_planes_from_{view_name}_overlay.glb"
        export_scene(overlay)
        assets.append((overlay, f"overlay_{view_name}"))

        preview = preview_directory / f"{view_name}.png"
        render_preview(preview, center, diagonal)
        preview_paths.append(preview)
        assets.append((preview, f"preview_{view_name}"))

    comparison = preview_directory / "comparison.png"
    create_contact_sheet(preview_paths, comparison)
    assets.append((comparison, "preview_comparison"))
    return assets


def export_roof_up_comparison_assets(
    input_path: Path,
    output_directory: Path,
    planes_by_roof_up: dict[str, list[dict[str, Any]]],
    data: dict[str, Any],
    diagonal: float,
    roof_ups: tuple[str, ...],
) -> list[tuple[Path, str]]:
    center = (data["vertices"].min(axis=0) + data["vertices"].max(axis=0)) * 0.5
    assets: list[tuple[Path, str]] = []
    preview_directory = output_directory / "previews"
    preview_directory.mkdir()
    preview_paths = []

    for roof_up in roof_ups:
        selected = planes_by_roof_up[roof_up]
        bpy.ops.wm.read_factory_settings(use_empty=True)
        add_plane_objects(selected, data, offset=0.0)
        add_comparison_camera(center, diagonal)
        planes_only = output_directory / f"roof_planes_roof_up_{roof_up}_only.glb"
        export_scene(planes_only)
        assets.append((planes_only, f"planes_only_roof_up_{roof_up}"))

        import_glb(input_path)
        add_plane_objects(selected, data, offset=diagonal * 0.0015)
        add_comparison_camera(center, diagonal)
        overlay = output_directory / f"roof_planes_roof_up_{roof_up}_overlay.glb"
        export_scene(overlay)
        assets.append((overlay, f"overlay_roof_up_{roof_up}"))

        preview = preview_directory / f"roof_up_{roof_up}.png"
        render_preview(preview, center, diagonal)
        preview_paths.append(preview)
        assets.append((preview, f"preview_roof_up_{roof_up}"))

    comparison = preview_directory / "comparison.png"
    create_contact_sheet(preview_paths, comparison)
    assets.append((comparison, "preview_comparison"))
    return assets


def create_run_directory(output_root: Path, timezone_name: str) -> tuple[Path, datetime]:
    timezone = ZoneInfo(timezone_name)
    created_at = datetime.now(timezone)
    output_root.mkdir(parents=True, exist_ok=True)
    directory = output_root / created_at.strftime("%Y%m%d_%H%M%S")
    if directory.exists():
        directory = output_root / created_at.strftime("%Y%m%d_%H%M%S_%f")
    directory.mkdir()
    return directory, created_at


def report_markdown(document: dict[str, Any]) -> str:
    visibility = document["view"]["visibility"]
    visible_part = document["view"]["visible_part"]
    simplified_summary = document["summary"].get("simplified_geometry", {})
    vertex_reduction = simplified_summary.get("vertex_reduction_percent")
    triangle_reduction = simplified_summary.get("triangle_reduction_percent")
    regularized_summary = document["summary"].get("regularized_geometry", {})
    lines = [
        "# Detección de planos de techo",
        "",
        f"- Entrada: `{document['source']['filename']}`",
        f"- Referencia de techo: `{document['parameters']['roof_up']}`",
        f"- Filtro de visibilidad: `{visibility}`",
        f"- Cámara de validación: `{document['view']['view_from']}`",
        f"- Porción conservada: `{visible_part if visible_part else 'no aplica'}`",
        f"- Planos detectados: `{document['summary']['roof_plane_count']}`",
        f"- Área detectada: `{document['summary']['detected_area']:.6g}`",
        f"- Cobertura de triángulos: `{document['summary']['triangle_coverage_percent']:.2f}%`",
        f"- Planos simplificados: `{simplified_summary.get('plane_count', 0)}`",
        f"- Planos con contorno regularizado: `{regularized_summary.get('planes_regularized', 0)}`",
        f"- Reducción de vértices: `{f'{vertex_reduction:.2f}%' if vertex_reduction is not None else 'no aplica'}`",
        f"- Reducción de triángulos: `{f'{triangle_reduction:.2f}%' if triangle_reduction is not None else 'no aplica'}`",
        "- Coordenadas: `glTF 2.0, Y-up`",
        "",
        "| Plano | Área | Pendiente | Azimut | Confianza | Visibilidad | RMS |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for plane in document["roof_planes"]:
        azimuth = "—" if plane["downslope_azimuth_degrees"] is None else f"{plane['downslope_azimuth_degrees']:.2f}°"
        view_visibility = (
            "—"
            if plane["view_visibility"] is None
            else f"{plane['view_visibility']:.3f}"
        )
        lines.append(
            f"| `{plane['id']}` | {plane['area']:.6g} | {plane['pitch_degrees']:.2f}° | "
            f"{azimuth} | {plane['confidence']:.3f} | {view_visibility} | "
            f"{plane['planarity']['rms_error']:.6g} |"
        )
    lines.extend(
        [
            "",
            "## Planos fusionados y vectorizados",
            "",
            "| Plano simplificado | Planos fuente | Área | Vértices | Triángulos | RMS |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for plane in document.get("simplified_roof_planes", []):
        lines.append(
            f"| `{plane['id']}` | {plane['source_plane_count']} | {plane['area']:.6g} | "
            f"{plane['vertex_count']} | {plane['triangle_count']} | "
            f"{plane['planarity']['rms_error']:.6g} |"
        )
    lines.extend(
        [
            "",
            "## Contornos regularizados",
            "",
            "| Plano | Aceptado | Vértices antes | Vértices después | IoU | Cambio de área |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for plane in document.get("regularized_roof_planes", []):
        validation = plane["regularization"]
        lines.append(
            f"| `{plane['id']}` | {'sí' if validation['accepted'] else 'no'} | "
            f"{validation['source_vertex_count']} | {plane['vertex_count']} | "
            f"{validation['iou']:.4f} | {100.0 * validation['area_change_ratio']:.2f}% |"
        )
    lines.extend(
        [
            "",
            "La clasificación se basa en orientación respecto a la referencia de techo, "
            "conectividad y ajuste plano. La visibilidad solo interviene cuando se usa el "
            "modo `directional`. Compare `roof_planes_exact_overlay.glb` con "
            "`roof_planes_simplified_overlay.glb` y "
            "`previews/comparison_exact_vs_simplified.png`.",
            "",
        ]
    )
    return "\n".join(lines)


def multiview_report_markdown(document: dict[str, Any]) -> str:
    views = document["view"]["views"]
    lines = [
        "# Calibración multivista de planos de techo",
        "",
        f"- Entrada: `{document['source']['filename']}`",
        f"- Planos globales: `{document['summary']['roof_plane_count']}`",
        f"- Área global: `{document['summary']['detected_area']:.6g}`",
        "- Modo: `full-plane`",
        "- Coordenadas: `glTF 2.0, Y-up`",
        "",
        "Orden de `previews/comparison.png`: primera fila `positive_x`, "
        "`negative_x`, `positive_y`; segunda fila `negative_y`, `positive_z`, "
        "`negative_z`.",
        "",
        "## Resultados por vista",
        "",
        "| Vista | Rayos | Planos | Área | Archivo |",
        "|---|---|---:|---:|---|",
    ]
    for view_name in views:
        summary = document["summary"]["by_view"][view_name]
        direction = document["view"]["ray_directions"][view_name]
        lines.append(
            f"| `{view_name}` | `{direction}` | {summary['plane_count']} | "
            f"{summary['area']:.6g} | `roof_planes_from_{view_name}_overlay.glb` |"
        )
    lines.extend(
        [
            "",
            "## Visibilidad por plano",
            "",
            "| Plano | Área | Pendiente | "
            + " | ".join(f"{view}" for view in views)
            + " |",
            "|---|---:|---:|" + "---:|" * len(views),
        ]
    )
    for plane in document["roof_planes"]:
        values = " | ".join(
            f"{plane['visibility_by_view'][view]:.3f}" for view in views
        )
        lines.append(
            f"| `{plane['id']}` | {plane['area']:.6g} | "
            f"{plane['pitch_degrees']:.2f}° | {values} |"
        )
    lines.extend(
        [
            "",
            "Los IDs y colores de los planos son estables entre los seis GLB. Un plano "
            "se incluye en una vista cuando su visibilidad alcanza el umbral configurado.",
            "",
        ]
    )
    return "\n".join(lines)


def roof_up_comparison_report_markdown(document: dict[str, Any]) -> str:
    lines = [
        "# Comparación de referencias roof-up",
        "",
        f"- Entrada: `{document['source']['filename']}`",
        f"- Dirección de normales: `{document['parameters']['normal_direction']}`",
        "- Filtro de visibilidad: `none`",
        "- Cámara: isométrica fija, usada únicamente para visualización",
        "- Coordenadas: `glTF 2.0, Y-up`",
        "",
        "Orden de `previews/comparison.png`: primera fila `positive_x`, "
        "`negative_x`, `positive_y`; segunda fila `negative_y`, `positive_z`, "
        "`negative_z`.",
        "",
        "| Roof-up | Planos | Triángulos | Área | Solo planos | Overlay | PNG |",
        "|---|---:|---:|---:|---|---|---|",
    ]
    for roof_up in document["roof_ups"]:
        summary = document["results"][roof_up]["summary"]
        lines.append(
            f"| `{roof_up}` | {summary['roof_plane_count']} | "
            f"{summary['detected_triangles']} | {summary['detected_area']:.6g} | "
            f"`roof_planes_roof_up_{roof_up}_only.glb` | "
            f"`roof_planes_roof_up_{roof_up}_overlay.glb` | "
            f"`previews/roof_up_{roof_up}.png` |"
        )
    lines.extend(
        [
            "",
            "Los seis resultados usan los mismos umbrales y la misma cámara. La única "
            "variable geométrica es la referencia `roof-up`. Los IDs y colores se "
            "mantienen estables cuando una misma región aparece en más de un resultado.",
            "",
        ]
    )
    return "\n".join(lines)


def run_roof_up_comparison(
    args: argparse.Namespace, roof_ups: tuple[str, ...]
) -> Path:
    input_path = resolve_input(args.input)
    output_root = Path(args.output_root).expanduser()
    if not output_root.is_absolute():
        output_root = (SCRIPT_DIR / output_root).resolve()
    run_directory, created_at = create_run_directory(output_root.resolve(), args.timezone)
    manifest_path = run_directory / "manifest.json"
    parameters = {
        "roof_ups": list(roof_ups),
        "normal_direction": args.normal_direction,
        "visibility": "none",
        "max_pitch": args.max_pitch,
        "angle_tolerance": args.angle_tolerance,
        "min_area": args.min_area,
        "min_faces": args.min_faces,
        "max_plane_rms": args.max_plane_rms,
        "boundary_simplify": args.boundary_simplify,
    }
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "task": "roof_up_comparison",
        "created_at": created_at.isoformat(),
        "source": {
            "path": str(input_path),
            "filename": input_path.name,
            "size_bytes": input_path.stat().st_size,
            "sha256": sha256_file(input_path),
        },
        "parameters": parameters,
        "outputs": [],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    try:
        import_glb(input_path)
        base_data = extract_world_triangles()
        vertices = base_data["vertices"]
        triangles = base_data["triangles"]
        diagonal = float(
            np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0))
        )
        planes_by_roof_up: dict[str, list[dict[str, Any]]] = {}
        statistics_by_roof_up: dict[str, dict[str, Any]] = {}
        signatures: set[tuple[int, ...]] = set()

        for roof_up in roof_ups:
            candidates, flipped, roof_up_blender, prepared_data = prepare_roof_candidates(
                base_data,
                roof_up,
                args.max_pitch,
                args.normal_direction,
            )
            groups = group_candidate_faces(
                triangles,
                prepared_data["normals"],
                candidates,
                args.angle_tolerance,
            )
            rejected = defaultdict(int)
            accepted = []
            for group in groups:
                if len(group) < args.min_faces:
                    rejected["min_faces"] += 1
                    continue
                plane = fit_plane(group, prepared_data, roof_up_blender)
                if plane["area"] < args.min_area:
                    rejected["min_area"] += 1
                    continue
                if plane["rms"] > args.max_plane_rms:
                    rejected["max_plane_rms"] += 1
                    continue
                plane["view_visibility"] = None
                plane["roof_up"] = roof_up
                plane["signature"] = tuple(sorted(plane["face_indices"]))
                signatures.add(plane["signature"])
                accepted.append(plane)

            accepted.sort(key=lambda plane: plane["area"], reverse=True)
            planes_by_roof_up[roof_up] = accepted
            detected_faces = {
                face_index
                for plane in accepted
                for face_index in plane["face_indices"]
            }
            statistics_by_roof_up[roof_up] = {
                "roof_orientation_candidate_triangles": len(candidates),
                "candidate_triangles_reoriented": int(
                    np.count_nonzero(flipped[candidates])
                ),
                "initial_connected_regions": len(groups),
                "roof_plane_count": len(accepted),
                "detected_triangles": len(detected_faces),
                "triangle_coverage_percent": 100.0
                * len(detected_faces)
                / max(len(triangles), 1),
                "detected_area": sum(plane["area"] for plane in accepted),
                "rejected_regions": dict(rejected),
            }

        signature_order = sorted(
            signatures,
            key=lambda signature: (
                -max(
                    plane["area"]
                    for planes in planes_by_roof_up.values()
                    for plane in planes
                    if plane["signature"] == signature
                ),
                signature,
            ),
        )
        identity_by_signature = {
            signature: (f"roof_plane_{index:03d}", index - 1)
            for index, signature in enumerate(signature_order, 1)
        }
        results = {}
        for roof_up in roof_ups:
            roof_up_gltf = np.asarray(VIEW_FROM_GLTF[roof_up], dtype=np.float64)
            plane_documents = []
            for plane in planes_by_roof_up[roof_up]:
                identifier, color_index = identity_by_signature[plane["signature"]]
                plane["id"] = identifier
                plane["color_index"] = color_index
                plane_json = plane_to_json(
                    plane,
                    base_data,
                    identifier,
                    args,
                    roof_up,
                    roof_up_gltf,
                )
                hue = (color_index * 0.61803398875) % 1.0
                plane_json["display_color_rgb"] = clean_vector(
                    colorsys.hsv_to_rgb(hue, 0.72, 1.0)
                )
                plane["json"] = plane_json
                plane["pitch_degrees"] = plane_json["pitch_degrees"]
                plane["confidence"] = plane_json["confidence"]
                plane_documents.append(plane_json)
            results[roof_up] = {
                "summary": statistics_by_roof_up[roof_up],
                "roof_planes": plane_documents,
            }

        vertices_gltf = blender_to_gltf_array(vertices)
        document = {
            "schema_version": 1,
            "description": "Signed comparison of six roof-up references without visibility filtering.",
            "source": manifest["source"],
            "coordinate_system": {
                "standard": "glTF 2.0",
                "up_axis": "Y",
                "handedness": "right-handed",
                "units": "source model units (not declared by glTF)",
            },
            "roof_ups": list(roof_ups),
            "comparison_order": [list(roof_ups[:3]), list(roof_ups[3:6])],
            "comparison_camera": {
                "role": "visualization_only",
                "projection": "orthographic",
                "direction_gltf": clean_vector(
                    np.asarray((-1.0, 0.7, -1.0), dtype=np.float64)
                    / np.linalg.norm(np.asarray((-1.0, 0.7, -1.0)))
                ),
            },
            "parameters": parameters,
            "summary": {
                "input_objects": base_data["objects"],
                "input_vertices": len(vertices),
                "input_triangles": len(triangles),
                "unique_plane_count": len(signatures),
                "bounds_gltf": {
                    "min": clean_vector(vertices_gltf.min(axis=0)),
                    "max": clean_vector(vertices_gltf.max(axis=0)),
                },
            },
            "results": results,
        }

        json_path = run_directory / "roof_planes_by_roof_up.json"
        json_path.write_text(json.dumps(document, indent=2), encoding="utf-8")
        report_path = run_directory / "roof_up_comparison_report.md"
        report_path.write_text(
            roof_up_comparison_report_markdown(document), encoding="utf-8"
        )
        assets = export_roof_up_comparison_assets(
            input_path,
            run_directory,
            planes_by_roof_up,
            base_data,
            diagonal,
            roof_ups,
        )
        assets.extend(
            (
                (json_path, "roof_up_comparison_json"),
                (report_path, "report"),
            )
        )
        for path, kind in assets:
            manifest["outputs"].append(
                {
                    "type": kind,
                    "filename": str(path.relative_to(run_directory)),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
        manifest["summary"] = {
            "unique_plane_count": len(signatures),
            "by_roof_up": statistics_by_roof_up,
        }
        manifest["status"] = "completed"
        manifest["completed_at"] = datetime.now(ZoneInfo(args.timezone)).isoformat()
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return run_directory
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = str(error)
        manifest["traceback"] = traceback.format_exc()
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        raise


def run_multiview(args: argparse.Namespace, views: tuple[str, ...]) -> Path:
    input_path = resolve_input(args.input)
    output_root = Path(args.output_root).expanduser()
    if not output_root.is_absolute():
        output_root = (SCRIPT_DIR / output_root).resolve()
    run_directory, created_at = create_run_directory(output_root.resolve(), args.timezone)
    manifest_path = run_directory / "manifest.json"
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "task": "roof_plane_multiview_calibration",
        "created_at": created_at.isoformat(),
        "source": {
            "path": str(input_path),
            "filename": input_path.name,
            "size_bytes": input_path.stat().st_size,
            "sha256": sha256_file(input_path),
        },
        "parameters": {
            "views": list(views),
            "roof_up": args.roof_up,
            "normal_direction": args.normal_direction,
            "visible_part": args.visible_part,
            "max_pitch": args.max_pitch,
            "angle_tolerance": args.angle_tolerance,
            "min_area": args.min_area,
            "min_faces": args.min_faces,
            "max_plane_rms": args.max_plane_rms,
            "min_view_visibility": args.min_view_visibility,
            "boundary_simplify": args.boundary_simplify,
        },
        "outputs": [],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    try:
        import_glb(input_path)
        data = extract_world_triangles()
        vertices = data["vertices"]
        triangles = data["triangles"]
        diagonal = float(
            np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0))
        )
        candidates, flipped_normals, roof_up_blender, data = prepare_roof_candidates(
            data, args.roof_up, args.max_pitch, args.normal_direction
        )
        groups = group_candidate_faces(
            triangles, data["normals"], candidates, args.angle_tolerance
        )
        bvh = BVHTree.FromPolygons(
            [Vector(tuple(point)) for point in vertices],
            [tuple(int(value) for value in triangle) for triangle in triangles],
            all_triangles=True,
        )

        rejection_counts = defaultdict(int)
        geometric_planes = []
        for group in groups:
            if len(group) < args.min_faces:
                rejection_counts["min_faces"] += 1
                continue
            plane = fit_plane(group, data, roof_up_blender)
            if plane["area"] < args.min_area:
                rejection_counts["min_area"] += 1
                continue
            if plane["rms"] > args.max_plane_rms:
                rejection_counts["max_plane_rms"] += 1
                continue
            visibility_by_view = {}
            for view_name in views:
                view_vector = gltf_to_blender_array(
                    np.asarray(VIEW_FROM_GLTF[view_name], dtype=np.float64)
                )
                visibility_by_view[view_name] = directional_visibility(
                    plane, data, bvh, diagonal, view_vector
                )
            plane["visibility_by_view"] = visibility_by_view
            if max(visibility_by_view.values(), default=0.0) < args.min_view_visibility:
                rejection_counts["all_view_visibility"] += 1
                continue
            geometric_planes.append(plane)

        geometric_planes.sort(key=lambda plane: plane["area"], reverse=True)
        for index, plane in enumerate(geometric_planes, 1):
            plane["id"] = f"roof_plane_{index:03d}"
            plane["color_index"] = index - 1
            plane["view_visibility"] = max(plane["visibility_by_view"].values())
            plane_json = plane_to_json(
                plane,
                data,
                plane["id"],
                args,
                args.roof_up,
                np.asarray(VIEW_FROM_GLTF[args.roof_up], dtype=np.float64),
            )
            hue = ((index - 1) * 0.61803398875) % 1.0
            plane_json["display_color_rgb"] = clean_vector(
                colorsys.hsv_to_rgb(hue, 0.72, 1.0)
            )
            plane_json["visibility_by_view"] = {
                view: clean_number(value)
                for view, value in plane["visibility_by_view"].items()
            }
            plane_json["visible_from"] = [
                view
                for view, value in plane["visibility_by_view"].items()
                if value >= args.min_view_visibility
            ]
            plane["json"] = plane_json
            plane["pitch_degrees"] = plane_json["pitch_degrees"]
            plane["confidence"] = plane_json["confidence"]

        if not geometric_planes:
            raise RuntimeError("No se detectaron planos visibles desde las vistas solicitadas.")

        vertices_gltf = blender_to_gltf_array(vertices)
        by_view = {}
        for view_name in views:
            selected = [
                plane
                for plane in geometric_planes
                if plane["visibility_by_view"][view_name] >= args.min_view_visibility
            ]
            by_view[view_name] = {
                "plane_count": len(selected),
                "triangle_count": sum(len(plane["face_indices"]) for plane in selected),
                "area": sum(plane["area"] for plane in selected),
                "plane_ids": [plane["id"] for plane in selected],
            }

        ray_directions = {
            view: clean_vector(-np.asarray(VIEW_FROM_GLTF[view], dtype=np.float64))
            for view in views
        }
        detected_faces = {
            face_index
            for plane in geometric_planes
            for face_index in plane["face_indices"]
        }
        document = {
            "schema_version": 1,
            "description": "Multiview calibration of exterior roof planes.",
            "source": manifest["source"],
            "coordinate_system": {
                "standard": "glTF 2.0",
                "up_axis": "Y",
                "handedness": "right-handed",
                "units": "source model units (not declared by glTF)",
            },
            "view": {
                "mode": "multiview",
                "views": list(views),
                "roof_up": args.roof_up,
                "visible_part": args.visible_part,
                "projection": "orthographic",
                "ray_directions": ray_directions,
            },
            "parameters": manifest["parameters"],
            "summary": {
                "input_objects": data["objects"],
                "input_vertices": len(vertices),
                "input_triangles": len(triangles),
                "roof_orientation_candidate_triangles": len(candidates),
                "candidate_triangles_reoriented_to_roof_up": int(
                    np.count_nonzero(flipped_normals[candidates])
                ),
                "initial_connected_regions": len(groups),
                "roof_plane_count": len(geometric_planes),
                "detected_triangles": len(detected_faces),
                "triangle_coverage_percent": 100.0
                * len(detected_faces)
                / max(len(triangles), 1),
                "detected_area": sum(
                    plane["area"] for plane in geometric_planes
                ),
                "rejected_regions": dict(rejection_counts),
                "by_view": by_view,
                "bounds_gltf": {
                    "min": clean_vector(vertices_gltf.min(axis=0)),
                    "max": clean_vector(vertices_gltf.max(axis=0)),
                },
            },
            "roof_planes": [plane["json"] for plane in geometric_planes],
        }

        json_path = run_directory / "roof_planes_multiview.json"
        json_path.write_text(json.dumps(document, indent=2), encoding="utf-8")
        report_path = run_directory / "roof_planes_report.md"
        report_path.write_text(multiview_report_markdown(document), encoding="utf-8")
        assets = export_multiview_assets(
            input_path,
            run_directory,
            geometric_planes,
            data,
            diagonal,
            views,
            args.min_view_visibility,
        )
        assets.extend(
            ((json_path, "roof_planes_multiview_json"), (report_path, "report"))
        )
        for path, kind in assets:
            manifest["outputs"].append(
                {
                    "type": kind,
                    "filename": str(path.relative_to(run_directory)),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
        manifest["summary"] = document["summary"]
        manifest["status"] = "completed"
        manifest["completed_at"] = datetime.now(ZoneInfo(args.timezone)).isoformat()
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return run_directory
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = str(error)
        manifest["traceback"] = traceback.format_exc()
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        raise


def run(args: argparse.Namespace) -> Path:
    input_path = resolve_input(args.input)
    output_root = Path(args.output_root).expanduser()
    if not output_root.is_absolute():
        output_root = (SCRIPT_DIR / output_root).resolve()
    run_directory, created_at = create_run_directory(output_root.resolve(), args.timezone)
    manifest_path = run_directory / "manifest.json"
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "task": "roof_plane_detection",
        "created_at": created_at.isoformat(),
        "source": {
            "path": str(input_path),
            "filename": input_path.name,
            "size_bytes": input_path.stat().st_size,
            "sha256": sha256_file(input_path),
        },
        "parameters": {
            key: getattr(args, key)
            for key in (
                "roof_up",
                "normal_direction",
                "visibility",
                "view_from",
                "visible_part",
                "max_pitch",
                "angle_tolerance",
                "min_area",
                "min_faces",
                "max_plane_rms",
                "min_view_visibility",
                "boundary_simplify",
                "geometry_output",
                "merge_angle",
                "merge_plane_distance",
                "merge_max_residual",
                "merge_residual_percentile",
                "merge_min_inlier_ratio",
                "merge_robust_max_residual",
                "merge_min_boundary_outlier_ratio",
                "merge_gap",
                "simplified_boundary_tolerance",
                "boundary_regularization",
                "regularization_tolerance",
                "regularization_min_iou",
                "regularization_max_area_change",
                "regularized_rotations",
                "min_hole_area",
            )
        },
        "outputs": [],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    try:
        import_glb(input_path)
        data = extract_world_triangles()
        vertices = data["vertices"]
        triangles = data["triangles"]
        minimum = vertices.min(axis=0)
        maximum = vertices.max(axis=0)
        diagonal = float(np.linalg.norm(maximum - minimum))
        candidates, flipped_normals, roof_up_blender, data = prepare_roof_candidates(
            data, args.roof_up, args.max_pitch, args.normal_direction
        )
        bvh = BVHTree.FromPolygons(
            [Vector(tuple(point)) for point in vertices],
            [tuple(int(value) for value in triangle) for triangle in triangles],
            all_triangles=True,
        )
        view_from_gltf = np.asarray(VIEW_FROM_GLTF[args.view_from], dtype=np.float64)
        view_from_blender = gltf_to_blender_array(view_from_gltf)
        view_from_blender /= np.linalg.norm(view_from_blender)
        if args.visibility == "directional" and args.visible_part == "only":
            grouping_candidates = visible_faces_from_direction(
                candidates, data, bvh, diagonal, view_from_blender
            )
        else:
            grouping_candidates = candidates
        groups = group_candidate_faces(
            triangles, data["normals"], grouping_candidates, args.angle_tolerance
        )

        accepted = []
        rejection_counts = defaultdict(int)
        for group in groups:
            if len(group) < args.min_faces:
                rejection_counts["min_faces"] += 1
                continue
            plane = fit_plane(group, data, roof_up_blender)
            if plane["area"] < args.min_area:
                rejection_counts["min_area"] += 1
                continue
            if plane["rms"] > args.max_plane_rms:
                rejection_counts["max_plane_rms"] += 1
                continue
            if args.visibility == "none":
                plane["view_visibility"] = None
            elif args.visible_part == "only":
                plane["view_visibility"] = 1.0
            else:
                plane["view_visibility"] = directional_visibility(
                    plane, data, bvh, diagonal, view_from_blender
                )
            if (
                plane["view_visibility"] is not None
                and plane["view_visibility"] < args.min_view_visibility
            ):
                rejection_counts["view_visibility"] += 1
                continue
            accepted.append(plane)

        accepted.sort(key=lambda plane: plane["area"], reverse=True)
        for index, plane in enumerate(accepted, 1):
            plane["id"] = f"roof_plane_{index:03d}"
            plane["color_index"] = index - 1
            plane["json"] = plane_to_json(
                plane,
                data,
                plane["id"],
                args,
                args.roof_up,
                np.asarray(VIEW_FROM_GLTF[args.roof_up], dtype=np.float64),
            )
            plane["pitch_degrees"] = plane["json"]["pitch_degrees"]
            plane["confidence"] = plane["json"]["confidence"]

        simplified = []
        if args.geometry_output in ("simplified", "both"):
            simplified = merge_coplanar_planes(
                accepted, data, roof_up_blender, args
            )
            roof_up_gltf = np.asarray(
                VIEW_FROM_GLTF[args.roof_up], dtype=np.float64
            )
            for index, plane in enumerate(simplified, 1):
                plane["id"] = f"simplified_roof_plane_{index:03d}"
                plane["color_index"] = index - 1
                vectorize_plane(plane, data, args)
                regularize_plane(plane, args)
                planarity_score = max(
                    0.0,
                    1.0 - plane["rms"] / max(args.merge_plane_distance, 1e-12),
                )
                plane["pitch_degrees"] = math.degrees(
                    math.acos(
                        max(
                            -1.0,
                            min(1.0, float(np.dot(plane["normal"], roof_up_blender))),
                        )
                    )
                )
                plane["confidence"] = 0.75 * planarity_score + 0.25 * min(
                    1.0, plane["simplified_area"] / max(args.min_area * 5.0, 1e-12)
                )
                plane["json"] = simplified_plane_to_json(
                    plane, args.roof_up, roof_up_gltf
                )
                plane["regularized_json"] = regularized_plane_to_json(
                    plane, args.roof_up, roof_up_gltf
                )

        detected_faces = sum(len(plane["face_indices"]) for plane in accepted)
        detected_area = sum(plane["area"] for plane in accepted)
        exact_mesh_vertices = sum(len(plane["vertex_ids"]) for plane in accepted)
        exact_mesh_triangles = sum(len(plane["face_indices"]) for plane in accepted)
        simplified_mesh_vertices = sum(
            len(plane["simplified_vertices"]) for plane in simplified
        )
        simplified_mesh_triangles = sum(
            len(plane["simplified_faces"]) for plane in simplified
        )
        regularized_mesh_vertices = sum(
            len(plane["regularized_vertices"]) for plane in simplified
        )
        regularized_mesh_triangles = sum(
            len(plane["regularized_faces"]) for plane in simplified
        )
        rotation_names = resolved_regularized_rotations(args)
        rotation_pivot_gltf = None
        rotation_results: dict[str, dict[str, Any]] = {}
        if simplified and rotation_names:
            rotation_pivot_gltf, rotation_results = build_regularized_rotation_results(
                simplified, rotation_names, args.roof_up
            )
        rotation_document = {
            "schema_version": 1,
            "description": (
                "Rigid 90-degree rotations of the regularized roof-plane geometry."
            ),
            "coordinate_system": {
                "standard": "glTF 2.0",
                "up_axis": "Y",
                "handedness": "right-handed",
                "units": "source model units (not declared by glTF)",
                "angle_convention": "right-hand rule",
            },
            "source_geometry": "roof_planes_regularized_only.glb",
            "pivot_definition": "center of the regularized geometry bounding box",
            "pivot_gltf": (
                None
                if rotation_pivot_gltf is None
                else clean_vector(rotation_pivot_gltf)
            ),
            "rotation_count": len(rotation_results),
            "rotations": [
                result["json"] for result in rotation_results.values()
            ],
        }
        vertices_gltf = blender_to_gltf_array(vertices)
        center_gltf = (vertices_gltf.min(axis=0) + vertices_gltf.max(axis=0)) * 0.5
        camera_position_gltf = center_gltf + view_from_gltf * diagonal * 1.5
        document = {
            "schema_version": 1,
            "description": "Exterior roof planes detected from a triangulated GLB mesh.",
            "source": manifest["source"],
            "coordinate_system": {
                "standard": "glTF 2.0",
                "up_axis": "Y",
                "handedness": "right-handed",
                "units": "source model units (not declared by glTF)",
            },
            "view": {
                "visibility": args.visibility,
                "view_from": args.view_from,
                "visible_part": (
                    args.visible_part if args.visibility == "directional" else None
                ),
                "view_from_role": (
                    "visibility_and_validation_camera"
                    if args.visibility == "directional"
                    else "validation_camera_only"
                ),
                "projection": "orthographic",
                "camera_position": clean_vector(camera_position_gltf),
                "camera_target": clean_vector(center_gltf),
                "ray_direction": clean_vector(-view_from_gltf),
            },
            "parameters": manifest["parameters"],
            "summary": {
                "input_objects": data["objects"],
                "input_vertices": len(vertices),
                "input_triangles": len(triangles),
                "roof_orientation_candidate_triangles": len(candidates),
                "candidate_triangles_reoriented_to_roof_up": int(
                    np.count_nonzero(flipped_normals[candidates])
                ),
                "visibility_filtered_candidate_triangles": (
                    len(grouping_candidates)
                    if args.visibility == "directional"
                    else None
                ),
                "initial_connected_regions": len(groups),
                "roof_plane_count": len(accepted),
                "detected_triangles": detected_faces,
                "triangle_coverage_percent": 100.0 * detected_faces / max(len(triangles), 1),
                "detected_area": detected_area,
                "exact_geometry": {
                    "plane_count": len(accepted),
                    "vertices": exact_mesh_vertices,
                    "triangles": exact_mesh_triangles,
                },
                "simplified_geometry": {
                    "plane_count": len(simplified),
                    "vertices": simplified_mesh_vertices,
                    "triangles": simplified_mesh_triangles,
                    "vertex_reduction_percent": (
                        100.0
                        * (1.0 - simplified_mesh_vertices / exact_mesh_vertices)
                        if exact_mesh_vertices and simplified
                        else None
                    ),
                    "triangle_reduction_percent": (
                        100.0
                        * (1.0 - simplified_mesh_triangles / exact_mesh_triangles)
                        if exact_mesh_triangles and simplified
                        else None
                    ),
                    "source_planes_fused": sum(
                        max(0, len(plane["source_plane_ids"]) - 1)
                        for plane in simplified
                    ),
                },
                "regularized_geometry": {
                    "plane_count": len(simplified),
                    "vertices": regularized_mesh_vertices,
                    "triangles": regularized_mesh_triangles,
                    "planes_regularized": sum(
                        bool(plane["regularization"]["accepted"])
                        for plane in simplified
                    ),
                    "vertex_reduction_from_simplified_percent": (
                        100.0
                        * (
                            1.0
                            - regularized_mesh_vertices
                            / simplified_mesh_vertices
                        )
                        if simplified_mesh_vertices and simplified
                        else None
                    ),
                },
                "regularized_rotations": {
                    "count": len(rotation_results),
                    "ids": list(rotation_results),
                    "pivot_gltf": (
                        None
                        if rotation_pivot_gltf is None
                        else clean_vector(rotation_pivot_gltf)
                    ),
                },
                "rejected_regions": dict(rejection_counts),
                "bounds_gltf": {
                    "min": clean_vector(vertices_gltf.min(axis=0)),
                    "max": clean_vector(vertices_gltf.max(axis=0)),
                },
            },
            "roof_planes": [plane["json"] for plane in accepted],
            "simplified_roof_planes": [plane["json"] for plane in simplified],
            "regularized_roof_planes": [
                plane["regularized_json"] for plane in simplified
            ],
            "regularized_rotations": rotation_document,
        }
        if not accepted:
            raise RuntimeError("No se detectaron planos con los parámetros actuales.")

        json_path = run_directory / "roof_planes.json"
        json_path.write_text(json.dumps(document, indent=2), encoding="utf-8")
        rotations_json_path = None
        if rotation_results:
            rotations_json_path = (
                run_directory / "roof_planes_regularized_rotations.json"
            )
            rotations_json_path.write_text(
                json.dumps(rotation_document, indent=2), encoding="utf-8"
            )
        report_path = run_directory / "roof_planes_report.md"
        report_path.write_text(report_markdown(document), encoding="utf-8")
        assets = export_exact_and_simplified_assets(
            input_path,
            run_directory,
            accepted,
            simplified,
            data,
            diagonal,
            args.geometry_output,
        )
        if rotation_results:
            assets.extend(
                export_regularized_rotation_assets(
                    run_directory,
                    rotation_pivot_gltf,
                    rotation_results,
                    diagonal,
                )
            )
        assets.extend(((json_path, "roof_planes_json"), (report_path, "report")))
        if rotations_json_path is not None:
            assets.append(
                (rotations_json_path, "regularized_rotations_json")
            )

        for path, kind in assets:
            manifest["outputs"].append(
                {
                    "type": kind,
                    "filename": str(path.relative_to(run_directory)),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
        manifest["summary"] = document["summary"]
        manifest["status"] = "completed"
        manifest["completed_at"] = datetime.now(ZoneInfo(args.timezone)).isoformat()
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return run_directory
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = str(error)
        manifest["traceback"] = traceback.format_exc()
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        raise


def main() -> int:
    raw_args = user_arguments()
    parser = build_parser()
    args = parser.parse_args(raw_args)
    try:
        validate_args(args)
    except ValueError as error:
        parser.error(str(error))
    if bpy is None:
        return launch_in_blender(args, raw_args)
    try:
        views = resolved_views(args)
        roof_ups = resolved_roof_ups(args)
        if roof_ups:
            output = run_roof_up_comparison(args, roof_ups)
        elif args.views:
            output = run_multiview(args, views)
        else:
            output = run(args)
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        traceback.print_exc()
        return 1
    print(f"RESULTADO_OK={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
