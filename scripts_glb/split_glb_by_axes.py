#!/usr/bin/env python3
"""Create selected axis half-model GLBs from one input GLB.

The script can be launched with regular Python; it relaunches itself inside
Blender because the geometric operations use Blender's Python API.

Examples (run from scripts_glb):
    python3 split_glb_by_axes.py input_glbs/roof_model.glb
    python3 split_glb_by_axes.py roof_model.glb --axes all
    python3 split_glb_by_axes.py roof_model.glb --axes x z
    python3 split_glb_by_axes.py roof_model.glb --keep positive
    python3 split_glb_by_axes.py roof_model.glb --cap no
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

try:
    import bmesh
    import bpy
    from mathutils import Vector
except ModuleNotFoundError:
    bmesh = None
    bpy = None
    Vector = None


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = SCRIPT_DIR / "input_glbs"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "outputs_glb"
AXES = ("x", "y", "z")

# glTF uses Y-up. Blender imports glTF scenes into its Z-up world using:
# (x, y, z)_gltf -> (x, -z, y)_blender.
GLTF_AXIS_TO_BLENDER = {
    "x": (1.0, 0.0, 0.0),
    "y": (0.0, 0.0, 1.0),
    "z": (0.0, -1.0, 0.0),
}


def user_arguments() -> list[str]:
    if bpy is not None and "--" in sys.argv:
        return sys.argv[sys.argv.index("--") + 1 :]
    return sys.argv[1:]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Corta un GLB por el centro de los ejes solicitados y conserva una "
            "mitad independiente por eje."
        )
    )
    parser.add_argument(
        "input",
        nargs="?",
        help=(
            "GLB de entrada. Si se omite, input_glbs debe contener exactamente "
            "un archivo .glb."
        ),
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Carpeta raíz de resultados (predeterminado: outputs_glb).",
    )
    parser.add_argument(
        "--axes",
        nargs="+",
        choices=(*AXES, "all"),
        default=["x"],
        help=(
            "Ejes que se procesan: x, y, z, una combinación o all "
            "(predeterminado: x)."
        ),
    )
    parser.add_argument(
        "--keep",
        choices=("positive", "negative"),
        default="negative",
        help="Mitad que se conserva en cada eje (predeterminado: negative).",
    )
    parser.add_argument(
        "--cap",
        choices=("auto", "yes", "no"),
        default="yes",
        help=(
            "Cierra la sección del corte. auto solo cierra objetos originalmente "
            "manifold/cerrados (predeterminado: yes)."
        ),
    )
    parser.add_argument(
        "--timezone",
        default="America/Bogota",
        help="Zona horaria para el nombre y manifiesto de la ejecución.",
    )
    parser.add_argument(
        "--blender",
        default=os.environ.get("BLENDER_EXECUTABLE", "blender"),
        help="Ejecutable de Blender usado al lanzar el script con Python.",
    )
    return parser


def normalize_axes(values: list[str]) -> tuple[str, ...]:
    if "all" in values:
        if len(values) != 1:
            raise ValueError("'all' no se puede combinar con x, y o z.")
        return AXES
    return tuple(dict.fromkeys(values))


def resolve_input(value: str | None) -> Path:
    if value is None:
        candidates = sorted(DEFAULT_INPUT_DIR.glob("*.glb"))
        if len(candidates) != 1:
            raise ValueError(
                "Sin argumento de entrada, input_glbs debe contener exactamente "
                f"un GLB; se encontraron {len(candidates)}."
            )
        return candidates[0].resolve()

    candidate = Path(value).expanduser()
    if not candidate.is_absolute() and not candidate.exists():
        candidate = DEFAULT_INPUT_DIR / candidate
    candidate = candidate.resolve()
    if not candidate.is_file():
        raise FileNotFoundError(f"No existe el GLB de entrada: {candidate}")
    if candidate.suffix.lower() != ".glb":
        raise ValueError(f"El archivo de entrada debe terminar en .glb: {candidate}")
    return candidate


def launch_in_blender(args: argparse.Namespace, raw_args: list[str]) -> int:
    executable = shutil.which(args.blender)
    if executable is None:
        raise RuntimeError(
            f"No se encontró Blender ('{args.blender}'). Instálalo o usa --blender."
        )
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


def gltf_to_blender(vector: Any) -> Any:
    return Vector((vector[0], -vector[2], vector[1]))


def blender_to_gltf(vector: Any) -> Any:
    return Vector((vector[0], vector[2], -vector[1]))


def mesh_objects() -> list[Any]:
    return [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]


def import_glb(path: Path) -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    result = bpy.ops.import_scene.gltf(filepath=str(path))
    if "FINISHED" not in result:
        raise RuntimeError(f"Blender no pudo importar {path}")
    if not mesh_objects():
        raise ValueError(f"El GLB no contiene objetos de malla: {path}")


def scene_bounds_gltf() -> tuple[Any, Any]:
    points = []
    for obj in mesh_objects():
        points.extend(
            blender_to_gltf(obj.matrix_world @ Vector(corner))
            for corner in obj.bound_box
        )
    if not points:
        raise ValueError("No hay geometría para calcular la caja envolvente.")
    minimum = Vector(min(point[i] for point in points) for i in range(3))
    maximum = Vector(max(point[i] for point in points) for i in range(3))
    return minimum, maximum


def scene_stats() -> dict[str, Any]:
    objects = mesh_objects()
    minimum, maximum = scene_bounds_gltf()
    return {
        "scene_objects": len(bpy.context.scene.objects),
        "mesh_objects": len(objects),
        "vertices": sum(len(obj.data.vertices) for obj in objects),
        "edges": sum(len(obj.data.edges) for obj in objects),
        "polygons": sum(len(obj.data.polygons) for obj in objects),
        "materials": len(bpy.data.materials),
        "images": len(bpy.data.images),
        "animations": len(bpy.data.actions),
        "armatures": sum(obj.type == "ARMATURE" for obj in bpy.context.scene.objects),
        "shape_key_meshes": sum(obj.data.shape_keys is not None for obj in objects),
        "bounds_gltf": {
            "min": list(minimum),
            "max": list(maximum),
            "center": list((minimum + maximum) * 0.5),
            "dimensions": list(maximum - minimum),
        },
    }


def is_manifold_mesh(bm: Any) -> bool:
    return bool(bm.faces) and all(edge.is_manifold for edge in bm.edges)


def clip_object(
    obj: Any,
    plane_point_world: Any,
    plane_normal_world: Any,
    keep: str,
    cap_mode: str,
    epsilon_world: float,
) -> dict[str, Any]:
    if obj.data.shape_keys is not None:
        raise ValueError(
            f"La malla '{obj.name}' tiene shape keys; no se puede cortar sin "
            "alterar sus morph targets de forma insegura."
        )

    if obj.data.users > 1:
        obj.data = obj.data.copy()

    bm = bmesh.new()
    bm.from_mesh(obj.data)
    before = {"vertices": len(bm.verts), "edges": len(bm.edges), "faces": len(bm.faces)}
    originally_manifold = is_manifold_mesh(bm)

    inverse_world = obj.matrix_world.inverted_safe()
    plane_point_local = inverse_world @ plane_point_world
    plane_normal_local = obj.matrix_world.to_3x3().transposed() @ plane_normal_world
    if plane_normal_local.length == 0:
        bm.free()
        raise ValueError(f"La transformación de '{obj.name}' aplasta el eje de corte.")
    plane_normal_local.normalize()

    scales = [max(abs(value), 1e-12) for value in obj.matrix_world.to_scale()]
    epsilon_local = max(epsilon_world / max(scales), 1e-9)
    geometry = [*bm.verts, *bm.edges, *bm.faces]
    bisect_result = bmesh.ops.bisect_plane(
        bm,
        geom=geometry,
        dist=epsilon_local,
        plane_co=plane_point_local,
        plane_no=plane_normal_local,
        use_snap_center=False,
        clear_outer=False,
        clear_inner=False,
    )

    keep_multiplier = 1.0 if keep == "positive" else -1.0

    def signed_distance(vertex: Any) -> float:
        world_position = obj.matrix_world @ vertex.co
        return (world_position - plane_point_world).dot(plane_normal_world)

    discard = [
        vertex
        for vertex in bm.verts
        if keep_multiplier * signed_distance(vertex) < -epsilon_world
    ]
    if discard:
        bmesh.ops.delete(bm, geom=discard, context="VERTS")

    cut_edges = [
        edge
        for edge in bm.edges
        if edge.is_boundary
        and all(abs(signed_distance(vertex)) <= epsilon_world * 10 for vertex in edge.verts)
    ]
    cap_requested = cap_mode == "yes" or (cap_mode == "auto" and originally_manifold)
    cap_faces = 0
    cap_error = None
    if cap_requested and cut_edges:
        try:
            fill_result = bmesh.ops.holes_fill(bm, edges=cut_edges, sides=0)
            cap_faces = len(fill_result.get("faces", []))
        except Exception as error:  # Preserve the valid open cut if filling fails.
            cap_error = str(error)

    bm.normal_update()
    after = {"vertices": len(bm.verts), "edges": len(bm.edges), "faces": len(bm.faces)}
    bm.to_mesh(obj.data)
    obj.data.update()
    bm.free()

    return {
        "object": obj.name,
        "originally_manifold": originally_manifold,
        "before": before,
        "after": after,
        "bisected_elements": len(bisect_result.get("geom_cut", [])),
        "discarded_vertices": len(discard),
        "cut_boundary_edges": len(cut_edges),
        "cap_requested": cap_requested,
        "cap_faces_created": cap_faces,
        "cap_error": cap_error,
    }


def remove_empty_mesh_objects() -> None:
    for obj in list(mesh_objects()):
        if len(obj.data.polygons) == 0:
            mesh_data = obj.data
            bpy.data.objects.remove(obj, do_unlink=True)
            if mesh_data.users == 0:
                bpy.data.meshes.remove(mesh_data)


def export_glb(path: Path) -> None:
    result = bpy.ops.export_scene.gltf(
        filepath=str(path),
        export_format="GLB",
        export_yup=True,
        export_apply=False,
        export_animations=True,
        export_materials="EXPORT",
        export_normals=True,
        export_texcoords=True,
        export_colors=True,
        export_cameras=False,
        export_lights=False,
    )
    if "FINISHED" not in result or not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"Blender no pudo exportar un GLB válido: {path}")


def make_run_directory(output_root: Path, timezone_name: str) -> tuple[Path, datetime]:
    try:
        timezone = ZoneInfo(timezone_name)
    except Exception as error:
        raise ValueError(f"Zona horaria no válida: {timezone_name}") from error
    created_at = datetime.now(timezone)
    output_root.mkdir(parents=True, exist_ok=True)
    run_directory = output_root / created_at.strftime("%Y%m%d_%H%M%S")
    if run_directory.exists():
        run_directory = output_root / created_at.strftime("%Y%m%d_%H%M%S_%f")
    run_directory.mkdir(parents=False, exist_ok=False)
    return run_directory, created_at


def validate_output(
    path: Path,
    axis: str,
    plane_coordinate: float,
    keep: str,
    tolerance: float,
    source_stats: dict[str, Any],
) -> dict[str, Any]:
    import_glb(path)
    stats = scene_stats()
    axis_index = AXES.index(axis)
    output_min = stats["bounds_gltf"]["min"][axis_index]
    output_max = stats["bounds_gltf"]["max"][axis_index]
    if keep == "positive":
        side_ok = output_min >= plane_coordinate - tolerance
    else:
        side_ok = output_max <= plane_coordinate + tolerance
    material_ok = stats["materials"] == source_stats["materials"]
    image_ok = stats["images"] == source_stats["images"]
    geometry_ok = stats["mesh_objects"] > 0 and stats["polygons"] > 0
    return {
        "passed": bool(side_ok and material_ok and image_ok and geometry_ok),
        "checks": {
            "geometry_present": geometry_ok,
            "kept_side_only": side_ok,
            "material_count_preserved": material_ok,
            "image_count_preserved": image_ok,
        },
        "stats": stats,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def markdown_manifest(manifest: dict[str, Any]) -> str:
    source = manifest["source"]
    lines = [
        "# Manifiesto de cortes GLB",
        "",
        f"- Ejecución: `{manifest['run']['directory_name']}`",
        f"- Fecha: `{manifest['run']['created_at']}`",
        f"- Entrada: `{source['path']}`",
        f"- SHA-256 de entrada: `{source['sha256']}`",
        f"- Ejes procesados: `{', '.join(manifest['settings']['axes'])}`",
        f"- Mitad conservada: `{manifest['settings']['keep']}`",
        f"- Cierre de cortes: `{manifest['settings']['cap']}`",
        "- Sistema de ejes: `glTF/GLB (Y-up)`",
        "",
        "## Archivos generados",
        "",
        "| Eje | Archivo | Plano | Validación |",
        "|---|---|---:|---|",
    ]
    for item in manifest["outputs"]:
        status = "correcta" if item["validation"]["passed"] else "fallida"
        lines.append(
            f"| {item['axis'].upper()} | `{item['filename']}` | "
            f"`{item['axis']} = {item['plane_coordinate_gltf']:.9g}` | {status} |"
        )
    lines.extend(
        [
            "",
            "Cada resultado se generó directamente desde el GLB original; los cortes no "
            "se aplicaron de forma acumulativa.",
            "",
        ]
    )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> Path:
    input_path = resolve_input(args.input)
    output_root = Path(args.output_root).expanduser()
    if not output_root.is_absolute():
        output_root = (SCRIPT_DIR / output_root).resolve()
    else:
        output_root = output_root.resolve()

    run_directory, created_at = make_run_directory(output_root, args.timezone)
    manifest_path = run_directory / "manifest.json"
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "run": {
            "directory_name": run_directory.name,
            "created_at": created_at.isoformat(),
            "timezone": args.timezone,
        },
        "source": {
            "path": str(input_path),
            "filename": input_path.name,
            "size_bytes": input_path.stat().st_size,
            "sha256": sha256_file(input_path),
        },
        "settings": {
            "axes": list(args.axes),
            "keep": args.keep,
            "cap": args.cap,
            "coordinate_system": "glTF/GLB Y-up",
            "cut_location": "center of the complete scene bounding box per axis",
        },
        "software": {
            "script": str(Path(__file__).resolve()),
            "python": sys.version.split()[0],
            "blender": bpy.app.version_string,
        },
        "outputs": [],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    try:
        import_glb(input_path)
        source_stats = scene_stats()
        manifest["source"]["stats"] = source_stats
        minimum = Vector(source_stats["bounds_gltf"]["min"])
        maximum = Vector(source_stats["bounds_gltf"]["max"])
        center = (minimum + maximum) * 0.5
        diagonal = (maximum - minimum).length
        tolerance = max(diagonal * 1e-6, 1e-7)
        epsilon = max(diagonal * 1e-8, 1e-9)

        for axis in args.axes:
            axis_index = AXES.index(axis)
            import_glb(input_path)
            plane_coordinate = center[axis_index]
            plane_point_gltf = center.copy()
            plane_point_gltf[axis_index] = plane_coordinate
            plane_point_blender = gltf_to_blender(plane_point_gltf)
            plane_normal_blender = Vector(GLTF_AXIS_TO_BLENDER[axis])

            object_operations = [
                clip_object(
                    obj=obj,
                    plane_point_world=plane_point_blender,
                    plane_normal_world=plane_normal_blender,
                    keep=args.keep,
                    cap_mode=args.cap,
                    epsilon_world=epsilon,
                )
                for obj in list(mesh_objects())
            ]
            remove_empty_mesh_objects()
            if not mesh_objects():
                raise RuntimeError(f"El corte {axis.upper()} eliminó toda la geometría.")

            output_name = f"{input_path.stem}_cut_{axis}_keep_{args.keep}.glb"
            output_path = run_directory / output_name
            export_glb(output_path)
            validation = validate_output(
                output_path,
                axis,
                plane_coordinate,
                args.keep,
                tolerance,
                source_stats,
            )
            manifest["outputs"].append(
                {
                    "axis": axis,
                    "filename": output_name,
                    "path": str(output_path),
                    "plane_coordinate_gltf": plane_coordinate,
                    "plane_normal_gltf": [
                        1.0 if index == axis_index else 0.0 for index in range(3)
                    ],
                    "kept_half": args.keep,
                    "removed_half": "negative" if args.keep == "positive" else "positive",
                    "object_operations": object_operations,
                    "validation": validation,
                }
            )
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        manifest["status"] = (
            "completed"
            if all(item["validation"]["passed"] for item in manifest["outputs"])
            else "validation_failed"
        )
        manifest["completed_at"] = datetime.now(ZoneInfo(args.timezone)).isoformat()
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        (run_directory / "manifest.md").write_text(
            markdown_manifest(manifest), encoding="utf-8"
        )
        if manifest["status"] != "completed":
            raise RuntimeError("Uno o más GLB no superaron la validación.")
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
        args.axes = normalize_axes(args.axes)
    except ValueError as error:
        parser.error(str(error))
    if bpy is None:
        return launch_in_blender(args, raw_args)

    try:
        run_directory = run(args)
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        traceback.print_exc()
        return 1
    print(f"RESULTADO_OK={run_directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
