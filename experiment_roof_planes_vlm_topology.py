from __future__ import annotations

import argparse
import base64
from dataclasses import asdict
import io
import json
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
from skimage.graph import route_through_array
from skimage.segmentation import relabel_sequential, watershed

from experiment_roof_planes import (
    make_contact_sheet,
    render_planes,
    save_label_mask,
    summarize_planes,
    write_geojson,
)
from run_artifacts import create_run_directory


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs"
STRUCTURAL_SUFFIX = "_roof_planes_experiment_02_structural"
EXPERIMENT_NAME = "roof_planes_experiment_04_vlm_topology"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Experiment 04: let a frontier multimodal model propose the complete "
            "roof topology, then snap and rasterize it locally."
        )
    )
    parser.add_argument(
        "--source-run",
        type=Path,
        help="Experiment 02 output directory. Defaults to the latest valid run.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument(
        "--detail",
        choices=("original", "high", "auto"),
        default="original",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=("medium", "high", "xhigh", "max"),
        default="high",
    )
    parser.add_argument(
        "--review-passes",
        type=int,
        choices=(0, 1),
        default=1,
        help="Run one independent topology-review pass after the initial proposal.",
    )
    parser.add_argument(
        "--pro",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Request pro reasoning mode; automatically retry standard mode if unavailable.",
    )
    parser.add_argument("--snap-margin", type=int, default=42)
    return parser.parse_args()


def find_latest_structural_run(output_root: Path) -> Path:
    required = {
        "input.png",
        "building_mask.png",
        "structural_boundary_map.png",
        "relative_depth_color.png",
        "surface_normals.png",
        "result.json",
    }
    candidates = [
        path
        for path in output_root.iterdir()
        if path.is_dir()
        and path.name.endswith(STRUCTURAL_SUFFIX)
        and all((path / name).is_file() for name in required)
    ]
    if not candidates:
        raise FileNotFoundError(
            "No valid experiment 02 output was found. Run "
            "experiment_roof_planes_structural.py first."
        )
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def load_mask(path: Path) -> np.ndarray:
    values = np.asarray(Image.open(path).convert("L"))
    return values > 0


def image_data_url(image: Image.Image, *, quality: int = 95) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=quality, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def point_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "x": {"type": "number"},
            "y": {"type": "number"},
        },
        "required": ["x", "y"],
        "additionalProperties": False,
    }


def topology_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "analysis_summary": {"type": "string"},
            "orientation_summary": {"type": "string"},
            "junctions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "junction_id": {"type": "string"},
                        "point": point_schema(),
                        "junction_type": {
                            "type": "string",
                            "enum": ["ridge_end", "valley_end", "hip_end", "intersection", "eave_contact", "uncertain"],
                        },
                        "confidence": {"type": "number"},
                    },
                    "required": ["junction_id", "point", "junction_type", "confidence"],
                    "additionalProperties": False,
                },
            },
            "boundaries": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "boundary_id": {"type": "string"},
                        "boundary_type": {
                            "type": "string",
                            "enum": ["ridge", "valley", "hip", "step", "eave", "uncertain"],
                        },
                        "points": {
                            "type": "array",
                            "items": point_schema(),
                        },
                        "plane_ids": {
                            "type": "array",
                            "items": {"type": "integer"},
                        },
                        "confidence": {"type": "number"},
                        "reason": {"type": "string"},
                    },
                    "required": [
                        "boundary_id",
                        "boundary_type",
                        "points",
                        "plane_ids",
                        "confidence",
                        "reason",
                    ],
                    "additionalProperties": False,
                },
            },
            "planes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "plane_id": {"type": "integer"},
                        "name": {"type": "string"},
                        "seed": point_schema(),
                        "outline": {
                            "type": "array",
                            "items": point_schema(),
                        },
                        "slope_direction_degrees": {"type": "number"},
                        "adjacent_plane_ids": {
                            "type": "array",
                            "items": {"type": "integer"},
                        },
                        "confidence": {"type": "number"},
                        "reason": {"type": "string"},
                    },
                    "required": [
                        "plane_id",
                        "name",
                        "seed",
                        "outline",
                        "slope_direction_degrees",
                        "adjacent_plane_ids",
                        "confidence",
                        "reason",
                    ],
                    "additionalProperties": False,
                },
            },
            "uncertainties": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "description": {"type": "string"},
                        "region": {
                            "type": "array",
                            "items": point_schema(),
                        },
                        "severity": {
                            "type": "string",
                            "enum": ["low", "medium", "high"],
                        },
                    },
                    "required": ["description", "region", "severity"],
                    "additionalProperties": False,
                },
            },
        },
        "required": [
            "analysis_summary",
            "orientation_summary",
            "junctions",
            "boundaries",
            "planes",
            "uncertainties",
        ],
        "additionalProperties": False,
    }


def first_pass_prompt() -> str:
    return """
Infer the complete physical roof-plane topology from scratch. Do not reuse,
select, merge, or refer to any precomputed candidate regions: none are provided.

The images are, in order: (1) the original overhead RGB image, (2) the exterior
building mask, (3) a local structural-boundary evidence map, (4) relative
monocular depth, and (5) estimated surface normals. Depth is relative, not
metric. The mask defines only the building exterior and does not define internal
planes.

Identify every visible planar roof face. Propose one interior seed and a closed
clockwise outline for each plane. Build the full internal boundary graph using
ridges, valleys, hips, steps, and relevant eave contacts. Represent curved or
piecewise boundaries with multiple points. Include junctions and adjacency.
Coordinates are normalized to the full image: x=0 is left, y=0 is top. Keep all
coordinates within [0,1]. Plane IDs must be unique positive integers and every
boundary plane ID and adjacency ID must refer to a returned plane. Each seed
must lie visibly inside its plane. Do not force a rectangular decomposition.

Use architectural evidence, roof shading, texture continuity, relative depth,
surface-normal changes, ridge/valley convergence, and drainage geometry. Ignore
trees, shadows, vents, chimneys, skylights, and image seams as plane boundaries
unless they coincide with a true roof break. Return your best complete topology,
including uncertain geometry rather than omitting a visible roof plane.
""".strip()


def review_prompt(draft: dict[str, Any]) -> str:
    return f"""
Act as a critical roof-topology reviewer. The images are: (1) original RGB,
(2) the current proposed topology overlay, (3) exterior building mask,
(4) structural-boundary evidence, (5) relative depth, and (6) surface normals.

Audit the draft for missing planes, false planes caused by shadows or objects,
incorrect ridge/valley/hip geometry, open boundaries, seeds outside their plane,
impossible junctions, and inconsistent adjacency. Then return a COMPLETE revised
topology in the same schema. Do not return a patch. You may freely redraw,
replace, add, or remove every plane, boundary, junction, seed, and outline. The
result must be your best independent architectural interpretation.

Draft topology:
{json.dumps(draft, separators=(",", ":"))}
""".strip()


def request_topology(
    client: OpenAI,
    *,
    model: str,
    detail: str,
    reasoning_effort: str,
    pro: bool,
    prompt: str,
    images: list[Image.Image],
) -> tuple[dict[str, Any], dict[str, Any]]:
    content: list[dict[str, Any]] = [{"type": "input_text", "text": prompt}]
    content.extend(
        {
            "type": "input_image",
            "image_url": image_data_url(image),
            "detail": detail,
        }
        for image in images
    )
    reasoning: dict[str, str] = {"effort": reasoning_effort}
    if pro:
        reasoning["mode"] = "pro"

    request = {
        "model": model,
        "reasoning": reasoning,
        "store": False,
        "input": [{"role": "user", "content": content}],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "complete_roof_topology",
                "strict": True,
                "schema": topology_schema(),
            }
        },
        "max_output_tokens": 48000,
    }
    pro_retry = False
    try:
        response = client.responses.create(**request)
    except Exception as exc:
        message = str(exc).lower()
        if not pro or ("pro" not in message and "reasoning.mode" not in message):
            raise
        pro_retry = True
        request["reasoning"] = {"effort": reasoning_effort}
        response = client.responses.create(**request)

    if not response.output_text:
        incomplete = getattr(response, "incomplete_details", None)
        incomplete_data = (
            incomplete.model_dump() if hasattr(incomplete, "model_dump") else incomplete
        )
        raise RuntimeError(
            "The multimodal model returned no structured topology. "
            f"status={response.status!r}; incomplete_details={incomplete_data!r}; "
            f"output_items={len(response.output)}"
        )
    parsed = json.loads(response.output_text)
    metadata = {
        "response_id": response.id,
        "model": model,
        "status": response.status,
        "requested_reasoning_effort": reasoning_effort,
        "requested_pro_mode": pro,
        "retried_without_pro_mode": pro_retry,
        "usage": response.usage.model_dump() if response.usage else None,
    }
    return parsed, metadata


def normalized_point(point: dict[str, float], width: int, height: int) -> tuple[int, int]:
    x = int(round(float(np.clip(point["x"], 0.0, 1.0)) * (width - 1)))
    y = int(round(float(np.clip(point["y"], 0.0, 1.0)) * (height - 1)))
    return x, y


def make_mask_overlay(image: Image.Image, mask: np.ndarray) -> Image.Image:
    rgb = np.asarray(image.convert("RGB")).copy()
    dark = (rgb.astype(np.float32) * 0.22).astype(np.uint8)
    dark[mask] = np.clip(
        0.55 * rgb[mask].astype(np.float32) + np.array([60, 175, 255]) * 0.45,
        0,
        255,
    ).astype(np.uint8)
    boundary = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((5, 5), np.uint8))
    dark[boundary > 0] = [255, 255, 255]
    return Image.fromarray(dark)


def create_evidence_map(
    image: Image.Image,
    structural: Image.Image,
    depth_edges: Image.Image | None,
    normal_edges: Image.Image | None,
    mask: np.ndarray,
) -> np.ndarray:
    gray = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    rgb_edge = cv2.magnitude(gx, gy)

    def normalized(values: np.ndarray) -> np.ndarray:
        values = values.astype(np.float32)
        sample = values[mask]
        high = float(np.percentile(sample, 98)) if sample.size else 1.0
        return np.clip(values / max(high, 1e-6), 0.0, 1.0)

    structural_gray = np.asarray(structural.convert("L"))
    components = [0.38 * normalized(structural_gray), 0.32 * normalized(rgb_edge)]
    if depth_edges is not None:
        components.append(0.14 * normalized(np.asarray(depth_edges.convert("L"))))
    if normal_edges is not None:
        components.append(0.16 * normalized(np.asarray(normal_edges.convert("L"))))
    evidence = np.clip(sum(components), 0.0, 1.0)
    evidence = cv2.GaussianBlur(evidence, (3, 3), 0)
    evidence[~ndimage.binary_dilation(mask, iterations=4)] = 0.0
    return evidence


def draw_topology(
    image: Image.Image,
    plan: dict[str, Any],
    *,
    snapped_paths: dict[str, list[tuple[int, int]]] | None = None,
) -> Image.Image:
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas, "RGBA")
    font = ImageFont.load_default(size=16)
    width, height = canvas.size
    boundary_colors = {
        "ridge": (255, 35, 90, 255),
        "valley": (0, 215, 255, 255),
        "hip": (255, 205, 20, 255),
        "step": (190, 80, 255, 255),
        "eave": (45, 255, 125, 255),
        "uncertain": (255, 255, 255, 255),
    }
    for plane in plan["planes"]:
        outline = [normalized_point(point, width, height) for point in plane["outline"]]
        if len(outline) >= 3:
            draw.line(outline + [outline[0]], fill=(255, 255, 255, 115), width=2)
        seed = normalized_point(plane["seed"], width, height)
        draw.ellipse((seed[0] - 7, seed[1] - 7, seed[0] + 7, seed[1] + 7), fill=(0, 0, 0, 220), outline=(255, 255, 255, 255), width=2)
        draw.text((seed[0] + 10, seed[1] - 9), f"P{plane['plane_id']}", font=font, fill=(255, 255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0, 255))
    for boundary in plan["boundaries"]:
        points = (
            snapped_paths.get(boundary["boundary_id"], [])
            if snapped_paths is not None
            else [normalized_point(point, width, height) for point in boundary["points"]]
        )
        if len(points) >= 2:
            draw.line(points, fill=boundary_colors[boundary["boundary_type"]], width=5, joint="curve")
    return canvas


def rasterize_raw_polygons(plan: dict[str, Any], mask: np.ndarray) -> np.ndarray:
    height, width = mask.shape
    labels = np.zeros(mask.shape, dtype=np.int32)
    planes = sorted(plan["planes"], key=lambda item: float(item["confidence"]))
    for plane in planes:
        points = np.asarray(
            [normalized_point(point, width, height) for point in plane["outline"]],
            dtype=np.int32,
        )
        if len(points) < 3:
            continue
        polygon = np.zeros(mask.shape, dtype=np.uint8)
        cv2.fillPoly(polygon, [points], 1)
        labels[(polygon > 0) & mask] = int(plane["plane_id"])
    return labels


def nearest_valid_pixel(
    point: tuple[int, int],
    valid: np.ndarray,
) -> tuple[int, int]:
    x, y = point
    if valid[y, x]:
        return x, y
    ys, xs = np.where(valid)
    if xs.size == 0:
        raise ValueError("No valid roof pixels are available")
    index = int(np.argmin((xs - x) ** 2 + (ys - y) ** 2))
    return int(xs[index]), int(ys[index])


def route_segment(
    start: tuple[int, int],
    end: tuple[int, int],
    evidence: np.ndarray,
    allowed: np.ndarray,
    margin: int,
) -> list[tuple[int, int]]:
    height, width = evidence.shape
    x1, y1 = start
    x2, y2 = end
    left = max(0, min(x1, x2) - margin)
    right = min(width, max(x1, x2) + margin + 1)
    top = max(0, min(y1, y2) - margin)
    bottom = min(height, max(y1, y2) + margin + 1)
    local_evidence = evidence[top:bottom, left:right]
    local_allowed = allowed[top:bottom, left:right]
    yy, xx = np.mgrid[top:bottom, left:right]
    dx = float(x2 - x1)
    dy = float(y2 - y1)
    denominator = max(dx * dx + dy * dy, 1.0)
    t = np.clip(((xx - x1) * dx + (yy - y1) * dy) / denominator, 0.0, 1.0)
    projection_x = x1 + t * dx
    projection_y = y1 + t * dy
    distance = np.hypot(xx - projection_x, yy - projection_y)
    cost = 1.0 + 14.0 * (1.0 - local_evidence) + 5.0 * np.clip(distance / max(margin, 1), 0.0, 2.0)
    cost[~local_allowed] += 35.0
    local_start = (y1 - top, x1 - left)
    local_end = (y2 - top, x2 - left)
    path, _ = route_through_array(cost, local_start, local_end, fully_connected=True)
    return [(int(column + left), int(row + top)) for row, column in path]


def snap_boundaries(
    plan: dict[str, Any],
    evidence: np.ndarray,
    mask: np.ndarray,
    margin: int,
) -> dict[str, list[tuple[int, int]]]:
    height, width = mask.shape
    allowed = ndimage.binary_dilation(mask, iterations=5)
    result: dict[str, list[tuple[int, int]]] = {}
    for boundary in plan["boundaries"]:
        proposed = [normalized_point(point, width, height) for point in boundary["points"]]
        if len(proposed) < 2:
            continue
        snapped: list[tuple[int, int]] = []
        for start, end in zip(proposed, proposed[1:]):
            start = nearest_valid_pixel(start, allowed)
            end = nearest_valid_pixel(end, allowed)
            segment = route_segment(start, end, evidence, allowed, margin)
            if snapped and segment and snapped[-1] == segment[0]:
                segment = segment[1:]
            snapped.extend(segment)
        result[boundary["boundary_id"]] = snapped
    return result


def build_markers(plan: dict[str, Any], mask: np.ndarray, barrier: np.ndarray) -> tuple[np.ndarray, dict[int, int]]:
    height, width = mask.shape
    markers = np.zeros(mask.shape, dtype=np.int32)
    label_to_plane: dict[int, int] = {}
    valid = mask & (barrier < 0.92)
    occupied = np.zeros(mask.shape, dtype=bool)
    for label, plane in enumerate(plan["planes"], start=1):
        point = normalized_point(plane["seed"], width, height)
        point = nearest_valid_pixel(point, valid & ~occupied if np.any(valid & ~occupied) else valid)
        cv2.circle(markers, point, 4, label, thickness=-1)
        marker_mask = markers == label
        marker_mask &= mask
        markers[markers == label] = 0
        markers[marker_mask] = label
        occupied |= ndimage.binary_dilation(marker_mask, iterations=8)
        label_to_plane[label] = int(plane["plane_id"])
    return markers, label_to_plane


def partition_roof(
    plan: dict[str, Any],
    mask: np.ndarray,
    evidence: np.ndarray,
    snapped_paths: dict[str, list[tuple[int, int]]],
) -> tuple[np.ndarray, np.ndarray, dict[int, int]]:
    barrier = np.clip(0.72 * evidence, 0.0, 1.0)
    path_mask = np.zeros(mask.shape, dtype=np.uint8)
    for path in snapped_paths.values():
        if len(path) >= 2:
            cv2.polylines(path_mask, [np.asarray(path, dtype=np.int32)], False, 255, 3, cv2.LINE_AA)
    barrier = np.maximum(barrier, path_mask.astype(np.float32) / 255.0)
    exterior = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    barrier[exterior > 0] = 1.0
    markers, label_to_plane = build_markers(plan, mask, barrier)
    if len(label_to_plane) < 2:
        raise ValueError("The proposed topology must contain at least two valid planes")
    labels = watershed(barrier, markers=markers, mask=mask, connectivity=2)
    missing = mask & (labels == 0)
    if np.any(missing):
        component_count, components = cv2.connectedComponents(
            missing.astype(np.uint8), 8
        )
        for component_id in range(1, component_count):
            component = components == component_id
            ring = ndimage.binary_dilation(component, iterations=1) & (labels > 0)
            neighbors, counts = np.unique(labels[ring], return_counts=True)
            if neighbors.size:
                labels[component] = int(neighbors[np.argmax(counts)])
    labels, _, _ = relabel_sequential(labels.astype(np.int32))
    return labels.astype(np.int32), barrier, label_to_plane


def realized_adjacency(labels: np.ndarray, label_to_plane: dict[int, int]) -> list[list[int]]:
    pairs: set[tuple[int, int]] = set()
    for first, second in ((labels[:, :-1], labels[:, 1:]), (labels[:-1, :], labels[1:, :])):
        different = (first != second) & (first > 0) & (second > 0)
        for left, right in zip(first[different], second[different]):
            a = label_to_plane.get(int(left), int(left))
            b = label_to_plane.get(int(right), int(right))
            pairs.add(tuple(sorted((a, b))))
    return [list(pair) for pair in sorted(pairs)]


def topology_validation(
    plan: dict[str, Any],
    labels: np.ndarray,
    mask: np.ndarray,
    label_to_plane: dict[int, int],
) -> dict[str, Any]:
    roof_pixels = int(np.count_nonzero(mask))
    covered_pixels = int(np.count_nonzero(labels))
    expected = {
        tuple(sorted((int(plane["plane_id"]), int(adjacent))))
        for plane in plan["planes"]
        for adjacent in plane["adjacent_plane_ids"]
        if int(adjacent) != int(plane["plane_id"])
    }
    realized = {tuple(pair) for pair in realized_adjacency(labels, label_to_plane)}
    disconnected: list[int] = []
    for label in (int(value) for value in np.unique(labels) if value):
        count, _ = cv2.connectedComponents((labels == label).astype(np.uint8), 8)
        if count > 2:
            disconnected.append(label_to_plane.get(label, label))
    return {
        "roof_pixels": roof_pixels,
        "covered_pixels": covered_pixels,
        "coverage_percentage": round(100.0 * covered_pixels / max(roof_pixels, 1), 4),
        "proposed_plane_count": len(plan["planes"]),
        "raster_plane_count": len([value for value in np.unique(labels) if value]),
        "expected_adjacencies": [list(pair) for pair in sorted(expected)],
        "realized_adjacencies": [list(pair) for pair in sorted(realized)],
        "missing_expected_adjacencies": [list(pair) for pair in sorted(expected - realized)],
        "unexpected_adjacencies": [list(pair) for pair in sorted(realized - expected)],
        "disconnected_plane_ids": sorted(disconnected),
        "overlap_pixels": 0,
    }


def save_scalar(values: np.ndarray, path: Path) -> None:
    image = np.clip(values * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(image).save(path)


def run(args: argparse.Namespace) -> Path:
    load_dotenv(PROJECT_ROOT / ".env")
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not defined in .env or the environment")

    output_root = args.output_root.resolve()
    source_run = args.source_run.resolve() if args.source_run else find_latest_structural_run(output_root)
    run_directory = create_run_directory(output_root, EXPERIMENT_NAME)

    image = Image.open(source_run / "input.png").convert("RGB")
    mask = load_mask(source_run / "building_mask.png")
    mask_overlay = make_mask_overlay(image, mask)
    structural = Image.open(source_run / "structural_boundary_map.png").convert("RGB")
    depth = Image.open(source_run / "relative_depth_color.png").convert("RGB")
    normals = Image.open(source_run / "surface_normals.png").convert("RGB")
    depth_edges = Image.open(source_run / "depth_edges.png").convert("L") if (source_run / "depth_edges.png").is_file() else None
    normal_edges = Image.open(source_run / "normal_edges.png").convert("L") if (source_run / "normal_edges.png").is_file() else None
    evidence = create_evidence_map(image, structural, depth_edges, normal_edges, mask)

    image.save(run_directory / "input.png")
    Image.fromarray((mask * 255).astype(np.uint8)).save(run_directory / "building_mask.png")
    mask_overlay.save(run_directory / "building_mask_overlay.png")
    structural.save(run_directory / "structural_boundary_map.png")
    depth.save(run_directory / "relative_depth_color.png")
    normals.save(run_directory / "surface_normals.png")
    save_scalar(evidence, run_directory / "combined_boundary_evidence.png")

    request_summary = {
        "experiment": EXPERIMENT_NAME,
        "source_run": str(source_run),
        "model": args.model,
        "detail": args.detail,
        "reasoning_effort": args.reasoning_effort,
        "pro_mode": args.pro,
        "review_passes": args.review_passes,
        "candidate_regions_supplied": False,
        "manual_reference_supplied_to_model": False,
        "first_pass_prompt": first_pass_prompt(),
    }
    (run_directory / "vlm_request_summary.json").write_text(json.dumps(request_summary, indent=2), encoding="utf-8")

    client = OpenAI()
    print(f"Requesting an unconstrained roof topology from {args.model}...")
    draft, draft_metadata = request_topology(
        client,
        model=args.model,
        detail=args.detail,
        reasoning_effort=args.reasoning_effort,
        pro=args.pro,
        prompt=first_pass_prompt(),
        images=[image, mask_overlay, structural, depth, normals],
    )
    (run_directory / "vlm_topology_draft.json").write_text(json.dumps(draft, indent=2), encoding="utf-8")
    draft_overlay = draw_topology(image, draft)
    draft_overlay.save(run_directory / "vlm_topology_draft_overlay.png")

    final_plan = draft
    review_metadata: dict[str, Any] | None = None
    if args.review_passes:
        print("Running an independent topology critique and full revision...")
        final_plan, review_metadata = request_topology(
            client,
            model=args.model,
            detail=args.detail,
            reasoning_effort=args.reasoning_effort,
            pro=args.pro,
            prompt=review_prompt(draft),
            images=[image, draft_overlay, mask_overlay, structural, depth, normals],
        )
    (run_directory / "vlm_topology_final.json").write_text(json.dumps(final_plan, indent=2), encoding="utf-8")

    raw_labels = rasterize_raw_polygons(final_plan, mask)
    raw_colors, raw_overlay, _ = render_planes(np.asarray(image), raw_labels)
    save_label_mask(raw_labels, run_directory / "vlm_raw_polygon_labels.png")
    Image.fromarray(raw_colors).save(run_directory / "vlm_raw_polygon_color.png")
    Image.fromarray(raw_overlay).save(run_directory / "vlm_raw_polygon_overlay.png")

    snapped_paths = snap_boundaries(final_plan, evidence, mask, args.snap_margin)
    raw_graph = draw_topology(image, final_plan)
    snapped_graph = draw_topology(image, final_plan, snapped_paths=snapped_paths)
    raw_graph.save(run_directory / "vlm_topology_raw_graph.png")
    snapped_graph.save(run_directory / "vlm_topology_snapped_graph.png")

    labels, barrier, label_to_plane = partition_roof(final_plan, mask, evidence, snapped_paths)
    colors, overlay, _ = render_planes(np.asarray(image), labels)
    summaries = summarize_planes(labels)
    save_label_mask(labels, run_directory / "topology_roof_planes_labels.png")
    Image.fromarray(colors).save(run_directory / "topology_roof_planes_color.png")
    Image.fromarray(overlay).save(run_directory / "topology_roof_planes_overlay.png")
    save_scalar(barrier, run_directory / "topology_watershed_barrier.png")
    write_geojson(labels, summaries, run_directory / "topology_roof_planes.geojson")

    validation = topology_validation(final_plan, labels, mask, label_to_plane)
    comparison_items = [
        ("AI draft topology", draft_overlay),
        ("AI final topology", raw_graph),
        ("Evidence-snapped graph", snapped_graph),
        ("Final local partition", Image.fromarray(overlay)),
    ]
    manual_reference = source_run / "manual_reference.png"
    if manual_reference.is_file():
        reference = Image.open(manual_reference).convert("RGB")
        reference.save(run_directory / "manual_reference.png")
        comparison_items.append(("Manual qualitative reference", reference))
    make_contact_sheet(comparison_items, run_directory / "qualitative_comparison.jpg")

    result = {
        "experiment": EXPERIMENT_NAME,
        "purpose": "Unconstrained VLM proposal of complete roof-plane topology followed by local evidence snapping and raster partitioning",
        "source_run": str(source_run),
        "candidate_regions_supplied_to_model": False,
        "manual_reference_supplied_to_model": False,
        "draft_response": draft_metadata,
        "review_response": review_metadata,
        "draft_plane_count": len(draft["planes"]),
        "final_proposed_plane_count": len(final_plan["planes"]),
        "final_raster_plane_count": len(summaries),
        "label_to_proposed_plane": {str(key): value for key, value in label_to_plane.items()},
        "validation": validation,
        "planes": [asdict(summary) for summary in summaries],
    }
    (run_directory / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Created {len(summaries)} roof planes from a complete AI-proposed topology")
    print(f"Coverage: {validation['coverage_percentage']:.2f}%")
    print(f"Output: {run_directory}")
    return run_directory


if __name__ == "__main__":
    run(parse_args())
