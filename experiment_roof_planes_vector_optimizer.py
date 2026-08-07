from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage

from experiment_roof_planes import (
    make_contact_sheet,
    render_planes,
    save_label_mask,
    summarize_planes,
    write_geojson,
)
from experiment_roof_planes_vlm_optimizer import (
    fit_all_planes,
    hypothesis_score,
    largest_connected_component,
)
from experiment_roof_planes_vlm_topology import (
    create_evidence_map,
    draw_topology,
    load_mask,
    partition_roof,
    save_scalar,
    snap_boundaries,
)
from run_artifacts import create_run_directory


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs"
EXPERIMENT_NAME = "roof_planes_experiment_06_vector_optimizer"
EXPERIMENT_05_SUFFIX = "_roof_planes_experiment_05_vlm_optimizer"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Experiment 06: optimize the selected Experiment 05 roof graph with "
            "shared vector boundaries, straight-line evidence, and junction consolidation."
        )
    )
    parser.add_argument("--source-run", type=Path)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--snap-margin", type=int, default=42)
    return parser.parse_args()


def find_latest_optimizer_run(output_root: Path) -> Path:
    candidates = sorted(
        (
            path
            for path in output_root.iterdir()
            if path.is_dir()
            and path.name.endswith(EXPERIMENT_05_SUFFIX)
            and (path / "result.json").is_file()
            and (path / "winner_roof_planes_labels.png").is_file()
        ),
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            "No completed Experiment 05 output was found. Run "
            "experiment_roof_planes_vlm_optimizer.py first."
        )
    return candidates[0]


def load_relative_depth(path: Path) -> np.ndarray:
    depth = np.asarray(Image.open(path), dtype=np.float32)
    if depth.ndim == 3:
        depth = depth[..., 0]
    maximum = float(depth.max())
    return depth / maximum if maximum > 0 else depth


def detect_structural_lines(
    evidence: np.ndarray,
    mask: np.ndarray,
) -> list[tuple[float, float, float, float]]:
    values = evidence[mask]
    threshold = float(np.percentile(values, 72.0)) if values.size else 0.5
    binary = ((evidence >= threshold) & ndimage.binary_dilation(mask, iterations=3)).astype(np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    lines = cv2.HoughLinesP(
        binary * 255,
        rho=1,
        theta=np.pi / 360.0,
        threshold=24,
        minLineLength=22,
        maxLineGap=12,
    )
    if lines is None:
        return []
    result = [
        tuple(float(value) for value in row)
        for row in np.asarray(lines).reshape(-1, 4)
    ]
    return sorted(
        result,
        key=lambda line: -math.hypot(line[2] - line[0], line[3] - line[1]),
    )[:240]


def orientation_difference(first: float, second: float) -> float:
    difference = abs(first - second) % math.pi
    return min(difference, math.pi - difference)


def project_point_to_line(
    point: np.ndarray,
    line: tuple[float, float, float, float],
) -> np.ndarray:
    start = np.asarray(line[:2], dtype=np.float64)
    end = np.asarray(line[2:], dtype=np.float64)
    direction = end - start
    denominator = float(np.dot(direction, direction))
    if denominator <= 1e-8:
        return point.copy()
    parameter = float(np.dot(point - start, direction) / denominator)
    return start + parameter * direction


def nearest_line(
    start: np.ndarray,
    end: np.ndarray,
    lines: list[tuple[float, float, float, float]],
    *,
    max_angle_degrees: float,
    max_distance: float,
) -> tuple[float, float, float, float] | None:
    direction = end - start
    if float(np.linalg.norm(direction)) < 3.0:
        return None
    angle = math.atan2(float(direction[1]), float(direction[0])) % math.pi
    midpoint = (start + end) / 2.0
    best: tuple[float, tuple[float, float, float, float]] | None = None
    for line in lines:
        line_direction = np.asarray(line[2:], dtype=np.float64) - np.asarray(line[:2], dtype=np.float64)
        line_angle = math.atan2(float(line_direction[1]), float(line_direction[0])) % math.pi
        angle_difference = math.degrees(orientation_difference(angle, line_angle))
        if angle_difference > max_angle_degrees:
            continue
        distance = float(np.linalg.norm(project_point_to_line(midpoint, line) - midpoint))
        if distance > max_distance:
            continue
        quality = angle_difference / max(max_angle_degrees, 1.0) + distance / max(max_distance, 1.0)
        if best is None or quality < best[0]:
            best = (quality, line)
    return best[1] if best else None


def simplify_path(path: list[tuple[int, int]], epsilon: float) -> np.ndarray:
    points = np.asarray(path, dtype=np.float32)
    if len(points) <= 2:
        return points
    simplified = cv2.approxPolyDP(points.reshape(-1, 1, 2), epsilon, False).reshape(-1, 2)
    if len(simplified) < 2:
        return np.vstack((points[0], points[-1]))
    return simplified.astype(np.float64)


def align_path_to_lines(
    points: np.ndarray,
    lines: list[tuple[float, float, float, float]],
    *,
    angle_degrees: float,
    max_distance: float,
    strength: float,
) -> np.ndarray:
    if len(points) < 2 or not lines or strength <= 0:
        return points.copy()
    proposals: list[list[np.ndarray]] = [[] for _ in range(len(points))]
    for index in range(len(points) - 1):
        line = nearest_line(
            points[index],
            points[index + 1],
            lines,
            max_angle_degrees=angle_degrees,
            max_distance=max_distance,
        )
        if line is None:
            continue
        proposals[index].append(project_point_to_line(points[index], line))
        proposals[index + 1].append(project_point_to_line(points[index + 1], line))
    aligned = points.copy()
    for index, values in enumerate(proposals):
        if values:
            target = np.mean(np.asarray(values), axis=0)
            aligned[index] = (1.0 - strength) * points[index] + strength * target
    return aligned


def best_evidence_point(
    center: np.ndarray,
    evidence: np.ndarray,
    allowed: np.ndarray,
    radius: int,
) -> np.ndarray:
    height, width = evidence.shape
    x = int(round(float(center[0])))
    y = int(round(float(center[1])))
    left, right = max(0, x - radius), min(width, x + radius + 1)
    top, bottom = max(0, y - radius), min(height, y + radius + 1)
    local_allowed = allowed[top:bottom, left:right]
    if not np.any(local_allowed):
        return np.asarray([np.clip(x, 0, width - 1), np.clip(y, 0, height - 1)], dtype=np.float64)
    yy, xx = np.mgrid[top:bottom, left:right]
    distance = np.hypot(xx - center[0], yy - center[1])
    quality = evidence[top:bottom, left:right] - 0.025 * distance
    quality[~local_allowed] = -np.inf
    row, column = np.unravel_index(int(np.argmax(quality)), quality.shape)
    return np.asarray([column + left, row + top], dtype=np.float64)


def consolidate_endpoints(
    paths: dict[str, np.ndarray],
    evidence: np.ndarray,
    mask: np.ndarray,
    radius: float,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    references: list[tuple[str, int]] = []
    coordinates: list[np.ndarray] = []
    for boundary_id, points in paths.items():
        if len(points) < 2:
            continue
        references.extend(((boundary_id, 0), (boundary_id, len(points) - 1)))
        coordinates.extend((points[0].copy(), points[-1].copy()))
    parent = list(range(len(coordinates)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first: int, second: int) -> None:
        root_first, root_second = find(first), find(second)
        if root_first != root_second:
            parent[root_second] = root_first

    for first in range(len(coordinates)):
        for second in range(first + 1, len(coordinates)):
            if float(np.linalg.norm(coordinates[first] - coordinates[second])) <= radius:
                union(first, second)

    groups: dict[int, list[int]] = {}
    for index in range(len(coordinates)):
        groups.setdefault(find(index), []).append(index)
    result = {key: value.copy() for key, value in paths.items()}
    records: list[dict[str, Any]] = []
    allowed = ndimage.binary_dilation(mask, iterations=5)
    for indices in groups.values():
        if len(indices) < 2:
            continue
        center = np.mean(np.asarray([coordinates[index] for index in indices]), axis=0)
        selected = best_evidence_point(center, evidence, allowed, max(3, int(round(radius / 2))))
        members: list[dict[str, Any]] = []
        for index in indices:
            boundary_id, point_index = references[index]
            result[boundary_id][point_index] = selected
            members.append({"boundary_id": boundary_id, "endpoint": "start" if point_index == 0 else "end"})
        records.append(
            {
                "coordinate": [round(float(selected[0]), 3), round(float(selected[1]), 3)],
                "members": members,
            }
        )
    return result, records


def finalize_paths(
    paths: dict[str, np.ndarray],
    mask: np.ndarray,
) -> dict[str, list[tuple[int, int]]]:
    allowed = ndimage.binary_dilation(mask, iterations=5)
    height, width = mask.shape
    ys, xs = np.where(allowed)
    result: dict[str, list[tuple[int, int]]] = {}
    for boundary_id, points in paths.items():
        finalized: list[tuple[int, int]] = []
        for point in points:
            x = int(np.clip(round(float(point[0])), 0, width - 1))
            y = int(np.clip(round(float(point[1])), 0, height - 1))
            if not allowed[y, x] and xs.size:
                index = int(np.argmin((xs - x) ** 2 + (ys - y) ** 2))
                x, y = int(xs[index]), int(ys[index])
            value = (x, y)
            if not finalized or finalized[-1] != value:
                finalized.append(value)
        if len(finalized) >= 2:
            result[boundary_id] = finalized
    return result


def optimize_paths(
    snapped_paths: dict[str, list[tuple[int, int]]],
    lines: list[tuple[float, float, float, float]],
    evidence: np.ndarray,
    mask: np.ndarray,
    settings: dict[str, float],
) -> tuple[dict[str, list[tuple[int, int]]], list[dict[str, Any]]]:
    paths: dict[str, np.ndarray] = {}
    for boundary_id, path in snapped_paths.items():
        points = simplify_path(path, settings["epsilon"])
        points = align_path_to_lines(
            points,
            lines,
            angle_degrees=settings["angle_degrees"],
            max_distance=settings["line_distance"],
            strength=settings["line_strength"],
        )
        paths[boundary_id] = points
    consolidated, junctions = consolidate_endpoints(
        paths,
        evidence,
        mask,
        settings["junction_radius"],
    )
    return finalize_paths(consolidated, mask), junctions


def rasterized_path_mask(
    paths: dict[str, list[tuple[int, int]]],
    shape: tuple[int, int],
) -> np.ndarray:
    result = np.zeros(shape, dtype=np.uint8)
    for path in paths.values():
        if len(path) >= 2:
            cv2.polylines(result, [np.asarray(path, dtype=np.int32)], False, 255, 1, cv2.LINE_AA)
    return result


def vector_metrics(
    paths: dict[str, list[tuple[int, int]]],
    original_paths: dict[str, list[tuple[int, int]]],
    evidence: np.ndarray,
) -> dict[str, float | int]:
    path_mask = rasterized_path_mask(paths, evidence.shape) > 0
    support = float(evidence[path_mask].mean()) if np.any(path_mask) else 0.0
    original_mask = rasterized_path_mask(original_paths, evidence.shape)
    distance = cv2.distanceTransform((original_mask == 0).astype(np.uint8), cv2.DIST_L2, 3)
    mean_drift = float(distance[path_mask].mean()) if np.any(path_mask) else 99.0
    original_vertices = sum(len(path) for path in original_paths.values())
    vertices = sum(len(path) for path in paths.values())
    minimum_vertices = 2 * max(len(paths), 1)
    reducible = max(original_vertices - minimum_vertices, 1)
    simplicity = float(np.clip((original_vertices - vertices) / reducible, 0.0, 1.0))
    drift_score = float(np.clip(1.0 - mean_drift / 16.0, 0.0, 1.0))
    return {
        "boundary_evidence_support": round(support, 8),
        "mean_drift_from_evidence_snapped_graph_pixels": round(mean_drift, 8),
        "drift_score": round(drift_score, 8),
        "path_count": len(paths),
        "vertex_count": vertices,
        "original_vertex_count": original_vertices,
        "simplicity_score": round(simplicity, 8),
    }


def serialize_paths(paths: dict[str, list[tuple[int, int]]]) -> dict[str, list[list[int]]]:
    return {
        boundary_id: [[int(x), int(y)] for x, y in points]
        for boundary_id, points in paths.items()
    }


def run(args: argparse.Namespace) -> Path:
    output_root = args.output_root.resolve()
    source_run = args.source_run.resolve() if args.source_run else find_latest_optimizer_run(output_root)
    source_result = json.loads((source_run / "result.json").read_text(encoding="utf-8"))
    winner_number = int(source_result["winner_hypothesis"])
    hypothesis_directory = source_run / f"hypothesis_{winner_number:02d}"
    hypothesis_result = json.loads((hypothesis_directory / "result.json").read_text(encoding="utf-8"))
    plan = json.loads((hypothesis_directory / "vlm_topology.json").read_text(encoding="utf-8"))
    structural_run = Path(source_result["source_run"])
    if not structural_run.is_absolute():
        structural_run = PROJECT_ROOT / structural_run

    image = Image.open(source_run / "input.png").convert("RGB")
    rgb = np.asarray(image)
    raw_mask = load_mask(structural_run / "building_mask.png")
    mask, mask_cleanup = largest_connected_component(raw_mask)
    structural = Image.open(structural_run / "structural_boundary_map.png").convert("RGB")
    depth_edges = Image.open(structural_run / "depth_edges.png").convert("L")
    normal_edges = Image.open(structural_run / "normal_edges.png").convert("L")
    relative_depth = load_relative_depth(structural_run / "relative_depth_16bit.png")
    evidence = create_evidence_map(image, structural, depth_edges, normal_edges, mask)
    snapped_paths = snap_boundaries(plan, evidence, mask, args.snap_margin)
    structural_lines = detect_structural_lines(evidence, mask)
    run_directory = create_run_directory(output_root, EXPERIMENT_NAME)

    image.save(run_directory / "input.png")
    Image.fromarray((mask * 255).astype(np.uint8)).save(run_directory / "building_mask.png")
    save_scalar(evidence, run_directory / "combined_boundary_evidence.png")
    baseline_labels = np.asarray(Image.open(source_run / "winner_roof_planes_labels.png"), dtype=np.int32)
    baseline_labels[~mask] = 0
    label_to_plane = {
        int(key): int(value)
        for key, value in hypothesis_result["label_to_proposed_plane"].items()
    }

    settings_by_name: dict[str, dict[str, float]] = {
        "conservative": {
            "epsilon": 3.0,
            "angle_degrees": 9.0,
            "line_distance": 8.0,
            "line_strength": 0.35,
            "junction_radius": 7.0,
        },
        "balanced": {
            "epsilon": 5.5,
            "angle_degrees": 14.0,
            "line_distance": 13.0,
            "line_strength": 0.62,
            "junction_radius": 10.0,
        },
        "structural": {
            "epsilon": 8.0,
            "angle_degrees": 19.0,
            "line_distance": 18.0,
            "line_strength": 0.82,
            "junction_radius": 14.0,
        },
    }

    candidates: list[dict[str, Any]] = []
    baseline_fits = fit_all_planes(baseline_labels, relative_depth)
    baseline_score, baseline_details = hypothesis_score(
        plan,
        baseline_labels,
        mask,
        label_to_plane,
        evidence,
        baseline_fits,
        [],
    )
    baseline_vector = vector_metrics(snapped_paths, snapped_paths, evidence)
    baseline_composite = (
        baseline_score
        + 0.04 * float(baseline_vector["boundary_evidence_support"])
        + 0.03 * float(baseline_vector["simplicity_score"])
        + 0.03 * float(baseline_vector["drift_score"])
    )
    _, baseline_overlay, _ = render_planes(rgb, baseline_labels)
    candidates.append(
        {
            "name": "experiment_05_hypothesis_2_baseline",
            "labels": baseline_labels,
            "paths": snapped_paths,
            "junctions": [],
            "base_score": baseline_score,
            "composite_score": baseline_composite,
            "score_details": baseline_details,
            "vector_metrics": baseline_vector,
            "overlay": Image.fromarray(baseline_overlay),
        }
    )

    for name, settings in settings_by_name.items():
        candidate_directory = run_directory / f"candidate_{name}"
        candidate_directory.mkdir(parents=True, exist_ok=True)
        paths, junctions = optimize_paths(
            snapped_paths,
            structural_lines,
            evidence,
            mask,
            settings,
        )
        labels, barrier, candidate_mapping = partition_roof(plan, mask, evidence, paths)
        fits = fit_all_planes(labels, relative_depth)
        base_score, score_details = hypothesis_score(
            plan,
            labels,
            mask,
            candidate_mapping,
            evidence,
            fits,
            [],
        )
        metrics = vector_metrics(paths, snapped_paths, evidence)
        composite = (
            base_score
            + 0.04 * float(metrics["boundary_evidence_support"])
            + 0.03 * float(metrics["simplicity_score"])
            + 0.03 * float(metrics["drift_score"])
        )
        colors, overlay, _ = render_planes(rgb, labels)
        graph = draw_topology(image, plan, snapped_paths=paths)
        summaries = summarize_planes(labels)
        save_label_mask(labels, candidate_directory / "roof_planes_labels.png")
        Image.fromarray(colors).save(candidate_directory / "roof_planes_color.png")
        Image.fromarray(overlay).save(candidate_directory / "roof_planes_overlay.png")
        graph.save(candidate_directory / "vector_graph_overlay.png")
        save_scalar(barrier, candidate_directory / "watershed_barrier.png")
        write_geojson(labels, summaries, candidate_directory / "roof_planes.geojson")
        candidate_result = {
            "name": name,
            "settings": settings,
            "base_score": round(float(base_score), 8),
            "composite_score": round(float(composite), 8),
            "score_details": score_details,
            "vector_metrics": metrics,
            "junction_clusters": junctions,
            "paths": serialize_paths(paths),
            "planes": [asdict(summary) for summary in summaries],
        }
        (candidate_directory / "result.json").write_text(
            json.dumps(candidate_result, indent=2), encoding="utf-8"
        )
        candidates.append(
            {
                "name": name,
                "labels": labels,
                "paths": paths,
                "junctions": junctions,
                "base_score": base_score,
                "composite_score": composite,
                "score_details": score_details,
                "vector_metrics": metrics,
                "overlay": Image.fromarray(overlay),
            }
        )

    winner = max(candidates, key=lambda item: float(item["composite_score"]))
    winner_labels = winner["labels"]
    winner_colors, winner_overlay, _ = render_planes(rgb, winner_labels)
    winner_graph = draw_topology(image, plan, snapped_paths=winner["paths"])
    winner_summaries = summarize_planes(winner_labels)
    save_label_mask(winner_labels, run_directory / "winner_roof_planes_labels.png")
    Image.fromarray(winner_colors).save(run_directory / "winner_roof_planes_color.png")
    Image.fromarray(winner_overlay).save(run_directory / "winner_roof_planes_overlay.png")
    winner_graph.save(run_directory / "winner_vector_graph_overlay.png")
    write_geojson(winner_labels, winner_summaries, run_directory / "winner_roof_planes.geojson")
    (run_directory / "winner_vector_graph.json").write_text(
        json.dumps(
            {
                "candidate": winner["name"],
                "paths": serialize_paths(winner["paths"]),
                "junction_clusters": winner["junctions"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    comparison = [
        (
            f"Experiment 05 hypothesis 2 | {baseline_composite:.3f}",
            candidates[0]["overlay"],
        ),
    ]
    comparison.extend(
        (
            f"Vector {candidate['name']} | {candidate['composite_score']:.3f}",
            candidate["overlay"],
        )
        for candidate in candidates[1:]
    )
    comparison.append((f"Selected: {winner['name']}", Image.fromarray(winner_overlay)))

    # The manual paint-over is intentionally accessed only after winner selection.
    # It is copied as a visual reference and never converted to labels or metrics.
    manual_reference_path = structural_run / "manual_reference.png"
    manual_visual_available = manual_reference_path.is_file()
    if manual_visual_available:
        manual = Image.open(manual_reference_path).convert("RGB")
        manual.save(run_directory / "manual_reference_visual_only.png")
        comparison.append(("Manual paint-over | visual reference only", manual))
    make_contact_sheet(comparison, run_directory / "qualitative_comparison.jpg")

    serializable_candidates = [
        {
            "name": candidate["name"],
            "base_score": round(float(candidate["base_score"]), 8),
            "composite_score": round(float(candidate["composite_score"]), 8),
            "score_details": candidate["score_details"],
            "vector_metrics": candidate["vector_metrics"],
        }
        for candidate in candidates
    ]
    result = {
        "experiment": EXPERIMENT_NAME,
        "purpose": (
            "Optimize the selected Experiment 05 hypothesis with simplified shared "
            "boundaries, structural-line alignment, and consolidated junctions."
        ),
        "source_run": str(source_run),
        "source_hypothesis": winner_number,
        "structural_source_run": str(structural_run),
        "building_mask_cleanup": mask_cleanup,
        "structural_line_count": len(structural_lines),
        "winner_candidate": winner["name"],
        "winner_base_score": round(float(winner["base_score"]), 8),
        "winner_composite_score": round(float(winner["composite_score"]), 8),
        "candidates": serializable_candidates,
        "manual_reference_available": manual_visual_available,
        "manual_reference_supplied_to_model": False,
        "manual_reference_used_for_optimization": False,
        "manual_reference_used_for_scoring": False,
        "manual_reference_used_for_winner_selection": False,
        "manual_reference_visual_comparison_only": True,
        "planes": [asdict(summary) for summary in winner_summaries],
    }
    (run_directory / "result.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(
        f"Selected {winner['name']} with composite score "
        f"{winner['composite_score']:.4f}"
    )
    print(f"Output: {run_directory}")
    return run_directory


if __name__ == "__main__":
    run(parse_args())
