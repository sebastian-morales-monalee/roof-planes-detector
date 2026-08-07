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
from skimage.segmentation import relabel_sequential, slic

from experiment_roof_planes import (
    make_contact_sheet,
    render_planes,
    save_label_mask,
    summarize_planes,
)
from experiment_roof_planes_google_solar_dsm import (
    aggregate_fit_metrics,
    fit_metric_planes,
    save_dsm_visualization,
)
from run_artifacts import create_run_directory


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs"
EXPERIMENT_NAME = "roof_planes_experiment_08_dsm_hypothesis_fusion"
EXPERIMENT_07_SUFFIX = "_roof_planes_experiment_07_google_solar_dsm"


class ExperimentError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate an independent roof-plane hypothesis from a registered "
            "Google Solar DSM and fuse it with the visual/vector topology."
        )
    )
    parser.add_argument(
        "--source-run",
        type=Path,
        help="Completed Experiment 07 directory. Defaults to the latest output.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dsm-superpixels", type=int, default=140)
    parser.add_argument("--dsm-compactness", type=float, default=0.09)
    parser.add_argument("--normal-angle-degrees", type=float, default=12.0)
    parser.add_argument("--boundary-step-meters", type=float, default=0.28)
    parser.add_argument("--minimum-region-percentage", type=float, default=0.45)
    parser.add_argument("--plane-residual-threshold-meters", type=float, default=0.18)
    parser.add_argument("--max-fit-samples", type=int, default=8000)
    parser.add_argument(
        "--manual-reference",
        type=Path,
        help="Optional paint-over used only in the qualitative comparison.",
    )
    return parser.parse_args()


def find_latest_solar_run(output_root: Path) -> Path:
    candidates: list[Path] = []
    for result_path in output_root.rglob("result.json"):
        directory = result_path.parent
        if not directory.name.endswith(EXPERIMENT_07_SUFFIX):
            continue
        required = (
            directory / "google_solar_dsm_registered.tif",
            directory / "vector_structural_labels.png",
            directory / "building_mask.png",
        )
        if all(path.is_file() for path in required):
            candidates.append(directory)
    if not candidates:
        raise FileNotFoundError(
            "No completed Experiment 07 run was found. Run "
            "experiment_roof_planes_google_solar_dsm.py first."
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def robust_unit(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    selected = values[mask & np.isfinite(values)]
    result = np.zeros(values.shape, dtype=np.float32)
    if selected.size == 0:
        return result
    low, high = np.percentile(selected, (2.0, 98.0))
    if float(high - low) <= 1e-8:
        return result
    result = np.clip((values - low) / float(high - low), 0.0, 1.0)
    result[~mask] = 0.0
    return result.astype(np.float32)


def fill_invalid(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    if not np.any(valid):
        raise ExperimentError("The registered Google Solar DSM contains no valid pixels")
    _, indices = ndimage.distance_transform_edt(~valid, return_indices=True)
    return values[tuple(indices)].astype(np.float32)


def load_registered_dsm(
    source_run: Path,
) -> tuple[np.ndarray, np.ndarray, Any, str | None]:
    try:
        import rasterio
    except ImportError as error:
        raise ExperimentError("Rasterio is required to read the registered DSM") from error

    path = source_run / "google_solar_dsm_registered.tif"
    with rasterio.open(path) as source:
        dsm = source.read(1).astype(np.float32)
        valid = source.read_masks(1) > 0
        if source.nodata is not None:
            valid &= ~np.isclose(dsm, float(source.nodata))
        valid &= np.isfinite(dsm)
        transform = source.transform
        crs = source.crs.to_string() if source.crs is not None else None
    return dsm, valid, transform, crs


def world_coordinates(shape: tuple[int, int], transform: Any) -> tuple[np.ndarray, np.ndarray]:
    rows, columns = np.indices(shape, dtype=np.float64)
    x = transform.a * columns + transform.b * rows + transform.c
    y = transform.d * columns + transform.e * rows + transform.f
    return x, y


def metric_gradients(dsm: np.ndarray, transform: Any) -> tuple[np.ndarray, np.ndarray]:
    row_gradient, column_gradient = np.gradient(dsm.astype(np.float64))
    jacobian = np.asarray(
        [[float(transform.a), float(transform.b)], [float(transform.d), float(transform.e)]],
        dtype=np.float64,
    )
    if abs(float(np.linalg.det(jacobian))) <= 1e-10:
        raise ExperimentError("The registered DSM has a singular affine transform")
    inverse_transpose = np.linalg.inv(jacobian.T)
    slope_x = inverse_transpose[0, 0] * column_gradient + inverse_transpose[0, 1] * row_gradient
    slope_y = inverse_transpose[1, 0] * column_gradient + inverse_transpose[1, 1] * row_gradient
    return slope_x.astype(np.float32), slope_y.astype(np.float32)


def dsm_evidence(
    dsm: np.ndarray,
    valid: np.ndarray,
    roof_mask: np.ndarray,
    transform: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    analysis_mask = roof_mask & valid
    filled = fill_invalid(dsm, valid)
    smoothed = cv2.GaussianBlur(filled, (0, 0), 1.15)
    slope_x, slope_y = metric_gradients(smoothed, transform)
    normal_change = np.hypot(
        cv2.Sobel(slope_x, cv2.CV_32F, 1, 0, ksize=3)
        + cv2.Sobel(slope_y, cv2.CV_32F, 1, 0, ksize=3),
        cv2.Sobel(slope_x, cv2.CV_32F, 0, 1, ksize=3)
        + cv2.Sobel(slope_y, cv2.CV_32F, 0, 1, ksize=3),
    )
    curvature = np.abs(cv2.Laplacian(smoothed, cv2.CV_32F, ksize=3))
    evidence = 0.72 * robust_unit(normal_change, analysis_mask) + 0.28 * robust_unit(
        curvature, analysis_mask
    )
    evidence[~roof_mask] = 0.0
    return smoothed, slope_x, slope_y, evidence.astype(np.float32)


def complete_mask(labels: np.ndarray, roof_mask: np.ndarray) -> np.ndarray:
    labels = labels.astype(np.int32)
    labels[~roof_mask] = 0
    missing = roof_mask & (labels == 0)
    if np.any(missing):
        known = labels > 0
        if not np.any(known):
            raise ExperimentError("Candidate generation produced no labeled roof pixels")
        _, indices = ndimage.distance_transform_edt(~known, return_indices=True)
        nearest = labels[tuple(indices)]
        labels[missing] = nearest[missing]
    return relabel_sequential(labels)[0].astype(np.int32)


def generate_dsm_superpixels(
    smoothed_dsm: np.ndarray,
    slope_x: np.ndarray,
    slope_y: np.ndarray,
    dsm_edges: np.ndarray,
    roof_mask: np.ndarray,
    segments: int,
    compactness: float,
) -> np.ndarray:
    features = np.dstack(
        (
            robust_unit(slope_x, roof_mask),
            robust_unit(slope_y, roof_mask),
            dsm_edges,
            0.20 * robust_unit(smoothed_dsm, roof_mask),
        )
    ).astype(np.float32)
    labels = slic(
        features,
        n_segments=segments,
        compactness=compactness,
        sigma=0.0,
        start_label=1,
        mask=roof_mask,
        channel_axis=-1,
        enforce_connectivity=True,
    )
    return complete_mask(labels, roof_mask)


def plane_normal(fit: dict[str, Any]) -> np.ndarray:
    normal = np.asarray(
        [-float(fit["slope_x"]), -float(fit["slope_y"]), 1.0], dtype=np.float64
    )
    return normal / max(float(np.linalg.norm(normal)), 1e-8)


def normal_angle(first: dict[str, Any], second: dict[str, Any]) -> float:
    cosine = float(np.clip(np.dot(plane_normal(first), plane_normal(second)), -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def adjacency_statistics(
    labels: np.ndarray,
    dsm: np.ndarray,
    valid: np.ndarray,
    evidence: np.ndarray,
) -> dict[tuple[int, int], dict[str, float | int]]:
    collected: dict[tuple[int, int], dict[str, list[float]]] = {}
    comparisons = (
        (labels[:, :-1], labels[:, 1:], dsm[:, :-1], dsm[:, 1:], valid[:, :-1] & valid[:, 1:], evidence[:, :-1], evidence[:, 1:]),
        (labels[:-1, :], labels[1:, :], dsm[:-1, :], dsm[1:, :], valid[:-1, :] & valid[1:, :], evidence[:-1, :], evidence[1:, :]),
    )
    for left, right, first_z, second_z, pair_valid, first_e, second_e in comparisons:
        boundary = (left > 0) & (right > 0) & (left != right) & pair_valid
        rows, columns = np.where(boundary)
        for row, column in zip(rows.tolist(), columns.tolist(), strict=True):
            first, second = int(left[row, column]), int(right[row, column])
            pair = (min(first, second), max(first, second))
            item = collected.setdefault(pair, {"step": [], "evidence": []})
            item["step"].append(abs(float(first_z[row, column] - second_z[row, column])))
            item["evidence"].append(float(max(first_e[row, column], second_e[row, column])))
    return {
        pair: {
            "samples": len(values["step"]),
            "median_step_meters": float(np.median(values["step"])),
            "mean_boundary_evidence": float(np.mean(values["evidence"])),
        }
        for pair, values in collected.items()
    }


def apply_unions(labels: np.ndarray, pairs: list[tuple[int, int]]) -> np.ndarray:
    values = sorted(int(value) for value in np.unique(labels) if value > 0)
    parent = {value: value for value in values}

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    for first, second in pairs:
        first_root, second_root = find(first), find(second)
        if first_root != second_root:
            parent[second_root] = first_root
    result = np.zeros(labels.shape, dtype=np.int32)
    for value in values:
        result[labels == value] = find(value)
    return relabel_sequential(result)[0].astype(np.int32)


def merge_planar_neighbors(
    labels: np.ndarray,
    dsm: np.ndarray,
    valid: np.ndarray,
    world_x: np.ndarray,
    world_y: np.ndarray,
    evidence: np.ndarray,
    residual_threshold: float,
    max_samples: int,
    angle_threshold: float,
    step_threshold: float,
    *,
    maximum_evidence: float = 1.0,
    iterations: int = 4,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    current = labels.copy()
    records: list[dict[str, Any]] = []
    for iteration in range(iterations):
        fits = fit_metric_planes(
            current,
            dsm,
            valid,
            world_x,
            world_y,
            residual_threshold,
            max_samples,
        )
        adjacency = adjacency_statistics(current, dsm, valid, evidence)
        accepted: list[tuple[int, int]] = []
        for pair, stats in adjacency.items():
            if pair[0] not in fits or pair[1] not in fits:
                continue
            angle = normal_angle(fits[pair[0]], fits[pair[1]])
            decision = bool(
                angle <= angle_threshold
                and float(stats["median_step_meters"]) <= step_threshold
                and float(stats["mean_boundary_evidence"]) <= maximum_evidence
            )
            records.append(
                {
                    "iteration": iteration + 1,
                    "labels": list(pair),
                    "normal_angle_degrees": round(angle, 6),
                    **{
                        key: round(float(value), 8) if isinstance(value, float) else value
                        for key, value in stats.items()
                    },
                    "merged": decision,
                }
            )
            if decision:
                accepted.append(pair)
        if not accepted:
            break
        current = apply_unions(current, accepted)
    return current, records


def merge_small_regions(
    labels: np.ndarray,
    roof_mask: np.ndarray,
    minimum_pixels: int,
) -> np.ndarray:
    current = labels.copy()
    kernel = np.ones((3, 3), np.uint8)
    for label in sorted(int(value) for value in np.unique(current) if value > 0):
        region = current == label
        if int(region.sum()) >= minimum_pixels:
            continue
        ring = cv2.dilate(region.astype(np.uint8), kernel, iterations=1).astype(bool) & ~region
        neighbors = current[ring & roof_mask]
        neighbors = neighbors[neighbors > 0]
        if neighbors.size:
            values, counts = np.unique(neighbors, return_counts=True)
            current[region] = int(values[int(np.argmax(counts))])
    return complete_mask(current, roof_mask)


def intersection_partition(
    vector_labels: np.ndarray,
    dsm_labels: np.ndarray,
    roof_mask: np.ndarray,
) -> np.ndarray:
    result = np.zeros(vector_labels.shape, dtype=np.int32)
    next_label = 1
    multiplier = int(dsm_labels.max()) + 1
    codes = vector_labels.astype(np.int64) * multiplier + dsm_labels.astype(np.int64)
    for code in sorted(int(value) for value in np.unique(codes[roof_mask])):
        region = roof_mask & (codes == code)
        components, count = ndimage.label(region)
        for component in range(1, count + 1):
            pixels = components == component
            if np.any(pixels):
                result[pixels] = next_label
                next_label += 1
    return complete_mask(result, roof_mask)


def internal_boundary(labels: np.ndarray) -> np.ndarray:
    boundary = np.zeros(labels.shape, dtype=bool)
    horizontal = (labels[:, :-1] != labels[:, 1:]) & (labels[:, :-1] > 0) & (labels[:, 1:] > 0)
    vertical = (labels[:-1, :] != labels[1:, :]) & (labels[:-1, :] > 0) & (labels[1:, :] > 0)
    boundary[:, :-1] |= horizontal
    boundary[:, 1:] |= horizontal
    boundary[:-1, :] |= vertical
    boundary[1:, :] |= vertical
    return boundary


def candidate_metrics(
    labels: np.ndarray,
    roof_mask: np.ndarray,
    dsm: np.ndarray,
    valid: np.ndarray,
    world_x: np.ndarray,
    world_y: np.ndarray,
    evidence: np.ndarray,
    residual_threshold: float,
    max_samples: int,
    minimum_pixels: int,
) -> tuple[float, dict[str, Any], dict[int, dict[str, Any]]]:
    fits = fit_metric_planes(
        labels,
        dsm,
        valid,
        world_x,
        world_y,
        residual_threshold,
        max_samples,
    )
    aggregate = aggregate_fit_metrics(fits)
    boundary = internal_boundary(labels)
    boundary_support = float(evidence[boundary].mean()) if np.any(boundary) else 0.0
    roof_pixels = max(int(roof_mask.sum()), 1)
    coverage = float(((labels > 0) & roof_mask).sum() / roof_pixels)
    fitted_labels = np.asarray(sorted(fits), dtype=np.int32)
    fitted_coverage = float(np.isin(labels, fitted_labels).sum() / roof_pixels) if fits else 0.0
    counts = np.asarray(
        [int((labels == value).sum()) for value in np.unique(labels) if value > 0],
        dtype=np.int64,
    )
    small_fraction = float(counts[counts < minimum_pixels].sum() / roof_pixels) if counts.size else 1.0
    rmse = aggregate["weighted_rmse_meters"]
    median = aggregate["weighted_median_absolute_residual_meters"]
    rmse_score = math.exp(-float(rmse) / 0.45) if rmse is not None else 0.0
    planarity_score = math.exp(-float(median) / 0.25) if median is not None else 0.0
    plane_count = int(len(counts))
    simplicity = float(np.clip(1.0 - max(plane_count - 30, 0) / 45.0, 0.0, 1.0))
    score = (
        0.28 * rmse_score
        + 0.25 * planarity_score
        + 0.20 * boundary_support
        + 0.12 * coverage
        + 0.10 * fitted_coverage
        + 0.05 * simplicity
        - 0.15 * small_fraction
    )
    details = {
        "score": round(score, 8),
        "plane_count": plane_count,
        "coverage_percentage": round(100.0 * coverage, 4),
        "fitted_coverage_percentage": round(100.0 * fitted_coverage, 4),
        "boundary_evidence_support": round(boundary_support, 8),
        "small_region_percentage": round(100.0 * small_fraction, 4),
        "rmse_score": round(rmse_score, 8),
        "planarity_score": round(planarity_score, 8),
        "simplicity_score": round(simplicity, 8),
        "metric_fit": aggregate,
    }
    return score, details, fits


def georeferenced_geojson(
    labels: np.ndarray,
    transform: Any,
    crs: str | None,
    fits: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    features: list[dict[str, Any]] = []
    for label in sorted(int(value) for value in np.unique(labels) if value > 0):
        binary = (labels == label).astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polygons: list[list[list[float]]] = []
        for contour in contours:
            if cv2.contourArea(contour) < 8.0:
                continue
            ring: list[list[float]] = []
            for column, row in contour[:, 0, :].tolist():
                x = transform.a * column + transform.b * row + transform.c
                y = transform.d * column + transform.e * row + transform.f
                ring.append([round(float(x), 6), round(float(y), 6)])
            if ring and ring[0] != ring[-1]:
                ring.append(ring[0])
            if len(ring) >= 4:
                polygons.append(ring)
        if not polygons:
            continue
        geometry: dict[str, Any]
        if len(polygons) == 1:
            geometry = {"type": "Polygon", "coordinates": [polygons[0]]}
        else:
            geometry = {"type": "MultiPolygon", "coordinates": [[[point for point in ring]] for ring in polygons]}
        properties: dict[str, Any] = {"roof_plane_id": label}
        if label in fits:
            properties.update(
                {
                    key: round(float(value), 8) if isinstance(value, float) else value
                    for key, value in fits[label].items()
                }
            )
        features.append({"type": "Feature", "geometry": geometry, "properties": properties})
    payload: dict[str, Any] = {"type": "FeatureCollection", "features": features}
    if crs:
        payload["coordinate_reference_system"] = crs
    return payload


def save_candidate(
    directory: Path,
    name: str,
    image: np.ndarray,
    labels: np.ndarray,
    score: float,
    details: dict[str, Any],
    fits: dict[int, dict[str, Any]],
    transform: Any,
    crs: str | None,
) -> Image.Image:
    directory.mkdir(parents=True, exist_ok=True)
    colors, overlay, _ = render_planes(image, labels)
    save_label_mask(labels, directory / "roof_planes_labels.png")
    Image.fromarray(colors).save(directory / "roof_planes_color.png")
    Image.fromarray(overlay).save(directory / "roof_planes_overlay.png")
    (directory / "roof_planes_georeferenced.geojson").write_text(
        json.dumps(georeferenced_geojson(labels, transform, crs, fits), indent=2),
        encoding="utf-8",
    )
    (directory / "result.json").write_text(
        json.dumps({"name": name, "score": round(score, 8), **details}, indent=2),
        encoding="utf-8",
    )
    return Image.fromarray(overlay)


def run(args: argparse.Namespace) -> Path:
    output_root = args.output_root.expanduser().resolve()
    source_run = (
        args.source_run.expanduser().resolve()
        if args.source_run is not None
        else find_latest_solar_run(output_root)
    )
    source_result = json.loads((source_run / "result.json").read_text(encoding="utf-8"))
    vector_run = Path(source_result["source_run"])
    vector_labels = np.asarray(
        Image.open(source_run / "vector_structural_labels.png"), dtype=np.int32
    )
    roof_mask = np.asarray(Image.open(source_run / "building_mask.png").convert("L")) > 0
    image = np.asarray(Image.open(source_run / "input.png").convert("RGB"))
    visual_evidence = np.asarray(
        Image.open(vector_run / "combined_boundary_evidence.png").convert("L"),
        dtype=np.float32,
    ) / 255.0
    dsm, dsm_valid, transform, crs = load_registered_dsm(source_run)
    if dsm.shape != roof_mask.shape:
        raise ExperimentError("Registered DSM and source image dimensions do not match")
    analysis_valid = dsm_valid & roof_mask
    smoothed, slope_x, slope_y, metric_edges = dsm_evidence(
        dsm, analysis_valid, roof_mask, transform
    )
    combined_evidence = np.clip(0.44 * visual_evidence + 0.56 * metric_edges, 0.0, 1.0)
    world_x, world_y = world_coordinates(roof_mask.shape, transform)
    minimum_pixels = max(
        20,
        int(round(float(roof_mask.sum()) * args.minimum_region_percentage / 100.0)),
    )

    dsm_initial = generate_dsm_superpixels(
        smoothed,
        slope_x,
        slope_y,
        metric_edges,
        roof_mask,
        args.dsm_superpixels,
        args.dsm_compactness,
    )
    dsm_labels, dsm_merge_records = merge_planar_neighbors(
        dsm_initial,
        dsm,
        analysis_valid,
        world_x,
        world_y,
        metric_edges,
        args.plane_residual_threshold_meters,
        args.max_fit_samples,
        args.normal_angle_degrees,
        args.boundary_step_meters,
    )
    dsm_labels = merge_small_regions(dsm_labels, roof_mask, minimum_pixels)

    fused_initial = intersection_partition(vector_labels, dsm_labels, roof_mask)
    fused_initial = merge_small_regions(fused_initial, roof_mask, minimum_pixels)
    fused_labels, fusion_merge_records = merge_planar_neighbors(
        fused_initial,
        dsm,
        analysis_valid,
        world_x,
        world_y,
        combined_evidence,
        args.plane_residual_threshold_meters,
        args.max_fit_samples,
        max(4.0, args.normal_angle_degrees - 1.5),
        args.boundary_step_meters,
        maximum_evidence=0.72,
    )
    fused_labels = merge_small_regions(fused_labels, roof_mask, minimum_pixels)

    run_directory = create_run_directory(output_root, EXPERIMENT_NAME)
    Image.fromarray(image).save(run_directory / "input.png")
    Image.fromarray((roof_mask * 255).astype(np.uint8)).save(run_directory / "building_mask.png")
    save_dsm_visualization(dsm, analysis_valid, run_directory / "registered_metric_dsm.png")
    Image.fromarray((robust_unit(slope_x, roof_mask) * 255).astype(np.uint8)).save(
        run_directory / "metric_slope_x.png"
    )
    Image.fromarray((robust_unit(slope_y, roof_mask) * 255).astype(np.uint8)).save(
        run_directory / "metric_slope_y.png"
    )
    Image.fromarray((metric_edges * 255).astype(np.uint8)).save(
        run_directory / "dsm_boundary_evidence.png"
    )
    Image.fromarray((combined_evidence * 255).astype(np.uint8)).save(
        run_directory / "combined_visual_dsm_evidence.png"
    )
    save_label_mask(dsm_initial, run_directory / "dsm_superpixels_initial.png")
    save_label_mask(fused_initial, run_directory / "fusion_intersections_initial.png")

    candidate_labels = {
        "vector_structural_baseline": complete_mask(vector_labels, roof_mask),
        "dsm_independent": dsm_labels,
        "visual_dsm_fusion": fused_labels,
    }
    candidates: list[dict[str, Any]] = []
    comparison: list[tuple[str, Image.Image]] = [
        ("Original input", Image.fromarray(image)),
        (
            "Google Solar metric DSM",
            Image.open(run_directory / "registered_metric_dsm.png").convert("RGB"),
        ),
        (
            "DSM slope/normal discontinuities",
            Image.open(run_directory / "dsm_boundary_evidence.png").convert("RGB"),
        ),
    ]
    for name, labels in candidate_labels.items():
        score, details, fits = candidate_metrics(
            labels,
            roof_mask,
            dsm,
            analysis_valid,
            world_x,
            world_y,
            combined_evidence,
            args.plane_residual_threshold_meters,
            args.max_fit_samples,
            minimum_pixels,
        )
        overlay = save_candidate(
            run_directory / f"candidate_{name}",
            name,
            image,
            labels,
            score,
            details,
            fits,
            transform,
            crs,
        )
        candidates.append(
            {
                "name": name,
                "labels": labels,
                "score": score,
                "details": details,
                "fits": fits,
                "overlay": overlay,
            }
        )
        comparison.append((f"{name} | {score:.3f}", overlay))

    winner = max(candidates, key=lambda item: float(item["score"]))
    winner_directory = run_directory / f"candidate_{winner['name']}"
    for source_name, target_name in (
        ("roof_planes_labels.png", "winner_roof_planes_labels.png"),
        ("roof_planes_color.png", "winner_roof_planes_color.png"),
        ("roof_planes_overlay.png", "winner_roof_planes_overlay.png"),
        ("roof_planes_georeferenced.geojson", "winner_roof_planes_georeferenced.geojson"),
    ):
        (run_directory / target_name).write_bytes((winner_directory / source_name).read_bytes())
    comparison.append((f"Selected: {winner['name']}", winner["overlay"]))

    manual_reference = (
        args.manual_reference.expanduser().resolve()
        if args.manual_reference is not None
        else None
    )
    manual_available = manual_reference is not None and manual_reference.is_file()
    if manual_available:
        assert manual_reference is not None
        manual = Image.open(manual_reference).convert("RGB")
        manual.save(run_directory / "manual_reference_visual_only.png")
        comparison.append(("Manual paint-over | visual reference only", manual))
    make_contact_sheet(comparison, run_directory / "qualitative_comparison.jpg")

    result = {
        "experiment": EXPERIMENT_NAME,
        "purpose": (
            "Generate an independent metric-DSM roof-plane hypothesis and fuse it "
            "with the LangSAM/OpenAI/vector structural result."
        ),
        "source_run": str(source_run),
        "source_vector_run": str(vector_run),
        "source_pipeline": [
            "LangSAM/SAM2 building segmentation",
            "Depth Anything V2 and local structural evidence",
            "OpenAI multimodal topology hypotheses",
            "vector structural optimization",
            "Google Solar metric DSM registration",
        ],
        "strategy": {
            "dsm_superpixels": args.dsm_superpixels,
            "dsm_compactness": args.dsm_compactness,
            "normal_angle_degrees": args.normal_angle_degrees,
            "boundary_step_meters": args.boundary_step_meters,
            "minimum_region_percentage": args.minimum_region_percentage,
            "plane_residual_threshold_meters": args.plane_residual_threshold_meters,
        },
        "dsm_merge_records": dsm_merge_records,
        "fusion_merge_records": fusion_merge_records,
        "candidates": [
            {"name": item["name"], "score": round(float(item["score"]), 8), **item["details"]}
            for item in candidates
        ],
        "winner": winner["name"],
        "winner_score": round(float(winner["score"]), 8),
        "winner_plane_count": len(summarize_planes(winner["labels"])),
        "manual_reference_available": manual_available,
        "manual_reference_used_for_generation": False,
        "manual_reference_used_for_scoring": False,
        "manual_reference_used_for_selection": False,
        "manual_reference_visual_comparison_only": True,
        "external_evidence": "Registered Google Solar metric DSM from Experiment 07",
    }
    (run_directory / "result.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(
        f"Selected {winner['name']} with score {winner['score']:.4f}; "
        f"planes={result['winner_plane_count']}"
    )
    print(f"Output: {run_directory}")
    return run_directory


if __name__ == "__main__":
    run(parse_args())
