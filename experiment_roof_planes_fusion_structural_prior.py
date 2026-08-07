from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
from skimage.segmentation import relabel_sequential, watershed

from experiment_roof_planes import render_planes, save_label_mask
from experiment_roof_planes_dsm_hypothesis_fusion import (
    candidate_metrics,
    complete_mask,
    dsm_evidence,
    georeferenced_geojson,
    internal_boundary,
    load_registered_dsm,
    merge_planar_neighbors,
    merge_small_regions,
    robust_unit,
    world_coordinates,
)
from experiment_roof_planes_google_solar_dsm import save_dsm_visualization
from run_artifacts import create_run_directory


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs"
EXPERIMENT_NAME = "roof_planes_experiment_09_fusion_structural_prior"
EXPERIMENT_08_SUFFIX = "_roof_planes_experiment_08_dsm_hypothesis_fusion"


class ExperimentError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Keep the Experiment 08 visual/DSM fusion as the primary topology "
            "and use the vector-structural baseline as a secondary geometric prior."
        )
    )
    parser.add_argument(
        "--source-run",
        type=Path,
        help="Completed Experiment 08 directory. Defaults to the latest output.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--maximum-snap-distance", type=float, default=20.0)
    parser.add_argument("--seed-inset", type=int, default=4)
    parser.add_argument("--minimum-region-percentage", type=float, default=0.45)
    parser.add_argument("--plane-residual-threshold-meters", type=float, default=0.18)
    parser.add_argument("--max-fit-samples", type=int, default=8000)
    parser.add_argument(
        "--manual-reference",
        type=Path,
        help="Optional paint-over used only in the qualitative comparison.",
    )
    return parser.parse_args()


def find_latest_fusion_run(output_root: Path) -> Path:
    candidates: list[Path] = []
    for result_path in output_root.rglob("result.json"):
        directory = result_path.parent
        if not directory.name.endswith(EXPERIMENT_08_SUFFIX):
            continue
        required = (
            directory / "candidate_visual_dsm_fusion" / "roof_planes_labels.png",
            directory / "candidate_vector_structural_baseline" / "roof_planes_labels.png",
            directory / "building_mask.png",
            directory / "input.png",
        )
        if all(path.is_file() for path in required):
            candidates.append(directory)
    if not candidates:
        raise FileNotFoundError(
            "No completed Experiment 08 run was found. Run "
            "experiment_roof_planes_dsm_hypothesis_fusion.py first."
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def load_labels(path: Path, roof_mask: np.ndarray) -> np.ndarray:
    labels = np.asarray(Image.open(path), dtype=np.int32)
    if labels.shape != roof_mask.shape:
        raise ExperimentError(f"Label dimensions do not match the roof mask: {path}")
    return complete_mask(labels, roof_mask)


def boundary_distance(boundary: np.ndarray) -> np.ndarray:
    if not np.any(boundary):
        return np.full(boundary.shape, np.inf, dtype=np.float32)
    return cv2.distanceTransform((~boundary).astype(np.uint8), cv2.DIST_L2, 5)


def supported_baseline_boundary(
    fusion_boundary: np.ndarray,
    baseline_boundary: np.ndarray,
    metric_evidence: np.ndarray,
    roof_mask: np.ndarray,
    maximum_distance: float,
) -> tuple[np.ndarray, dict[str, float]]:
    distance = boundary_distance(fusion_boundary)
    values = metric_evidence[baseline_boundary & roof_mask]
    threshold = float(np.percentile(values, 62.0)) if values.size else 1.0
    near_fusion = distance <= maximum_distance
    dsm_supported = metric_evidence >= threshold
    supported = baseline_boundary & roof_mask & (near_fusion | dsm_supported)
    diagnostics = {
        "metric_support_threshold": round(threshold, 8),
        "baseline_boundary_pixels": int((baseline_boundary & roof_mask).sum()),
        "supported_baseline_boundary_pixels": int(supported.sum()),
        "supported_baseline_percentage": round(
            100.0 * float(supported.sum()) / max(int((baseline_boundary & roof_mask).sum()), 1),
            4,
        ),
    }
    return supported, diagnostics


def region_markers(
    labels: np.ndarray,
    roof_mask: np.ndarray,
    inset: int,
) -> np.ndarray:
    markers = np.zeros(labels.shape, dtype=np.int32)
    kernel = np.ones((3, 3), np.uint8)
    next_marker = 1
    for label in sorted(int(value) for value in np.unique(labels) if value > 0):
        region = labels == label
        seed = cv2.erode(region.astype(np.uint8), kernel, iterations=max(inset, 0)) > 0
        if not np.any(seed):
            distance = cv2.distanceTransform(region.astype(np.uint8), cv2.DIST_L2, 5)
            row, column = np.unravel_index(int(np.argmax(distance)), distance.shape)
            seed[row, column] = True
        components, count = ndimage.label(seed)
        for component in range(1, count + 1):
            pixels = components == component
            if np.any(pixels):
                markers[pixels] = next_marker
                next_marker += 1
    if next_marker == 1:
        raise ExperimentError("The visual/DSM fusion did not produce watershed seeds")
    markers[~roof_mask] = 0
    return markers


def blurred_boundary(boundary: np.ndarray, sigma: float) -> np.ndarray:
    values = cv2.GaussianBlur(boundary.astype(np.float32), (0, 0), sigma)
    maximum = float(values.max())
    return values / maximum if maximum > 0 else values


def snap_partition(
    fusion_labels: np.ndarray,
    roof_mask: np.ndarray,
    metric_evidence: np.ndarray,
    visual_evidence: np.ndarray,
    supported_baseline: np.ndarray,
    *,
    fusion_weight: float,
    dsm_weight: float,
    baseline_weight: float,
    seed_inset: int,
) -> tuple[np.ndarray, np.ndarray]:
    markers = region_markers(fusion_labels, roof_mask, seed_inset)
    fusion_prior = blurred_boundary(internal_boundary(fusion_labels), 1.6)
    baseline_prior = blurred_boundary(supported_baseline, 1.35)
    external_evidence = np.clip(0.72 * metric_evidence + 0.28 * visual_evidence, 0.0, 1.0)
    barrier = np.clip(
        fusion_weight * fusion_prior
        + dsm_weight * external_evidence
        + baseline_weight * baseline_prior,
        0.0,
        1.0,
    )
    barrier = cv2.GaussianBlur(barrier.astype(np.float32), (0, 0), 0.65)
    labels = watershed(barrier, markers=markers, mask=roof_mask, connectivity=1)
    return complete_mask(labels.astype(np.int32), roof_mask), barrier


def boundary_straightness(labels: np.ndarray) -> tuple[float, dict[str, float | int]]:
    boundary = internal_boundary(labels)
    boundary_pixels = int(boundary.sum())
    if boundary_pixels == 0:
        return 0.0, {"boundary_pixels": 0, "line_supported_pixels": 0}
    source = (boundary.astype(np.uint8) * 255)
    lines = cv2.HoughLinesP(
        source,
        rho=1,
        theta=np.pi / 360.0,
        threshold=18,
        minLineLength=14,
        maxLineGap=7,
    )
    line_mask = np.zeros(labels.shape, dtype=np.uint8)
    line_count = 0
    if lines is not None:
        for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):
            cv2.line(line_mask, (int(x1), int(y1)), (int(x2), int(y2)), 255, 1, cv2.LINE_8)
            line_count += 1
    supported = boundary & (cv2.dilate(line_mask, np.ones((3, 3), np.uint8)) > 0)
    score = float(supported.sum() / boundary_pixels)
    return score, {
        "boundary_pixels": boundary_pixels,
        "line_supported_pixels": int(supported.sum()),
        "detected_line_count": line_count,
        "straightness_score": round(score, 8),
    }


def displacement_metrics(
    labels: np.ndarray,
    fusion_boundary: np.ndarray,
    baseline_boundary: np.ndarray,
) -> dict[str, float]:
    boundary = internal_boundary(labels)
    if not np.any(boundary):
        return {
            "mean_distance_to_fusion_pixels": 0.0,
            "mean_distance_to_baseline_pixels": 0.0,
        }
    return {
        "mean_distance_to_fusion_pixels": round(
            float(boundary_distance(fusion_boundary)[boundary].mean()), 8
        ),
        "mean_distance_to_baseline_pixels": round(
            float(boundary_distance(baseline_boundary)[boundary].mean()), 8
        ),
    }


def evaluate_candidate(
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
    fusion_boundary: np.ndarray,
    baseline_boundary: np.ndarray,
) -> tuple[float, dict[str, Any], dict[int, dict[str, Any]]]:
    metric_score, details, fits = candidate_metrics(
        labels,
        roof_mask,
        dsm,
        valid,
        world_x,
        world_y,
        evidence,
        residual_threshold,
        max_samples,
        minimum_pixels,
    )
    straightness, straightness_details = boundary_straightness(labels)
    final_score = 0.88 * metric_score + 0.12 * straightness
    details = {
        **{key: value for key, value in details.items() if key != "score"},
        "dsm_quality_score": round(metric_score, 8),
        "straightness": straightness_details,
        "boundary_displacement": displacement_metrics(
            labels, fusion_boundary, baseline_boundary
        ),
        "final_score": round(final_score, 8),
        "score_weights": {"dsm_quality": 0.88, "structural_straightness": 0.12},
    }
    return final_score, details, fits


def make_complete_contact_sheet(
    items: list[tuple[str, Image.Image]],
    output: Path,
) -> None:
    width, height = 420, 420
    header = 34
    columns = min(3, max(len(items), 1))
    rows = math.ceil(len(items) / columns)
    sheet = Image.new("RGB", (width * columns, (height + header) * rows), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for index, (title, image) in enumerate(items):
        x = (index % columns) * width
        y = (index // columns) * (height + header)
        fitted = image.convert("RGB").copy()
        fitted.thumbnail((width, height), Image.Resampling.LANCZOS)
        offset_x = x + (width - fitted.width) // 2
        offset_y = y + header + (height - fitted.height) // 2
        sheet.paste(fitted, (offset_x, offset_y))
        draw.text((x + 10, y + 10), title, fill="black", font=font)
    sheet.save(output, format="JPEG", quality=94, optimize=True)


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
        json.dumps({**details, "name": name, "score": round(score, 8)}, indent=2),
        encoding="utf-8",
    )
    return Image.fromarray(overlay)


def run(args: argparse.Namespace) -> Path:
    output_root = args.output_root.expanduser().resolve()
    source_run = (
        args.source_run.expanduser().resolve()
        if args.source_run is not None
        else find_latest_fusion_run(output_root)
    )
    source_result = json.loads((source_run / "result.json").read_text(encoding="utf-8"))
    solar_run = Path(source_result["source_run"])
    vector_run = Path(source_result["source_vector_run"])

    image = np.asarray(Image.open(source_run / "input.png").convert("RGB"))
    roof_mask = np.asarray(Image.open(source_run / "building_mask.png").convert("L")) > 0
    fusion_labels = load_labels(
        source_run / "candidate_visual_dsm_fusion" / "roof_planes_labels.png",
        roof_mask,
    )
    baseline_labels = load_labels(
        source_run / "candidate_vector_structural_baseline" / "roof_planes_labels.png",
        roof_mask,
    )
    visual_evidence = np.asarray(
        Image.open(vector_run / "combined_boundary_evidence.png").convert("L"),
        dtype=np.float32,
    ) / 255.0

    dsm, dsm_valid, transform, crs = load_registered_dsm(solar_run)
    if dsm.shape != roof_mask.shape:
        raise ExperimentError("Registered DSM and source image dimensions do not match")
    analysis_valid = dsm_valid & roof_mask
    _, _, _, metric_evidence = dsm_evidence(dsm, analysis_valid, roof_mask, transform)
    scoring_evidence = np.clip(0.44 * visual_evidence + 0.56 * metric_evidence, 0.0, 1.0)
    world_x, world_y = world_coordinates(roof_mask.shape, transform)
    minimum_pixels = max(
        20,
        int(round(float(roof_mask.sum()) * args.minimum_region_percentage / 100.0)),
    )

    fusion_boundary = internal_boundary(fusion_labels)
    baseline_boundary = internal_boundary(baseline_labels)
    supported_baseline, support_diagnostics = supported_baseline_boundary(
        fusion_boundary,
        baseline_boundary,
        metric_evidence,
        roof_mask,
        args.maximum_snap_distance,
    )

    variants = {
        "fusion_consensus_snap": {
            "fusion_weight": 0.52,
            "dsm_weight": 0.34,
            "baseline_weight": 0.14,
        },
        "fusion_balanced_snap": {
            "fusion_weight": 0.42,
            "dsm_weight": 0.36,
            "baseline_weight": 0.22,
        },
        "fusion_structural_snap": {
            "fusion_weight": 0.34,
            "dsm_weight": 0.38,
            "baseline_weight": 0.28,
        },
    }
    candidate_labels: dict[str, np.ndarray] = {
        "visual_dsm_fusion_control": fusion_labels,
    }
    barriers: dict[str, np.ndarray] = {}
    for name, settings in variants.items():
        labels, barrier = snap_partition(
            fusion_labels,
            roof_mask,
            metric_evidence,
            visual_evidence,
            supported_baseline,
            seed_inset=args.seed_inset,
            **settings,
        )
        labels = merge_small_regions(labels, roof_mask, minimum_pixels)
        candidate_labels[name] = labels
        barriers[name] = barrier

    regularized, regularization_records = merge_planar_neighbors(
        candidate_labels["fusion_balanced_snap"],
        dsm,
        analysis_valid,
        world_x,
        world_y,
        np.clip(scoring_evidence + 0.18 * supported_baseline.astype(np.float32), 0.0, 1.0),
        args.plane_residual_threshold_meters,
        args.max_fit_samples,
        angle_threshold=7.0,
        step_threshold=0.20,
        maximum_evidence=0.62,
        iterations=3,
    )
    candidate_labels["fusion_topology_regularized"] = merge_small_regions(
        regularized, roof_mask, minimum_pixels
    )

    run_directory = create_run_directory(output_root, EXPERIMENT_NAME)
    Image.fromarray(image).save(run_directory / "input.png")
    Image.fromarray((roof_mask * 255).astype(np.uint8)).save(
        run_directory / "building_mask.png"
    )
    save_dsm_visualization(dsm, analysis_valid, run_directory / "registered_metric_dsm.png")
    Image.fromarray((metric_evidence * 255).astype(np.uint8)).save(
        run_directory / "dsm_boundary_evidence.png"
    )
    Image.fromarray((fusion_boundary * 255).astype(np.uint8)).save(
        run_directory / "visual_dsm_fusion_boundary.png"
    )
    Image.fromarray((baseline_boundary * 255).astype(np.uint8)).save(
        run_directory / "vector_structural_baseline_boundary.png"
    )
    Image.fromarray((supported_baseline * 255).astype(np.uint8)).save(
        run_directory / "supported_baseline_boundary_prior.png"
    )
    for name, barrier in barriers.items():
        Image.fromarray((robust_unit(barrier, roof_mask) * 255).astype(np.uint8)).save(
            run_directory / f"barrier_{name}.png"
        )

    candidates: list[dict[str, Any]] = []
    comparison: list[tuple[str, Image.Image]] = [
        ("Original input", Image.fromarray(image)),
        (
            "Visual + DSM fusion | primary topology",
            Image.open(
                source_run / "candidate_visual_dsm_fusion" / "roof_planes_overlay.png"
            ).convert("RGB"),
        ),
        (
            "Vector structural baseline | secondary prior",
            Image.open(
                source_run
                / "candidate_vector_structural_baseline"
                / "roof_planes_overlay.png"
            ).convert("RGB"),
        ),
    ]
    for name, labels in candidate_labels.items():
        score, details, fits = evaluate_candidate(
            labels,
            roof_mask,
            dsm,
            analysis_valid,
            world_x,
            world_y,
            scoring_evidence,
            args.plane_residual_threshold_meters,
            args.max_fit_samples,
            minimum_pixels,
            fusion_boundary,
            baseline_boundary,
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
    make_complete_contact_sheet(comparison, run_directory / "qualitative_comparison.jpg")

    result = {
        "experiment": EXPERIMENT_NAME,
        "purpose": (
            "Preserve visual_dsm_fusion as the primary topology while using the "
            "vector structural baseline as a DSM-gated geometric prior."
        ),
        "source_run": str(source_run),
        "source_solar_run": str(solar_run),
        "source_vector_run": str(vector_run),
        "strategy": {
            "primary_topology": "visual_dsm_fusion",
            "secondary_prior": "vector_structural_baseline",
            "maximum_snap_distance": args.maximum_snap_distance,
            "seed_inset": args.seed_inset,
            "minimum_region_percentage": args.minimum_region_percentage,
            "candidate_weights": variants,
            "selection": (
                "88% independent DSM/visual metric quality and 12% boundary straightness"
            ),
        },
        "baseline_support": support_diagnostics,
        "regularization_records": regularization_records,
        "candidates": [
            {
                **item["details"],
                "name": item["name"],
                "score": round(float(item["score"]), 8),
            }
            for item in candidates
        ],
        "winner": winner["name"],
        "winner_score": round(float(winner["score"]), 8),
        "winner_plane_count": int(len(np.unique(winner["labels"])) - 1),
        "manual_reference_available": manual_available,
        "manual_reference_used_for_generation": False,
        "manual_reference_used_for_scoring": False,
        "manual_reference_used_for_selection": False,
        "manual_reference_visual_comparison_only": True,
    }
    (run_directory / "result.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(
        f"Selected {winner['name']} with score {winner['score']:.4f}; "
        f"planes={result['winner_plane_count']}"
    )
    for item in candidates:
        print(
            f"  {item['name']}: score={item['score']:.4f}; "
            f"planes={item['details']['plane_count']}"
        )
    print(f"Output: {run_directory}")
    return run_directory


if __name__ == "__main__":
    run(parse_args())
