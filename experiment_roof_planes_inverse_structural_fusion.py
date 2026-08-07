from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage
from skimage.segmentation import relabel_sequential

from experiment_roof_planes_dsm_hypothesis_fusion import (
    complete_mask,
    dsm_evidence,
    internal_boundary,
    load_registered_dsm,
    merge_planar_neighbors,
    merge_small_regions,
    world_coordinates,
)
from experiment_roof_planes_fusion_structural_prior import (
    boundary_distance,
    evaluate_candidate,
    make_complete_contact_sheet,
    save_candidate,
    snap_partition,
)
from experiment_roof_planes_google_solar_dsm import (
    aggregate_fit_metrics,
    fit_metric_planes,
    save_dsm_visualization,
)
from run_artifacts import create_run_directory


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs"
EXPERIMENT_NAME = "roof_planes_experiment_10_inverse_structural_fusion"
EXPERIMENT_09_SUFFIX = "_roof_planes_experiment_09_fusion_structural_prior"


class ExperimentError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Use the vector-structural baseline as the primary roof topology and "
            "visual/DSM fusion boundaries as a secondary, DSM-gated prior."
        )
    )
    parser.add_argument(
        "--source-run",
        type=Path,
        help="Completed Experiment 09 directory. Defaults to the latest output.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--maximum-snap-distance", type=float, default=18.0)
    parser.add_argument("--seed-inset", type=int, default=4)
    parser.add_argument("--minimum-region-percentage", type=float, default=0.45)
    parser.add_argument("--plane-residual-threshold-meters", type=float, default=0.18)
    parser.add_argument("--max-fit-samples", type=int, default=8000)
    parser.add_argument("--minimum-split-improvement-meters", type=float, default=0.018)
    parser.add_argument("--minimum-split-improvement-ratio", type=float, default=0.07)
    parser.add_argument("--maximum-subplanes-per-region", type=int, default=7)
    parser.add_argument(
        "--manual-reference",
        type=Path,
        help="Optional paint-over used only in the qualitative comparison.",
    )
    return parser.parse_args()


def find_latest_experiment_09(output_root: Path) -> Path:
    candidates: list[Path] = []
    for result_path in output_root.rglob("result.json"):
        directory = result_path.parent
        if not directory.name.endswith(EXPERIMENT_09_SUFFIX):
            continue
        if (
            (directory / "result.json").is_file()
            and (directory / "winner_roof_planes_labels.png").is_file()
        ):
            candidates.append(directory)
    if not candidates:
        raise FileNotFoundError(
            "No completed Experiment 09 run was found. Run "
            "experiment_roof_planes_fusion_structural_prior.py first."
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def load_labels(path: Path, roof_mask: np.ndarray) -> np.ndarray:
    labels = np.asarray(Image.open(path), dtype=np.int32)
    if labels.shape != roof_mask.shape:
        raise ExperimentError(f"Label dimensions do not match the roof mask: {path}")
    return complete_mask(labels, roof_mask)


def supported_secondary_boundary(
    primary_boundary: np.ndarray,
    secondary_boundary: np.ndarray,
    metric_evidence: np.ndarray,
    roof_mask: np.ndarray,
    maximum_distance: float,
) -> tuple[np.ndarray, dict[str, float | int]]:
    distance = boundary_distance(primary_boundary)
    secondary_values = metric_evidence[secondary_boundary & roof_mask]
    metric_threshold = (
        float(np.percentile(secondary_values, 60.0))
        if secondary_values.size
        else 1.0
    )
    near_primary = distance <= maximum_distance
    metric_supported = metric_evidence >= metric_threshold
    supported = secondary_boundary & roof_mask & (near_primary | metric_supported)
    total = max(int((secondary_boundary & roof_mask).sum()), 1)
    return supported, {
        "metric_support_threshold": round(metric_threshold, 8),
        "secondary_boundary_pixels": int((secondary_boundary & roof_mask).sum()),
        "supported_secondary_boundary_pixels": int(supported.sum()),
        "supported_secondary_percentage": round(100.0 * float(supported.sum()) / total, 4),
    }


def local_intersection_partition(
    primary_region: np.ndarray,
    secondary_labels: np.ndarray,
    minimum_pixels: int,
) -> np.ndarray:
    local = np.zeros(secondary_labels.shape, dtype=np.int32)
    next_label = 1
    for secondary_label in sorted(
        int(value) for value in np.unique(secondary_labels[primary_region]) if value > 0
    ):
        region = primary_region & (secondary_labels == secondary_label)
        components, count = ndimage.label(region)
        for component in range(1, count + 1):
            pixels = components == component
            if np.any(pixels):
                local[pixels] = next_label
                next_label += 1
    if next_label == 1:
        local[primary_region] = 1
        return local
    local = complete_mask(local, primary_region)
    local = merge_small_regions(local, primary_region, minimum_pixels)
    return relabel_sequential(local)[0].astype(np.int32)


def metric_summary(
    labels: np.ndarray,
    dsm: np.ndarray,
    valid: np.ndarray,
    world_x: np.ndarray,
    world_y: np.ndarray,
    residual_threshold: float,
    max_samples: int,
) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    fits = fit_metric_planes(
        labels,
        dsm,
        valid,
        world_x,
        world_y,
        residual_threshold,
        max_samples,
    )
    return aggregate_fit_metrics(fits), fits


def controlled_enrichment(
    primary_labels: np.ndarray,
    secondary_labels: np.ndarray,
    roof_mask: np.ndarray,
    dsm: np.ndarray,
    valid: np.ndarray,
    world_x: np.ndarray,
    world_y: np.ndarray,
    metric_evidence: np.ndarray,
    minimum_pixels: int,
    residual_threshold: float,
    max_samples: int,
    minimum_improvement_meters: float,
    minimum_improvement_ratio: float,
    maximum_subplanes: int,
    *,
    evidence_percentile: float,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    result = np.zeros(primary_labels.shape, dtype=np.int32)
    records: list[dict[str, Any]] = []
    next_label = 1
    roof_evidence = metric_evidence[roof_mask]
    evidence_threshold = (
        float(np.percentile(roof_evidence, evidence_percentile))
        if roof_evidence.size
        else 1.0
    )

    for primary_label in sorted(
        int(value) for value in np.unique(primary_labels) if value > 0
    ):
        primary_region = primary_labels == primary_label
        single = np.zeros(primary_labels.shape, dtype=np.int32)
        single[primary_region] = 1
        partition = local_intersection_partition(
            primary_region,
            secondary_labels,
            minimum_pixels,
        )
        subplane_count = int(partition.max())
        single_metrics, _ = metric_summary(
            single,
            dsm,
            valid,
            world_x,
            world_y,
            residual_threshold,
            max_samples,
        )
        split_metrics, split_fits = metric_summary(
            partition,
            dsm,
            valid,
            world_x,
            world_y,
            residual_threshold,
            max_samples,
        )
        single_rmse = single_metrics["weighted_rmse_meters"]
        split_rmse = split_metrics["weighted_rmse_meters"]
        split_boundary = internal_boundary(partition) & primary_region
        boundary_support = (
            float(metric_evidence[split_boundary].mean())
            if np.any(split_boundary)
            else 0.0
        )
        improvement = (
            float(single_rmse) - float(split_rmse)
            if single_rmse is not None and split_rmse is not None
            else 0.0
        )
        improvement_ratio = (
            improvement / max(float(single_rmse), 1e-8)
            if single_rmse is not None
            else 0.0
        )
        enough_fits = len(split_fits) == subplane_count
        accepted = bool(
            2 <= subplane_count <= maximum_subplanes
            and enough_fits
            and boundary_support >= evidence_threshold
            and (
                improvement >= minimum_improvement_meters
                or improvement_ratio >= minimum_improvement_ratio
            )
        )
        records.append(
            {
                "primary_plane": primary_label,
                "proposed_subplanes": subplane_count,
                "single_rmse_meters": single_rmse,
                "split_rmse_meters": split_rmse,
                "improvement_meters": round(improvement, 8),
                "improvement_ratio": round(improvement_ratio, 8),
                "boundary_support": round(boundary_support, 8),
                "required_boundary_support": round(evidence_threshold, 8),
                "all_subplanes_fitted": enough_fits,
                "accepted": accepted,
            }
        )
        if accepted:
            for subplane in range(1, subplane_count + 1):
                pixels = partition == subplane
                if np.any(pixels):
                    result[pixels] = next_label
                    next_label += 1
        else:
            result[primary_region] = next_label
            next_label += 1
    return complete_mask(result, roof_mask), records


def run(args: argparse.Namespace) -> Path:
    output_root = args.output_root.expanduser().resolve()
    source_run = (
        args.source_run.expanduser().resolve()
        if args.source_run is not None
        else find_latest_experiment_09(output_root)
    )
    experiment_09_result = json.loads(
        (source_run / "result.json").read_text(encoding="utf-8")
    )
    experiment_08_run = Path(experiment_09_result["source_run"])
    experiment_08_result = json.loads(
        (experiment_08_run / "result.json").read_text(encoding="utf-8")
    )
    solar_run = Path(experiment_08_result["source_run"])
    vector_run = Path(experiment_08_result["source_vector_run"])

    image = np.asarray(Image.open(experiment_08_run / "input.png").convert("RGB"))
    roof_mask = (
        np.asarray(Image.open(experiment_08_run / "building_mask.png").convert("L"))
        > 0
    )
    baseline_labels = load_labels(
        experiment_08_run
        / "candidate_vector_structural_baseline"
        / "roof_planes_labels.png",
        roof_mask,
    )
    fusion_labels = load_labels(
        experiment_08_run
        / "candidate_visual_dsm_fusion"
        / "roof_planes_labels.png",
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

    primary_boundary = internal_boundary(baseline_labels)
    secondary_boundary = internal_boundary(fusion_labels)
    supported_secondary, support_diagnostics = supported_secondary_boundary(
        primary_boundary,
        secondary_boundary,
        metric_evidence,
        roof_mask,
        args.maximum_snap_distance,
    )

    strict_variants = {
        "baseline_consensus_snap": {
            "fusion_weight": 0.58,
            "dsm_weight": 0.31,
            "baseline_weight": 0.11,
        },
        "baseline_balanced_snap": {
            "fusion_weight": 0.49,
            "dsm_weight": 0.34,
            "baseline_weight": 0.17,
        },
        "baseline_fusion_snap": {
            "fusion_weight": 0.41,
            "dsm_weight": 0.36,
            "baseline_weight": 0.23,
        },
    }
    candidate_labels: dict[str, np.ndarray] = {
        "vector_structural_baseline_control": baseline_labels,
    }
    barriers: dict[str, np.ndarray] = {}
    for name, weights in strict_variants.items():
        labels, barrier = snap_partition(
            baseline_labels,
            roof_mask,
            metric_evidence,
            visual_evidence,
            supported_secondary,
            seed_inset=args.seed_inset,
            **weights,
        )
        candidate_labels[name] = labels
        barriers[name] = barrier

    enriched, conservative_records = controlled_enrichment(
        candidate_labels["baseline_balanced_snap"],
        fusion_labels,
        roof_mask,
        dsm,
        analysis_valid,
        world_x,
        world_y,
        metric_evidence,
        minimum_pixels,
        args.plane_residual_threshold_meters,
        args.max_fit_samples,
        args.minimum_split_improvement_meters,
        args.minimum_split_improvement_ratio,
        args.maximum_subplanes_per_region,
        evidence_percentile=58.0,
    )
    candidate_labels["baseline_controlled_enrichment"] = enriched

    exploratory_enriched, exploratory_records = controlled_enrichment(
        candidate_labels["baseline_fusion_snap"],
        fusion_labels,
        roof_mask,
        dsm,
        analysis_valid,
        world_x,
        world_y,
        metric_evidence,
        minimum_pixels,
        args.plane_residual_threshold_meters,
        args.max_fit_samples,
        0.65 * args.minimum_split_improvement_meters,
        0.65 * args.minimum_split_improvement_ratio,
        args.maximum_subplanes_per_region,
        evidence_percentile=50.0,
    )
    candidate_labels["baseline_dsm_enrichment"] = exploratory_enriched

    regularized, regularization_records = merge_planar_neighbors(
        exploratory_enriched,
        dsm,
        analysis_valid,
        world_x,
        world_y,
        np.clip(scoring_evidence + 0.16 * primary_boundary.astype(np.float32), 0.0, 1.0),
        args.plane_residual_threshold_meters,
        args.max_fit_samples,
        angle_threshold=7.0,
        step_threshold=0.20,
        maximum_evidence=0.60,
        iterations=3,
    )
    candidate_labels["baseline_enrichment_regularized"] = merge_small_regions(
        regularized,
        roof_mask,
        minimum_pixels,
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
    Image.fromarray((primary_boundary * 255).astype(np.uint8)).save(
        run_directory / "primary_baseline_boundary.png"
    )
    Image.fromarray((secondary_boundary * 255).astype(np.uint8)).save(
        run_directory / "secondary_visual_dsm_fusion_boundary.png"
    )
    Image.fromarray((supported_secondary * 255).astype(np.uint8)).save(
        run_directory / "supported_secondary_boundary_prior.png"
    )
    for name, barrier in barriers.items():
        normalized = np.zeros(barrier.shape, dtype=np.uint8)
        values = barrier[roof_mask]
        if values.size and float(values.max() - values.min()) > 1e-8:
            scaled = np.clip(
                (barrier - float(values.min())) / float(values.max() - values.min()),
                0.0,
                1.0,
            )
            normalized = (scaled * 255).astype(np.uint8)
        Image.fromarray(normalized).save(run_directory / f"barrier_{name}.png")

    comparison: list[tuple[str, Image.Image]] = [
        ("Original input", Image.fromarray(image)),
        (
            "Vector structural baseline | primary topology",
            Image.open(
                experiment_08_run
                / "candidate_vector_structural_baseline"
                / "roof_planes_overlay.png"
            ).convert("RGB"),
        ),
        (
            "Visual + DSM fusion | secondary prior",
            Image.open(
                experiment_08_run
                / "candidate_visual_dsm_fusion"
                / "roof_planes_overlay.png"
            ).convert("RGB"),
        ),
    ]
    candidates: list[dict[str, Any]] = []
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
            primary_boundary,
            secondary_boundary,
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
            "Test the inverse hierarchy: vector structural baseline as primary "
            "topology and visual/DSM fusion as a DSM-gated secondary prior."
        ),
        "source_run": str(source_run),
        "source_experiment_08_run": str(experiment_08_run),
        "source_solar_run": str(solar_run),
        "source_vector_run": str(vector_run),
        "strategy": {
            "primary_topology": "vector_structural_baseline",
            "secondary_prior": "visual_dsm_fusion",
            "strict_topology_plane_count": int(baseline_labels.max()),
            "maximum_snap_distance": args.maximum_snap_distance,
            "candidate_weights": strict_variants,
            "controlled_enrichment": {
                "minimum_improvement_meters": args.minimum_split_improvement_meters,
                "minimum_improvement_ratio": args.minimum_split_improvement_ratio,
                "maximum_subplanes_per_region": args.maximum_subplanes_per_region,
            },
            "selection": (
                "88% independent DSM/visual metric quality and 12% boundary straightness"
            ),
        },
        "secondary_support": support_diagnostics,
        "controlled_enrichment_records": conservative_records,
        "exploratory_enrichment_records": exploratory_records,
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
    accepted_conservative = sum(1 for record in conservative_records if record["accepted"])
    accepted_exploratory = sum(1 for record in exploratory_records if record["accepted"])
    print(
        "Accepted enriched primary regions: "
        f"controlled={accepted_conservative}; exploratory={accepted_exploratory}"
    )
    print(f"Output: {run_directory}")
    return run_directory


if __name__ == "__main__":
    run(parse_args())
