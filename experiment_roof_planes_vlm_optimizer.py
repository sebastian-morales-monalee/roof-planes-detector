from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import MiniBatchKMeans
from skimage.segmentation import relabel_sequential, watershed

from experiment_roof_planes import (
    make_contact_sheet,
    render_planes,
    save_label_mask,
    summarize_planes,
    write_geojson,
)
from experiment_roof_planes_vlm_topology import (
    create_evidence_map,
    find_latest_structural_run,
    first_pass_prompt,
    load_mask,
    make_mask_overlay,
    normalized_point,
    partition_roof,
    request_topology,
    save_scalar,
    snap_boundaries,
    topology_validation,
)
from run_artifacts import create_run_directory
from segment_image import prepare_cache, resolve_device


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs"
EXPERIMENT_NAME = "roof_planes_experiment_05_vlm_optimizer"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Experiment 05: generate multiple independent VLM roof topologies, "
            "refine their planes with SAM2, fit relative-depth planes, and select "
            "the strongest globally consistent result."
        )
    )
    parser.add_argument("--source-run", type=Path)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument("--detail", choices=("original", "high", "auto"), default="original")
    parser.add_argument(
        "--reasoning-effort",
        choices=("medium", "high", "xhigh", "max"),
        default="high",
    )
    parser.add_argument("--hypotheses", type=int, choices=(2, 3), default=2)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--sam-type", default="sam2.1_hiera_small")
    parser.add_argument("--snap-margin", type=int, default=42)
    parser.add_argument("--reference-planes", type=int, default=14)
    parser.add_argument(
        "--merge-coplanar",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Merge only very strongly supported near-coplanar neighboring regions.",
    )
    return parser.parse_args()


def hypothesis_prompt(index: int) -> str:
    strategies = (
        "Prioritize architectural roof grammar, junction validity, and a parsimonious plane graph.",
        "Prioritize depth/normal discontinuities and geometric consistency while rejecting shadows and objects.",
        "Prioritize complete coverage and coherent shared ridges, hips, valleys, and step boundaries.",
    )
    return (
        first_pass_prompt()
        + "\n\nThis is an independent hypothesis. Do not assume another solution exists. "
        + strategies[index % len(strategies)]
    )


def largest_connected_component(mask: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if count <= 1:
        return mask.copy(), {"component_count": 0, "removed_pixels": 0}
    areas = stats[1:, cv2.CC_STAT_AREA]
    selected = int(np.argmax(areas)) + 1
    cleaned = labels == selected
    return cleaned, {
        "component_count": int(count - 1),
        "selected_component_pixels": int(np.count_nonzero(cleaned)),
        "removed_component_count": int(count - 2),
        "removed_pixels": int(np.count_nonzero(mask & ~cleaned)),
    }


def adjacency_pairs(labels: np.ndarray) -> set[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    for first, second in (
        (labels[:, :-1], labels[:, 1:]),
        (labels[:-1, :], labels[1:, :]),
    ):
        changed = (first != second) & (first > 0) & (second > 0)
        for a, b in zip(first[changed], second[changed], strict=True):
            one, two = sorted((int(a), int(b)))
            pairs.add((one, two))
    return pairs


def boundary_support(labels: np.ndarray, evidence: np.ndarray) -> float:
    boundary = cv2.morphologyEx(
        labels.astype(np.float32), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)
    ) > 0
    boundary &= labels > 0
    values = evidence[boundary]
    return float(values.mean()) if values.size else 0.0


def negative_seed_points(
    plan: dict[str, Any],
    plane_id: int,
    width: int,
    height: int,
) -> list[tuple[int, int]]:
    planes = {int(item["plane_id"]): item for item in plan["planes"]}
    plane = planes[plane_id]
    adjacent = [int(value) for value in plane["adjacent_plane_ids"] if int(value) in planes]
    if not adjacent:
        origin = normalized_point(plane["seed"], width, height)
        candidates = sorted(
            (
                math.dist(origin, normalized_point(other["seed"], width, height)),
                normalized_point(other["seed"], width, height),
            )
            for other_id, other in planes.items()
            if other_id != plane_id
        )
        return [point for _, point in candidates[:3]]
    return [normalized_point(planes[value]["seed"], width, height) for value in adjacent[:6]]


def refine_partition_with_sam2(
    predictor: Any,
    rgb: np.ndarray,
    mask: np.ndarray,
    initial_labels: np.ndarray,
    label_to_plane: dict[int, int],
    plan: dict[str, Any],
    evidence: np.ndarray,
    output_directory: Path,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    predictor.set_image(np.ascontiguousarray(rgb))
    height, width = mask.shape
    accepted: dict[int, tuple[np.ndarray, float]] = {}
    records: list[dict[str, Any]] = []
    mask_directory = output_directory / "sam2_plane_masks"
    mask_directory.mkdir(parents=True, exist_ok=True)

    plan_by_id = {int(item["plane_id"]): item for item in plan["planes"]}
    for label, plane_id in sorted(label_to_plane.items()):
        initial = initial_labels == label
        if not np.any(initial) or plane_id not in plan_by_id:
            continue
        ys, xs = np.where(initial)
        positive = normalized_point(plan_by_id[plane_id]["seed"], width, height)
        if not initial[positive[1], positive[0]]:
            positive = (int(np.median(xs)), int(np.median(ys)))
        negatives = negative_seed_points(plan, plane_id, width, height)
        points = np.asarray([positive, *negatives], dtype=np.float32)
        point_labels = np.asarray([1, *([0] * len(negatives))], dtype=np.int32)
        margin = 18
        box = np.asarray(
            [
                max(int(xs.min()) - margin, 0),
                max(int(ys.min()) - margin, 0),
                min(int(xs.max()) + margin, width - 1),
                min(int(ys.max()) + margin, height - 1),
            ],
            dtype=np.float32,
        )
        masks, scores, _ = predictor.predict(
            point_coords=points,
            point_labels=point_labels,
            box=box,
            multimask_output=True,
        )

        initial_pixels = max(int(initial.sum()), 1)
        best: tuple[float, np.ndarray, float, float, float] | None = None
        for raw_candidate, sam_score in zip(masks, scores, strict=True):
            candidate = np.asarray(raw_candidate, dtype=bool) & mask
            if not candidate[positive[1], positive[0]]:
                continue
            negative_hits = sum(candidate[y, x] for x, y in negatives)
            intersection = int(np.count_nonzero(candidate & initial))
            union = max(int(np.count_nonzero(candidate | initial)), 1)
            iou = intersection / union
            area_ratio = int(candidate.sum()) / initial_pixels
            candidate_boundary = cv2.morphologyEx(
                candidate.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)
            ).astype(bool)
            support = float(evidence[candidate_boundary].mean()) if np.any(candidate_boundary) else 0.0
            objective = (
                0.42 * iou
                + 0.28 * float(sam_score)
                + 0.20 * support
                + 0.10 * (1.0 - negative_hits / max(len(negatives), 1))
            )
            if best is None or objective > best[0]:
                best = (objective, candidate, iou, area_ratio, support)

        accepted_quality = False
        if best is not None:
            objective, candidate, iou, area_ratio, support = best
            accepted_quality = iou >= 0.32 and 0.42 <= area_ratio <= 1.90
            if accepted_quality:
                accepted[label] = (candidate, objective)
                Image.fromarray((candidate * 255).astype(np.uint8)).save(
                    mask_directory / f"plane_{plane_id:02d}_label_{label:02d}.png"
                )
            records.append(
                {
                    "label": label,
                    "plane_id": plane_id,
                    "accepted": accepted_quality,
                    "objective": round(objective, 5),
                    "iou_with_initial_partition": round(iou, 5),
                    "area_ratio": round(area_ratio, 5),
                    "boundary_support": round(support, 5),
                    "negative_seed_count": len(negatives),
                }
            )

    if not accepted:
        return initial_labels.copy(), records

    # Non-overlapping SAM2 cores become watershed markers. The final watershed
    # guarantees one owner per building pixel and preserves closed regions.
    marker_stack = np.zeros((len(accepted), height, width), dtype=bool)
    labels_in_stack = list(sorted(accepted))
    for index, label in enumerate(labels_in_stack):
        candidate, _ = accepted[label]
        marker_stack[index] = cv2.erode(
            candidate.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1
        ).astype(bool)
    ownership_count = marker_stack.sum(axis=0)
    markers = np.zeros(mask.shape, dtype=np.int32)
    for index, label in enumerate(labels_in_stack):
        core = marker_stack[index] & (ownership_count == 1)
        if np.count_nonzero(core) < 8:
            core = ndimage.binary_erosion(initial_labels == label, iterations=2)
        if not np.any(core):
            ys, xs = np.where(initial_labels == label)
            if xs.size:
                core[int(np.median(ys)), int(np.median(xs))] = True
        markers[core] = label

    missing_labels = sorted(set(int(v) for v in np.unique(initial_labels) if v > 0) - set(labels_in_stack))
    for label in missing_labels:
        core = ndimage.binary_erosion(initial_labels == label, iterations=2)
        if not np.any(core):
            ys, xs = np.where(initial_labels == label)
            if xs.size:
                core[int(np.median(ys)), int(np.median(xs))] = True
        markers[core] = label

    barrier = np.clip(0.72 * evidence + 0.28 * (initial_labels == 0), 0.0, 1.0)
    refined = watershed(barrier, markers=markers, mask=mask, compactness=0.0005)
    missing = mask & (refined == 0)
    if np.any(missing) and np.any(refined > 0):
        _, nearest = ndimage.distance_transform_edt(
            refined == 0,
            return_distances=True,
            return_indices=True,
        )
        refined[missing] = refined[nearest[0][missing], nearest[1][missing]]
    return np.asarray(refined, dtype=np.int32), records


def robust_plane_fit(
    depth: np.ndarray,
    region: np.ndarray,
    *,
    max_points: int = 7000,
) -> dict[str, Any]:
    ys, xs = np.where(region)
    if xs.size < 12:
        return {"valid": False, "pixel_count": int(xs.size)}
    rng = np.random.default_rng(20260729)
    if xs.size > max_points:
        selected = rng.choice(xs.size, max_points, replace=False)
        xs, ys = xs[selected], ys[selected]
    height, width = depth.shape
    x = xs.astype(np.float64) / max(width - 1, 1)
    y = ys.astype(np.float64) / max(height - 1, 1)
    z = depth[ys, xs].astype(np.float64)
    design = np.column_stack((x, y, np.ones_like(x)))
    keep = np.ones(z.size, dtype=bool)
    coefficients = np.zeros(3, dtype=np.float64)
    for _ in range(5):
        coefficients, *_ = np.linalg.lstsq(design[keep], z[keep], rcond=None)
        residual = np.abs(z - design @ coefficients)
        median = float(np.median(residual[keep]))
        mad = float(np.median(np.abs(residual[keep] - median)))
        threshold = max(median + 2.8 * 1.4826 * mad, 0.008)
        updated = residual <= threshold
        if np.array_equal(updated, keep) or np.count_nonzero(updated) < 8:
            break
        keep = updated
    prediction = design @ coefficients
    rmse = float(np.sqrt(np.mean((z[keep] - prediction[keep]) ** 2)))
    normal = np.asarray([-coefficients[0], -coefficients[1], 1.0])
    normal /= max(float(np.linalg.norm(normal)), 1e-12)
    return {
        "valid": True,
        "pixel_count": int(region.sum()),
        "sample_count": int(z.size),
        "coefficients": [round(float(value), 8) for value in coefficients],
        "normal": [round(float(value), 8) for value in normal],
        "rmse": round(rmse, 8),
        "inlier_ratio": round(float(keep.mean()), 8),
    }


def fit_all_planes(labels: np.ndarray, depth: np.ndarray) -> dict[int, dict[str, Any]]:
    return {
        int(label): robust_plane_fit(depth, labels == label)
        for label in np.unique(labels)
        if label > 0
    }


def normal_angle(first: dict[str, Any], second: dict[str, Any]) -> float:
    if not first.get("valid") or not second.get("valid"):
        return 180.0
    one = np.asarray(first["normal"], dtype=np.float64)
    two = np.asarray(second["normal"], dtype=np.float64)
    cosine = float(np.clip(abs(np.dot(one, two)), -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def shared_boundary_support(labels: np.ndarray, a: int, b: int, evidence: np.ndarray) -> float:
    region_a = labels == a
    region_b = labels == b
    contact = cv2.dilate(region_a.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & region_b
    return float(evidence[contact].mean()) if np.any(contact) else 1.0


def merge_near_coplanar(
    labels: np.ndarray,
    depth: np.ndarray,
    evidence: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, Any]], list[dict[str, Any]]]:
    merged = labels.copy()
    decisions: list[dict[str, Any]] = []
    split_flags: list[dict[str, Any]] = []
    fits = fit_all_planes(merged, depth)
    roof_pixels = max(int(np.count_nonzero(merged)), 1)
    for label, fit in fits.items():
        area_fraction = int(np.count_nonzero(merged == label)) / roof_pixels
        if fit.get("valid") and area_fraction >= 0.035 and float(fit["rmse"]) >= 0.055:
            split_flags.append(
                {
                    "label": label,
                    "reason": "high_relative_depth_plane_residual",
                    "rmse": fit["rmse"],
                    "area_fraction": round(area_fraction, 6),
                }
            )

    for a, b in sorted(adjacency_pairs(merged)):
        fits = fit_all_planes(merged, depth)
        angle = normal_angle(fits.get(a, {}), fits.get(b, {}))
        support = shared_boundary_support(merged, a, b, evidence)
        first = fits.get(a, {})
        second = fits.get(b, {})
        intercept_delta = (
            abs(float(first["coefficients"][2]) - float(second["coefficients"][2]))
            if first.get("valid") and second.get("valid")
            else 1.0
        )
        accepted = angle <= 3.0 and support <= 0.18 and intercept_delta <= 0.025
        decisions.append(
            {
                "labels": [a, b],
                "normal_angle_degrees": round(angle, 5),
                "boundary_support": round(support, 5),
                "intercept_delta": round(intercept_delta, 5),
                "merged": accepted,
            }
        )
        if accepted:
            destination, source = (a, b) if np.count_nonzero(merged == a) >= np.count_nonzero(merged == b) else (b, a)
            merged[merged == source] = destination
    return np.asarray(merged, dtype=np.int32), decisions, split_flags


def hypothesis_score(
    plan: dict[str, Any],
    labels: np.ndarray,
    mask: np.ndarray,
    label_to_plane: dict[int, int],
    evidence: np.ndarray,
    fits: dict[int, dict[str, Any]],
    sam_records: list[dict[str, Any]],
) -> tuple[float, dict[str, Any]]:
    validation = topology_validation(plan, labels, mask, label_to_plane)
    expected = len(validation["expected_adjacencies"])
    missing = len(validation["missing_expected_adjacencies"])
    unexpected = len(validation["unexpected_adjacencies"])
    adjacency_recall = 1.0 - missing / max(expected, 1)
    boundary = boundary_support(labels, evidence)
    valid_fits = [item for item in fits.values() if item.get("valid")]
    planar = (
        float(np.mean([float(item["inlier_ratio"]) * max(0.0, 1.0 - 5.0 * float(item["rmse"])) for item in valid_fits]))
        if valid_fits
        else 0.0
    )
    sam_acceptance = sum(bool(item["accepted"]) for item in sam_records) / max(len(sam_records), 1)
    disconnected = len(validation["disconnected_plane_ids"])
    plane_count = len([value for value in np.unique(labels) if value > 0])
    complexity_penalty = max(0, plane_count - 22) / 22
    score = (
        0.22 * float(validation["coverage_percentage"]) / 100.0
        + 0.22 * adjacency_recall
        + 0.24 * boundary
        + 0.32 * planar
        - 0.025 * unexpected
        - 0.05 * disconnected
        - 0.04 * complexity_penalty
    )
    details = {
        "score": round(score, 8),
        "coverage_percentage": validation["coverage_percentage"],
        "adjacency_recall": round(adjacency_recall, 8),
        "missing_expected_adjacencies": missing,
        "unexpected_realized_adjacencies": unexpected,
        "boundary_evidence_support": round(boundary, 8),
        "planarity_score": round(planar, 8),
        "sam2_acceptance_ratio": round(sam_acceptance, 8),
        "disconnected_label_count": disconnected,
        "plane_count": plane_count,
        "validation": validation,
    }
    return score, details


def approximate_manual_labels(
    original: np.ndarray,
    reference: np.ndarray,
    mask: np.ndarray,
    plane_count: int,
) -> np.ndarray:
    # The supplied reference is a translucent paint-over, not an indexed mask.
    # Cluster its color displacement from the original to obtain an explicitly
    # marked approximate ground truth for quantitative diagnostics.
    delta = (reference.astype(np.float32) - original.astype(np.float32)) / 255.0
    hsv = cv2.cvtColor(reference, cv2.COLOR_RGB2HSV).astype(np.float32)
    features = np.column_stack(
        (
            delta[mask],
            hsv[..., 0][mask, None] / 180.0,
            hsv[..., 1][mask, None] / 255.0,
        )
    )
    model = MiniBatchKMeans(
        n_clusters=plane_count,
        random_state=20260729,
        batch_size=4096,
        n_init=12,
    )
    clustered = model.fit_predict(features) + 1
    labels = np.zeros(mask.shape, dtype=np.int32)
    labels[mask] = clustered
    labels = ndimage.median_filter(labels, size=3)
    labels[~mask] = 0
    relabeled, _, _ = relabel_sequential(labels)
    return np.asarray(relabeled, dtype=np.int32)


def boundary_mask(labels: np.ndarray) -> np.ndarray:
    return cv2.morphologyEx(
        labels.astype(np.float32), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)
    ) > 0


def evaluate_labels(predicted: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    pred_ids = [int(value) for value in np.unique(predicted) if value > 0]
    ref_ids = [int(value) for value in np.unique(reference) if value > 0]
    iou = np.zeros((len(pred_ids), len(ref_ids)), dtype=np.float64)
    for row, pred_id in enumerate(pred_ids):
        pred = predicted == pred_id
        for column, ref_id in enumerate(ref_ids):
            ref = reference == ref_id
            union = np.count_nonzero(pred | ref)
            iou[row, column] = np.count_nonzero(pred & ref) / max(union, 1)
    rows, columns = linear_sum_assignment(-iou)
    matches = [
        {"predicted_label": pred_ids[row], "reference_label": ref_ids[column], "iou": round(float(iou[row, column]), 8)}
        for row, column in zip(rows, columns, strict=True)
    ]
    pred_boundary = boundary_mask(predicted)
    ref_boundary = boundary_mask(reference)
    tolerance = np.ones((7, 7), np.uint8)
    pred_hit = pred_boundary & cv2.dilate(ref_boundary.astype(np.uint8), tolerance).astype(bool)
    ref_hit = ref_boundary & cv2.dilate(pred_boundary.astype(np.uint8), tolerance).astype(bool)
    precision = np.count_nonzero(pred_hit) / max(np.count_nonzero(pred_boundary), 1)
    recall = np.count_nonzero(ref_hit) / max(np.count_nonzero(ref_boundary), 1)
    boundary_f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "reference_type": "approximate_indexed_labels_derived_from_translucent_manual_overlay",
        "predicted_plane_count": len(pred_ids),
        "reference_plane_count": len(ref_ids),
        "matched_mean_iou": round(float(np.mean([item["iou"] for item in matches])) if matches else 0.0, 8),
        "boundary_precision_at_3px": round(float(precision), 8),
        "boundary_recall_at_3px": round(float(recall), 8),
        "boundary_f1_at_3px": round(float(boundary_f1), 8),
        "matches": matches,
    }


def run(args: argparse.Namespace) -> Path:
    load_dotenv(PROJECT_ROOT / ".env")
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not defined in .env or the environment")
    prepare_cache(PROJECT_ROOT)
    device = resolve_device(args.device)
    output_root = args.output_root.resolve()
    source_run = args.source_run.resolve() if args.source_run else find_latest_structural_run(output_root)
    run_directory = create_run_directory(output_root, EXPERIMENT_NAME)

    image = Image.open(source_run / "input.png").convert("RGB")
    rgb = np.array(image)
    raw_mask = load_mask(source_run / "building_mask.png")
    mask, mask_cleanup = largest_connected_component(raw_mask)
    mask_overlay = make_mask_overlay(image, mask)
    structural = Image.open(source_run / "structural_boundary_map.png").convert("RGB")
    depth_color = Image.open(source_run / "relative_depth_color.png").convert("RGB")
    normals = Image.open(source_run / "surface_normals.png").convert("RGB")
    depth_edges = Image.open(source_run / "depth_edges.png").convert("L") if (source_run / "depth_edges.png").is_file() else None
    normal_edges = Image.open(source_run / "normal_edges.png").convert("L") if (source_run / "normal_edges.png").is_file() else None
    evidence = create_evidence_map(image, structural, depth_edges, normal_edges, mask)
    depth_raw = np.asarray(Image.open(source_run / "relative_depth_16bit.png"), dtype=np.float32)
    depth = np.zeros_like(depth_raw, dtype=np.float32)
    if np.any(mask):
        values = depth_raw[mask]
        depth[mask] = (values - values.min()) / max(float(values.max() - values.min()), 1.0)

    image.save(run_directory / "input.png")
    Image.fromarray((mask * 255).astype(np.uint8)).save(run_directory / "building_mask.png")
    save_scalar(evidence, run_directory / "combined_boundary_evidence.png")

    from lang_sam import LangSAM

    print(f"Loading SAM2 {args.sam_type} on {device}...")
    lang_sam = LangSAM(sam_type=args.sam_type, device=device)
    predictor = lang_sam.sam.predictor
    client = OpenAI()
    hypothesis_results: list[dict[str, Any]] = []

    for index in range(args.hypotheses):
        number = index + 1
        hypothesis_directory = run_directory / f"hypothesis_{number:02d}"
        hypothesis_directory.mkdir(parents=True, exist_ok=True)
        print(f"Requesting independent topology hypothesis {number}/{args.hypotheses}...")
        plan, metadata = request_topology(
            client,
            model=args.model,
            detail=args.detail,
            reasoning_effort=args.reasoning_effort,
            pro=False,
            prompt=hypothesis_prompt(index),
            images=[image, mask_overlay, structural, depth_color, normals],
        )
        (hypothesis_directory / "vlm_topology.json").write_text(
            json.dumps(plan, indent=2), encoding="utf-8"
        )
        snapped = snap_boundaries(plan, evidence, mask, args.snap_margin)
        initial, barrier, label_to_plane = partition_roof(plan, mask, evidence, snapped)
        refined, sam_records = refine_partition_with_sam2(
            predictor,
            rgb,
            mask,
            initial,
            label_to_plane,
            plan,
            evidence,
            hypothesis_directory,
        )
        merge_records: list[dict[str, Any]] = []
        split_flags: list[dict[str, Any]] = []
        merged_labels = refined
        if args.merge_coplanar:
            merged_labels, merge_records, split_flags = merge_near_coplanar(refined, depth, evidence)
        variants: dict[str, np.ndarray] = {
            "initial_vlm_partition": initial,
            "sam2_refined_partition": refined,
            "coplanar_merged_partition": merged_labels,
        }
        variant_results: dict[str, dict[str, Any]] = {}
        for variant_name, variant_labels in variants.items():
            variant_fits = fit_all_planes(variant_labels, depth)
            variant_score, variant_details = hypothesis_score(
                plan,
                variant_labels,
                mask,
                label_to_plane,
                evidence,
                variant_fits,
                sam_records,
            )
            variant_results[variant_name] = {
                "score": variant_score,
                "details": variant_details,
                "fits": variant_fits,
            }
        selected_variant = max(
            variant_results,
            key=lambda name: float(variant_results[name]["score"]),
        )
        final_labels = variants[selected_variant]
        score = float(variant_results[selected_variant]["score"])
        score_details = variant_results[selected_variant]["details"]
        fits = variant_results[selected_variant]["fits"]

        initial_colors, initial_overlay, _ = render_planes(rgb, initial)
        refined_colors, refined_overlay, _ = render_planes(rgb, refined)
        final_colors, final_overlay, _ = render_planes(rgb, final_labels)
        save_label_mask(initial, hypothesis_directory / "initial_topology_labels.png")
        save_label_mask(refined, hypothesis_directory / "sam2_refined_labels.png")
        save_label_mask(merged_labels, hypothesis_directory / "coplanar_merged_labels.png")
        save_label_mask(final_labels, hypothesis_directory / "optimized_labels.png")
        Image.fromarray(initial_colors).save(hypothesis_directory / "initial_topology_color.png")
        Image.fromarray(initial_overlay).save(hypothesis_directory / "initial_topology_overlay.png")
        Image.fromarray(refined_overlay).save(hypothesis_directory / "sam2_refined_overlay.png")
        Image.fromarray(final_colors).save(hypothesis_directory / "optimized_color.png")
        Image.fromarray(final_overlay).save(hypothesis_directory / "optimized_overlay.png")
        delta = initial != refined
        delta_visual = (rgb.astype(np.float32) * 0.35).astype(np.uint8)
        delta_visual[delta] = np.asarray([255, 215, 0], dtype=np.uint8)
        Image.fromarray((delta * 255).astype(np.uint8)).save(
            hypothesis_directory / "sam2_refinement_delta_mask.png"
        )
        Image.fromarray(delta_visual).save(
            hypothesis_directory / "sam2_refinement_difference.png"
        )
        save_scalar(barrier, hypothesis_directory / "watershed_barrier.png")
        summaries = summarize_planes(final_labels)
        write_geojson(final_labels, summaries, hypothesis_directory / "optimized_planes.geojson")
        (hypothesis_directory / "sam2_refinement.json").write_text(json.dumps(sam_records, indent=2), encoding="utf-8")
        (hypothesis_directory / "relative_depth_plane_fits.json").write_text(
            json.dumps({str(key): value for key, value in fits.items()}, indent=2), encoding="utf-8"
        )
        local_result = {
            "hypothesis": number,
            "vlm_response": metadata,
            "selected_variant": selected_variant,
            "score": score_details,
            "variant_scores": {
                name: {
                    "score": round(float(values["score"]), 8),
                    "details": values["details"],
                }
                for name, values in variant_results.items()
            },
            "coplanar_merge_decisions": merge_records,
            "split_flags": split_flags,
            "sam2_refinement": sam_records,
            "label_to_proposed_plane": {str(key): value for key, value in label_to_plane.items()},
            "planes": [asdict(summary) for summary in summaries],
        }
        (hypothesis_directory / "result.json").write_text(json.dumps(local_result, indent=2), encoding="utf-8")
        make_contact_sheet(
            [
                ("Initial VLM topology", Image.fromarray(initial_overlay)),
                ("SAM2 positive/negative refinement", Image.fromarray(refined_overlay)),
                (f"Selected: {selected_variant}", Image.fromarray(final_overlay)),
                ("Pixels changed by SAM2", Image.fromarray(delta_visual)),
            ],
            hypothesis_directory / "refinement_comparison.jpg",
        )
        hypothesis_results.append(
            {
                "number": number,
                "directory": str(hypothesis_directory),
                "score": score,
                "score_details": score_details,
                "labels": final_labels,
                "overlay": Image.fromarray(final_overlay),
                "result": local_result,
            }
        )

    winner = max(hypothesis_results, key=lambda item: item["score"])
    winner_labels = winner["labels"]
    winner_colors, winner_overlay, _ = render_planes(rgb, winner_labels)
    save_label_mask(winner_labels, run_directory / "winner_roof_planes_labels.png")
    Image.fromarray(winner_colors).save(run_directory / "winner_roof_planes_color.png")
    Image.fromarray(winner_overlay).save(run_directory / "winner_roof_planes_overlay.png")
    winner_summaries = summarize_planes(winner_labels)
    write_geojson(winner_labels, winner_summaries, run_directory / "winner_roof_planes.geojson")

    evaluation: dict[str, Any] | None = None
    comparison = [
        (f"Hypothesis {item['number']} | score {item['score']:.3f}", item["overlay"])
        for item in hypothesis_results
    ]
    comparison.append((f"Selected hypothesis {winner['number']}", Image.fromarray(winner_overlay)))
    manual_reference_path = source_run / "manual_reference.png"
    if manual_reference_path.is_file():
        manual = np.asarray(Image.open(manual_reference_path).convert("RGB"))
        reference_labels = approximate_manual_labels(rgb, manual, mask, args.reference_planes)
        ref_colors, ref_overlay, _ = render_planes(rgb, reference_labels)
        save_label_mask(reference_labels, run_directory / "manual_reference_indexed_approx.png")
        Image.fromarray(ref_colors).save(run_directory / "manual_reference_indexed_approx_color.png")
        Image.fromarray(ref_overlay).save(run_directory / "manual_reference_indexed_approx_overlay.png")
        Image.fromarray(manual).save(run_directory / "manual_reference.png")
        evaluation = evaluate_labels(winner_labels, reference_labels)
        (run_directory / "quantitative_evaluation.json").write_text(
            json.dumps(evaluation, indent=2), encoding="utf-8"
        )
        comparison.extend(
            [
                ("Approximate indexed manual reference", Image.fromarray(ref_overlay)),
                ("Original manual paint-over", Image.fromarray(manual)),
            ]
        )
    make_contact_sheet(comparison, run_directory / "qualitative_comparison.jpg")

    serializable_hypotheses = [
        {
            "hypothesis": item["number"],
            "score": round(float(item["score"]), 8),
            "score_details": item["score_details"],
            "directory": item["directory"],
        }
        for item in hypothesis_results
    ]
    result = {
        "experiment": EXPERIMENT_NAME,
        "purpose": "Multi-hypothesis VLM topology, SAM2 positive/negative refinement, robust relative-depth plane fitting, guarded global optimization, and automatic winner selection",
        "source_run": str(source_run),
        "device": str(device),
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "hypothesis_count": args.hypotheses,
        "building_mask_cleanup": mask_cleanup,
        "winner_hypothesis": winner["number"],
        "winner_score": round(float(winner["score"]), 8),
        "winner_selected_without_manual_reference": True,
        "hypotheses": serializable_hypotheses,
        "manual_evaluation": evaluation,
        "manual_reference_supplied_to_model": False,
        "manual_reference_used_for_winner_selection": False,
        "manual_reference_warning": "The current manual reference is a translucent paint-over; indexed labels are an approximate color-displacement clustering used only for diagnostics.",
        "planes": [asdict(summary) for summary in winner_summaries],
    }
    (run_directory / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Selected hypothesis {winner['number']} with score {winner['score']:.4f}")
    if evaluation:
        print(
            "Approximate manual evaluation: "
            f"mIoU={evaluation['matched_mean_iou']:.4f}; "
            f"boundary F1={evaluation['boundary_f1_at_3px']:.4f}"
        )
    print(f"Output: {run_directory}")
    return run_directory


if __name__ == "__main__":
    run(parse_args())
