from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from scipy import sparse
from scipy.ndimage import distance_transform_edt
from skimage.segmentation import mark_boundaries

from experiment_roof_planes import (
    DEFAULT_DEPTH_MODEL,
    DEFAULT_INPUT,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_REFERENCE,
    build_region_features,
    cluster_superpixels,
    colorize_scalar,
    compute_geometry,
    estimate_depth,
    make_contact_sheet,
    make_superpixels,
    normalize_inside_mask,
    render_planes,
    save_label_mask,
    split_and_merge_small_components,
    summarize_planes,
    write_geojson,
)
from run_artifacts import create_run_directory
from segment_image import (
    create_overlay as create_langsam_overlay,
    prepare_cache,
    resolve_device,
    select_prediction,
)

PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Experiment 02: generate roof-plane candidates using LangSAM, relative "
            "depth, structural line barriers, and optional SAM2 refinement."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--prompt", default="building.")
    parser.add_argument("--sam-type", default="sam2.1_hiera_small")
    parser.add_argument("--box-threshold", type=float, default=0.25)
    parser.add_argument("--text-threshold", type=float, default=0.20)
    parser.add_argument("--depth-model", default=DEFAULT_DEPTH_MODEL)
    parser.add_argument("--superpixels", type=int, default=180)
    parser.add_argument("--compactness", type=float, default=10.0)
    parser.add_argument("--cluster-threshold", type=float, default=7.0)
    parser.add_argument("--min-plane-percentage", type=float, default=2.0)
    parser.add_argument("--barrier-threshold", type=float, default=0.47)
    parser.add_argument("--minimum-line-length", type=float, default=26.0)
    parser.add_argument("--line-support-threshold", type=float, default=0.12)
    parser.add_argument("--sam-proposal-min-iou", type=float, default=0.76)
    parser.add_argument("--sam-proposal-min-stability", type=float, default=0.86)
    parser.add_argument(
        "--disable-sam-refinement",
        action="store_true",
        help="Generate structural candidates without point/box refinement by SAM2.",
    )
    return parser.parse_args()


def mask_bbox(mask: np.ndarray, margin: int = 0) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if xs.size == 0:
        raise ValueError("The LangSAM building mask is empty")
    height, width = mask.shape
    return (
        max(int(xs.min()) - margin, 0),
        max(int(ys.min()) - margin, 0),
        min(int(xs.max()) + margin + 1, width),
        min(int(ys.max()) + margin + 1, height),
    )


def gradient_map(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    x = cv2.Sobel(image.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    y = cv2.Sobel(image.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    return normalize_inside_mask(np.hypot(x, y), mask)


def absorb_isolated_tiny_planes(
    planes: np.ndarray,
    roof_mask: np.ndarray,
    minimum_percentage: float,
) -> np.ndarray:
    """Assign isolated fragments to the nearest retained plane and relabel densely."""
    minimum_pixels = max(
        16,
        int(np.count_nonzero(roof_mask) * minimum_percentage / 100.0),
    )
    cleaned = planes.copy()
    small_ids = [
        int(plane_id)
        for plane_id in np.unique(cleaned)
        if plane_id > 0 and np.count_nonzero(cleaned == plane_id) < minimum_pixels
    ]
    if small_ids:
        small_mask = np.isin(cleaned, small_ids)
        retained = (cleaned > 0) & ~small_mask
        if np.any(retained):
            _, nearest = distance_transform_edt(~retained, return_indices=True)
            cleaned[small_mask] = cleaned[
                nearest[0][small_mask], nearest[1][small_mask]
            ]

    relabeled = np.zeros_like(cleaned, dtype=np.int32)
    for new_id, old_id in enumerate(
        [int(value) for value in np.unique(cleaned) if value > 0], start=1
    ):
        relabeled[cleaned == old_id] = new_id
    relabeled[~roof_mask] = 0
    return relabeled


def normal_discontinuity(normals: np.ndarray, mask: np.ndarray) -> np.ndarray:
    strength = np.zeros(mask.shape, dtype=np.float32)
    for channel in range(3):
        strength += gradient_map(normals[..., channel], mask) ** 2
    return normalize_inside_mask(np.sqrt(strength), mask)


def proposal_boundaries(
    lang_sam: Any,
    rgb: np.ndarray,
    roof_mask: np.ndarray,
    min_iou: float,
    min_stability: float,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    x0, y0, x1, y1 = mask_bbox(roof_mask, margin=12)
    crop = np.ascontiguousarray(rgb[y0:y1, x0:x1])
    crop_roof = roof_mask[y0:y1, x0:x1]
    roof_pixels = max(int(np.count_nonzero(crop_roof)), 1)
    generated = lang_sam.sam.generate(crop)
    boundary = np.zeros(roof_mask.shape, dtype=np.uint8)
    accepted: list[dict[str, Any]] = []
    kernel = np.ones((3, 3), dtype=np.uint8)

    for index, item in enumerate(generated):
        segmentation = np.asarray(item["segmentation"], dtype=bool)
        intersection = segmentation & crop_roof
        intersection_pixels = int(np.count_nonzero(intersection))
        segmentation_pixels = max(int(np.count_nonzero(segmentation)), 1)
        purity = intersection_pixels / segmentation_pixels
        roof_fraction = intersection_pixels / roof_pixels
        predicted_iou = float(item.get("predicted_iou", 0.0))
        stability = float(item.get("stability_score", 0.0))
        if not (
            predicted_iou >= min_iou
            and stability >= min_stability
            and purity >= 0.72
            and 0.004 <= roof_fraction <= 0.68
        ):
            continue

        local_boundary = cv2.morphologyEx(
            intersection.astype(np.uint8), cv2.MORPH_GRADIENT, kernel
        )
        boundary[y0:y1, x0:x1] |= local_boundary
        accepted.append(
            {
                "proposal_id": index,
                "predicted_iou": round(predicted_iou, 4),
                "stability_score": round(stability, 4),
                "roof_fraction": round(roof_fraction, 4),
                "purity": round(purity, 4),
            }
        )

    boundary[~roof_mask] = 0
    return boundary.astype(bool), accepted


def detect_structural_lines(
    gray: np.ndarray,
    roof_mask: np.ndarray,
    support_map: np.ndarray,
    minimum_length: float,
    support_threshold: float,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    encoded = np.clip(gray * 255.0, 0, 255).astype(np.uint8)
    detector = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    detected = detector.detect(encoded)[0]
    line_mask = np.zeros(roof_mask.shape, dtype=np.uint8)
    records: list[dict[str, Any]] = []
    if detected is None:
        return line_mask.astype(bool), records

    height, width = roof_mask.shape
    for raw_line in np.asarray(detected).reshape(-1, 4):
        x0, y0, x1, y1 = [float(value) for value in raw_line]
        length = math.hypot(x1 - x0, y1 - y0)
        if length < minimum_length:
            continue
        samples = max(int(length), 12)
        xs = np.clip(np.rint(np.linspace(x0, x1, samples)).astype(int), 0, width - 1)
        ys = np.clip(np.rint(np.linspace(y0, y1, samples)).astype(int), 0, height - 1)
        inside_fraction = float(np.mean(roof_mask[ys, xs]))
        support = float(np.mean(support_map[ys, xs][roof_mask[ys, xs]])) if np.any(
            roof_mask[ys, xs]
        ) else 0.0
        if inside_fraction < 0.72 or support < support_threshold:
            continue

        extension = min(length * 0.10, 18.0)
        direction_x = (x1 - x0) / length
        direction_y = (y1 - y0) / length
        start = (
            int(np.clip(round(x0 - direction_x * extension), 0, width - 1)),
            int(np.clip(round(y0 - direction_y * extension), 0, height - 1)),
        )
        end = (
            int(np.clip(round(x1 + direction_x * extension), 0, width - 1)),
            int(np.clip(round(y1 + direction_y * extension), 0, height - 1)),
        )
        cv2.line(line_mask, start, end, 255, 2, cv2.LINE_AA)
        records.append(
            {
                "xyxy": [start[0], start[1], end[0], end[1]],
                "length_pixels": round(length, 2),
                "angle_degrees": round(math.degrees(math.atan2(y1 - y0, x1 - x0)), 2),
                "inside_fraction": round(inside_fraction, 4),
                "support": round(support, 4),
            }
        )

    line_mask[~roof_mask] = 0
    return line_mask > 0, records


def build_structural_map(
    rgb: np.ndarray,
    depth: np.ndarray,
    normals: np.ndarray,
    roof_mask: np.ndarray,
    sam_boundaries: np.ndarray,
    minimum_line_length: float,
    line_support_threshold: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], list[dict[str, Any]]]:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    rgb_edges = gradient_map(gray, roof_mask)
    depth_edges = gradient_map(depth, roof_mask)
    normal_edges = normal_discontinuity(normals, roof_mask)
    canny = cv2.Canny((gray * 255).astype(np.uint8), 45, 125).astype(np.float32) / 255.0
    canny[~roof_mask] = 0.0
    support = normalize_inside_mask(
        0.25 * rgb_edges + 0.34 * depth_edges + 0.31 * normal_edges + 0.10 * canny,
        roof_mask,
    )
    line_mask, lines = detect_structural_lines(
        gray,
        roof_mask,
        support,
        minimum_line_length,
        line_support_threshold,
    )
    structural = normalize_inside_mask(
        0.20 * rgb_edges
        + 0.28 * depth_edges
        + 0.28 * normal_edges
        + 0.09 * canny
        + 0.15 * sam_boundaries.astype(np.float32),
        roof_mask,
    )
    structural[line_mask] = 1.0
    structural[~roof_mask] = 0.0
    diagnostics = {
        "rgb_edges": rgb_edges,
        "depth_edges": depth_edges,
        "normal_edges": normal_edges,
        "canny": canny,
        "sam_boundaries": sam_boundaries.astype(np.float32),
    }
    return structural, line_mask, diagnostics, lines


def build_barrier_connectivity(
    superpixels: np.ndarray,
    region_ids: list[int],
    structural_map: np.ndarray,
    line_mask: np.ndarray,
    barrier_threshold: float,
) -> tuple[sparse.csr_matrix, list[dict[str, Any]]]:
    index_by_id = {region_id: index for index, region_id in enumerate(region_ids)}
    observations: dict[tuple[int, int], list[float]] = {}
    hard_barriers: set[tuple[int, int]] = set()

    comparisons = (
        (superpixels[:, :-1], superpixels[:, 1:], structural_map[:, :-1], structural_map[:, 1:], line_mask[:, :-1], line_mask[:, 1:]),
        (superpixels[:-1, :], superpixels[1:, :], structural_map[:-1, :], structural_map[1:, :], line_mask[:-1, :], line_mask[1:, :]),
    )
    for first, second, first_strength, second_strength, first_line, second_line in comparisons:
        changed = (first != second) & (first > 0) & (second > 0)
        left_values = first[changed]
        right_values = second[changed]
        strengths = np.maximum(first_strength[changed], second_strength[changed])
        barriers = first_line[changed] | second_line[changed]
        for left, right, strength, hard in zip(
            left_values, right_values, strengths, barriers, strict=False
        ):
            pair = tuple(sorted((int(left), int(right))))
            observations.setdefault(pair, []).append(float(strength))
            if bool(hard):
                hard_barriers.add(pair)

    row: list[int] = []
    column: list[int] = []
    records: list[dict[str, Any]] = []
    for pair, strengths in observations.items():
        score = float(np.percentile(strengths, 75.0))
        blocked = pair in hard_barriers or score >= barrier_threshold
        records.append(
            {
                "regions": list(pair),
                "boundary_score": round(score, 4),
                "hard_line": pair in hard_barriers,
                "blocked": blocked,
            }
        )
        if blocked:
            continue
        a = index_by_id[pair[0]]
        b = index_by_id[pair[1]]
        row.extend((a, b))
        column.extend((b, a))

    data = np.ones(len(row), dtype=np.uint8)
    connectivity = sparse.csr_matrix(
        (data, (row, column)), shape=(len(region_ids), len(region_ids))
    )
    return connectivity, records


def boundary_energy(mask: np.ndarray, structural_map: np.ndarray) -> float:
    boundary = cv2.morphologyEx(
        mask.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), dtype=np.uint8)
    ).astype(bool)
    values = structural_map[boundary]
    return float(values.mean()) if values.size else 0.0


def refine_with_sam2(
    lang_sam: Any,
    rgb: np.ndarray,
    roof_mask: np.ndarray,
    initial_planes: np.ndarray,
    structural_map: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    predictor = lang_sam.sam.predictor
    predictor.set_image(np.ascontiguousarray(rgb))
    accepted_masks: dict[int, tuple[np.ndarray, float, float]] = {}
    records: list[dict[str, Any]] = []
    plane_ids = [int(value) for value in np.unique(initial_planes) if value > 0]

    for plane_id in plane_ids:
        initial = initial_planes == plane_id
        ys, xs = np.where(initial)
        if xs.size == 0:
            continue
        centroid = np.array([[float(xs.mean()), float(ys.mean())]], dtype=np.float32)
        point_labels = np.array([1], dtype=np.int32)
        margin = 8
        box = np.array(
            [
                max(int(xs.min()) - margin, 0),
                max(int(ys.min()) - margin, 0),
                min(int(xs.max()) + margin, rgb.shape[1] - 1),
                min(int(ys.max()) + margin, rgb.shape[0] - 1),
            ],
            dtype=np.float32,
        )
        masks, scores, _ = predictor.predict(
            point_coords=centroid,
            point_labels=point_labels,
            box=box,
            multimask_output=True,
        )
        initial_pixels = max(int(np.count_nonzero(initial)), 1)
        initial_energy = boundary_energy(initial, structural_map)
        best: tuple[float, np.ndarray, float, float, float] | None = None
        for mask, sam_score in zip(masks, scores, strict=True):
            candidate = np.asarray(mask, dtype=bool) & roof_mask
            intersection = int(np.count_nonzero(candidate & initial))
            union = max(int(np.count_nonzero(candidate | initial)), 1)
            iou = intersection / union
            area_ratio = int(np.count_nonzero(candidate)) / initial_pixels
            energy = boundary_energy(candidate, structural_map)
            objective = 0.52 * iou + 0.28 * float(sam_score) + 0.20 * energy
            if best is None or objective > best[0]:
                best = (objective, candidate, iou, area_ratio, energy)

        if best is None:
            continue
        objective, candidate, iou, area_ratio, energy = best
        quality_accepted = (
            iou >= 0.48
            and 0.62 <= area_ratio <= 1.55
            and energy >= max(initial_energy * 0.90, 0.08)
        )
        if quality_accepted:
            accepted_masks[plane_id] = (candidate, objective, energy)
        records.append(
            {
                "plane_id": plane_id,
                "quality_accepted": quality_accepted,
                "accepted": False,
                "objective": round(objective, 4),
                "iou_with_structural_candidate": round(iou, 4),
                "area_ratio": round(area_ratio, 4),
                "initial_boundary_energy": round(initial_energy, 4),
                "refined_boundary_energy": round(energy, 4),
                "gained_pixels": 0,
                "changed_pixels": 0,
                "rejection_reason": None if quality_accepted else "quality_gate",
            }
        )

    if not accepted_masks:
        return initial_planes.copy(), records

    # Every pixel starts with its structural owner. A quality-approved SAM2 mask
    # may claim neighboring pixels, with nearer and higher-quality claims winning.
    # This preserves a complete partition while allowing boundaries to move.
    refined = initial_planes.copy()
    winning_priority = np.zeros(roof_mask.shape, dtype=np.float32)
    winning_owner = initial_planes.copy()
    for plane_id, (candidate, objective, energy) in accepted_masks.items():
        initial = initial_planes == plane_id
        distance = cv2.distanceTransform((~initial).astype(np.uint8), cv2.DIST_L2, 3)
        candidate_distance = distance[candidate]
        distance_scale = max(float(candidate_distance.max()), 1.0)
        priority = (
            float(objective)
            + 0.12 * float(energy)
            - 0.30 * distance / distance_scale
        )
        claim = candidate & roof_mask & (priority > winning_priority)
        winning_priority[claim] = priority[claim]
        winning_owner[claim] = plane_id

    refined[roof_mask] = winning_owner[roof_mask]
    refined[~roof_mask] = 0

    for record in records:
        plane_id = int(record["plane_id"])
        initial = initial_planes == plane_id
        final = refined == plane_id
        gained_pixels = int(np.count_nonzero(final & ~initial))
        changed_pixels = int(np.count_nonzero(final != initial))
        effectively_applied = bool(record["quality_accepted"] and gained_pixels > 0)
        record["accepted"] = effectively_applied
        record["gained_pixels"] = gained_pixels
        record["changed_pixels"] = changed_pixels
        if record["quality_accepted"] and not effectively_applied:
            record["rejection_reason"] = "no_effect_after_conflict_resolution"
    return refined, records


def create_refinement_difference(
    rgb: np.ndarray,
    initial_planes: np.ndarray,
    refined_planes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    changed = initial_planes != refined_planes
    visual = (rgb.astype(np.float32) * 0.35).astype(np.uint8)
    visual[changed] = np.array([255, 215, 0], dtype=np.uint8)
    boundary = cv2.morphologyEx(
        changed.astype(np.uint8),
        cv2.MORPH_GRADIENT,
        np.ones((3, 3), dtype=np.uint8),
    ).astype(bool)
    visual[boundary] = np.array([220, 20, 60], dtype=np.uint8)
    return changed, visual


def save_scalar_diagnostic(values: np.ndarray, mask: np.ndarray, path: Path) -> None:
    Image.fromarray(colorize_scalar(values, mask)).save(path)


def run(args: argparse.Namespace) -> Path:
    prepare_cache(PROJECT_ROOT)
    device = resolve_device(args.device)
    if not args.input.is_file():
        raise FileNotFoundError(f"Input image not found: {args.input}")

    run_directory = create_run_directory(
        args.output_root, "roof_planes_experiment_02_structural"
    )
    started_total = time.perf_counter()
    timings: dict[str, float] = {}
    image = Image.open(args.input).convert("RGB")
    rgb = np.array(image)
    image.save(run_directory / "input.png")

    from lang_sam import LangSAM

    print(f"Run directory: {run_directory}")
    print(f"Loading LangSAM/SAM2 {args.sam_type} on {device}...")
    started = time.perf_counter()
    lang_sam = LangSAM(sam_type=args.sam_type, device=device)
    result = lang_sam.predict(
        [image],
        [args.prompt],
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
    )[0]
    prediction = select_prediction(result, image)
    roof_mask_image = prediction.mask.resize(image.size, Image.Resampling.NEAREST)
    roof_mask = np.asarray(roof_mask_image) > 0
    timings["langsam_seconds"] = round(time.perf_counter() - started, 3)
    roof_mask_image.save(run_directory / "building_mask.png")
    cutout = image.convert("RGBA")
    cutout.putalpha(roof_mask_image)
    cutout.save(run_directory / "building_cutout.png")
    create_langsam_overlay(image, roof_mask_image, prediction).save(
        run_directory / "building_overlay.jpg", quality=95
    )

    x0, y0, x1, y1 = mask_bbox(roof_mask, margin=12)
    image.crop((x0, y0, x1, y1)).save(run_directory / "roof_crop.png")

    print("Generating SAM2 interior-mask proposals...")
    started = time.perf_counter()
    sam_boundaries, sam_proposals = proposal_boundaries(
        lang_sam,
        rgb,
        roof_mask,
        args.sam_proposal_min_iou,
        args.sam_proposal_min_stability,
    )
    timings["sam_proposals_seconds"] = round(time.perf_counter() - started, 3)
    Image.fromarray((sam_boundaries * 255).astype(np.uint8)).save(
        run_directory / "sam2_proposal_boundaries.png"
    )

    print(f"Estimating relative depth with {args.depth_model}...")
    started = time.perf_counter()
    raw_depth = estimate_depth(image, args.depth_model, device)
    depth = normalize_inside_mask(raw_depth, roof_mask)
    timings["depth_seconds"] = round(time.perf_counter() - started, 3)
    Image.fromarray((depth * 65535).astype(np.uint16)).save(
        run_directory / "relative_depth_16bit.png"
    )
    save_scalar_diagnostic(depth, roof_mask, run_directory / "relative_depth_color.png")

    _, _, normals, normal_scale = compute_geometry(depth, roof_mask)
    normal_rgb = np.clip((normals + 1.0) * 127.5, 0, 255).astype(np.uint8)
    normal_rgb[~roof_mask] = 255
    Image.fromarray(normal_rgb).save(run_directory / "surface_normals.png")

    print("Detecting roof structure and constrained plane candidates...")
    started = time.perf_counter()
    structural_map, line_mask, diagnostics, lines = build_structural_map(
        rgb,
        depth,
        normals,
        roof_mask,
        sam_boundaries,
        args.minimum_line_length,
        args.line_support_threshold,
    )
    save_scalar_diagnostic(
        structural_map, roof_mask, run_directory / "structural_boundary_map.png"
    )
    Image.fromarray((line_mask * 255).astype(np.uint8)).save(
        run_directory / "structural_lines.png"
    )
    for name, values in diagnostics.items():
        save_scalar_diagnostic(values, roof_mask, run_directory / f"{name}.png")

    superpixels = make_superpixels(
        rgb, roof_mask, args.superpixels, args.compactness
    )
    superpixel_overlay = (
        mark_boundaries(rgb, superpixels, color=(1, 0, 0)) * 255
    ).astype(np.uint8)
    superpixel_overlay[~roof_mask] = 255
    Image.fromarray(superpixel_overlay).save(run_directory / "superpixels.png")

    features, region_ids = build_region_features(rgb, depth, normals, superpixels)
    connectivity, adjacency = build_barrier_connectivity(
        superpixels,
        region_ids,
        structural_map,
        line_mask,
        args.barrier_threshold,
    )
    clustered = cluster_superpixels(
        features,
        region_ids,
        connectivity,
        superpixels,
        args.cluster_threshold,
    )
    structural_planes = split_and_merge_small_components(
        clustered, roof_mask, args.min_plane_percentage
    )
    structural_planes = absorb_isolated_tiny_planes(
        structural_planes, roof_mask, args.min_plane_percentage
    )
    structural_colors, structural_overlay, _ = render_planes(rgb, structural_planes)
    structural_summaries = summarize_planes(structural_planes)
    save_label_mask(
        structural_planes, run_directory / "structural_planes_labels.png"
    )
    Image.fromarray(structural_colors).save(
        run_directory / "structural_planes_color.png"
    )
    Image.fromarray(structural_overlay).save(
        run_directory / "structural_planes_overlay.png"
    )
    write_geojson(
        structural_planes,
        structural_summaries,
        run_directory / "structural_planes.geojson",
    )
    timings["structural_candidates_seconds"] = round(
        time.perf_counter() - started, 3
    )

    refinement_records: list[dict[str, Any]] = []
    refined_planes = structural_planes.copy()
    if not args.disable_sam_refinement:
        print("Refining structural candidates with SAM2 point and box prompts...")
        started = time.perf_counter()
        refined_planes, refinement_records = refine_with_sam2(
            lang_sam,
            rgb,
            roof_mask,
            structural_planes,
            structural_map,
        )
        timings["sam_refinement_seconds"] = round(time.perf_counter() - started, 3)

    refined_colors, refined_overlay, _ = render_planes(rgb, refined_planes)
    refined_summaries = summarize_planes(refined_planes)
    save_label_mask(refined_planes, run_directory / "roof_planes_labels.png")
    Image.fromarray(refined_colors).save(run_directory / "roof_planes_color.png")
    Image.fromarray(refined_overlay).save(run_directory / "roof_planes_overlay.png")
    refinement_delta, refinement_difference = create_refinement_difference(
        rgb, structural_planes, refined_planes
    )
    Image.fromarray((refinement_delta.astype(np.uint8) * 255)).save(
        run_directory / "refinement_delta_mask.png"
    )
    Image.fromarray(refinement_difference).save(
        run_directory / "refinement_difference.png"
    )
    write_geojson(
        refined_planes, refined_summaries, run_directory / "roof_planes.geojson"
    )

    reference_available = args.reference.is_file()
    if reference_available:
        shutil.copy2(args.reference, run_directory / "manual_reference.png")
    make_contact_sheet(
        [
            ("Original input", image),
            ("LangSAM building mask", cutout),
            ("Structural boundary map", Image.open(run_directory / "structural_boundary_map.png")),
            ("Detected straight lines", Image.open(run_directory / "structural_lines.png")),
            ("Structural candidates", Image.fromarray(structural_overlay)),
            ("SAM2-refined candidates", Image.fromarray(refined_overlay)),
            ("Effective SAM2 changes", Image.fromarray(refinement_difference)),
        ],
        run_directory / "diagnostic_contact_sheet.jpg",
    )
    comparison_items = [
        ("Line-aware structural candidates", Image.fromarray(structural_overlay)),
        ("SAM2-refined candidates", Image.fromarray(refined_overlay)),
        ("Effective refinement difference", Image.fromarray(refinement_difference)),
    ]
    if reference_available:
        comparison_items.append(("Manual qualitative reference", Image.open(args.reference)))
    make_contact_sheet(
        comparison_items, run_directory / "qualitative_comparison.jpg"
    )

    accepted_refinements = sum(
        1 for record in refinement_records if record["accepted"]
    )
    metadata = {
        "experiment": "roof_planes_experiment_02_structural",
        "purpose": (
            "RGB-only line-aware roof-plane candidates with SAM2 refinement; "
            "not metric geometry"
        ),
        "input": str(args.input.resolve()),
        "reference": str(args.reference.resolve()) if reference_available else None,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "langsam": {
            "prompt": args.prompt,
            "sam_type": args.sam_type,
            "selected_label": prediction.label,
            "selected_score": prediction.score,
            "selected_mask_score": prediction.mask_score,
            "roof_coverage_percentage": round(
                np.count_nonzero(roof_mask) / roof_mask.size * 100.0, 3
            ),
        },
        "depth": {
            "model": args.depth_model,
            "type": "relative_monocular_depth",
            "metric": False,
            "georeferenced": False,
            "normal_scale": normal_scale,
        },
        "sam2_proposals": {
            "accepted_count": len(sam_proposals),
            "proposals": sam_proposals,
        },
        "structural_lines": {
            "accepted_count": len(lines),
            "minimum_length": args.minimum_line_length,
            "support_threshold": args.line_support_threshold,
            "lines": lines,
        },
        "candidate_generation": {
            "requested_superpixels": args.superpixels,
            "actual_superpixels": len(region_ids),
            "cluster_threshold": args.cluster_threshold,
            "barrier_threshold": args.barrier_threshold,
            "blocked_adjacencies": sum(1 for item in adjacency if item["blocked"]),
            "total_adjacencies": len(adjacency),
            "structural_plane_count": len(structural_summaries),
            "refined_plane_count": len(refined_summaries),
            "planes": [asdict(summary) for summary in refined_summaries],
        },
        "sam2_refinement": {
            "enabled": not args.disable_sam_refinement,
            "accepted_count": accepted_refinements,
            "changed_pixel_count": int(np.count_nonzero(refinement_delta)),
            "changed_percentage_of_roof": round(
                np.count_nonzero(refinement_delta)
                / max(np.count_nonzero(roof_mask), 1)
                * 100.0,
                4,
            ),
            "records": refinement_records,
        },
        "evaluation": {
            "mode": "qualitative_only",
            "reason": (
                "The manual reference is a color overlay rather than an indexed "
                "machine-readable ground-truth mask."
            ),
        },
        "timings": {
            **timings,
            "total_seconds": round(time.perf_counter() - started_total, 3),
        },
        "assets": sorted(
            [path.name for path in run_directory.iterdir()] + ["result.json"]
        ),
    }
    (run_directory / "result.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    del lang_sam
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(
        f"Detected {len(structural_summaries)} structural candidates; "
        f"accepted {accepted_refinements} SAM2 refinements; "
        f"final labels: {len(refined_summaries)}"
    )
    print(f"Results written to {run_directory}")
    return run_directory


if __name__ == "__main__":
    run(parse_args())
