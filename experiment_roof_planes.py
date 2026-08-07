from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from scipy import sparse
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from skimage.segmentation import mark_boundaries, slic
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

from run_artifacts import create_run_directory
from segment_image import (
    create_overlay as create_langsam_overlay,
    prepare_cache,
    resolve_device,
    select_prediction,
)

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = PROJECT_ROOT / "start_input" / "UP_original.png"
DEFAULT_REFERENCE = (
    PROJECT_ROOT / "start_example_roof_planes" / "image_with_roof_planes.png"
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs"
DEFAULT_DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"


@dataclass(frozen=True)
class PlaneSummary:
    id: int
    area_pixels: int
    area_percentage_of_roof: float
    bbox_xyxy: list[int]
    centroid_xy: list[float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate experimental roof-plane candidates from an aerial RGB image "
            "using LangSAM, Depth Anything V2, and local image processing."
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
    parser.add_argument("--superpixels", type=int, default=80)
    parser.add_argument("--compactness", type=float, default=12.0)
    parser.add_argument("--cluster-threshold", type=float, default=4.5)
    parser.add_argument("--min-plane-percentage", type=float, default=1.25)
    return parser.parse_args()


def normalize_inside_mask(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    normalized = np.zeros(values.shape, dtype=np.float32)
    selected = values[mask]
    if selected.size == 0:
        return normalized
    low, high = np.percentile(selected, (2.0, 98.0))
    if math.isclose(float(low), float(high)):
        normalized[mask] = 0.5
        return normalized
    normalized[mask] = np.clip((values[mask] - low) / (high - low), 0.0, 1.0)
    return normalized


def colorize_scalar(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    encoded = np.clip(values * 255.0, 0, 255).astype(np.uint8)
    color = cv2.applyColorMap(encoded, cv2.COLORMAP_TURBO)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    color[~mask] = 255
    return color


def estimate_depth(
    image: Image.Image,
    model_name: str,
    device: torch.device,
) -> np.ndarray:
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForDepthEstimation.from_pretrained(model_name).to(device)
    model.eval()
    inputs = processor(images=image, return_tensors="pt")
    inputs = {name: value.to(device) for name, value in inputs.items()}
    with torch.inference_mode():
        prediction = model(**inputs).predicted_depth
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=(image.height, image.width),
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    depth = prediction.float().cpu().numpy()
    del model, processor, inputs, prediction
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return depth


def compute_geometry(
    depth: np.ndarray,
    roof_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    smooth = cv2.GaussianBlur(depth.astype(np.float32), (0, 0), sigmaX=2.0)
    gradient_x = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=5)
    gradient_y = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=5)
    magnitude = np.hypot(gradient_x, gradient_y)

    selected = magnitude[roof_mask]
    normal_scale = 1.0 / max(float(np.percentile(selected, 75.0)), 1e-6)
    normal_x = -gradient_x * normal_scale
    normal_y = -gradient_y * normal_scale
    normal_z = np.ones_like(normal_x)
    norm = np.sqrt(normal_x**2 + normal_y**2 + normal_z**2)
    normals = np.stack(
        (normal_x / norm, normal_y / norm, normal_z / norm), axis=-1
    )
    normals[~roof_mask] = 0.0
    magnitude = normalize_inside_mask(magnitude, roof_mask)
    return smooth, magnitude, normals, normal_scale


def make_superpixels(
    rgb: np.ndarray,
    roof_mask: np.ndarray,
    count: int,
    compactness: float,
) -> np.ndarray:
    labels = slic(
        rgb,
        n_segments=count,
        compactness=compactness,
        sigma=1.0,
        mask=roof_mask,
        start_label=1,
        channel_axis=-1,
    )
    labels[~roof_mask] = 0
    return labels.astype(np.int32)


def build_region_features(
    rgb: np.ndarray,
    depth: np.ndarray,
    normals: np.ndarray,
    superpixels: np.ndarray,
) -> tuple[np.ndarray, list[int]]:
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32) / 255.0
    region_ids = [int(value) for value in np.unique(superpixels) if value > 0]
    rows: list[np.ndarray] = []
    for region_id in region_ids:
        selected = superpixels == region_id
        rows.append(
            np.concatenate(
                (
                    normals[selected].mean(axis=0) * 3.0,
                    np.array([depth[selected].mean() * 1.5], dtype=np.float32),
                    lab[selected].mean(axis=0) * 0.45,
                )
            )
        )
    return np.asarray(rows, dtype=np.float32), region_ids


def build_connectivity(superpixels: np.ndarray, region_ids: list[int]) -> sparse.csr_matrix:
    index_by_id = {region_id: index for index, region_id in enumerate(region_ids)}
    pairs: set[tuple[int, int]] = set()
    for first, second in (
        (superpixels[:, :-1], superpixels[:, 1:]),
        (superpixels[:-1, :], superpixels[1:, :]),
    ):
        changed = (first != second) & (first > 0) & (second > 0)
        for left, right in zip(first[changed], second[changed], strict=False):
            a = index_by_id[int(left)]
            b = index_by_id[int(right)]
            if a != b:
                pairs.add((min(a, b), max(a, b)))

    row: list[int] = []
    column: list[int] = []
    for a, b in pairs:
        row.extend((a, b))
        column.extend((b, a))
    data = np.ones(len(row), dtype=np.uint8)
    return sparse.csr_matrix((data, (row, column)), shape=(len(region_ids),) * 2)


def cluster_superpixels(
    features: np.ndarray,
    region_ids: list[int],
    connectivity: sparse.csr_matrix,
    superpixels: np.ndarray,
    distance_threshold: float,
) -> np.ndarray:
    standardized = StandardScaler().fit_transform(features)
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        connectivity=connectivity,
        linkage="ward",
    )
    region_clusters = clustering.fit_predict(standardized)
    clustered = np.zeros(superpixels.shape, dtype=np.int32)
    for region_id, cluster_id in zip(region_ids, region_clusters, strict=True):
        clustered[superpixels == region_id] = int(cluster_id) + 1
    return clustered


def split_and_merge_small_components(
    clustered: np.ndarray,
    roof_mask: np.ndarray,
    minimum_percentage: float,
) -> np.ndarray:
    components = np.zeros_like(clustered, dtype=np.int32)
    next_id = 1
    for cluster_id in [int(value) for value in np.unique(clustered) if value > 0]:
        count, labels = cv2.connectedComponents(
            (clustered == cluster_id).astype(np.uint8), connectivity=8
        )
        for component_id in range(1, count):
            components[labels == component_id] = next_id
            next_id += 1

    minimum_pixels = max(
        16, int(np.count_nonzero(roof_mask) * minimum_percentage / 100.0)
    )
    kernel = np.ones((3, 3), dtype=np.uint8)
    changed = True
    while changed:
        changed = False
        for plane_id in [int(value) for value in np.unique(components) if value > 0]:
            region = components == plane_id
            if np.count_nonzero(region) >= minimum_pixels:
                continue
            ring = cv2.dilate(region.astype(np.uint8), kernel, iterations=1).astype(bool)
            neighbors = components[ring & ~region]
            neighbors = neighbors[neighbors > 0]
            if neighbors.size:
                target = int(np.bincount(neighbors).argmax())
                components[region] = target
                changed = True

    relabeled = np.zeros_like(components)
    for new_id, old_id in enumerate(
        [int(value) for value in np.unique(components) if value > 0], start=1
    ):
        relabeled[components == old_id] = new_id
    relabeled[~roof_mask] = 0
    return relabeled


def plane_palette(count: int) -> np.ndarray:
    palette = np.zeros((count + 1, 3), dtype=np.uint8)
    for plane_id in range(1, count + 1):
        hue = int(((plane_id - 1) * 179 / max(count, 1)) % 180)
        hsv = np.uint8([[[hue, 180, 235]]])
        palette[plane_id] = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0, 0]
    return palette


def render_planes(
    rgb: np.ndarray,
    planes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = int(planes.max())
    colors = plane_palette(count)[planes]
    colors[planes == 0] = 255
    overlay = rgb.copy()
    selected = planes > 0
    overlay[selected] = (
        rgb[selected].astype(np.float32) * 0.42
        + colors[selected].astype(np.float32) * 0.58
    ).astype(np.uint8)
    boundaries = np.zeros(planes.shape, dtype=bool)
    boundaries[:, 1:] |= planes[:, 1:] != planes[:, :-1]
    boundaries[1:, :] |= planes[1:, :] != planes[:-1, :]
    boundaries &= selected
    overlay[boundaries] = 20
    return colors, overlay, boundaries


def summarize_planes(planes: np.ndarray) -> list[PlaneSummary]:
    roof_pixels = max(int(np.count_nonzero(planes)), 1)
    summaries: list[PlaneSummary] = []
    for plane_id in [int(value) for value in np.unique(planes) if value > 0]:
        ys, xs = np.where(planes == plane_id)
        summaries.append(
            PlaneSummary(
                id=plane_id,
                area_pixels=int(len(xs)),
                area_percentage_of_roof=round(len(xs) / roof_pixels * 100.0, 3),
                bbox_xyxy=[int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
                centroid_xy=[round(float(xs.mean()), 2), round(float(ys.mean()), 2)],
            )
        )
    return summaries


def write_geojson(planes: np.ndarray, summaries: list[PlaneSummary], path: Path) -> None:
    features: list[dict[str, Any]] = []
    for summary in summaries:
        contours, _ = cv2.findContours(
            (planes == summary.id).astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        polygons = []
        for contour in contours:
            epsilon = max(1.0, 0.005 * cv2.arcLength(contour, True))
            simplified = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
            if len(simplified) < 3:
                continue
            ring = [[int(x), int(y)] for x, y in simplified]
            ring.append(ring[0])
            polygons.append([ring])
        geometry: dict[str, Any]
        if len(polygons) == 1:
            geometry = {"type": "Polygon", "coordinates": polygons[0]}
        else:
            geometry = {"type": "MultiPolygon", "coordinates": polygons}
        features.append(
            {
                "type": "Feature",
                "properties": asdict(summary),
                "geometry": geometry,
            }
        )
    path.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "coordinate_system": "image_pixels_origin_top_left",
                "features": features,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def save_label_mask(planes: np.ndarray, path: Path) -> None:
    if int(planes.max()) <= 255:
        Image.fromarray(planes.astype(np.uint8), mode="L").save(path)
    else:
        Image.fromarray(planes.astype(np.uint16), mode="I;16").save(path)


def make_contact_sheet(items: list[tuple[str, Image.Image]], output: Path) -> None:
    width, height = 420, 420
    header = 34
    columns = min(3, max(len(items), 1))
    rows = math.ceil(len(items) / columns)
    sheet = Image.new("RGB", (width * columns, (height + header) * rows), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for index, (title, image) in enumerate(items[:6]):
        x = (index % columns) * width
        y = (index // columns) * (height + header)
        fitted = image.convert("RGB").copy()
        fitted.thumbnail((width, height), Image.Resampling.LANCZOS)
        offset_x = x + (width - fitted.width) // 2
        offset_y = y + header + (height - fitted.height) // 2
        sheet.paste(fitted, (offset_x, offset_y))
        draw.text((x + 10, y + 10), title, fill="black", font=font)
    sheet.save(output, format="JPEG", quality=94, optimize=True)


def run(args: argparse.Namespace) -> Path:
    prepare_cache(PROJECT_ROOT)
    device = resolve_device(args.device)
    if not args.input.is_file():
        raise FileNotFoundError(f"Input image not found: {args.input}")

    run_directory = create_run_directory(args.output_root, "roof_planes_experiment_01")
    started_total = time.perf_counter()
    timings: dict[str, float] = {}
    image = Image.open(args.input).convert("RGB")
    rgb = np.asarray(image)
    image.save(run_directory / "input.png")

    from lang_sam import LangSAM

    print(f"Run directory: {run_directory}")
    print(f"Loading LangSAM {args.sam_type} on {device}...")
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

    del lang_sam
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print(f"Estimating relative depth with {args.depth_model}...")
    started = time.perf_counter()
    raw_depth = estimate_depth(image, args.depth_model, device)
    depth = normalize_inside_mask(raw_depth, roof_mask)
    timings["depth_seconds"] = round(time.perf_counter() - started, 3)
    Image.fromarray((depth * 65535).astype(np.uint16)).save(
        run_directory / "relative_depth_16bit.png"
    )
    depth_color = colorize_scalar(depth, roof_mask)
    Image.fromarray(depth_color).save(run_directory / "relative_depth_color.png")

    print("Computing normals, edges, superpixels, and plane candidates...")
    started = time.perf_counter()
    _, gradient, normals, normal_scale = compute_geometry(depth, roof_mask)
    normal_rgb = np.clip((normals + 1.0) * 127.5, 0, 255).astype(np.uint8)
    normal_rgb[~roof_mask] = 255
    Image.fromarray(normal_rgb).save(run_directory / "surface_normals.png")
    Image.fromarray(colorize_scalar(gradient, roof_mask)).save(
        run_directory / "depth_gradient.png"
    )

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    rgb_gradient = np.hypot(
        cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3),
        cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3),
    )
    rgb_gradient = normalize_inside_mask(rgb_gradient, roof_mask)
    combined_edges = normalize_inside_mask(0.65 * gradient + 0.35 * rgb_gradient, roof_mask)
    Image.fromarray(colorize_scalar(combined_edges, roof_mask)).save(
        run_directory / "combined_edges.png"
    )

    superpixels = make_superpixels(
        rgb, roof_mask, args.superpixels, args.compactness
    )
    superpixel_overlay = (mark_boundaries(rgb, superpixels, color=(1, 0, 0)) * 255).astype(
        np.uint8
    )
    superpixel_overlay[~roof_mask] = 255
    Image.fromarray(superpixel_overlay).save(run_directory / "superpixels.png")

    features, region_ids = build_region_features(rgb, depth, normals, superpixels)
    connectivity = build_connectivity(superpixels, region_ids)
    clustered = cluster_superpixels(
        features,
        region_ids,
        connectivity,
        superpixels,
        args.cluster_threshold,
    )
    planes = split_and_merge_small_components(
        clustered, roof_mask, args.min_plane_percentage
    )
    colors, plane_overlay, _ = render_planes(rgb, planes)
    summaries = summarize_planes(planes)
    timings["candidate_generation_seconds"] = round(time.perf_counter() - started, 3)

    save_label_mask(planes, run_directory / "roof_planes_labels.png")
    Image.fromarray(colors).save(run_directory / "roof_planes_color.png")
    Image.fromarray(plane_overlay).save(run_directory / "roof_planes_overlay.png")
    write_geojson(planes, summaries, run_directory / "roof_planes.geojson")

    reference_available = args.reference.is_file()
    if reference_available:
        shutil.copy2(args.reference, run_directory / "manual_reference.png")

    contact_items = [
        ("Original input", image),
        ("LangSAM building mask", Image.fromarray(np.asarray(cutout))),
        ("Relative depth", Image.fromarray(depth_color)),
        ("Estimated surface normals", Image.fromarray(normal_rgb)),
        ("Superpixels", Image.fromarray(superpixel_overlay)),
        ("Roof-plane candidates", Image.fromarray(plane_overlay)),
    ]
    make_contact_sheet(contact_items, run_directory / "diagnostic_contact_sheet.jpg")

    if reference_available:
        make_contact_sheet(
            [
                ("Automatic roof-plane candidates", Image.fromarray(plane_overlay)),
                ("Manual qualitative reference", Image.open(args.reference)),
            ],
            run_directory / "qualitative_comparison.jpg",
        )

    metadata = {
        "experiment": "roof_planes_experiment_01",
        "purpose": "RGB-only roof-plane candidate generation; not metric geometry",
        "input": str(args.input.resolve()),
        "reference": str(args.reference.resolve()) if reference_available else None,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "langsam": {
            "prompt": args.prompt,
            "sam_type": args.sam_type,
            "box_threshold": args.box_threshold,
            "text_threshold": args.text_threshold,
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
        "candidate_generation": {
            "requested_superpixels": args.superpixels,
            "actual_superpixels": len(region_ids),
            "compactness": args.compactness,
            "cluster_threshold": args.cluster_threshold,
            "min_plane_percentage": args.min_plane_percentage,
            "plane_count": len(summaries),
            "planes": [asdict(summary) for summary in summaries],
        },
        "evaluation": {
            "mode": "qualitative_only",
            "reason": (
                "The supplied manual reference is a color overlay, not a machine-readable "
                "indexed ground-truth mask."
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
    print(f"Detected {len(summaries)} roof-plane candidates")
    print(f"Results written to {run_directory}")
    return run_directory


if __name__ == "__main__":
    run(parse_args())
