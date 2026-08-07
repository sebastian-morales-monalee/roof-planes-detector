from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import cv2
import httpx
import numpy as np
from dotenv import load_dotenv
from PIL import Image
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression, RANSACRegressor

from experiment_roof_planes import (
    make_contact_sheet,
    render_planes,
    save_label_mask,
    summarize_planes,
)
from experiment_roof_planes_vlm_topology import draw_topology, partition_roof
from run_artifacts import create_run_directory


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs"
DEFAULT_LOCATION = "40.6898556050961, -111.79091797620747"
EXPERIMENT_NAME = "roof_planes_experiment_07_google_solar_dsm"
EXPERIMENT_06_SUFFIX = "_roof_planes_experiment_06_vector_optimizer"
BUILDING_INSIGHTS_ENDPOINT = (
    "https://solar.googleapis.com/v1/buildingInsights:findClosest"
)
DATA_LAYERS_ENDPOINT = "https://solar.googleapis.com/v1/dataLayers:get"
MAX_DOWNLOAD_BYTES = 100 * 1024 * 1024


class ExperimentError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Experiment 07: register the human-selected vector-structural roof "
            "topology against Google Solar RGB/mask layers and refine its shared "
            "boundaries using metric DSM elevation."
        )
    )
    parser.add_argument("--location", default=DEFAULT_LOCATION)
    parser.add_argument("--source-run", type=Path)
    parser.add_argument(
        "--manual-reference",
        type=Path,
        help="Optional paint-over used only in the qualitative comparison.",
    )
    parser.add_argument(
        "--source-candidate",
        default="structural",
        choices=("conservative", "balanced", "structural"),
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--minimum-registration-iou", type=float, default=0.42)
    parser.add_argument("--registration-angle-step", type=float, default=2.0)
    parser.add_argument("--plane-residual-threshold-meters", type=float, default=0.18)
    parser.add_argument("--max-fit-samples", type=int, default=8000)
    parser.add_argument("--max-vector-shift-pixels", type=float, default=10.0)
    parser.add_argument("--maximum-boundary-complexity-ratio", type=float, default=1.08)
    return parser.parse_args()


def parse_location(value: str) -> tuple[float, float]:
    pieces = [piece.strip() for piece in value.split(",")]
    if len(pieces) != 2:
        raise ExperimentError("Location must use '<latitude>, <longitude>' format")
    try:
        latitude, longitude = (float(piece) for piece in pieces)
    except ValueError as error:
        raise ExperimentError("Location contains a non-numeric coordinate") from error
    if not -90.0 <= latitude <= 90.0 or not -180.0 <= longitude <= 180.0:
        raise ExperimentError("Location is outside valid latitude/longitude ranges")
    return latitude, longitude


def find_latest_vector_run(output_root: Path) -> Path:
    candidates = sorted(
        (
            path
            for path in output_root.iterdir()
            if path.is_dir()
            and path.name.endswith(EXPERIMENT_06_SUFFIX)
            and (path / "result.json").is_file()
            and (path / "candidate_structural" / "roof_planes_labels.png").is_file()
        ),
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            "No completed Experiment 06 output was found. Run "
            "experiment_roof_planes_vector_optimizer.py first."
        )
    return candidates[0]


def request_json(
    client: httpx.Client,
    endpoint: str,
    params: dict[str, Any],
    provider: str,
) -> dict[str, Any]:
    response = client.get(endpoint, params=params, timeout=60.0)
    if response.status_code != 200:
        raise ExperimentError(f"{provider} returned HTTP {response.status_code}")
    try:
        payload = response.json()
    except ValueError as error:
        raise ExperimentError(f"{provider} returned invalid JSON") from error
    if not isinstance(payload, dict):
        raise ExperimentError(f"{provider} returned an unexpected payload")
    return payload


def download_solar_layer(
    client: httpx.Client,
    url: str,
    api_key: str,
    output_path: Path,
) -> None:
    parsed = httpx.URL(url)
    if parsed.scheme != "https" or parsed.host != "solar.googleapis.com":
        raise ExperimentError("Google Solar returned an unsafe layer URL")
    response = client.get(
        str(parsed.copy_with(query=None)),
        params={**dict(parsed.params), "key": api_key},
        timeout=120.0,
    )
    if response.status_code != 200:
        raise ExperimentError(
            f"Google Solar layer download returned HTTP {response.status_code}"
        )
    if not response.content or len(response.content) > MAX_DOWNLOAD_BYTES:
        raise ExperimentError("Google Solar layer download has an invalid size")
    output_path.write_bytes(response.content)


def fetch_google_solar_layers(
    latitude: float,
    longitude: float,
    api_key: str,
    output_directory: Path,
) -> dict[str, Any]:
    output_directory.mkdir(parents=True, exist_ok=True)
    with httpx.Client(headers={"User-Agent": "langsam-roof-planes-experiment/1.0"}) as client:
        insights = request_json(
            client,
            BUILDING_INSIGHTS_ENDPOINT,
            {
                "location.latitude": latitude,
                "location.longitude": longitude,
                "requiredQuality": "MEDIUM",
                "key": api_key,
            },
            "Google Solar Building Insights",
        )
        center = insights.get("center")
        bounding_box = insights.get("boundingBox")
        if not isinstance(center, dict) or not isinstance(bounding_box, dict):
            raise ExperimentError("Building Insights is missing center or bounding box")
        center_latitude = float(center.get("latitude", center.get("lat")))
        center_longitude = float(center.get("longitude", center.get("lng")))
        layers = request_json(
            client,
            DATA_LAYERS_ENDPOINT,
            {
                "location.latitude": center_latitude,
                "location.longitude": center_longitude,
                "radiusMeters": 50.0,
                "view": "FULL_LAYERS",
                "requiredQuality": "MEDIUM",
                "pixelSizeMeters": 0.1,
                "key": api_key,
            },
            "Google Solar Data Layers",
        )
        required = {
            "rgbUrl": "google_solar_rgb.tif",
            "dsmUrl": "google_solar_dsm.tif",
            "maskUrl": "google_solar_mask.tif",
        }
        for field, name in required.items():
            url = layers.get(field)
            if not isinstance(url, str) or not url:
                raise ExperimentError(f"Google Solar Data Layers is missing {field}")
            download_solar_layer(client, url, api_key, output_directory / name)

    public_metadata = {
        "requested_location": {"latitude": latitude, "longitude": longitude},
        "building_center": {
            "latitude": center_latitude,
            "longitude": center_longitude,
        },
        "building_bounding_box": bounding_box,
        "imagery_quality": layers.get("imageryQuality"),
        "imagery_date": layers.get("imageryDate"),
        "imagery_processed_date": layers.get("imageryProcessedDate"),
        "pixel_size_meters": 0.1,
        "view": "FULL_LAYERS",
        "downloaded_assets": list(required.values()),
        "ephemeral_layer_urls_persisted": False,
    }
    (output_directory / "metadata.json").write_text(
        json.dumps(public_metadata, indent=2), encoding="utf-8"
    )
    (output_directory / "building_insights.json").write_text(
        json.dumps(insights, indent=2), encoding="utf-8"
    )
    return public_metadata


def read_solar_crop(
    solar_directory: Path,
    metadata: dict[str, Any],
    padding_meters: float = 5.0,
) -> dict[str, Any]:
    try:
        import rasterio
        from rasterio.windows import Window, from_bounds
        from rasterio.warp import transform_bounds
    except ImportError as error:
        raise ExperimentError("Rasterio is required to read Google Solar GeoTIFFs") from error

    dsm_path = solar_directory / "google_solar_dsm.tif"
    mask_path = solar_directory / "google_solar_mask.tif"
    rgb_path = solar_directory / "google_solar_rgb.tif"
    bbox = metadata["building_bounding_box"]
    southwest = bbox.get("sw", bbox.get("southwest"))
    northeast = bbox.get("ne", bbox.get("northeast"))
    if not isinstance(southwest, dict) or not isinstance(northeast, dict):
        raise ExperimentError("Building bounding box has an unsupported format")
    west = float(southwest.get("longitude", southwest.get("lng")))
    south = float(southwest.get("latitude", southwest.get("lat")))
    east = float(northeast.get("longitude", northeast.get("lng")))
    north = float(northeast.get("latitude", northeast.get("lat")))

    with rasterio.open(dsm_path) as source:
        if source.crs is None:
            raise ExperimentError("Google Solar DSM has no CRS")
        projected = transform_bounds("EPSG:4326", source.crs, west, south, east, north)
        bounds = (
            projected[0] - padding_meters,
            projected[1] - padding_meters,
            projected[2] + padding_meters,
            projected[3] + padding_meters,
        )
        window = from_bounds(*bounds, transform=source.transform).round_offsets().round_lengths()
        full_window = Window(0, 0, source.width, source.height)
        window = window.intersection(full_window)
        dsm = source.read(1, window=window).astype(np.float32)
        dsm_valid = source.read_masks(1, window=window) > 0
        if source.nodata is not None:
            dsm_valid &= ~np.isclose(dsm, float(source.nodata))
        dsm_valid &= np.isfinite(dsm)
        crop_transform = source.window_transform(window)
        crs = source.crs

    with rasterio.open(mask_path) as source:
        mask = source.read(1, window=window) > 0

    with rasterio.open(rgb_path) as source:
        rgb_bands = source.read(window=window)
        if rgb_bands.shape[0] < 3:
            raise ExperimentError("Google Solar RGB layer has fewer than three bands")
        rgb = np.moveaxis(rgb_bands[:3], 0, -1)
        if rgb.dtype != np.uint8:
            low, high = np.percentile(rgb, (1.0, 99.0))
            rgb = np.clip((rgb - low) * 255.0 / max(high - low, 1e-6), 0, 255).astype(np.uint8)

    center = metadata["building_center"]
    with rasterio.open(dsm_path) as source:
        from rasterio.warp import transform

        xs, ys = transform(
            "EPSG:4326",
            crs,
            [float(center["longitude"])],
            [float(center["latitude"])],
        )
    inverse = ~crop_transform
    center_column, center_row = inverse * (xs[0], ys[0])
    target_mask = select_target_component(mask, center_row, center_column)
    return {
        "dsm": dsm,
        "dsm_valid": dsm_valid,
        "rgb": rgb,
        "mask": target_mask,
        "all_buildings_mask": mask,
        "transform": crop_transform,
        "crs": crs,
        "window": {
            "column_offset": int(window.col_off),
            "row_offset": int(window.row_off),
            "width": int(window.width),
            "height": int(window.height),
        },
    }


def select_target_component(
    mask: np.ndarray,
    center_row: float,
    center_column: float,
) -> np.ndarray:
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    if count <= 1:
        raise ExperimentError("Google Solar mask contains no building")
    row = int(np.clip(round(center_row), 0, mask.shape[0] - 1))
    column = int(np.clip(round(center_column), 0, mask.shape[1] - 1))
    selected = int(labels[row, column])
    if selected == 0:
        distances = []
        for label in range(1, count):
            x, y = centroids[label]
            distances.append((math.hypot(x - center_column, y - center_row), label))
        selected = min(distances)[1]
    component = labels == selected
    if int(stats[selected, cv2.CC_STAT_AREA]) < 100:
        raise ExperimentError("The selected Google Solar building mask is too small")
    return component


def mask_center(mask: np.ndarray) -> np.ndarray:
    moments = cv2.moments(mask.astype(np.uint8))
    if moments["m00"] <= 0:
        raise ExperimentError("Cannot calculate the center of an empty mask")
    return np.asarray(
        [moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]],
        dtype=np.float64,
    )


def affine_matrix(
    source_center: np.ndarray,
    target_center: np.ndarray,
    angle_degrees: float,
    scale_x: float,
    scale_y: float,
    shift_x: float = 0.0,
    shift_y: float = 0.0,
) -> np.ndarray:
    angle = math.radians(angle_degrees)
    cosine, sine = math.cos(angle), math.sin(angle)
    linear = np.asarray(
        [[scale_x * cosine, -scale_y * sine], [scale_x * sine, scale_y * cosine]],
        dtype=np.float64,
    )
    translation = target_center + np.asarray([shift_x, shift_y]) - linear @ source_center
    return np.column_stack((linear, translation)).astype(np.float32)


def warp_mask(mask: np.ndarray, matrix: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    return cv2.warpAffine(
        mask.astype(np.uint8),
        matrix,
        (shape[1], shape[0]),
        flags=cv2.INTER_NEAREST,
        borderValue=0,
    ) > 0


def registration_score(warped: np.ndarray, target: np.ndarray) -> dict[str, float]:
    intersection = int(np.logical_and(warped, target).sum())
    union = int(np.logical_or(warped, target).sum())
    iou = intersection / union if union else 0.0
    precision = intersection / int(warped.sum()) if warped.any() else 0.0
    recall = intersection / int(target.sum()) if target.any() else 0.0
    return {"iou": iou, "precision": precision, "recall": recall}


def register_masks(
    source: np.ndarray,
    target: np.ndarray,
    angle_step: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    source_center = mask_center(source)
    target_center = mask_center(target)
    base_scale = math.sqrt(float(target.sum()) / float(source.sum()))
    best: dict[str, Any] | None = None
    for angle in np.arange(0.0, 360.0, angle_step):
        matrix = affine_matrix(
            source_center, target_center, float(angle), base_scale, base_scale
        )
        metrics = registration_score(warp_mask(source, matrix, target.shape), target)
        candidate = {"matrix": matrix, "angle": float(angle), "metrics": metrics}
        if best is None or metrics["iou"] > best["metrics"]["iou"]:
            best = candidate
    assert best is not None

    initial_angle = float(best["angle"])

    def objective(parameters: np.ndarray) -> float:
        angle, log_scale_x, log_scale_y, shift_x, shift_y = parameters
        matrix = affine_matrix(
            source_center,
            target_center,
            float(angle),
            base_scale * math.exp(float(log_scale_x)),
            base_scale * math.exp(float(log_scale_y)),
            float(shift_x),
            float(shift_y),
        )
        metrics = registration_score(warp_mask(source, matrix, target.shape), target)
        anisotropy = abs(float(log_scale_x) - float(log_scale_y))
        return -metrics["iou"] + 0.015 * anisotropy

    optimized = minimize(
        objective,
        np.asarray([initial_angle, 0.0, 0.0, 0.0, 0.0]),
        method="Powell",
        bounds=[
            (initial_angle - max(angle_step * 1.5, 3.0), initial_angle + max(angle_step * 1.5, 3.0)),
            (math.log(0.82), math.log(1.18)),
            (math.log(0.82), math.log(1.18)),
            (-12.0, 12.0),
            (-12.0, 12.0),
        ],
        options={"maxiter": 180, "xtol": 0.05, "ftol": 1e-5},
    )
    angle, log_scale_x, log_scale_y, shift_x, shift_y = optimized.x
    matrix = affine_matrix(
        source_center,
        target_center,
        float(angle),
        base_scale * math.exp(float(log_scale_x)),
        base_scale * math.exp(float(log_scale_y)),
        float(shift_x),
        float(shift_y),
    )
    determinant = float(np.linalg.det(matrix[:, :2]))
    if determinant <= 0:
        raise ExperimentError("Registration attempted an invalid reflection")
    metrics = registration_score(warp_mask(source, matrix, target.shape), target)
    metadata = {
        **{key: round(float(value), 8) for key, value in metrics.items()},
        "angle_degrees": round(float(angle) % 360.0, 6),
        "scale_x": round(base_scale * math.exp(float(log_scale_x)), 8),
        "scale_y": round(base_scale * math.exp(float(log_scale_y)), 8),
        "shift_x_pixels": round(float(shift_x), 6),
        "shift_y_pixels": round(float(shift_y), 6),
        "determinant": round(determinant, 8),
        "reflection_allowed": False,
        "optimizer_success": bool(optimized.success),
        "matrix_source_pixels_to_solar_crop_pixels": matrix.tolist(),
    }
    return matrix, metadata


def warp_solar_to_source(
    values: np.ndarray,
    matrix_source_to_solar: np.ndarray,
    source_shape: tuple[int, int],
    interpolation: int,
    border_value: float | tuple[int, int, int] = 0,
) -> np.ndarray:
    inverse = cv2.invertAffineTransform(matrix_source_to_solar)
    return cv2.warpAffine(
        values,
        inverse,
        (source_shape[1], source_shape[0]),
        flags=interpolation,
        borderValue=border_value,
    )


def source_world_coordinates(
    shape: tuple[int, int],
    source_to_solar: np.ndarray,
    crop_transform: Any,
) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.indices(shape, dtype=np.float64)
    solar_x = source_to_solar[0, 0] * xx + source_to_solar[0, 1] * yy + source_to_solar[0, 2]
    solar_y = source_to_solar[1, 0] * xx + source_to_solar[1, 1] * yy + source_to_solar[1, 2]
    world_x = crop_transform.a * solar_x + crop_transform.b * solar_y + crop_transform.c
    world_y = crop_transform.d * solar_x + crop_transform.e * solar_y + crop_transform.f
    return world_x, world_y


def fit_metric_planes(
    labels: np.ndarray,
    dsm: np.ndarray,
    valid: np.ndarray,
    world_x: np.ndarray,
    world_y: np.ndarray,
    residual_threshold: float,
    max_samples: int,
) -> dict[int, dict[str, Any]]:
    fits: dict[int, dict[str, Any]] = {}
    random = np.random.default_rng(1729)
    for label in sorted(int(value) for value in np.unique(labels) if value > 0):
        region = (labels == label) & valid
        rows, columns = np.where(region)
        if len(rows) < 30:
            continue
        if len(rows) > max_samples:
            selected = random.choice(len(rows), max_samples, replace=False)
            rows, columns = rows[selected], columns[selected]
        origin_x = float(np.median(world_x[rows, columns]))
        origin_y = float(np.median(world_y[rows, columns]))
        features = np.column_stack(
            (world_x[rows, columns] - origin_x, world_y[rows, columns] - origin_y)
        )
        elevations = dsm[rows, columns]
        model = RANSACRegressor(
            estimator=LinearRegression(),
            min_samples=max(3, int(0.35 * len(elevations))),
            residual_threshold=residual_threshold,
            max_trials=100,
            random_state=1729,
        )
        try:
            model.fit(features, elevations)
        except (ValueError, np.linalg.LinAlgError):
            continue
        estimator = model.estimator_
        slope_x, slope_y = (float(value) for value in estimator.coef_)
        intercept = float(estimator.intercept_)
        predictions = model.predict(features)
        residuals = elevations - predictions
        inliers = model.inlier_mask_
        slope = math.hypot(slope_x, slope_y)
        pitch = math.degrees(math.atan(slope))
        downhill_east, downhill_north = -slope_x, -slope_y
        aspect = (
            math.degrees(math.atan2(downhill_east, downhill_north)) + 360.0
        ) % 360.0
        fits[label] = {
            "label": label,
            "origin_x": origin_x,
            "origin_y": origin_y,
            "slope_x": slope_x,
            "slope_y": slope_y,
            "intercept": intercept,
            "pitch_degrees": pitch,
            "aspect_degrees": aspect,
            "sample_count": int(len(elevations)),
            "inlier_count": int(inliers.sum()),
            "inlier_ratio": float(inliers.mean()),
            "median_absolute_residual_meters": float(np.median(np.abs(residuals))),
            "rmse_meters": float(np.sqrt(np.mean(residuals**2))),
        }
    return fits


def absolute_plane_coefficients(fit: dict[str, Any]) -> tuple[float, float, float]:
    slope_x = float(fit["slope_x"])
    slope_y = float(fit["slope_y"])
    intercept = (
        float(fit["intercept"])
        - slope_x * float(fit["origin_x"])
        - slope_y * float(fit["origin_y"])
    )
    return slope_x, slope_y, intercept


def plane_intersection_in_source_pixels(
    first: dict[str, Any],
    second: dict[str, Any],
    source_to_solar: np.ndarray,
    crop_transform: Any,
) -> np.ndarray | None:
    first_a, first_b, first_c = absolute_plane_coefficients(first)
    second_a, second_b, second_c = absolute_plane_coefficients(second)
    delta_a, delta_b, delta_c = (
        first_a - second_a,
        first_b - second_b,
        first_c - second_c,
    )
    world_u = np.asarray(
        [
            crop_transform.a * source_to_solar[0, 0]
            + crop_transform.b * source_to_solar[1, 0],
            crop_transform.d * source_to_solar[0, 0]
            + crop_transform.e * source_to_solar[1, 0],
        ]
    )
    world_v = np.asarray(
        [
            crop_transform.a * source_to_solar[0, 1]
            + crop_transform.b * source_to_solar[1, 1],
            crop_transform.d * source_to_solar[0, 1]
            + crop_transform.e * source_to_solar[1, 1],
        ]
    )
    world_origin = np.asarray(
        [
            crop_transform.a * source_to_solar[0, 2]
            + crop_transform.b * source_to_solar[1, 2]
            + crop_transform.c,
            crop_transform.d * source_to_solar[0, 2]
            + crop_transform.e * source_to_solar[1, 2]
            + crop_transform.f,
        ]
    )
    coefficients = np.asarray(
        [
            delta_a * world_u[0] + delta_b * world_u[1],
            delta_a * world_v[0] + delta_b * world_v[1],
            delta_a * world_origin[0] + delta_b * world_origin[1] + delta_c,
        ],
        dtype=np.float64,
    )
    if float(np.hypot(coefficients[0], coefficients[1])) < 1e-8:
        return None
    return coefficients


def project_to_implicit_line(
    point: np.ndarray,
    coefficients: np.ndarray,
    max_shift: float,
) -> tuple[np.ndarray, float]:
    normal = coefficients[:2]
    denominator = float(np.dot(normal, normal))
    displacement = -(
        float(np.dot(normal, point)) + float(coefficients[2])
    ) * normal / denominator
    distance = float(np.linalg.norm(displacement))
    if distance > max_shift:
        displacement *= max_shift / distance
    return point + displacement, distance


def line_orientation_degrees(points: np.ndarray) -> float:
    direction = points[-1] - points[0]
    return math.degrees(math.atan2(float(direction[1]), float(direction[0]))) % 180.0


def implicit_line_orientation_degrees(coefficients: np.ndarray) -> float:
    return math.degrees(math.atan2(float(-coefficients[0]), float(coefficients[1]))) % 180.0


def angle_difference_degrees(first: float, second: float) -> float:
    difference = abs(first - second) % 180.0
    return min(difference, 180.0 - difference)


def consolidate_vector_junctions(
    original: dict[str, np.ndarray],
    proposed: dict[str, np.ndarray],
    radius: float = 3.0,
) -> dict[str, np.ndarray]:
    references: list[tuple[str, int]] = []
    coordinates: list[np.ndarray] = []
    for boundary_id, points in original.items():
        references.extend(((boundary_id, 0), (boundary_id, len(points) - 1)))
        coordinates.extend((points[0], points[-1]))
    parent = list(range(len(coordinates)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first: int, second: int) -> None:
        first_root, second_root = find(first), find(second)
        if first_root != second_root:
            parent[second_root] = first_root

    for first in range(len(coordinates)):
        for second in range(first + 1, len(coordinates)):
            if float(np.linalg.norm(coordinates[first] - coordinates[second])) <= radius:
                union(first, second)
    groups: dict[int, list[int]] = {}
    for index in range(len(coordinates)):
        groups.setdefault(find(index), []).append(index)
    result = {key: value.copy() for key, value in proposed.items()}
    for indices in groups.values():
        if len(indices) < 2:
            continue
        targets = []
        for index in indices:
            boundary_id, endpoint = references[index]
            targets.append(result[boundary_id][endpoint])
        shared = np.mean(np.asarray(targets), axis=0)
        for index in indices:
            boundary_id, endpoint = references[index]
            result[boundary_id][endpoint] = shared
    return result


def refine_vector_paths_with_dsm(
    paths: dict[str, list[list[int]]],
    plan: dict[str, Any],
    fits: dict[int, dict[str, Any]],
    source_to_solar: np.ndarray,
    crop_transform: Any,
    max_shift: float,
) -> tuple[dict[str, list[tuple[int, int]]], dict[str, Any]]:
    boundary_lookup = {
        str(boundary["boundary_id"]): boundary
        for boundary in plan.get("boundaries", [])
        if isinstance(boundary, dict) and boundary.get("boundary_id")
    }
    original = {
        boundary_id: np.asarray(points, dtype=np.float64)
        for boundary_id, points in paths.items()
        if len(points) >= 2
    }
    proposed = {key: value.copy() for key, value in original.items()}
    adjustments: list[dict[str, Any]] = []
    for boundary_id, points in original.items():
        boundary = boundary_lookup.get(boundary_id)
        plane_ids = boundary.get("plane_ids", []) if boundary else []
        if len(plane_ids) != 2 or any(int(label) not in fits for label in plane_ids):
            continue
        first, second = (fits[int(label)] for label in plane_ids)
        coefficients = plane_intersection_in_source_pixels(
            first, second, source_to_solar, crop_transform
        )
        if coefficients is None:
            continue
        original_angle = line_orientation_degrees(points)
        dsm_angle = implicit_line_orientation_degrees(coefficients)
        angle_difference = angle_difference_degrees(original_angle, dsm_angle)
        if angle_difference > 28.0:
            adjustments.append(
                {
                    "boundary_id": boundary_id,
                    "plane_ids": plane_ids,
                    "accepted": False,
                    "reason": "intersection_angle_conflicts_with_vector_topology",
                    "angle_difference_degrees": round(angle_difference, 6),
                }
            )
            continue
        projected_start, start_distance = project_to_implicit_line(
            points[0], coefficients, max_shift
        )
        projected_end, end_distance = project_to_implicit_line(
            points[-1], coefficients, max_shift
        )
        adjusted = points.copy()
        adjusted[0] = projected_start
        adjusted[-1] = projected_end
        if len(points) > 2:
            for index in range(1, len(points) - 1):
                adjusted[index], _ = project_to_implicit_line(
                    points[index], coefficients, max_shift
                )
        proposed[boundary_id] = adjusted
        adjustments.append(
            {
                "boundary_id": boundary_id,
                "plane_ids": plane_ids,
                "accepted": True,
                "angle_difference_degrees": round(angle_difference, 6),
                "unclamped_start_shift_pixels": round(start_distance, 6),
                "unclamped_end_shift_pixels": round(end_distance, 6),
            }
        )
    consolidated = consolidate_vector_junctions(original, proposed)
    serialized = {
        boundary_id: [
            (int(round(float(point[0]))), int(round(float(point[1]))))
            for point in points
        ]
        for boundary_id, points in consolidated.items()
    }
    accepted = sum(bool(item["accepted"]) for item in adjustments)
    return serialized, {
        "path_count": len(serialized),
        "evaluated_boundary_count": len(adjustments),
        "adjusted_boundary_count": accepted,
        "rejected_boundary_count": len(adjustments) - accepted,
        "max_vector_shift_pixels": max_shift,
        "adjustments": adjustments,
    }


def boundary_complexity(labels: np.ndarray) -> int:
    horizontal = (
        (labels[:, 1:] != labels[:, :-1])
        & (labels[:, 1:] > 0)
        & (labels[:, :-1] > 0)
    )
    vertical = (
        (labels[1:, :] != labels[:-1, :])
        & (labels[1:, :] > 0)
        & (labels[:-1, :] > 0)
    )
    return int(horizontal.sum() + vertical.sum())


def aggregate_fit_metrics(fits: dict[int, dict[str, Any]]) -> dict[str, Any]:
    if not fits:
        return {
            "fitted_plane_count": 0,
            "weighted_median_absolute_residual_meters": None,
            "weighted_rmse_meters": None,
        }
    weights = np.asarray([fit["sample_count"] for fit in fits.values()], dtype=np.float64)
    medians = np.asarray(
        [fit["median_absolute_residual_meters"] for fit in fits.values()], dtype=np.float64
    )
    rmses = np.asarray([fit["rmse_meters"] for fit in fits.values()], dtype=np.float64)
    return {
        "fitted_plane_count": len(fits),
        "weighted_median_absolute_residual_meters": round(
            float(np.average(medians, weights=weights)), 8
        ),
        "weighted_rmse_meters": round(float(np.average(rmses, weights=weights)), 8),
    }


def save_dsm_visualization(
    dsm: np.ndarray,
    valid: np.ndarray,
    output_path: Path,
) -> Image.Image:
    valid_values = dsm[valid]
    if valid_values.size:
        low, high = np.percentile(valid_values, (2.0, 98.0))
    else:
        low, high = 0.0, 1.0
    normalized = np.clip((dsm - low) / max(float(high - low), 1e-6), 0.0, 1.0)
    colored = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    colored[~valid] = 255
    image = Image.fromarray(colored)
    image.save(output_path)
    return image


def save_registered_geotiff(
    output_path: Path,
    dsm: np.ndarray,
    valid: np.ndarray,
    source_to_solar: np.ndarray,
    crop_transform: Any,
    crs: Any,
) -> None:
    import rasterio
    from rasterio import Affine

    source_pixel_to_crop = Affine(
        float(source_to_solar[0, 0]),
        float(source_to_solar[0, 1]),
        float(source_to_solar[0, 2]),
        float(source_to_solar[1, 0]),
        float(source_to_solar[1, 1]),
        float(source_to_solar[1, 2]),
    )
    transform = crop_transform * source_pixel_to_crop
    nodata = -9999.0
    output = np.where(valid, dsm, nodata).astype(np.float32)
    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=output.shape[0],
        width=output.shape[1],
        count=1,
        dtype="float32",
        crs=crs,
        transform=transform,
        nodata=nodata,
        compress="deflate",
    ) as destination:
        destination.write(output, 1)


def georeferenced_geojson(
    labels: np.ndarray,
    source_to_solar: np.ndarray,
    crop_transform: Any,
    crs: Any,
    fits: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    from rasterio.warp import transform

    features: list[dict[str, Any]] = []
    for label in sorted(int(value) for value in np.unique(labels) if value > 0):
        contours, _ = cv2.findContours(
            (labels == label).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea).reshape(-1, 2)
        if len(contour) < 3:
            continue
        points = contour.astype(np.float64)
        solar_x = (
            source_to_solar[0, 0] * points[:, 0]
            + source_to_solar[0, 1] * points[:, 1]
            + source_to_solar[0, 2]
        )
        solar_y = (
            source_to_solar[1, 0] * points[:, 0]
            + source_to_solar[1, 1] * points[:, 1]
            + source_to_solar[1, 2]
        )
        world_x = crop_transform.a * solar_x + crop_transform.b * solar_y + crop_transform.c
        world_y = crop_transform.d * solar_x + crop_transform.e * solar_y + crop_transform.f
        longitudes, latitudes = transform(crs, "EPSG:4326", world_x.tolist(), world_y.tolist())
        ring = [[longitude, latitude] for longitude, latitude in zip(longitudes, latitudes)]
        ring.append(ring[0])
        properties = {"plane_id": label, "pixel_area": int((labels == label).sum())}
        if label in fits:
            properties.update(
                {
                    "pitch_degrees": round(float(fits[label]["pitch_degrees"]), 6),
                    "aspect_degrees": round(float(fits[label]["aspect_degrees"]), 6),
                    "rmse_meters": round(float(fits[label]["rmse_meters"]), 6),
                    "inlier_ratio": round(float(fits[label]["inlier_ratio"]), 6),
                }
            )
        features.append(
            {
                "type": "Feature",
                "properties": properties,
                "geometry": {"type": "Polygon", "coordinates": [ring]},
            }
        )
    return {"type": "FeatureCollection", "features": features}


def run(args: argparse.Namespace) -> Path:
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("GOOGLE_MAPS_API_KEY", "").strip()
    if not api_key:
        raise ExperimentError("GOOGLE_MAPS_API_KEY is not configured in .env")
    latitude, longitude = parse_location(args.location)
    output_root = args.output_root.resolve()
    source_run = (
        args.source_run.resolve()
        if args.source_run
        else find_latest_vector_run(output_root)
    )
    candidate_directory = source_run / f"candidate_{args.source_candidate}"
    labels_path = candidate_directory / "roof_planes_labels.png"
    if not labels_path.is_file():
        raise FileNotFoundError(f"Candidate labels were not found: {labels_path}")
    run_directory = create_run_directory(output_root, EXPERIMENT_NAME)
    solar_directory = run_directory / "google_solar"

    source_result = json.loads((source_run / "result.json").read_text(encoding="utf-8"))
    candidate_result = json.loads(
        (candidate_directory / "result.json").read_text(encoding="utf-8")
    )
    optimizer_run = Path(source_result["source_run"])
    source_hypothesis = int(source_result["source_hypothesis"])
    hypothesis_directory = optimizer_run / f"hypothesis_{source_hypothesis:02d}"
    plan = json.loads((hypothesis_directory / "vlm_topology.json").read_text(encoding="utf-8"))
    evidence = np.asarray(
        Image.open(source_run / "combined_boundary_evidence.png").convert("L"),
        dtype=np.float32,
    ) / 255.0

    image = Image.open(source_run / "input.png").convert("RGB")
    rgb = np.asarray(image)
    labels = np.asarray(Image.open(labels_path), dtype=np.int32)
    roof_mask = np.asarray(Image.open(source_run / "building_mask.png").convert("L")) > 0
    labels[~roof_mask] = 0
    image.save(run_directory / "input.png")
    Image.fromarray((roof_mask * 255).astype(np.uint8)).save(
        run_directory / "building_mask.png"
    )
    save_label_mask(labels, run_directory / "vector_structural_labels.png")
    _, structural_overlay, _ = render_planes(rgb, labels)
    Image.fromarray(structural_overlay).save(run_directory / "vector_structural_overlay.png")

    solar_metadata = fetch_google_solar_layers(
        latitude, longitude, api_key, solar_directory
    )
    solar = read_solar_crop(solar_directory, solar_metadata)
    Image.fromarray(solar["rgb"]).save(run_directory / "google_solar_rgb_crop.png")
    Image.fromarray((solar["mask"] * 255).astype(np.uint8)).save(
        run_directory / "google_solar_target_mask_crop.png"
    )

    matrix, registration = register_masks(
        roof_mask,
        solar["mask"],
        args.registration_angle_step,
    )
    warped_source_mask = warp_mask(roof_mask, matrix, solar["mask"].shape)
    registration_overlay = solar["rgb"].copy()
    registration_overlay[warped_source_mask] = (
        0.55 * registration_overlay[warped_source_mask]
        + 0.45 * np.asarray([110, 65, 220])
    ).astype(np.uint8)
    target_outline = cv2.morphologyEx(
        solar["mask"].astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)
    ) > 0
    registration_overlay[target_outline] = np.asarray([0, 255, 90], dtype=np.uint8)
    Image.fromarray(registration_overlay).save(run_directory / "registration_overlay.png")

    registration_accepted = registration["iou"] >= args.minimum_registration_iou
    source_shape = roof_mask.shape
    registered_dsm = warp_solar_to_source(
        solar["dsm"], matrix, source_shape, cv2.INTER_LINEAR, 0.0
    ).astype(np.float32)
    registered_valid = warp_solar_to_source(
        solar["dsm_valid"].astype(np.uint8), matrix, source_shape, cv2.INTER_NEAREST, 0
    ) > 0
    registered_solar_mask = warp_solar_to_source(
        solar["mask"].astype(np.uint8), matrix, source_shape, cv2.INTER_NEAREST, 0
    ) > 0
    registered_valid &= registered_solar_mask & roof_mask
    registered_rgb = warp_solar_to_source(
        solar["rgb"], matrix, source_shape, cv2.INTER_LINEAR, (255, 255, 255)
    )
    Image.fromarray(registered_rgb).save(run_directory / "google_solar_rgb_registered.png")
    dsm_visual = save_dsm_visualization(
        registered_dsm, registered_valid, run_directory / "google_solar_dsm_registered.png"
    )
    save_registered_geotiff(
        run_directory / "google_solar_dsm_registered.tif",
        registered_dsm,
        registered_valid,
        matrix,
        solar["transform"],
        solar["crs"],
    )

    world_x, world_y = source_world_coordinates(
        source_shape, matrix, solar["transform"]
    )
    baseline_fits = fit_metric_planes(
        labels,
        registered_dsm,
        registered_valid,
        world_x,
        world_y,
        args.plane_residual_threshold_meters,
        args.max_fit_samples,
    )
    refined_paths, refinement = refine_vector_paths_with_dsm(
        candidate_result["paths"],
        plan,
        baseline_fits,
        matrix,
        solar["transform"],
        args.max_vector_shift_pixels,
    )
    refined_labels, _, _ = partition_roof(
        plan, roof_mask, evidence, refined_paths
    )
    refined_fits = fit_metric_planes(
        refined_labels,
        registered_dsm,
        registered_valid,
        world_x,
        world_y,
        args.plane_residual_threshold_meters,
        args.max_fit_samples,
    )
    baseline_metrics = aggregate_fit_metrics(baseline_fits)
    refined_metrics = aggregate_fit_metrics(refined_fits)
    residual_improved = (
        refined_metrics["weighted_rmse_meters"] is not None
        and baseline_metrics["weighted_rmse_meters"] is not None
        and refined_metrics["weighted_rmse_meters"]
        <= baseline_metrics["weighted_rmse_meters"] + 1e-6
    )
    labels_preserved = set(np.unique(labels)) - {0} <= set(np.unique(refined_labels))
    baseline_complexity = boundary_complexity(labels)
    refined_complexity = boundary_complexity(refined_labels)
    complexity_ratio = refined_complexity / max(baseline_complexity, 1)
    complexity_accepted = complexity_ratio <= args.maximum_boundary_complexity_ratio
    dsm_refinement_accepted = bool(
        registration_accepted
        and residual_improved
        and labels_preserved
        and complexity_accepted
    )
    final_labels = refined_labels if dsm_refinement_accepted else labels
    final_fits = refined_fits if dsm_refinement_accepted else baseline_fits
    selected_result = (
        "google_solar_dsm_refined" if dsm_refinement_accepted else "vector_structural_fallback"
    )

    final_colors, final_overlay, _ = render_planes(rgb, final_labels)
    save_label_mask(final_labels, run_directory / "final_roof_planes_labels.png")
    Image.fromarray(final_colors).save(run_directory / "final_roof_planes_color.png")
    Image.fromarray(final_overlay).save(run_directory / "final_roof_planes_overlay.png")
    draw_topology(image, plan, snapped_paths=refined_paths).save(
        run_directory / "dsm_refined_vector_graph_overlay.png"
    )
    (run_directory / "dsm_refined_vector_graph.json").write_text(
        json.dumps(
            {
                "paths": {
                    boundary_id: [[int(x), int(y)] for x, y in points]
                    for boundary_id, points in refined_paths.items()
                },
                **refinement,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    geojson = georeferenced_geojson(
        final_labels, matrix, solar["transform"], solar["crs"], final_fits
    )
    (run_directory / "final_roof_planes_georeferenced.geojson").write_text(
        json.dumps(geojson, indent=2), encoding="utf-8"
    )
    plane_metrics = {
        str(label): {
            key: round(float(value), 8) if isinstance(value, float) else value
            for key, value in fit.items()
        }
        for label, fit in final_fits.items()
    }
    (run_directory / "metric_plane_fits.json").write_text(
        json.dumps(plane_metrics, indent=2), encoding="utf-8"
    )

    registration_comparison: list[tuple[str, Image.Image]] = [
        ("Original input", image),
        ("Google Solar RGB crop", Image.fromarray(solar["rgb"])),
        (
            f"Mask registration | IoU {registration['iou']:.3f}",
            Image.fromarray(registration_overlay),
        ),
        ("Google Solar RGB registered", Image.fromarray(registered_rgb)),
    ]
    make_contact_sheet(
        registration_comparison, run_directory / "registration_comparison.jpg"
    )
    comparison: list[tuple[str, Image.Image]] = [
        ("Original input", image),
        ("Human-selected vector structural", Image.fromarray(structural_overlay)),
        ("Google Solar metric DSM registered", dsm_visual),
        (f"Selected: {selected_result}", Image.fromarray(final_overlay)),
    ]
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
            "Use Google Solar metric DSM evidence to validate and refine the "
            "human-selected vector-structural roof-plane topology."
        ),
        "source_run": str(source_run),
        "source_candidate": args.source_candidate,
        "location": {"latitude": latitude, "longitude": longitude},
        "solar_metadata": solar_metadata,
        "solar_crop": solar["window"],
        "registration": registration,
        "minimum_registration_iou": args.minimum_registration_iou,
        "registration_accepted": registration_accepted,
        "baseline_metric_fit": baseline_metrics,
        "refined_metric_fit": refined_metrics,
        "refinement": refinement,
        "boundary_complexity": {
            "baseline": baseline_complexity,
            "refined": refined_complexity,
            "ratio": round(complexity_ratio, 8),
            "maximum_accepted_ratio": args.maximum_boundary_complexity_ratio,
            "accepted": complexity_accepted,
        },
        "dsm_refinement_accepted": dsm_refinement_accepted,
        "selected_result": selected_result,
        "final_plane_count": len(summarize_planes(final_labels)),
        "manual_reference_available": manual_available,
        "manual_reference_used_for_registration": False,
        "manual_reference_used_for_optimization": False,
        "manual_reference_used_for_scoring": False,
        "manual_reference_used_for_selection": False,
        "manual_reference_visual_comparison_only": True,
        "external_services": ["Google Solar Building Insights", "Google Solar Data Layers"],
    }
    (run_directory / "result.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(
        f"Registration IoU: {registration['iou']:.4f}; "
        f"selected result: {selected_result}"
    )
    print(f"Output: {run_directory}")
    return run_directory


if __name__ == "__main__":
    run(parse_args())
