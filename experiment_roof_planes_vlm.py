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
EXPERIMENT_NAME = "roof_planes_experiment_03_vlm"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Experiment 03: use a multimodal model to interpret and regroup the "
            "local structural roof-plane candidates from experiment 02."
        )
    )
    parser.add_argument(
        "--source-run",
        type=Path,
        help="Experiment 02 output directory. Defaults to the latest valid run.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument(
        "--detail", choices=("low", "high", "auto"), default="high"
    )
    return parser.parse_args()


def find_latest_structural_run(output_root: Path) -> Path:
    required = {
        "input.png",
        "building_mask.png",
        "structural_planes_labels.png",
        "structural_planes_overlay.png",
        "structural_lines.png",
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


def load_label_mask(path: Path) -> np.ndarray:
    labels = np.asarray(Image.open(path))
    if labels.ndim != 2:
        raise ValueError(f"Expected a single-channel label mask: {path}")
    return labels.astype(np.int32)


def candidate_metadata(labels: np.ndarray) -> list[dict[str, Any]]:
    roof_pixels = max(int(np.count_nonzero(labels)), 1)
    records: list[dict[str, Any]] = []
    for candidate_id in sorted(int(value) for value in np.unique(labels) if value):
        ys, xs = np.where(labels == candidate_id)
        records.append(
            {
                "candidate_id": candidate_id,
                "area_pixels": int(xs.size),
                "roof_percentage": round(100.0 * xs.size / roof_pixels, 3),
                "centroid_normalized": [
                    round(float(xs.mean()) / labels.shape[1], 4),
                    round(float(ys.mean()) / labels.shape[0], 4),
                ],
                "bbox_normalized": [
                    round(float(xs.min()) / labels.shape[1], 4),
                    round(float(ys.min()) / labels.shape[0], 4),
                    round(float(xs.max()) / labels.shape[1], 4),
                    round(float(ys.max()) / labels.shape[0], 4),
                ],
            }
        )
    return records


def outlined_text(
    draw: ImageDraw.ImageDraw,
    position: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
) -> None:
    x, y = position
    for dx, dy in ((-2, 0), (2, 0), (0, -2), (0, 2)):
        draw.text((x + dx, y + dy), text, font=font, fill="black", anchor="mm")
    draw.text((x, y), text, font=font, fill="white", anchor="mm")


def create_numbered_candidate_map(
    image: Image.Image, labels: np.ndarray
) -> Image.Image:
    _, overlay, _ = render_planes(np.asarray(image.convert("RGB")), labels)
    canvas = Image.fromarray(overlay)
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=18)
    for record in candidate_metadata(labels):
        candidate_id = record["candidate_id"]
        ys, xs = np.where(labels == candidate_id)
        outlined_text(
            draw,
            (int(round(xs.mean())), int(round(ys.mean()))),
            f"C{candidate_id}",
            font,
        )
    return canvas


def image_data_url(image: Image.Image, *, quality: int = 92) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=quality, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def response_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "analysis_summary": {"type": "string"},
            "roof_planes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "plane_id": {"type": "string"},
                        "candidate_ids": {
                            "type": "array",
                            "items": {"type": "integer"},
                        },
                        "confidence": {"type": "number"},
                        "reason": {"type": "string"},
                    },
                    "required": [
                        "plane_id",
                        "candidate_ids",
                        "confidence",
                        "reason",
                    ],
                    "additionalProperties": False,
                },
            },
            "split_requests": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "candidate_id": {"type": "integer"},
                        "boundary_type": {
                            "type": "string",
                            "enum": ["ridge", "valley", "hip", "unknown"],
                        },
                        "line": {
                            "type": "object",
                            "properties": {
                                "x1": {"type": "number"},
                                "y1": {"type": "number"},
                                "x2": {"type": "number"},
                                "y2": {"type": "number"},
                            },
                            "required": ["x1", "y1", "x2", "y2"],
                            "additionalProperties": False,
                        },
                        "confidence": {"type": "number"},
                        "reason": {"type": "string"},
                    },
                    "required": [
                        "candidate_id",
                        "boundary_type",
                        "line",
                        "confidence",
                        "reason",
                    ],
                    "additionalProperties": False,
                },
            },
            "uncertain_candidate_ids": {
                "type": "array",
                "items": {"type": "integer"},
            },
        },
        "required": [
            "analysis_summary",
            "roof_planes",
            "split_requests",
            "uncertain_candidate_ids",
        ],
        "additionalProperties": False,
    }


def build_prompt(metadata: list[dict[str, Any]]) -> str:
    return f"""
You are analyzing roof geometry from an overhead aerial image. The first image is
the original RGB image. The second image is an algorithmic roof segmentation in
which each colored region has a stable label C1, C2, and so on. The third image
shows locally detected structural lines.

Determine which candidate regions represent the same physical planar roof face.
Merge candidates only when they are visibly coplanar and separated by an
algorithmic artifact, shadow, texture, or weak false boundary. Never merge across
a visible ridge, valley, hip, or strong change in roof slope. If one candidate
clearly crosses a real roof boundary, add one split request with a normalized
line segment that follows that boundary. Coordinates must be in [0, 1] relative
to the full image. Do not invent candidates or use pixels outside the building.

Every candidate should occur at most once across roof_planes. Candidates needing
a split should be omitted from roof_planes and listed in split_requests. Put
ambiguous candidates in uncertain_candidate_ids. Confidence must be from 0 to 1.

Candidate metadata:
{json.dumps(metadata, indent=2)}
""".strip()


def request_vlm_analysis(
    client: OpenAI,
    model: str,
    detail: str,
    image: Image.Image,
    candidate_map: Image.Image,
    structural_lines: Image.Image,
    metadata: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    response = client.responses.create(
        model=model,
        reasoning={"effort": "medium"},
        store=False,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": build_prompt(metadata)},
                    {
                        "type": "input_image",
                        "image_url": image_data_url(image),
                        "detail": detail,
                    },
                    {
                        "type": "input_image",
                        "image_url": image_data_url(candidate_map),
                        "detail": detail,
                    },
                    {
                        "type": "input_image",
                        "image_url": image_data_url(structural_lines),
                        "detail": detail,
                    },
                ],
            }
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "roof_plane_topology",
                "strict": True,
                "schema": response_schema(),
            }
        },
        max_output_tokens=6000,
    )
    parsed = json.loads(response.output_text)
    metadata_out = {
        "response_id": response.id,
        "model": model,
        "status": response.status,
        "usage": response.usage.model_dump() if response.usage else None,
    }
    return parsed, metadata_out


def normalized_line_pixels(
    line: dict[str, float], width: int, height: int
) -> tuple[tuple[int, int], tuple[int, int]]:
    values = [
        float(np.clip(line[name], 0.0, 1.0))
        for name in ("x1", "y1", "x2", "y2")
    ]
    x1, y1, x2, y2 = values
    return (
        (int(round(x1 * (width - 1))), int(round(y1 * (height - 1)))),
        (int(round(x2 * (width - 1))), int(round(y2 * (height - 1)))),
    )


def split_candidate(
    candidate_mask: np.ndarray,
    line: dict[str, float],
    thickness: int = 5,
) -> list[np.ndarray]:
    height, width = candidate_mask.shape
    p1, p2 = normalized_line_pixels(line, width, height)
    if p1 == p2:
        return []

    barrier = np.zeros_like(candidate_mask, dtype=np.uint8)
    cv2.line(barrier, p1, p2, color=1, thickness=thickness, lineType=cv2.LINE_8)
    separated = candidate_mask & ~barrier.astype(bool)
    count, components = cv2.connectedComponents(separated.astype(np.uint8), 8)
    pieces = [components == value for value in range(1, count)]
    minimum = max(25, int(np.count_nonzero(candidate_mask) * 0.04))
    pieces = [piece for piece in pieces if np.count_nonzero(piece) >= minimum]
    if len(pieces) < 2:
        return []

    ownership = np.zeros_like(components, dtype=np.int32)
    for index, piece in enumerate(pieces, start=1):
        ownership[piece] = index
    missing = candidate_mask & (ownership == 0)
    if np.any(missing):
        _, nearest = ndimage.distance_transform_edt(
            ownership == 0, return_indices=True
        )
        nearest_owner = ownership[tuple(nearest)]
        ownership[missing] = nearest_owner[missing]
    return [ownership == value for value in range(1, len(pieces) + 1)]


def apply_vlm_plan(
    structural_labels: np.ndarray,
    plan: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    valid_ids = {int(value) for value in np.unique(structural_labels) if value}
    used: set[int] = set()
    output = np.zeros_like(structural_labels, dtype=np.int32)
    next_label = 1
    rejected: list[dict[str, Any]] = []
    applied_splits: list[dict[str, Any]] = []

    split_ids = {
        int(item["candidate_id"])
        for item in plan["split_requests"]
        if int(item["candidate_id"]) in valid_ids
    }
    for plane in plan["roof_planes"]:
        requested = [int(value) for value in plane["candidate_ids"]]
        accepted = [
            value
            for value in requested
            if value in valid_ids and value not in used and value not in split_ids
        ]
        if not accepted:
            rejected.append({"type": "empty_plane", "plane": plane})
            continue
        mask = np.isin(structural_labels, accepted)
        output[mask] = next_label
        used.update(accepted)
        next_label += 1

    for request in plan["split_requests"]:
        candidate_id = int(request["candidate_id"])
        if candidate_id not in valid_ids or candidate_id in used:
            rejected.append({"type": "invalid_split", "request": request})
            continue
        pieces = split_candidate(
            structural_labels == candidate_id,
            request["line"],
        )
        if len(pieces) < 2:
            rejected.append({"type": "ineffective_split", "request": request})
            continue
        for piece in pieces:
            output[piece] = next_label
            next_label += 1
        used.add(candidate_id)
        applied_splits.append(
            {"candidate_id": candidate_id, "piece_count": len(pieces)}
        )

    for candidate_id in sorted(valid_ids - used):
        output[structural_labels == candidate_id] = next_label
        next_label += 1

    return output, {
        "valid_candidate_ids": sorted(valid_ids),
        "used_candidate_ids": sorted(used),
        "preserved_candidate_ids": sorted(valid_ids - used),
        "applied_splits": applied_splits,
        "rejected_operations": rejected,
    }


def draw_guidance(
    candidate_map: Image.Image,
    structural_labels: np.ndarray,
    plan: dict[str, Any],
) -> Image.Image:
    canvas = candidate_map.copy()
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=16)
    width, height = canvas.size
    colors = {
        "ridge": "#ff2d55",
        "valley": "#00d4ff",
        "hip": "#ffd60a",
        "unknown": "#ffffff",
    }
    centers: dict[int, tuple[int, int]] = {}
    for candidate_id in (int(value) for value in np.unique(structural_labels) if value):
        ys, xs = np.where(structural_labels == candidate_id)
        centers[candidate_id] = (int(round(xs.mean())), int(round(ys.mean())))

    for plane in plan["roof_planes"]:
        candidate_ids = [
            int(value)
            for value in plane["candidate_ids"]
            if int(value) in centers
        ]
        if len(candidate_ids) < 2:
            continue
        anchor = centers[candidate_ids[0]]
        for candidate_id in candidate_ids[1:]:
            draw.line((anchor, centers[candidate_id]), fill="#ff00c8", width=5)
        outlined_text(draw, anchor, f"merge {plane['plane_id']}", font)

    for candidate_id in plan["uncertain_candidate_ids"]:
        center = centers.get(int(candidate_id))
        if center is None:
            continue
        radius = 18
        draw.ellipse(
            (
                center[0] - radius,
                center[1] - radius,
                center[0] + radius,
                center[1] + radius,
            ),
            outline="#ff9500",
            width=4,
        )
    for request in plan["split_requests"]:
        p1, p2 = normalized_line_pixels(request["line"], width, height)
        color = colors[request["boundary_type"]]
        draw.line((p1, p2), fill=color, width=5)
        midpoint = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
        outlined_text(
            draw,
            midpoint,
            f"split C{request['candidate_id']}",
            font,
        )
    return canvas


def run(args: argparse.Namespace) -> Path:
    load_dotenv(PROJECT_ROOT / ".env")
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not defined in .env or the environment")

    output_root = args.output_root.resolve()
    source_run = (
        args.source_run.resolve()
        if args.source_run
        else find_latest_structural_run(output_root)
    )
    run_directory = create_run_directory(output_root, EXPERIMENT_NAME)

    image = Image.open(source_run / "input.png").convert("RGB")
    structural_labels = load_label_mask(
        source_run / "structural_planes_labels.png"
    )
    structural_lines = Image.open(source_run / "structural_lines.png").convert("RGB")
    candidate_map = create_numbered_candidate_map(image, structural_labels)
    metadata = candidate_metadata(structural_labels)

    image.save(run_directory / "input.png")
    Image.open(source_run / "building_mask.png").save(
        run_directory / "building_mask.png"
    )
    candidate_map.save(run_directory / "numbered_structural_candidates.png")
    structural_lines.save(run_directory / "structural_lines.png")

    request_summary = {
        "model": args.model,
        "detail": args.detail,
        "source_run": str(source_run),
        "prompt": build_prompt(metadata),
        "candidate_metadata": metadata,
        "images": [
            "input.png",
            "numbered_structural_candidates.png",
            "structural_lines.png",
        ],
    }
    (run_directory / "vlm_request_summary.json").write_text(
        json.dumps(request_summary, indent=2), encoding="utf-8"
    )

    print(f"Requesting semantic roof topology from {args.model}...")
    plan, response_metadata = request_vlm_analysis(
        OpenAI(),
        args.model,
        args.detail,
        image,
        candidate_map,
        structural_lines,
        metadata,
    )
    (run_directory / "vlm_response.json").write_text(
        json.dumps(plan, indent=2), encoding="utf-8"
    )

    hybrid_labels, validation = apply_vlm_plan(structural_labels, plan)
    colors, overlay, _ = render_planes(np.asarray(image), hybrid_labels)
    summaries = summarize_planes(hybrid_labels)
    guidance = draw_guidance(candidate_map, structural_labels, plan)

    save_label_mask(hybrid_labels, run_directory / "hybrid_roof_planes_labels.png")
    Image.fromarray(colors).save(run_directory / "hybrid_roof_planes_color.png")
    Image.fromarray(overlay).save(run_directory / "hybrid_roof_planes_overlay.png")
    guidance.save(run_directory / "vlm_guidance_overlay.png")
    write_geojson(
        hybrid_labels,
        summaries,
        run_directory / "hybrid_roof_planes.geojson",
    )

    comparison_items = [
        ("Numbered local candidates", candidate_map),
        ("VLM guidance", guidance),
        ("VLM-guided local result", Image.fromarray(overlay)),
    ]
    manual_reference = source_run / "manual_reference.png"
    if manual_reference.is_file():
        comparison_items.append(
            ("Manual qualitative reference", Image.open(manual_reference).convert("RGB"))
        )
        Image.open(manual_reference).save(run_directory / "manual_reference.png")
    make_contact_sheet(comparison_items, run_directory / "qualitative_comparison.jpg")

    result = {
        "experiment": EXPERIMENT_NAME,
        "purpose": (
            "External multimodal semantic interpretation of local structural "
            "roof-plane candidates; not metric or georeferenced geometry"
        ),
        "source_run": str(source_run),
        "model_response": response_metadata,
        "candidate_count": len(metadata),
        "hybrid_plane_count": len(summaries),
        "vlm_plan": plan,
        "validation": validation,
        "planes": [asdict(summary) for summary in summaries],
    }
    (run_directory / "result.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(f"Created {len(summaries)} VLM-guided roof-plane candidates")
    print(f"Output: {run_directory}")
    return run_directory


if __name__ == "__main__":
    run(parse_args())
