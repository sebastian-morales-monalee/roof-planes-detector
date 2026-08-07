from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw

from run_artifacts import create_run_directory

DEFAULT_INPUT = Path(__file__).parent / "start_input" / "UP_original.png"
DEFAULT_OUTPUT = Path(__file__).parent / "outputs"
PROMPTS = ("building.", "roof.")
MIN_MASK_PERCENTAGE = 5.0
MAX_MASK_PERCENTAGE = 85.0
MIN_DIMENSION_RATIO = 0.12
MAX_HORIZONTAL_CENTER_OFFSET_RATIO = 0.30
MAX_VERTICAL_CENTER_OFFSET_RATIO = 0.40


@dataclass(frozen=True)
class Prediction:
    mask: Image.Image
    box: tuple[float, float, float, float]
    score: float | None
    mask_score: float | None
    label: str
    labels: list[str]
    scores: list[float]
    boxes: list[list[float]]
    mask_scores: list[float]
    selection_strategy: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare LangSAM building and roof segmentations."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--output-root",
        "--output",
        dest="output_root",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Root directory where a timestamped run folder will be created.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="auto selects CUDA when available and otherwise uses CPU.",
    )
    parser.add_argument("--sam-type", default="sam2.1_hiera_small")
    parser.add_argument("--box-threshold", type=float, default=0.25)
    parser.add_argument("--text-threshold", type=float, default=0.20)
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but PyTorch cannot access a CUDA GPU"
            )
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prepare_cache(project_root: Path) -> None:
    cache_root = project_root / ".cache"
    values = {
        "XDG_CACHE_HOME": cache_root,
        "TORCH_HOME": cache_root / "torch",
        "HF_HOME": cache_root / "huggingface",
    }
    for name, path in values.items():
        os.environ.setdefault(name, str(path))
        Path(os.environ[name]).mkdir(parents=True, exist_ok=True)


def select_prediction(result: dict[str, Any], image: Image.Image) -> Prediction:
    boxes = np.asarray(result.get("boxes", []), dtype=np.float32).reshape(-1, 4)
    scores = np.asarray(result.get("scores", []), dtype=np.float32).reshape(-1)
    mask_scores = np.asarray(result.get("mask_scores", []), dtype=np.float32).reshape(
        -1
    )
    masks = np.asarray(result.get("masks", []))
    labels = [str(value) for value in (result.get("labels") or [])]

    if masks.size == 0:
        raise RuntimeError("LangSAM did not return any masks")

    masks = masks.reshape((-1, masks.shape[-2], masks.shape[-1])).astype(bool)
    center = (image.width / 2.0, image.height / 2.0)
    center_x = max(0, min(image.width - 1, int(round(center[0]))))
    center_y = max(0, min(image.height - 1, int(round(center[1]))))
    indexes = list(range(len(masks)))
    mask_matches = [index for index in indexes if masks[index, center_y, center_x]]
    box_matches = [
        index
        for index in indexes
        if index < len(boxes)
        and boxes[index][0] <= center[0] <= boxes[index][2]
        and boxes[index][1] <= center[1] <= boxes[index][3]
    ]
    strategy = "highest_score"
    if mask_matches:
        indexes = mask_matches
        strategy = "center_inside_mask"
    elif box_matches:
        indexes = box_matches
        strategy = "center_inside_box"

    best_index = max(
        indexes,
        key=lambda index: float(scores[index]) if index < len(scores) else -1.0,
    )
    box = (
        tuple(float(value) for value in boxes[best_index])
        if best_index < len(boxes)
        else (0.0, 0.0, float(image.width), float(image.height))
    )
    return Prediction(
        mask=Image.fromarray(masks[best_index].astype(np.uint8) * 255, mode="L"),
        box=box,
        score=float(scores[best_index]) if best_index < len(scores) else None,
        mask_score=(
            float(mask_scores[best_index]) if best_index < len(mask_scores) else None
        ),
        label=labels[best_index] if best_index < len(labels) else "unknown",
        labels=labels,
        scores=[float(value) for value in scores.tolist()],
        boxes=[[float(value) for value in row] for row in boxes.tolist()],
        mask_scores=[float(value) for value in mask_scores.tolist()],
        selection_strategy=strategy,
    )


def evaluate_quality(mask: Image.Image, prediction: Prediction) -> dict[str, Any]:
    binary = mask.convert("L").point(lambda value: 255 if value else 0)
    histogram = binary.histogram()
    mask_pixels = sum(histogram[1:])
    image_pixels = binary.width * binary.height
    mask_percentage = mask_pixels / image_pixels * 100.0
    raw_box = binary.getbbox()
    reasons: list[str] = []

    if raw_box is None:
        foreground_box = None
        margins = None
        reasons.append("mask is empty")
    else:
        left, top, right, bottom = raw_box
        foreground_box = [left, top, right - 1, bottom - 1]
        margins = {
            "left": left,
            "top": top,
            "right": binary.width - right,
            "bottom": binary.height - bottom,
        }
        width_ratio = (right - left) / binary.width
        height_ratio = (bottom - top) / binary.height
        center_x = (left + right - 1) / 2.0
        center_y = (top + bottom - 1) / 2.0
        horizontal_offset = abs(center_x - binary.width / 2.0) / binary.width
        vertical_offset = abs(center_y - binary.height / 2.0) / binary.height
        if width_ratio < MIN_DIMENSION_RATIO:
            reasons.append("mask is too narrow")
        if height_ratio < MIN_DIMENSION_RATIO:
            reasons.append("mask is too short")
        if horizontal_offset > MAX_HORIZONTAL_CENTER_OFFSET_RATIO:
            reasons.append("mask is too far from the horizontal center")
        if vertical_offset > MAX_VERTICAL_CENTER_OFFSET_RATIO:
            reasons.append("mask is too far from the vertical center")

    if mask_percentage < MIN_MASK_PERCENTAGE:
        reasons.append(
            f"mask occupies {mask_percentage:.2f}% (minimum {MIN_MASK_PERCENTAGE:.2f}%)"
        )
    if mask_percentage > MAX_MASK_PERCENTAGE:
        reasons.append(
            f"mask occupies {mask_percentage:.2f}% (maximum {MAX_MASK_PERCENTAGE:.2f}%)"
        )

    touching_edges = []
    edge_pixels = {}
    crops = {
        "left": binary.crop((0, 0, 1, binary.height)),
        "top": binary.crop((0, 0, binary.width, 1)),
        "right": binary.crop((binary.width - 1, 0, binary.width, binary.height)),
        "bottom": binary.crop((0, binary.height - 1, binary.width, binary.height)),
    }
    for name, edge in crops.items():
        pixels = sum(edge.histogram()[1:])
        edge_pixels[name] = pixels
        if pixels:
            touching_edges.append(name)

    return {
        "accepted": not reasons,
        "reasons": reasons,
        "mask_percentage": round(mask_percentage, 2),
        "foreground_box": foreground_box,
        "margins_pixels": margins,
        "touching_edges": touching_edges,
        "edge_foreground_pixels": edge_pixels,
        "selected_box": [round(value, 3) for value in prediction.box],
    }


def create_overlay(
    image: Image.Image, mask: Image.Image, prediction: Prediction
) -> Image.Image:
    base = image.convert("RGBA")
    layer = Image.new("RGBA", image.size, (0, 255, 0, 0))
    layer.putalpha(mask.point(lambda value: int(value * 0.55)))
    overlay = Image.alpha_composite(base, layer).convert("RGB")
    draw = ImageDraw.Draw(overlay)
    draw.rectangle(prediction.box, outline=(255, 0, 0), width=3)
    score = f" {prediction.score:.3f}" if prediction.score is not None else ""
    draw.text(
        (prediction.box[0] + 4, prediction.box[1] + 4),
        f"{prediction.label}{score}",
        fill=(255, 0, 0),
    )
    return overlay


def write_outputs(
    image: Image.Image,
    prediction: Prediction,
    prompt: str,
    output_directory: Path,
    elapsed_seconds: float,
    device: torch.device,
    sam_type: str,
) -> dict[str, Any]:
    output_directory.mkdir(parents=True, exist_ok=True)
    mask = prediction.mask.resize(image.size, Image.Resampling.NEAREST)
    quality = evaluate_quality(mask, prediction)
    rgba = image.convert("RGBA")
    rgba.putalpha(mask)

    image.save(output_directory / "input.png", format="PNG")
    mask.save(output_directory / "mask.png", format="PNG")
    rgba.save(output_directory / "cutout.png", format="PNG")
    create_overlay(image, mask, prediction).save(
        output_directory / "overlay.jpg",
        format="JPEG",
        quality=95,
        subsampling=0,
        optimize=True,
    )

    metadata = {
        "prompt": prompt,
        "device": str(device),
        "sam_type": sam_type,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "selected_label": prediction.label,
        "selected_score": prediction.score,
        "selected_mask_score": prediction.mask_score,
        "selection_strategy": prediction.selection_strategy,
        "quality": quality,
        "detections": {
            "labels": prediction.labels,
            "scores": prediction.scores,
            "boxes": prediction.boxes,
            "mask_scores": prediction.mask_scores,
        },
        "assets": ["input.png", "mask.png", "cutout.png", "overlay.jpg"],
    }
    (output_directory / "result.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return metadata


def run(args: argparse.Namespace) -> dict[str, Any]:
    project_root = Path(__file__).resolve().parent
    prepare_cache(project_root)
    device = resolve_device(args.device)
    if not args.input.is_file():
        raise FileNotFoundError(f"Input image not found: {args.input}")

    from lang_sam import LangSAM

    image = Image.open(args.input).convert("RGB")
    run_directory = create_run_directory(
        args.output_root, "langsam_building_roof_comparison"
    )
    print(f"Loading {args.sam_type} on {device}...")
    model = LangSAM(sam_type=args.sam_type, device=device)

    comparison: dict[str, Any] = {
        "input": str(args.input.resolve()),
        "image_size": [image.width, image.height],
        "requested_device": args.device,
        "resolved_device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "sam_type": args.sam_type,
        "box_threshold": args.box_threshold,
        "text_threshold": args.text_threshold,
        "results": {},
    }
    for prompt in PROMPTS:
        name = prompt.rstrip(".")
        print(f"Running prompt {prompt!r}...")
        started = time.perf_counter()
        result = model.predict(
            [image],
            [prompt],
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
        )[0]
        prediction = select_prediction(result, image)
        elapsed = time.perf_counter() - started
        metadata = write_outputs(
            image=image,
            prediction=prediction,
            prompt=prompt,
            output_directory=run_directory / name,
            elapsed_seconds=elapsed,
            device=device,
            sam_type=args.sam_type,
        )
        comparison["results"][name] = {
            "accepted": metadata["quality"]["accepted"],
            "mask_percentage": metadata["quality"]["mask_percentage"],
            "score": metadata["selected_score"],
            "mask_score": metadata["selected_mask_score"],
            "elapsed_seconds": metadata["elapsed_seconds"],
            "directory": name,
        }

    comparison["run_directory"] = str(run_directory.resolve())
    (run_directory / "comparison.json").write_text(
        json.dumps(comparison, indent=2), encoding="utf-8"
    )
    print(f"Results written to {run_directory.resolve()}")
    return comparison


if __name__ == "__main__":
    run(parse_args())
