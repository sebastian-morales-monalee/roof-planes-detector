from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv
from PIL import Image

import experiment_roof_planes_google_solar_dsm as solar_dsm
import experiment_roof_planes_structural as structural
import experiment_roof_planes_vector_optimizer as vector_optimizer
import experiment_roof_planes_vlm_optimizer as vlm_optimizer
from experiment_roof_planes import make_contact_sheet
from run_artifacts import create_run_directory


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs"
EXPERIMENT_NAME = "roof_planes_end_to_end"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the effective roof-plane pipeline end to end: LangSAM and local "
            "structural evidence, multi-hypothesis VLM optimization, vector "
            "regularization, and Google Solar metric DSM refinement."
        )
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--location",
        required=True,
        help="Google Solar location in '<latitude>, <longitude>' format.",
    )
    parser.add_argument(
        "--manual-reference",
        type=Path,
        help=(
            "Optional paint-over included only in the final qualitative comparison. "
            "It is never supplied to a model, optimizer, metric, or selector."
        ),
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--prompt", default="building.")
    parser.add_argument("--sam-type", default="sam2.1_hiera_small")
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument(
        "--reasoning-effort",
        choices=("medium", "high", "xhigh", "max"),
        default="high",
    )
    parser.add_argument("--hypotheses", type=int, choices=(2, 3), default=2)
    parser.add_argument(
        "--source-candidate",
        choices=("conservative", "balanced", "structural"),
        default="structural",
    )
    return parser.parse_args()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def release_accelerator_cache() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def run_stage(
    manifest: dict[str, Any],
    manifest_path: Path,
    stage_id: str,
    function: Callable[[argparse.Namespace], Path],
    arguments: argparse.Namespace,
) -> Path:
    record = {
        "id": stage_id,
        "status": "processing",
        "started_at_unix": time.time(),
        "output_directory": None,
        "elapsed_seconds": None,
        "error": None,
    }
    manifest["stages"].append(record)
    manifest["current_stage"] = stage_id
    write_json(manifest_path, manifest)
    started = time.perf_counter()
    print(f"\n[{stage_id}] starting")
    try:
        output = function(arguments).resolve()
    except Exception as error:
        record["status"] = "failed"
        record["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        record["error"] = f"{type(error).__name__}: {error}"
        manifest["status"] = "failed"
        manifest["current_stage"] = None
        manifest["error"] = record["error"]
        write_json(manifest_path, manifest)
        raise
    record["status"] = "completed"
    record["output_directory"] = str(output)
    record["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    write_json(manifest_path, manifest)
    print(f"[{stage_id}] completed in {record['elapsed_seconds']:.1f}s")
    release_accelerator_cache()
    return output


def copy_final_artifacts(source: Path, destination: Path) -> list[str]:
    destination.mkdir(parents=True, exist_ok=False)
    names = (
        "final_roof_planes_labels.png",
        "final_roof_planes_color.png",
        "final_roof_planes_overlay.png",
        "final_roof_planes_georeferenced.geojson",
        "google_solar_dsm_registered.tif",
        "google_solar_dsm_registered.png",
        "google_solar_rgb_registered.png",
        "metric_plane_fits.json",
        "dsm_refined_vector_graph.json",
        "dsm_refined_vector_graph_overlay.png",
        "registration_comparison.jpg",
        "result.json",
    )
    copied: list[str] = []
    for name in names:
        source_path = source / name
        if not source_path.is_file():
            raise FileNotFoundError(f"Expected final artifact was not created: {source_path}")
        target_name = "experiment_07_result.json" if name == "result.json" else name
        shutil.copy2(source_path, destination / target_name)
        copied.append(target_name)
    return copied


def create_final_comparison(
    input_path: Path,
    structural_run: Path,
    vector_run: Path,
    solar_run: Path,
    manual_reference: Path | None,
    output_path: Path,
) -> bool:
    items = [
        ("Original input", Image.open(input_path).convert("RGB")),
        (
            "Initial structural candidates",
            Image.open(structural_run / "structural_planes_overlay.png").convert("RGB"),
        ),
        (
            "Human-selected vector structural",
            Image.open(
                vector_run / "candidate_structural" / "roof_planes_overlay.png"
            ).convert("RGB"),
        ),
        (
            "Google Solar metric DSM registered",
            Image.open(solar_run / "google_solar_dsm_registered.png").convert("RGB"),
        ),
        (
            "Final DSM-guided roof planes",
            Image.open(solar_run / "final_roof_planes_overlay.png").convert("RGB"),
        ),
    ]
    reference_available = manual_reference is not None
    if manual_reference is not None:
        items.append(
            (
                "Manual paint-over | visual reference only",
                Image.open(manual_reference).convert("RGB"),
            )
        )
    make_contact_sheet(items, output_path)
    return reference_available


def validate_inputs(args: argparse.Namespace) -> tuple[Path, Path | None]:
    input_path = args.input.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input image was not found: {input_path}")
    with Image.open(input_path) as image:
        image.verify()
    manual_reference = None
    if args.manual_reference is not None:
        manual_reference = args.manual_reference.expanduser().resolve()
        if not manual_reference.is_file():
            raise FileNotFoundError(
                f"Manual reference was not found: {manual_reference}"
            )
        with Image.open(manual_reference) as image:
            image.verify()
    solar_dsm.parse_location(args.location)
    load_dotenv(PROJECT_ROOT / ".env")
    missing = [
        name
        for name in ("OPENAI_API_KEY", "GOOGLE_MAPS_API_KEY")
        if not os.getenv(name, "").strip()
    ]
    if missing:
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")
    return input_path, manual_reference


def run(args: argparse.Namespace) -> Path:
    input_path, manual_reference = validate_inputs(args)
    output_root = args.output_root.expanduser().resolve()
    run_directory = create_run_directory(output_root, EXPERIMENT_NAME)
    stages_root = run_directory / "stages"
    manifest_path = run_directory / "pipeline_result.json"
    started = time.perf_counter()
    manifest: dict[str, Any] = {
        "pipeline": EXPERIMENT_NAME,
        "status": "processing",
        "current_stage": None,
        "input": str(input_path),
        "location": args.location,
        "device": args.device,
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "hypotheses": args.hypotheses,
        "source_candidate": args.source_candidate,
        "manual_reference": str(manual_reference) if manual_reference else None,
        "manual_reference_computational_role": False,
        "stages": [],
        "final_directory": None,
        "final_artifacts": [],
        "elapsed_seconds": None,
        "error": None,
    }
    write_json(manifest_path, manifest)
    shutil.copy2(input_path, run_directory / "input.png")

    disabled_reference = run_directory / "manual_reference_disabled"
    structural_run = run_stage(
        manifest,
        manifest_path,
        "01_structural_evidence",
        structural.run,
        argparse.Namespace(
            input=input_path,
            reference=disabled_reference,
            output_root=stages_root,
            device=args.device,
            prompt=args.prompt,
            sam_type=args.sam_type,
            box_threshold=0.25,
            text_threshold=0.20,
            depth_model=structural.DEFAULT_DEPTH_MODEL,
            superpixels=180,
            compactness=10.0,
            cluster_threshold=7.0,
            min_plane_percentage=2.0,
            barrier_threshold=0.47,
            minimum_line_length=26.0,
            line_support_threshold=0.12,
            sam_proposal_min_iou=0.76,
            sam_proposal_min_stability=0.86,
            disable_sam_refinement=False,
        ),
    )
    optimizer_run = run_stage(
        manifest,
        manifest_path,
        "02_vlm_multi_hypothesis",
        vlm_optimizer.run,
        argparse.Namespace(
            source_run=structural_run,
            output_root=stages_root,
            model=args.model,
            detail="original",
            reasoning_effort=args.reasoning_effort,
            hypotheses=args.hypotheses,
            device=args.device,
            sam_type=args.sam_type,
            snap_margin=42,
            reference_planes=14,
            merge_coplanar=True,
        ),
    )
    vector_run = run_stage(
        manifest,
        manifest_path,
        "03_vector_optimizer",
        vector_optimizer.run,
        argparse.Namespace(
            source_run=optimizer_run,
            output_root=stages_root,
            snap_margin=42,
        ),
    )
    solar_run = run_stage(
        manifest,
        manifest_path,
        "04_google_solar_dsm",
        solar_dsm.run,
        argparse.Namespace(
            location=args.location,
            source_run=vector_run,
            source_candidate=args.source_candidate,
            output_root=stages_root,
            minimum_registration_iou=0.42,
            registration_angle_step=2.0,
            plane_residual_threshold_meters=0.18,
            max_fit_samples=8000,
            max_vector_shift_pixels=10.0,
            maximum_boundary_complexity_ratio=1.08,
            manual_reference=None,
        ),
    )

    final_directory = run_directory / "final"
    copied = copy_final_artifacts(solar_run, final_directory)
    reference_available = create_final_comparison(
        input_path,
        structural_run,
        vector_run,
        solar_run,
        manual_reference,
        final_directory / "qualitative_comparison.jpg",
    )
    copied.append("qualitative_comparison.jpg")
    if manual_reference is not None:
        shutil.copy2(
            manual_reference, final_directory / "manual_reference_visual_only.png"
        )
        copied.append("manual_reference_visual_only.png")

    solar_result = json.loads((solar_run / "result.json").read_text(encoding="utf-8"))
    manifest.update(
        {
            "status": "completed",
            "current_stage": None,
            "final_directory": str(final_directory),
            "final_artifacts": copied,
            "selected_result": solar_result["selected_result"],
            "final_plane_count": solar_result["final_plane_count"],
            "registration_iou": solar_result["registration"]["iou"],
            "manual_reference_available": reference_available,
            "manual_reference_used_for_registration": False,
            "manual_reference_used_for_optimization": False,
            "manual_reference_used_for_scoring": False,
            "manual_reference_used_for_selection": False,
            "manual_reference_visual_comparison_only": True,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        }
    )
    write_json(manifest_path, manifest)
    print(f"\nPipeline completed: {run_directory}")
    print(f"Final artifacts: {final_directory}")
    return run_directory


if __name__ == "__main__":
    run(parse_args())
