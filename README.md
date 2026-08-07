# LangSAM Roof Segmentation Experiments

Roof-segmentation experiments that combine local vision models with an optional
external multimodal analysis stage. Local model weights are downloaded once and
cached locally.

The project currently provides:

1. A LangSAM comparison using the prompts `building.` and `roof.`.
2. An RGB-only roof-plane candidate experiment using LangSAM, Depth Anything
   V2, surface normals, superpixels, and constrained clustering.
3. A structural roof-plane experiment that adds straight-line barriers, SAM2
   interior proposals, and guarded point/box refinement.
4. A hybrid VLM experiment that asks an external multimodal model to interpret
   numbered local candidates, then applies its merge and split plan locally.
5. An unconstrained topology experiment that asks a frontier multimodal model
   to draw the complete roof graph from scratch, critiques that proposal in a
   second pass, and rasterizes the revised topology against local evidence.
6. A multi-hypothesis optimizer that refines each proposed plane with positive
   and negative SAM2 points, fits robust relative-depth planes, applies guarded
   coplanar merging, and selects the strongest globally consistent topology.
7. A vector-graph optimizer that simplifies shared boundaries, aligns supported
   segments to local structural lines, consolidates junctions, and preserves the
   previous winning hypothesis as a control candidate.

## Requirements

- Ubuntu/WSL with Python 3.10 or newer
- Internet access for the initial dependency and model downloads
- Optional NVIDIA GPU with a compatible driver

The tested machine uses an NVIDIA GeForce RTX 5060 Laptop GPU with 8 GB VRAM.

## Install

Create the local environment file before running experiments that use OpenAI
or Google Solar:

```bash
cp .env.example .env
```

Then replace the placeholders in `.env` with valid credentials. The real
`.env` file is ignored by Git and must never be committed.

GPU environment:

```bash
chmod +x setup_env.sh
./setup_env.sh gpu
```

CPU-only environment:

```bash
./setup_env.sh cpu
```

The setup script creates an isolated `.venv`. Running it again with a different
mode replaces the installed PyTorch build inside that environment.

## Run

Automatically use CUDA when available and otherwise use CPU:

```bash
.venv/bin/python segment_image.py --device auto
```

Force GPU:

```bash
.venv/bin/python segment_image.py --device cuda
```

Force CPU:

```bash
.venv/bin/python segment_image.py --device cpu
```

Custom paths:

```bash
.venv/bin/python segment_image.py \
  --input start_input/image.png \
  --output-root outputs \
  --device auto
```

## Roof-plane experiment

Run the first roof-plane experiment with automatic GPU selection:

```bash
.venv/bin/python experiment_roof_planes.py --device auto
```

Force CUDA:

```bash
.venv/bin/python experiment_roof_planes.py --device cuda
```

The first run downloads `Depth-Anything-V2-Small-hf`. Later runs reuse the
model under `.cache/huggingface/`. This experiment estimates relative depth;
it does not produce metric, surveyed, or georeferenced roof geometry.

Useful tuning controls:

```bash
.venv/bin/python experiment_roof_planes.py \
  --device cuda \
  --superpixels 180 \
  --cluster-threshold 1.35 \
  --min-plane-percentage 0.65
```

## Structural roof-plane experiment

Run the second experiment after the baseline experiment:

```bash
.venv/bin/python experiment_roof_planes_structural.py --device cuda
```

This experiment keeps the LangSAM exterior building mask and relative depth
from the previous approach, then detects RGB, depth, and normal discontinuities.
Straight lines and SAM2 automatic-mask boundaries become barriers during
superpixel clustering. Each structural candidate is optionally passed through
SAM2 using a centroid point and bounding box; a refinement is accepted only
when overlap, area, and boundary-quality checks pass and conflict resolution
actually changes the final plane ownership.

To inspect only the structural candidates without point/box refinement:

```bash
.venv/bin/python experiment_roof_planes_structural.py \
  --device cuda \
  --disable-sam-refinement
```

The structural and refined masks are both preserved in the output so their
effect can be compared rather than hidden. `refinement_delta_mask.png` contains
only changed pixels, while `refinement_difference.png` highlights effective
changes in yellow with crimson boundaries. `result.json` records the changed
pixel count and percentage of the roof.

## VLM-guided roof-plane experiment

Experiment 03 builds on a completed structural run. It sends the original
image, a numbered candidate overlay, and the local structural-line diagnostic
to an OpenAI vision-capable model. The model returns strict JSON describing
which candidates should be merged, which candidates require a structural
split, and which regions remain uncertain. Raster operations, clipping, split
application, and artifact generation remain local.

OpenAI experiments read the following variable from the local `.env` file:

```dotenv
OPENAI_API_KEY=your_api_key
```

Run against the latest valid experiment 02 output:

```bash
.venv/bin/python experiment_roof_planes_vlm.py
```

Run against a specific structural output or model:

```bash
.venv/bin/python experiment_roof_planes_vlm.py \
  --source-run outputs/<experiment-02-directory> \
  --model gpt-5-mini
```

This is the first experiment that calls an external API. The request summary
never contains the API key. Outputs include the numbered candidate map, parsed
VLM response, guidance overlay, indexed hybrid mask, colored result, GeoJSON,
validation details, token usage, and a qualitative comparison sheet.

## Complete VLM topology experiment

Experiment 04 does not send the numbered experiment-02 candidates to the model.
Instead, it sends the original image, exterior building mask, structural-boundary
diagnostic, relative depth, and surface normals. The model proposes all roof
planes, interior seeds, outlines, junctions, structural boundaries, and
adjacencies from scratch. A second model pass audits and may fully redraw that
topology. The manual reference is never included in either API request.

Run the quality-first configuration against the latest structural output:

```bash
.venv/bin/python experiment_roof_planes_vlm_topology.py
```

Run against a specific structural output or disable the second model pass:

```bash
.venv/bin/python experiment_roof_planes_vlm_topology.py \
  --source-run outputs/<experiment-02-directory> \
  --review-passes 0
```

The default uses `gpt-5.6-sol`, original image detail, high reasoning effort, and
an independent review pass. The more expensive maximum-effort pro configuration
remains available explicitly:

```bash
.venv/bin/python experiment_roof_planes_vlm_topology.py \
  --reasoning-effort max \
  --pro
```

If pro mode is unavailable to the API project, the script retries that request
with the same model and effort in standard mode. Local processing then snaps the
proposed graph toward RGB, depth, normal, and structural evidence; a
marker-controlled watershed creates a complete, non-overlapping partition of the
LangSAM building mask. Outputs preserve the draft, reviewed topology, raw
polygons, raw and snapped graphs, final indexed mask, GeoJSON, validation report,
token usage, and qualitative comparison.

## Multi-hypothesis topology optimizer

Experiment 05 generates two independent complete roof-topology hypotheses by
default. Each hypothesis is first snapped to local RGB, depth, normal, and
structural evidence. SAM2 then receives one positive seed per proposed roof
plane and negative seeds from adjacent planes. The accepted masks become cores
for a global watershed, so every building pixel has exactly one owner and the
final regions remain closed and non-overlapping.

Each region is also fitted to the relative-depth field using iterative robust
least squares. Very strongly supported near-coplanar neighbors may be merged;
large regions with poor planar residuals are explicitly flagged as possible
missing splits. The initial VLM partition, SAM2-refined partition, and guarded
coplanar-merge result are scored separately, so SAM2 is retained only when it
improves global quality. A score combining coverage, topology consistency,
boundary support, depth planarity, connectivity, and complexity selects first
the best variant and then the winning hypothesis. The manual reference is not
sent to the model and is not used to choose the winner.

Before partitioning, disconnected LangSAM mask fragments are measured and the
largest connected building component is retained. This prevents isolated
one-pixel segmentation noise from being counted as an uncovered roof or being
assigned artificially to a distant plane.

Run the default two-hypothesis experiment:

```bash
.venv/bin/python experiment_roof_planes_vlm_optimizer.py --device cuda
```

Generate three independent hypotheses or disable guarded coplanar merging:

```bash
.venv/bin/python experiment_roof_planes_vlm_optimizer.py \
  --device cuda \
  --hypotheses 3 \
  --no-merge-coplanar
```

The current manual example is a translucent paint-over rather than an indexed
ground-truth mask. Experiment 05 therefore derives an explicitly named
`manual_reference_indexed_approx` diagnostic through color-displacement
clustering. Hungarian IoU matching and boundary F1 are reported separately and
never influence hypothesis selection. For authoritative benchmark metrics,
replace this approximation in a future experiment with manually indexed plane
polygons or labels.

## Vector roof-graph optimizer

Experiment 06 starts from the winning Experiment 05 hypothesis and performs a
fully local geometric optimization. It detects straight structural evidence,
creates conservative, balanced, and structural vector variants, simplifies
shared boundaries, aligns supported segments, consolidates nearby endpoints
into common junctions, and rasterizes every candidate as a complete watershed
partition. The unchanged Experiment 05 winner remains a control candidate, so
the script does not silently accept a geometric edit that scores worse.

Run it against the latest completed Experiment 05 output:

```bash
.venv/bin/python experiment_roof_planes_vector_optimizer.py
```

Or select an Experiment 05 run explicitly:

```bash
.venv/bin/python experiment_roof_planes_vector_optimizer.py \
  --source-run outputs/<experiment-05-directory>
```

The manual paint-over has no computational role in Experiment 06. It is loaded
only after the winning candidate has already been selected and is appended to
`qualitative_comparison.jpg` as a visual reference. It is never converted into
labels and is not used for optimization, scoring, metrics, or winner selection.

## Google Solar DSM-guided vector refinement

Experiment 07 treats the manually selected `vector structural` result from
Experiment 06 as the initial roof topology and adds independent metric geometry
from Google Solar. It calls Building Insights and Data Layers, downloads the
RGB, building mask, and DSM GeoTIFFs, registers the LangSAM building footprint
without allowing reflections, and fits one robust metric plane per roof region.
It then moves only eligible straight boundaries toward the mathematical
intersection of their adjacent DSM planes and consolidates shared junctions.

The default location belongs to the current sample image. A different input must
provide its own location explicitly:

```bash
.venv/bin/python experiment_roof_planes_google_solar_dsm.py
```

```bash
.venv/bin/python experiment_roof_planes_google_solar_dsm.py \
  --location "<latitude>, <longitude>" \
  --source-run outputs/<experiment-06-directory>
```

`GOOGLE_MAPS_API_KEY` must exist in `.env` and have Google Solar API access.
Every execution saves the downloaded Solar layers, registration diagnostics,
the DSM reprojected into the source-image grid, per-plane pitch/aspect and fit
residuals, a georeferenced GeoJSON, and a qualitative comparison. If mask
registration is below the configured IoU threshold, metric fit does not improve,
plane labels are lost, or boundary complexity increases beyond the configured
limit, the selected output remains the original vector-structural candidate.
The manual paint-over remains visual-only and has no role in registration,
optimization, scoring, or selection.

## Google Solar DSM hypothesis fusion

Experiment 08 uses the registered Google Solar DSM as an independent source of
roof-plane hypotheses instead of limiting it to small movements of boundaries
inherited from the visual solution. It derives metric slope, surface normals,
normal discontinuities, curvature, and elevation-step evidence from the DSM;
creates a DSM-only partition; and builds a second candidate by intersecting and
merging that partition with the Experiment 06 vector-structural topology.

Three candidates are evaluated independently: the unchanged vector-structural
baseline, the DSM-only hypothesis, and the visual-plus-DSM fusion. Selection is
based on metric plane residuals, planarity, support from DSM boundaries,
building coverage, fitted coverage, geometric simplicity, and small-region
penalties. This allows DSM evidence to create, split, merge, or reject roof
planes rather than merely shifting existing lines.

Run it against the latest completed Experiment 07 output:

```bash
.venv/bin/python experiment_roof_planes_dsm_hypothesis_fusion.py
```

Or select the source explicitly and include the manual reference for visual
comparison only:

```bash
.venv/bin/python experiment_roof_planes_dsm_hypothesis_fusion.py \
  --source-run outputs/<experiment-07-directory> \
  --manual-reference start_example_roof_planes/image_with_roof_planes.png
```

The output includes indexed masks, overlays, metric plane fits, candidate
scores, a georeferenced GeoJSON for every hypothesis, and explicit winner
artifacts. The manual paint-over is loaded only after scoring and winner
selection; it never influences segmentation, fusion, metrics, or ranking.

## Visual/DSM fusion with a structural prior

Experiment 09 keeps the Experiment 08 `visual_dsm_fusion` result as the primary
topology and uses the earlier `vector_structural_baseline` only as a secondary
source of clean lines and junction placement. Baseline boundaries are eligible
only when they are close to the fusion boundary or independently supported by
the Google Solar DSM. Seeded watershed produces conservative, balanced, and
structural snapping variants without changing the primary set of roof regions;
an additional candidate applies cautious planar-neighbor regularization.

```bash
.venv/bin/python experiment_roof_planes_fusion_structural_prior.py
```

```bash
.venv/bin/python experiment_roof_planes_fusion_structural_prior.py \
  --source-run outputs/<experiment-08-directory> \
  --manual-reference start_example_roof_planes/image_with_roof_planes.png
```

Winner selection remains dominated by independent metric evidence: 88% of the
score comes from DSM/visual plane quality and 12% from measured boundary
straightness. Similarity with the old baseline is recorded for diagnostics but
is not rewarded. The manual paint-over remains visual-only.

## Inverse structural fusion

Experiment 10 reverses the hierarchy tested in Experiment 09. The 21-plane
`vector_structural_baseline` becomes the primary topology and the 32-plane
`visual_dsm_fusion` becomes a secondary source of candidate boundaries. Strict
variants preserve the original baseline region count while moving boundaries
only toward fusion lines that are nearby or supported by the Google Solar DSM.

Two enrichment variants additionally allow the secondary topology to split a
baseline region. A split is accepted only when all proposed subplanes can be
fit, their shared boundary has sufficient DSM evidence, and separate metric
planes reduce the elevation residual by the configured absolute or relative
threshold. This tests whether the baseline's cleaner structure can be retained
without permanently discarding additional DSM-supported roof planes.

```bash
.venv/bin/python experiment_roof_planes_inverse_structural_fusion.py
```

```bash
.venv/bin/python experiment_roof_planes_inverse_structural_fusion.py \
  --source-run outputs/<experiment-09-directory> \
  --manual-reference start_example_roof_planes/image_with_roof_planes.png
```

As in Experiments 08 and 09, the manual paint-over is appended only after the
winner has been selected and never participates in generation or scoring.

## End-to-end runner

`run_roof_planes_pipeline.py` executes the effective accumulated pipeline from
a new source image in one command. It runs structural LangSAM/SAM2 processing,
multi-hypothesis VLM topology optimization, vector regularization, and Google
Solar DSM refinement. Earlier experiments 01, 03, and 04 remain available for
research traceability but are superseded in this effective chain.

```bash
.venv/bin/python run_roof_planes_pipeline.py \
  --input path/to/image.png \
  --location "<latitude>, <longitude>" \
  --device auto
```

An optional manual paint-over can be appended to the final comparison:

```bash
.venv/bin/python run_roof_planes_pipeline.py \
  --input path/to/image.png \
  --location "<latitude>, <longitude>" \
  --manual-reference path/to/manual_reference.png \
  --device auto
```

The manual reference is withheld from every computational stage. It is copied
only after the final result has been selected and appears exclusively in
`final/qualitative_comparison.jpg`. Both `OPENAI_API_KEY` and
`GOOGLE_MAPS_API_KEY` must be configured in `.env`.

Each invocation creates one timestamped `roof_planes_end_to_end` directory:

```text
outputs/<timestamp>_roof_planes_end_to_end/
|-- input.png
|-- pipeline_result.json
|-- stages/
|   |-- <timestamp>_roof_planes_experiment_02_structural/
|   |-- <timestamp>_roof_planes_experiment_05_vlm_optimizer/
|   |-- <timestamp>_roof_planes_experiment_06_vector_optimizer/
|   `-- <timestamp>_roof_planes_experiment_07_google_solar_dsm/
`-- final/
    |-- final_roof_planes_labels.png
    |-- final_roof_planes_overlay.png
    |-- final_roof_planes_georeferenced.geojson
    |-- google_solar_dsm_registered.tif
    |-- metric_plane_fits.json
    `-- qualitative_comparison.jpg
```

`pipeline_result.json` records stage progress, elapsed time, source directories,
the selected result, registration IoU, and the final artifact list. If a stage
fails, the manifest preserves the failed stage and error for diagnosis.

## Outputs

Every execution creates a unique UTC-timestamped directory:

```text
outputs/
|-- 20260729T120000_000000Z_langsam_building_roof_comparison/
|-- 20260729T121500_000000Z_roof_planes_experiment_01/
|-- 20260729T123000_000000Z_roof_planes_experiment_02_structural/
|-- 20260729T124500_000000Z_roof_planes_experiment_03_vlm/
|-- 20260729T130000_000000Z_roof_planes_experiment_04_vlm_topology/
|-- 20260729T140000_000000Z_roof_planes_experiment_05_vlm_optimizer/
|-- 20260729T150000_000000Z_roof_planes_experiment_06_vector_optimizer/
|-- 20260729T160000_000000Z_roof_planes_experiment_07_google_solar_dsm/
|-- 20260729T170000_000000Z_roof_planes_experiment_08_dsm_hypothesis_fusion/
|-- 20260729T180000_000000Z_roof_planes_experiment_09_fusion_structural_prior/
`-- 20260729T190000_000000Z_roof_planes_experiment_10_inverse_structural_fusion/
```

The roof-plane experiment writes the LangSAM mask, relative depth, estimated
surface normals, edge diagnostics, superpixels, an indexed plane-label mask,
a colored overlay, pixel-coordinate GeoJSON, JSON metadata, and visual contact
sheets. The manual example is copied into the run only for qualitative review;
it is not currently a machine-readable ground-truth label mask.

`cutout.png` preserves only the selected mask and makes all other pixels fully
transparent. `result.json` records detection scores, mask coverage, quality
checks, execution time, and the selected device.

Models downloaded by LangSAM and Depth Anything are cached under `.cache/` so
subsequent runs do not need to download them again.
