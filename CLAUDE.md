# CLAUDE.md — Claude Code Development Guide for M.I.R.A.G.E.

## Quick Commands
- **Run test suite**: `pytest tests/ -v`
- **Generate synthetic dataset**: `python scripts/generate_dataset.py --config configs/industrial_anomaly.yaml --num-scenes 10 --output-dir ./datasets/dev_check`
- **Run renderer benchmark**: `python scripts/benchmark_renderer.py --profiles eevee_fast_640,cycles_optix_640 --num-frames 20`
- **Run pipeline benchmark**: `python scripts/benchmark.py --profiles eevee_fast_640 --num-frames 20`

## Architectural Modules
- `src/mirage/contracts.py`: Pydantic v2 data contracts (`DomainRandomizationSpec`, `CameraIntrinsics`, `Pose6DoF`, `InstanceAnnotation`, `DatasetManifest`).
- `src/mirage/sampler.py`: Stratified Latin Hypercube Sampling (LHS) over continuous scene randomization intervals using `PCG64`.
- `src/mirage/projection.py`: $SE(3)$ transformations, $\mathbf{C}_{B\to C} = \text{diag}(1, -1, -1)$ frame conversion, $\mathbf{p}_{\text{uv}} = \mathbf{K}[\mathbf{R}\mid\mathbf{t}]\tilde{\mathbf{P}}_{\text{world}}$ pinhole projection, and modal/amodal visibility ratio calculation.
- `src/mirage/compositor.py`: Programmatic Blender `bpy` compositor graph builder for synchronized RGB, Float32 EXR depth, Float16 EXR surface normals, and 16-bit instance ID masks.
- `src/mirage/qa_gate.py`: Rejects frames with `NaN`/`Inf` depth buffers, degenerate bounding boxes, or excessive occlusion, and computes SHA-256 provenance manifests.
- `src/mirage/exporters/`: Format-specific dataset writers for Ultralytics YOLO11 (`yolo.py`), COCO JSON (`coco.py`), and WebDataset/LeRobot (`webdataset.py`).

## Critical Rules
- **No emojis** anywhere in code, comments, or documentation.
- **No marketing language**: Write concise, objective systems engineering prose.
- **Blender `bpy` Optional Fallback**: Unit tests in `tests/` must be runnable both inside a full `bpy` environment and in lightweight CI environments via deterministic software rasterizer fallbacks in `compositor.py`.
- **Memory Hygiene**: On RTX 4060 (8 GB VRAM), always purge orphaned Blender data-blocks (`meshes`, `materials`, `images`, `worlds`) between scene generations to avoid CUDA out-of-memory faults.
