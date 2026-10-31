# AGENTS.md — Autonomous Engineering Agent Instructions for M.I.R.A.G.E.

## 1. Repository Overview
M.I.R.A.G.E. (Multimodal Intelligent Rendering & Automated Generation Engine) is a deterministic, headless synthetic perception dataset generation pipeline targeting single-workstation NVIDIA RTX 4060 (8 GB VRAM) hardware. It generates physically based multi-modal ground truth (RGB, Float32 Z-depth, surface normals, 16-bit instance segmentation masks, 2D modal/amodal bounding boxes, and 6-DoF poses) and exports directly to Ultralytics YOLO11 ([R.A.P.T.O.R.](https://github.com/madhavmadupu/raptor)), COCO JSON, and WebDataset/LeRobot ([P.R.I.M.E.](https://github.com/madhavmadupu/prime)).

## 2. Non-Negotiable Engineering Invariants
1. **Strict Determinism**: All random number generation in scene construction, Latin Hypercube Sampling (`src/mirage/sampler.py`), and physics perturbation must derive from an explicit `numpy.random.Generator(numpy.random.PCG64(master_seed))`. Never call unseeded global `random.*` or `np.random.*` functions.
2. **Coordinate Frame Exactness**: Always apply the basis change matrix $\mathbf{C}_{B\to C} = \text{diag}(1, -1, -1)$ when converting between Blender's camera coordinate frame ($+X$ right, $+Y$ up, $-Z$ forward) and the standard computer vision optical frame ($+X$ right, $+Y$ down, $+Z$ forward). Any modification to `src/mirage/projection.py` must maintain 0.0-pixel drift in `tests/test_projection.py`.
3. **VRAM Budget Enforcement (8 GB Ceiling)**: Worker processes executing headless Blender `bpy` must not exceed 7.2 GB of allocated GPU memory on an NVIDIA RTX 4060. Ensure persistent data (`use_persistent_data = False` between distinct scene geometries) and orphaned mesh/texture blocks (`bpy.ops.outliner.orphans_purge`) are cleaned up after each scene cycle.
4. **Zero Fake Telemetry**: Never hardcode fabricated benchmark numbers in documentation or code. All latency and throughput tables in documentation represent the **Design Budget / Target Envelope (RTX 4060 8GB)** and must be paired with `python scripts/benchmark.py` (`python scripts/benchmark_renderer.py`) for empirical measurement.
5. **Documentation Style**: Do not use emojis or promotional/marketing adjectives in any source file, docstring, commit message, or markdown document. Maintain rigorous, plain technical prose.

## 3. Standard Verification Commands
Before concluding any task or submitting changes, execute:
```bash
# Run mathematical projection, LHS stratification, and exporter tests
pytest tests/ -v

# Run benchmark dry-run verification
python scripts/benchmark.py --dry-run
```

## 4. Physical AI Suite Context
When updating cross-repository interfaces or export schemas, preserve compatibility with:
- `https://github.com/madhavmadupu/raptor` — R.A.P.T.O.R. (YOLO11 label & `data.yaml` consumer)
- `https://github.com/madhavmadupu/sonar` — S.O.N.A.R. (shared mesh inertial/frame conventions)
- `https://github.com/madhavmadupu/aegis` — A.E.G.I.S. (synthetic depth obstacle verification)
- `https://github.com/madhavmadupu/prime` — P.R.I.M.E. (WebDataset/LeRobot shard consumer)
- `https://github.com/madhavmadupu/mirage` — M.I.R.A.G.E. (this repository)
