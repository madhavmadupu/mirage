# ADR 0001: Headless Blender `bpy` 4.2 LTS with NVIDIA OptiX on 8 GB VRAM

- **Status**: Accepted
- **Date**: 2026-10-31
- **Author**: Madhav Madupu (`madhav.madupu@gmail.com`)

---

## Context and Problem Statement

M.I.R.A.G.E. requires a programmable, physically based 3D rendering and rigid-body physics runtime capable of generating synchronized multi-modal ground truth (RGB, Float32 metric depth, camera-space surface normals, 16-bit instance segmentation masks, and 6-DoF poses) in headless mode on a single workstation equipped with an **NVIDIA RTX 4060 (8 GB GDDR6 VRAM)**.

We evaluated three candidate rendering engines for this hardware envelope:
1. **NVIDIA Isaac Sim / Omniverse Replicator**: Provides high-fidelity RTX path tracing and PhysX 5 integration, but its base USD Kit runtime and RTX renderer allocate 7.5 GB to 11.5 GB of VRAM at startup, leading to frequent out-of-memory (OOM) crashes or severe PCIe page-thrashing on an 8 GB RTX 4060 card.
2. **Unity Perception / HDRP**: Supports synthetic annotation generation, but requires Editor/Player binary builds, C# shader compilation toolchains, and non-trivial headless Linux container licensing.
3. **Headless Blender 4.2 LTS (`bpy` Python Wheel) + Cycles OptiX / EEVEE Next**: Installs directly into a standard Python 3.11 virtual environment (`pip install bpy`), exposes full C-level scene graph and compositor APIs, includes built-in Bullet rigid-body physics, and runs Cycles hardware raytracing via NVIDIA OptiX within a 3.5 GB to 6.8 GB VRAM footprint.

## Decision

We select **Headless Blender 4.2 LTS (`bpy`)** with dual **Cycles OptiX** and **EEVEE Next** render backends, executed inside a recycled isolated worker process pool with an explicit 7.2 GB VRAM ceiling guard.

## Consequences

### Positive
- **8 GB VRAM Compatibility**: Cycles OptiX renders 640x640 scenes in ~4.5–5.4 GB VRAM and 1280x720 scenes in ~5.8–6.8 GB VRAM on an RTX 4060, leaving > 1.2 GB of safety headroom for host OS and display buffers.
- **Pure Python Deployment**: Installing `bpy>=4.2.0` via `pip` eliminates external GUI installation steps and enables reproducible CI/CD execution.
- **Single-Pass Multi-Layer Compositor**: Blender's `CompositorNodeRLayers` outputs denoised RGB alongside unfiltered Float32 OpenEXR depth, Float16 normals, and 16-bit integer `PassIndex` masks in a single BVH traversal.

### Negative & Mitigations
- **Process-Global C State in `bpy`**: `bpy` cannot run multiple independent scenes concurrently across Python threads in a single process. *Mitigation*: M.I.R.A.G.E. uses process-level isolation (`multiprocessing` / subprocess worker recycling every $M=25$ scenes) and purges orphaned data-blocks after every scene.
- **Coordinate System Difference**: Blender uses $+Y$ up, $-Z$ forward for cameras, whereas OpenCV/[R.A.P.T.O.R.](https://github.com/madhavmadupu/raptor)/[P.R.I.M.E.](https://github.com/madhavmadupu/prime) use $+Y$ down, $+Z$ forward. *Mitigation*: Enforced an explicit, unit-tested basis transformation $\mathbf{C}_{B\to C} = \text{diag}(1, -1, -1)$ in `src/mirage/projection.py` verified to $0.0\text{ px}$ projection drift.
