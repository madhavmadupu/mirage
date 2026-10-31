# Contributing to M.I.R.A.G.E. (`CONTRIBUTING.md`)

Thank you for contributing to **M.I.R.A.G.E. (Multimodal Intelligent Rendering & Automated Generation Engine)**. This document defines the technical standards, mathematical verification gates, and workflow conventions required for all pull requests.

---

## 1. Engineering Standards

1. **Deterministic Execution**:
   - Every stochastic operation in scene generation, camera sampling, lighting perturbation, and clutter placement must use an explicit `numpy.random.Generator` initialized with `numpy.random.PCG64(seed)`.
   - Given identical `DomainRandomizationSpec` and `--seed`, two runs must produce identical sampled scene parameters and identical projected 2D/3D ground-truth coordinates.
2. **Coordinate System Rigor**:
   - All functions manipulating 3D poses or projections must explicitly document in their docstring whether input/output matrices reside in the World Frame ($\mathcal{F}_W$), Blender Camera Frame ($\mathcal{F}_B$), or OpenCV Optical Frame ($\mathcal{F}_C$).
   - Any change to `src/mirage/projection.py` must pass the zero-pixel projection drift verification suite in `tests/test_projection.py`.
3. **VRAM Budget Compliance (NVIDIA RTX 4060 8 GB)**:
   - Any change to `src/mirage/compositor.py` or shader generation must maintain peak VRAM usage below 7.2 GB for the `cycles_optix_720p` profile.
   - All dynamically allocated Blender meshes, materials, images, and node trees must be purged between scene iterations.
4. **Documentation & Prose Style**:
   - Do not use emojis anywhere in code, comments, commit messages, or documentation.
   - Avoid promotional or marketing language. State algorithmic complexity, mathematical assumptions, and hardware constraints in plain technical prose.

---

## 2. Local Development Setup

```bash
git clone https://github.com/madhavmadupu/mirage.git
cd mirage
python -m venv .venv
# Activate virtual environment:
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# Linux / macOS: source .venv/bin/activate

pip install --upgrade pip
pip install numpy scipy pydantic pyyaml pillow pytest
# Optional for full GPU rendering (requires Python 3.11):
pip install bpy>=4.2.0 OpenEXR>=3.2.0
```

---

## 3. Pre-Submission Verification Checklist

Run the following verification commands locally before opening a pull request:

```bash
# 1. Execute unit tests (projection accuracy, LHS stratification, YOLO/COCO export schemas)
pytest tests/ -v

# 2. Verify dataset generation pipeline end-to-end
python scripts/generate_dataset.py \
  --config configs/industrial_anomaly.yaml \
  --num-scenes 5 \
  --views-per-scene 2 \
  --output-dir ./datasets/pr_check

# 3. Verify cryptographic manifest integrity
python scripts/generate_dataset.py \
  --verify-manifest ./datasets/pr_check/manifest.json \
  --check-projection-drift
```

---

## 4. Pull Request & Commit Conventions

Follow Conventional Commits:
- `feat(projection): add radial-tangential Brown-Conrady distortion model`
- `fix(compositor): purge orphaned EXR image buffers after scene bake`
- `docs(spec): clarify amodal vs in-frame visibility ratio denominator`
- `test(sampler): verify Kolmogorov-Smirnov uniformity on 1D LHS margins`

## License Agreement

By submitting a contribution to this repository, you agree that your contribution will be licensed under the [Apache License, Version 2.0](LICENSE).
