# M.I.R.A.G.E. — Benchmark & Verification Methodology (`docs/BENCHMARK_METHODOLOGY.md`)

This document specifies the reproducible methodology for evaluating M.I.R.A.G.E. across three axes:
1. **Rendering Throughput & VRAM Telemetry** on an NVIDIA RTX 4060 (8 GB VRAM).
2. **Zero-Pixel 3D-to-2D Annotation Projection Accuracy**.
3. **Downstream Perception Transfer Gain (mAP@50-95)** when fine-tuning [R.A.P.T.O.R.](https://github.com/madhavmadupu/raptor) Tier 1 YOLO11 detectors on M.I.R.A.G.E. synthetic data.

---

## 1. Rendering Throughput and VRAM Profiling

### 1.1 Execution Command
To measure empirical frame generation latency (p50, p95, p99), sustained frames-per-second (FPS), and peak GPU memory consumption on your host machine, execute:

```bash
python scripts/benchmark_renderer.py \
  --profiles eevee_fast_640,cycles_optix_640,cycles_optix_720p \
  --num-frames 100 \
  --warmup-frames 10 \
  --seed 42 \
  --output reports/benchmark_results.json
```

*(Or via the unified benchmark entrypoint: `python scripts/benchmark.py`)*

### 1.2 Measurement Protocol
- **Warmup Exclusion**: The first `warmup_frames` (default: 10) are excluded from latency percentiles to separate one-time OptiX BVH kernel compilation and shader cache initialization from steady-state rendering throughput.
- **Stage-Decomposed Timing**: Every frame is instrumented using `time.perf_counter_ns()` across four distinct sub-stages:
  1. $t_{\text{scene}}$: LHS parameter sampling, procedural mesh placement, PBR shader graph mutation, and Bullet rigid-body physics step amortized per view.
  2. $t_{\text{render}}$: GPU rendering execution (`bpy.ops.render.render(write_still=True)`) and OptiX AI denoising.
  3. $t_{\text{qa}}$: 3D-to-2D pinhole projection, 16-bit instance mask tight bounding box extraction, modal/amodal visibility ratio calculation, and `NaN` depth buffer verification.
  4. $t_{\text{export}}$: Serialization to YOLO11, COCO JSON, WebDataset tar shards, and SHA-256 manifest hashing.
- **VRAM Telemetry**: Sampled at peak scene complexity via NVML (`pynvml`) or Blender's render stats hook.

### 1.3 Design Budget / Target Envelope (NVIDIA RTX 4060 8GB)

| Profile ID | Resolution | Backend | $t_{\text{render}}$ Budget (p50) | $t_{\text{render}}$ Budget (p95) | $t_{\text{qa}} + t_{\text{export}}$ Budget (p95) | Peak VRAM Ceiling |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `eevee_fast_640` | 640 x 640 | EEVEE Next (16 spp) | <= 85.0 ms | <= 120.0 ms | <= 18.0 ms | <= 3.8 GB |
| `cycles_optix_640` | 640 x 640 | Cycles OptiX (32 spp) | <= 240.0 ms | <= 310.0 ms | <= 18.0 ms | <= 5.4 GB |
| `cycles_optix_720p` | 1280 x 720 | Cycles OptiX (64 spp) | <= 580.0 ms | <= 750.0 ms | <= 25.0 ms | <= 6.8 GB |

---

## 2. Zero-Pixel Annotation Projection Verification

A common failure mode in synthetic data pipelines is mismatch between the camera intrinsic/extrinsic matrices used by the 3D renderer and the mathematical projection matrices used by the bounding-box exporter (due to sensor fit mode discrepancies, principal point off-by-half-pixel shifts, or Blender-to-OpenCV axis inversions).

### 2.1 Verification Algorithm
For every unoccluded object $k$ ($V_k = 1.0$, $T_k = 0.0$) with known local mesh vertices $\mathcal{V}_k = \{\mathbf{v}_1, \dots, \mathbf{v}_M\}$:
1. Transform all mesh vertices into the OpenCV camera frame $\mathcal{F}_C$ via $\mathbf{T}_C^{O_k}$ and project via $\mathbf{K}$:
   $$\mathbf{p}_{\text{uv}, m} = \mathbf{K} \mathbf{T}_C^{O_k} \tilde{\mathbf{v}}_m, \quad m \in \{1, \dots, M\}$$
2. Compute the analytical vertex-hull bounding box in continuous pixel coordinates and quantize to integer pixel bounds:
   $$\hat{u}_{\min} = \lfloor \min_m u_m \rfloor, \quad \hat{v}_{\min} = \lfloor \min_m v_m \rfloor, \quad \hat{u}_{\max} = \lceil \max_m u_m \rceil, \quad \hat{v}_{\max} = \lceil \max_m v_m \rceil$$
3. Extract the empirical rasterized bounding box $(u_{\min}^{\text{tight}}, v_{\min}^{\text{tight}}, u_{\max}^{\text{tight}}, v_{\max}^{\text{tight}})$ directly from the non-zero pixels of object $k$ in the 16-bit instance mask buffer $\mathbf{I}_{\text{inst}}$.
4. Compute the $L_\infty$ boundary drift:
   $$\Delta_{\text{drift}}^{(k)} = \max\left( |\hat{u}_{\min} - u_{\min}^{\text{tight}}|, |\hat{v}_{\min} - v_{\min}^{\text{tight}}|, |\hat{u}_{\max} - u_{\max}^{\text{tight}}|, |\hat{v}_{\max} - v_{\max}^{\text{tight}}| \right)$$

### 2.2 Acceptance Criterion
- On exact vertex rasterization (`tests/test_projection.py`), $\Delta_{\text{drift}}^{(k)}$ must equal **$0.0\text{ pixels}$** across random camera poses, focal lengths, and aspect ratios.
- On 8-corner 3D oriented bounding box projections (`bbox_2d_proj` vs `bbox_2d_tight`), the modal tight box must be strictly contained within the projected 3D OBB ($\hat{u}_{\min}^{\text{OBB}} \le u_{\min}^{\text{tight}}$ and $u_{\max}^{\text{tight}} \le \hat{u}_{\max}^{\text{OBB}}$) with zero out-of-bounds leakage.

---

## 3. Downstream Sim-to-Real / Perception Transfer Evaluation (R.A.P.T.O.R.)

To evaluate the effectiveness of M.I.R.A.G.E. synthetic data on downstream robotic anomaly and obstacle detection in [R.A.P.T.O.R.](https://github.com/madhavmadupu/raptor):

### 3.1 Experimental Protocol
1. **Baseline Model**: Ultralytics `yolo11s.pt` trained exclusively on a limited real-world seed dataset $\mathcal{D}_{\text{real-train}}$ (e.g., 200 real annotated industrial frames) and evaluated on a held-out real test split $\mathcal{D}_{\text{real-test}}$ (500 real frames across specular glare, heavy occlusion, and low-light conditions).
2. **LHS Synthetic Augmentation**: Generate $\mathcal{D}_{\text{mirage}}$ (2,000 frames across 500 LHS scenes using `configs/industrial_anomaly.yaml`).
3. **Mixed Training**: Fine-tune `yolo11s.pt` on $\mathcal{D}_{\text{mirage}} \cup \mathcal{D}_{\text{real-train}}$ for 50 epochs at $640 \times 640$ resolution, followed by 10 epochs of real-domain fine-tuning on $\mathcal{D}_{\text{real-train}}$.
4. **Ablation Comparisons**:
   - **Uniform Random vs. Stratified LHS**: Compare downstream mAP@50-95 on $\mathcal{D}_{\text{real-test}}$ when $\mathcal{D}_{\text{mirage}}$ is sampled with independent uniform sampling vs. Stratified Latin Hypercube Sampling.
   - **With vs. Without QA Occlusion Filtering**: Compare mAP@50-95 and False Positive Rate when training with unfiltered bounding boxes ($V_{\min} = 0.0$) vs. M.I.R.A.G.E. QA-filtered tight modal boxes ($V_{\min} = 0.15$, $T_{\max} = 0.80$).
