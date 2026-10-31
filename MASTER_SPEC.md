# M.I.R.A.G.E. — Master Technical Specification (`MASTER_SPEC.md`)

**System**: Multimodal Intelligent Rendering & Automated Generation Engine (M.I.R.A.G.E.)  
**Version**: 1.0.0  
**Author**: Madhav Madupu (`madhav.madupu@gmail.com`)  
**Target Hardware**: Single-Workstation NVIDIA RTX 4060 (8 GB VRAM, AD107 Architecture)  

---

## 1. Coordinate Frames and SE(3) Conventions

M.I.R.A.G.E. enforces explicit coordinate frame transformations between the Blender internal scene representation and standard computer vision camera conventions (OpenCV / COLMAP / ROS `optical_frame` convention):

1. **World Frame ($\mathcal{F}_W$)**: Right-handed Cartesian frame where $+X$ is East, $+Y$ is North, and $+Z$ is Up (gravity vector $\mathbf{g} = [0, 0, -9.80665]^\top\,\text{m/s}^2$).
2. **Blender Camera Frame ($\mathcal{F}_{B}$)**: Right-handed frame where $+X$ is right, $+Y$ is up, and $-Z$ points along the optical axis into the scene.
3. **Computer Vision Optical Frame ($\mathcal{F}_{C}$)**: Right-handed pinhole camera frame where $+X$ points right along image columns $u$, $+Y$ points down along image rows $v$, and $+Z$ points forward along the optical axis into the scene.

The fixed basis change matrix $\mathbf{C}_{B\to C} \in SO(3)$ mapping coordinates from the Blender camera frame $\mathcal{F}_B$ to the OpenCV optical frame $\mathcal{F}_C$ is:

$$\mathbf{C}_{B\to C} = \begin{bmatrix} 1 & 0 & 0 \\ 0 & -1 & 0 \\ 0 & 0 & -1 \end{bmatrix}, \quad \det(\mathbf{C}_{B\to C}) = +1$$

Given the Blender camera's world transformation matrix $\mathbf{T}_{W}^{B} \in SE(3)$ (`camera.matrix_world`), the world-to-camera extrinsic matrix $\mathbf{E} = [\mathbf{R} \mid \mathbf{t}] \in \mathbb{R}^{3\times 4}$ in standard computer vision convention is computed as:

$$\mathbf{T}_{C}^{W} = \begin{bmatrix} \mathbf{C}_{B\to C} & \mathbf{0} \\ \mathbf{0}^\top & 1 \end{bmatrix} \left(\mathbf{T}_{W}^{B}\right)^{-1} = \begin{bmatrix} \mathbf{R} & \mathbf{t} \\ \mathbf{0}^\top & 1 \end{bmatrix}$$

For any rigid object $k$ with world pose $\mathbf{T}_{W}^{O_k} \in SE(3)$, its 6-DoF pose in the camera optical frame $\mathcal{F}_C$ is:

$$\mathbf{T}_{C}^{O_k} = \mathbf{T}_{C}^{W} \mathbf{T}_{W}^{O_k} = \begin{bmatrix} \mathbf{R}_{C}^{O_k} & \mathbf{t}_{C}^{O_k} \\ \mathbf{0}^\top & 1 \end{bmatrix}$$

---

## 2. Pinhole Camera Projection Mathematics

Let a 3D point in the world frame be denoted in homogeneous coordinates as $\tilde{\mathbf{P}}_{\text{world}} = [X_W, Y_W, Z_W, 1]^\top \in \mathbb{P}^3$. Its inhomogeneous coordinates in the camera optical frame $\mathcal{F}_C$ are:

$$\mathbf{P}_{\text{cam}} = \begin{bmatrix} X_C \\ Y_C \\ Z_C \end{bmatrix} = [\mathbf{R} \mid \mathbf{t}] \tilde{\mathbf{P}}_{\text{world}}$$

Given image width $W \in \mathbb{Z}^+$, image height $H \in \mathbb{Z}^+$, horizontal sensor width $s_x$ (mm), vertical sensor height $s_y$ (mm), and focal length $f$ (mm) configured in Blender (`Camera.lens`, `Camera.sensor_width`, `Camera.sensor_height` with `sensor_fit = 'HORIZONTAL'`), the pixel focal lengths $(f_x, f_y)$ and principal point $(c_x, c_y)$ with lens shift $(\Delta_x, \Delta_y)$ are:

$$f_x = \frac{f \cdot W}{s_x}, \quad f_y = \frac{f \cdot H}{s_y} \cdot \text{pixel\_aspect\_ratio}$$

$$c_x = \frac{W}{2} - \Delta_x \cdot W, \quad c_y = \frac{H}{2} + \Delta_y \cdot W$$

The upper-triangular camera intrinsic matrix $\mathbf{K} \in \mathbb{R}^{3\times 3}$ is:

$$\mathbf{K} = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}$$

For any point in front of the principal plane ($Z_C > z_{\text{near}} > 0$), the homogeneous pixel projection is:

$$Z_C \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = \mathbf{K} [\mathbf{R} \mid \mathbf{t}] \tilde{\mathbf{P}}_{\text{world}} \implies \mathbf{p}_{\text{uv}} = \begin{bmatrix} u \\ v \end{bmatrix} = \begin{bmatrix} f_x \frac{X_C}{Z_C} + c_x \\ f_y \frac{Y_C}{Z_C} + c_y \end{bmatrix}$$

### 2.1 3D Oriented Bounding Box (OBB) to 2D Bounding Box Projection

Each mesh asset $k$ has a local-frame axis-aligned bounding box defined by its minimum and maximum local vertex extents $\mathbf{b}_{\min}^{(k)}, \mathbf{b}_{\max}^{(k)} \in \mathbb{R}^3$, yielding 8 local corner vertices $\mathcal{V}_{\text{local}}^{(k)} = \{\mathbf{v}_1, \dots, \mathbf{v}_8\}$.

Transforming each vertex via $\mathbf{T}_{C}^{O_k}$ and projecting via $\mathbf{K}$ produces 8 image-plane coordinates $\{\mathbf{p}_{\text{uv}, j}^{(k)}\}_{j=1}^8$. The projected 3D bounding box Clipped to image dimensions $[0, W] \times [0, H]$ is:

$$u_{\min}^{\text{proj}} = \max\left(0, \min_{j=1..8} u_j^{(k)}\right), \quad u_{\max}^{\text{proj}} = \min\left(W, \max_{j=1..8} u_j^{(k)}\right)$$

$$v_{\min}^{\text{proj}} = \max\left(0, \min_{j=1..8} v_j^{(k)}\right), \quad v_{\max}^{\text{proj}} = \min\left(H, \max_{j=1..8} v_j^{(k)}\right)$$

### 2.2 Tight Instance Mask Bounding Box vs. Projected 3D Box

Because projecting the 8 corners of a 3D bounding box overestimates the tight 2D pixel silhouette of non-cuboid meshes or partially occluded objects, M.I.R.A.G.E. extracts both:
1. `bbox_2d_proj`: $[u_{\min}^{\text{proj}}, v_{\min}^{\text{proj}}, u_{\max}^{\text{proj}} - u_{\min}^{\text{proj}}, v_{\max}^{\text{proj}} - v_{\min}^{\text{proj}}]$ from the 3D OBB projection.
2. `bbox_2d_tight` (Modal Bounding Box): Derived directly from the 16-bit instance segmentation buffer $\mathbf{I}_{\text{inst}} \in \mathbb{Z}_{\ge 0}^{H\times W}$ where pixel $(r, c)$ equals the unique object `pass_index` $k$:

$$\Omega_k = \left\{ (r, c) \in \{0, \dots, H-1\} \times \{0, \dots, W-1\} \;\middle|\; \mathbf{I}_{\text{inst}}[r, c] = k \right\}$$

If $|\Omega_k| > 0$, the exact pixel-tight modal bounding box is:

$$u_{\min}^{\text{tight}} = \min_{(r,c) \in \Omega_k} c, \quad v_{\min}^{\text{tight}} = \min_{(r,c) \in \Omega_k} r, \quad u_{\max}^{\text{tight}} = \max_{(r,c) \in \Omega_k} c + 1, \quad v_{\max}^{\text{tight}} = \max_{(r,c) \in \Omega_k} r + 1$$

---

## 3. Visibility, Occlusion, and Truncation Formulation

Synthetic datasets often degrade downstream detector precision when heavily occluded or image-border-truncated object fragments are labeled as positive full-confidence training instances. M.I.R.A.G.E. computes three analytical visibility metrics for every object $k$ in every camera view:

1. **Modal Visible Pixel Area ($A_{\text{modal}}^{(k)}$)**:
   $$A_{\text{modal}}^{(k)} = |\Omega_k| = \sum_{r=0}^{H-1} \sum_{c=0}^{W-1} \mathbb{I}\left(\mathbf{I}_{\text{inst}}[r, c] == k\right)$$

2. **Amodal Unoccluded Pixel Area ($A_{\text{amodal}}^{(k)}$)**:
   Computed by rasterizing the mesh of object $k$ alone in camera frame $\mathcal{F}_C$ onto an unclipped virtual viewport (or within the $[0, W) \times [0, H)$ frustum for pure occlusion vs. truncation separation). Let $\Omega_k^{\text{unocc, in-frame}}$ denote the set of pixels inside $[0, W) \times [0, H)$ covered by object $k$ when rendered in isolation without other scene geometry, and let $\Omega_k^{\text{unocc, full}}$ denote the total pixel count on an unbounded canvas.

3. **Visibility Ratio ($V_k \in [0, 1]$)** and **Occlusion Ratio ($O_k \in [0, 1]$)**:
   $$V_k = \frac{A_{\text{modal}}^{(k)}}{\max\left(1, \left|\Omega_k^{\text{unocc, in-frame}}\right|\right)}, \quad O_k = 1 - V_k$$

4. **Frustum Truncation Ratio ($T_k \in [0, 1]$)**:
   $$T_k = 1 - \frac{\left|\Omega_k^{\text{unocc, in-frame}}\right|}{\max\left(1, \left|\Omega_k^{\text{unocc, full}}\right|\right)}$$

**QA Filtering Rule**: An instance annotation $k$ is included in exported training labels if and only if:
- $A_{\text{modal}}^{(k)} \ge A_{\min}$ (default: $16\text{ pixels}^2$),
- $V_k \ge V_{\min}$ (default: $0.15$, i.e., at most $85\%$ occluded),
- $T_k \le T_{\max}$ (default: $0.80$, i.e., at least $20\%$ inside the camera frustum),
- $\min(w_{\text{tight}}, h_{\text{tight}}) \ge 4\text{ pixels}$.

---

## 4. Stratified Latin Hypercube Sampling (LHS) for Domain Randomization

Independent uniform random sampling ($\theta_d \sim \mathcal{U}(a_d, b_d)$) in a $D$-dimensional domain randomization space exhibits $O(N^{-1/2})$ discrepancy and leaves large void regions and clusters when generating moderate dataset sizes ($N \in [250, 5000]$ scenes).

M.I.R.A.G.E. implements **Stratified Latin Hypercube Sampling (LHS)** across $D$ continuous randomization dimensions (camera azimuth $\phi$, camera elevation $\theta$, camera radius $\rho$, HDRI yaw $\psi_{\text{hdri}}$, key light log-intensity, surface roughness $\alpha$, metallic factor $m$, and distractor count):

Given $N$ scenes and $D$ parameters:
1. Partition each parameter's unit interval $[0, 1]$ into $N$ equal strata: $I_i = \left[\frac{i-1}{N}, \frac{i}{N}\right]$ for $i \in \{1, \dots, N\}$.
2. For each dimension $d \in \{1, \dots, D\}$, generate an independent random permutation $\pi_d: \{1, \dots, N\} \to \{1, \dots, N\}$ seeded deterministically from the master 64-bit seed $S_0$.
3. Draw uniform jitter $u_{i, d} \sim \mathcal{U}(0, 1)$ for each stratum $(i, d)$ and compute the normalized sample coordinate:
   $$\hat{x}_{i, d} = \frac{\pi_d(i) - 1 + u_{i, d}}{N} \in [0, 1]$$
4. Map $\hat{x}_{i, d}$ through the inverse cumulative distribution function $F_d^{-1}(\hat{x}_{i, d})$ of parameter $d$'s target distribution (e.g., continuous uniform $\mathcal{U}(a_d, b_d)$, log-uniform for light exposure, or truncated normal for camera focal length jitter).

By construction, every one-dimensional projection of the $N$ samples contains exactly one sample per stratum $\left[\frac{i-1}{N}, \frac{i}{N}\right]$, reducing variance of Monte Carlo estimators by $O(N^{-1})$ on additive functions compared to crude Monte Carlo.

---

## 5. Metric Depth and Surface Normal Verification Invariants

### 5.1 Metric Z-Depth Buffer ($\mathbf{D} \in \mathbb{R}^{H\times W}$)
- Extracted from the Blender Render Layer `Depth` socket in Float32 OpenEXR format.
- Represents the orthogonal distance $Z_C$ (in meters) along the camera optical axis, **not** Euclidean ray length $\|\mathbf{P}_{\text{cam}}\|_2$ (Cycles ray-length pass is converted via $Z_C = d_{\text{ray}} \cos\theta_{\text{ray}}$ when required).
- Background/sky pixels where no geometry is intersected are clamped from Blender's sentinel value ($10^{10}$) to `0.0` (or user-configured `max_depth_m`) and accompanied by a binary validity mask $\mathbf{M}_{\text{depth}} = (\mathbf{D} > z_{\text{near}}) \land (\mathbf{D} < z_{\text{far}})$.

### 5.2 Camera-Space Surface Normals ($\mathbf{N}_C \in [-1, 1]^{H\times W\times 3}$)
- World-space geometric/shading normals $\mathbf{n}_W(r, c)$ extracted from the `Normal` render pass are rotated into the camera optical frame $\mathcal{F}_C$:
  $$\mathbf{n}_C(r, c) = \frac{\mathbf{R} \, \mathbf{n}_W(r, c)}{\left\|\mathbf{R} \, \mathbf{n}_W(r, c)\right\|_2}$$
- Valid foreground pixels must satisfy two mathematical invariants in the QA Gate:
  1. **Unit norm invariant**: $\left| \|\mathbf{n}_C(r, c)\|_2 - 1.0 \right| < 10^{-3}$.
  2. **Front-facing visibility invariant**: Because visible surfaces face toward the camera origin in $\mathcal{F}_C$ (where $+Z$ points away from the camera), the dot product with the viewing ray $\mathbf{v}_C(r, c)$ must satisfy $\mathbf{n}_C(r, c) \cdot \mathbf{v}_C(r, c) \le \epsilon_{\text{grazing}}$.

---

## 6. Dataset Provenance and Cryptographic Manifest Verification

Every generated dataset directory contains a root `manifest.json` file that binds the input configuration, software environment, PRNG seed, and output file hashes into an immutable cryptographic audit trail:

1. For every rendered frame $i \in \{0, \dots, M-1\}$, M.I.R.A.G.E. computes the SHA-256 digest of each modality file on disk (`rgb/{frame_id}.png`, `depth/{frame_id}.exr`, `normals/{frame_id}.exr`, `masks/{frame_id}.png`, `labels/{frame_id}.txt`).
2. The frame digest $H_i$ is computed as:
   $$H_i = \text{SHA256}\left( \text{concat}\left( h_{\text{rgb}, i} \parallel h_{\text{depth}, i} \parallel h_{\text{norm}, i} \parallel h_{\text{mask}, i} \parallel h_{\text{label}, i} \right) \right)$$
3. The root dataset Merkle/manifest digest $H_{\text{dataset}}$ is computed over the sorted concatenation of all frame digests plus the canonicalized JSON representation of `DomainRandomizationSpec`:
   $$H_{\text{dataset}} = \text{SHA256}\left( \text{SHA256}(\text{canonical\_json}(\text{spec})) \parallel H_0 \parallel H_1 \parallel \dots \parallel H_{M-1} \right)$$

Downstream training pipelines in [R.A.P.T.O.R.](https://github.com/madhavmadupu/raptor) and [P.R.I.M.E.](https://github.com/madhavmadupu/prime) record $H_{\text{dataset}}$ in their model checkpoint metadata to guarantee bit-level lineage between synthetic training shards and deployed perception weights.
