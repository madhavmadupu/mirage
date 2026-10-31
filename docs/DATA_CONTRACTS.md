# M.I.R.A.G.E. — Data Contracts & Schemas (`docs/DATA_CONTRACTS.md`)

All data structures crossing module boundaries in M.I.R.A.G.E. are strictly typed using **Pydantic v2** (`extra="forbid"`, `frozen=True`) and implemented in [`src/mirage/contracts.py`](../src/mirage/contracts.py).

---

## 1. `DomainRandomizationSpec`

Defines the declarative configuration for a synthetic dataset generation campaign, including stratified Latin Hypercube Sampling (LHS) bounds and Quality Assurance thresholds.

```python
class DomainRandomizationSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_name: str = Field(..., min_length=1, max_length=128)
    master_seed: int = Field(default=42, ge=0, le=2**63 - 1)
    num_scenes: int = Field(default=100, ge=1, le=100000)
    views_per_scene: int = Field(default=4, ge=1, le=64)
    resolution_wh: tuple[int, int] = Field(default=(640, 640))
    render_backend: Literal["eevee_next", "cycles_optix", "software_raster"] = "cycles_optix"
    samples_per_pixel: int = Field(default=32, ge=1, le=4096)
    max_vram_gb: float = Field(default=7.2, gt=1.0, le=24.0)
    train_split_ratio: float = Field(default=0.85, gt=0.0, lt=1.0)

    # LHS Continuous Domain Randomization Intervals
    camera_radius_range_m: tuple[float, float] = (0.55, 1.35)
    camera_elevation_deg_range: tuple[float, float] = (20.0, 75.0)
    camera_azimuth_deg_range: tuple[float, float] = (0.0, 360.0)
    focal_length_mm_range: tuple[float, float] = (28.0, 50.0)
    hdri_exposure_ev_range: tuple[float, float] = (-1.5, 1.5)
    hdri_rotation_deg_range: tuple[float, float] = (0.0, 360.0)
    pbr_metallic_range: tuple[float, float] = (0.0, 0.95)
    pbr_roughness_range: tuple[float, float] = (0.05, 0.95)
    distractor_count_range: tuple[int, int] = (2, 10)

    # Quality Assurance Gate Thresholds
    min_visibility_ratio: float = Field(default=0.15, ge=0.0, le=1.0)
    max_truncation_ratio: float = Field(default=0.80, ge=0.0, le=1.0)
    min_modal_area_px: int = Field(default=16, ge=1)
    min_bbox_dim_px: float = Field(default=4.0, ge=1.0)
    categories: list[ObjectCategorySpec]
```

---

## 2. `CameraIntrinsics` & `CameraExtrinsics`

Encodes the pinhole camera model in the standard OpenCV optical frame $\mathcal{F}_C$ ($+X$ right, $+Y$ down, $+Z$ forward).

```python
class CameraIntrinsics(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    width: int = Field(..., ge=16, le=4096)
    height: int = Field(..., ge=16, le=4096)
    fx: float = Field(..., gt=0.0)
    fy: float = Field(..., gt=0.0)
    cx: float = Field(..., gt=0.0)
    cy: float = Field(..., gt=0.0)
    near_clip_m: float = Field(default=0.05, gt=0.0)
    far_clip_m: float = Field(default=25.0, gt=0.1)

    def to_matrix_3x3(self) -> list[list[float]]:
        return [
            [self.fx, 0.0, self.cx],
            [0.0, self.fy, self.cy],
            [0.0, 0.0, 1.0],
        ]
```

---

## 3. `Pose6DoF`

Represents the rigid-body transformation $\mathbf{T}_{C}^{O_k} \in SE(3)$ of an object $k$ relative to the camera optical frame $\mathcal{F}_C$.

```python
class Pose6DoF(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    instance_id: int = Field(..., ge=1, le=65535)
    class_id: int = Field(..., ge=0)
    class_name: str
    translation_xyz_m: tuple[float, float, float]
    quaternion_wxyz: tuple[float, float, float, float]
    rotation_matrix_3x3: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]
    local_bbox_min_xyz_m: tuple[float, float, float]
    local_bbox_max_xyz_m: tuple[float, float, float]
```

### Invariants Verified on Construction
- Unit quaternion norm: $\left| w^2 + x^2 + y^2 + z^2 - 1.0 \right| < 10^{-5}$.
- Proper rotation matrix: $\mathbf{R}^\top \mathbf{R} = \mathbf{I}_3$ within $10^{-5}$ and $\det(\mathbf{R}) = +1$.
- Positive optical depth: $t_z > 0$ for any object in front of the camera principal plane.

---

## 4. `InstanceAnnotation`

Stores complete 2D and 3D ground truth for a single visible object instance within a rendered frame.

```python
class InstanceAnnotation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    instance_id: int = Field(..., ge=1, le=65535)
    class_id: int = Field(..., ge=0)
    class_name: str
    bbox_2d_tight_xywh: tuple[float, float, float, float]
    bbox_2d_amodal_xywh: tuple[float, float, float, float]
    bbox_2d_proj_xywh: tuple[float, float, float, float]
    bbox_3d_corners_uv: list[tuple[float, float]]  # 8 projected OBB corners in pixels
    modal_pixel_area: int = Field(..., ge=0)
    amodal_inframe_pixel_area: int = Field(..., ge=1)
    visibility_ratio: float = Field(..., ge=0.0, le=1.0)
    occlusion_ratio: float = Field(..., ge=0.0, le=1.0)
    truncation_ratio: float = Field(..., ge=0.0, le=1.0)
    projection_drift_px: float = Field(..., ge=0.0)
    pose_cam: Pose6DoF
    polygon_uv: list[tuple[float, float]] = Field(default_factory=list)
```

---

## 5. `DatasetManifest` & `FrameManifestEntry`

Provides cryptographic provenance and SHA-256 verification across all generated files.

```python
class FrameManifestEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    frame_id: str
    scene_index: int
    view_index: int
    split: Literal["train", "val"]
    rgb_relpath: str
    depth_relpath: str
    normals_relpath: str
    mask_relpath: str
    rgb_sha256: str
    depth_sha256: str
    normals_sha256: str
    mask_sha256: str
    frame_merkle_sha256: str
    num_valid_instances: int
    intrinsics: CameraIntrinsics
    annotations: list[InstanceAnnotation]


class DatasetManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "1.0.0"
    dataset_name: str
    created_utc: str
    master_seed: int
    spec_sha256: str
    total_scenes: int
    total_frames: int
    rejected_frames: int
    dataset_merkle_sha256: str
    frames: list[FrameManifestEntry]
```

---

## 6. Downstream Export Schemas

### 6.1 Ultralytics YOLO11 Format (for [R.A.P.T.O.R.](https://github.com/madhavmadupu/raptor))
- **Directory Layout**:
  ```text
  yolo11/
  ├── data.yaml
  ├── images/
  │   ├── train/{frame_id}.png
  │   └── val/{frame_id}.png
  └── labels/
      ├── train/{frame_id}.txt
      └── val/{frame_id}.txt
  ```
- **Label Line Schema** (`labels/{split}/{frame_id}.txt`):
  ```text
  <class_id> <cx_norm> <cy_norm> <w_norm> <h_norm>
  ```
  where:
  $$\text{cx}_{\text{norm}} = \frac{u_{\min}^{\text{tight}} + w_{\text{tight}} / 2}{W}, \quad \text{cy}_{\text{norm}} = \frac{v_{\min}^{\text{tight}} + h_{\text{tight}} / 2}{H}, \quad w_{\text{norm}} = \frac{w_{\text{tight}}}{W}, \quad h_{\text{norm}} = \frac{h_{\text{tight}}}{H}$$
  All normalized coordinates are strictly clamped and verified in $[0.0, 1.0]$ with 6 decimal places of precision.

### 6.2 COCO Instance Format (`coco/instances_{split}.json`)
Conforms to the standard MS-COCO object detection and instance segmentation JSON schema with an additional `"mirage_6dof"` dictionary on each annotation entry:
```json
{
  "id": 1001,
  "image_id": 42,
  "category_id": 1,
  "bbox": [184.0, 210.0, 96.0, 112.0],
  "bbox_amodal": [180.0, 206.0, 104.0, 118.0],
  "area": 8420,
  "iscrowd": 0,
  "segmentation": [[184.0, 210.0, 280.0, 210.0, 280.0, 322.0, 184.0, 322.0]],
  "visibility_ratio": 0.912,
  "truncation_ratio": 0.0,
  "mirage_6dof": {
    "translation_xyz_m": [0.042, -0.018, 0.785],
    "quaternion_wxyz": [0.92388, 0.38268, 0.0, 0.0]
  }
}
```

### 6.3 WebDataset / LeRobot Shard Format (for [P.R.I.M.E.](https://github.com/madhavmadupu/prime))
- Packaged as uncompressed POSIX tar archives (`webdataset/shard-{000000..000099}.tar`) where consecutive members share a common key `{frame_id}`:
  - `{frame_id}.rgb.png`
  - `{frame_id}.depth.npy` (Float32 metric depth array $(H, W)$)
  - `{frame_id}.normals.npy` (Float16 camera-space normal array $(H, W, 3)$)
  - `{frame_id}.mask.png` (16-bit instance mask)
  - `{frame_id}.json` (intrinsics, 6-DoF object poses, and visibility metadata)
