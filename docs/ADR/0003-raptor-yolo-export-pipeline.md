# ADR 0003: Multi-Format Export Pipeline for R.A.P.T.O.R. and P.R.I.M.E.

- **Status**: Accepted
- **Date**: 2026-10-31
- **Author**: Madhav Madupu (`madhav.madupu@gmail.com`)

---

## Context and Problem Statement

M.I.R.A.G.E. serves two distinct consumers within the Physical AI portfolio:
1. **[R.A.P.T.O.R.](https://github.com/madhavmadupu/raptor)** (Real-time Anomaly Perception & Tactical Operational Reconnaissance): Requires 2D tight bounding boxes and instance polygon contours in **Ultralytics YOLO11** directory format (`images/{train,val}`, `labels/{train,val}`, `data.yaml`) as well as **COCO JSON** for standardized `pycocotools` mAP@50-95 evaluation.
2. **[P.R.I.M.E.](https://github.com/madhavmadupu/prime)** (Physical Reasoning & Intelligent Motor Execution): Requires synchronized multi-modal tensors (RGB + Float32 metric depth + camera-space normals + 16-bit segmentation masks + $SE(3)$ 6-DoF object poses + camera intrinsics) packaged in sequential **WebDataset / LeRobot `.tar` shards** to avoid filesystem inode bottlenecks during high-throughput DataLoader streaming.

Historically, synthetic pipelines wrote custom intermediate formats requiring slow, error-prone post-processing conversion scripts that often introduced bounding-box coordinate rounding errors or train/validation data leakage across multi-view camera renders.

## Decision

We implement a unified, single-pass **Multi-Format Dataset Exporter** (`src/mirage/exporters/`) invoked directly after the Deterministic Quality Assurance Gate:
1. **Scene-Level Train/Val Split**: Train/validation partitioning is computed at the *scene index* level (`scene_index < int(num_scenes * train_split_ratio)`), never at the individual frame level. All $K$ camera views of a physical scene are assigned to the same split to prevent multi-view data leakage.
2. **Tight Modal Bounding Box Default for YOLO11**: YOLO11 labels use `bbox_2d_tight_xywh` extracted from the 16-bit instance mask buffer (rather than the projected 8-corner 3D bounding box) so that detectors are trained on pixel-tight visible object boundaries. Both modal (`bbox`) and amodal (`bbox_amodal`) boxes are preserved in the COCO JSON export.
3. **Cryptographic Lineage Binding**: The generated `yolo11/data.yaml`, `coco/instances_*.json`, and `webdataset/shard-*.tar` headers embed the `dataset_merkle_sha256` digest from `manifest.json`.

## Consequences

### Positive
- **Zero-Friction R.A.P.T.O.R. Fine-Tuning**: Output directories can be passed directly to Ultralytics YOLO11 (`yolo detect train data=.../yolo11/data.yaml`) with zero file restructuring.
- **Zero Multi-View Leakage**: Scene-level splitting guarantees that validation mAP reflects true generalization to unseen object arrangements and shader combinations.
- **Auditable Provenance**: Every model checkpoint trained in R.A.P.T.O.R. or P.R.I.M.E. can verify the exact synthetic dataset state via `manifest.json`.

### Negative & Mitigations
- **Disk Usage Across Multiple Formats**: Writing both unzipped YOLO11 images and WebDataset `.tar` shards duplicates RGB storage if both exporters are enabled simultaneously. *Mitigation*: CLI `--export-formats` flag allows users to select only the required target formats (`yolo11`, `coco`, `webdataset`) per run.
