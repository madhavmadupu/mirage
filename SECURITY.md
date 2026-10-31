# Security Policy (`SECURITY.md`)

## Supported Versions

| Version | Supported | Target Runtime |
| :--- | :--- | :--- |
| `1.0.x` | Yes | Python 3.11+, Blender `bpy` 4.2 LTS, NVIDIA Driver >= 550.54 |
| `< 1.0.0` | No | Deprecated pre-release prototypes |

## Threat Model & Security Boundaries for Synthetic Rendering Pipelines

Synthetic dataset engines ingest external 3D asset files (`.usd`, `.glb`, `.obj`, `.blend`), HDRI environment textures (`.exr`, `.hdr`), and YAML scene configurations, and execute GPU shaders and compositor pipelines. M.I.R.A.G.E. enforces four security controls across this surface:

### 1. Untrusted `.blend` Auto-Execution Prevention
Blender `.blend` files can embed Python driver scripts that execute automatically upon file load if auto-execution is enabled. M.I.R.A.G.E. strictly disables embedded script execution in all headless `bpy` workers (`bpy.context.preferences.filepaths.use_scripts_auto_execute = False`) and restricts external asset ingestion to declarative geometry formats (`.glb`/`.gltf`, `.obj`, `.stl`, and USD geometry layers without executable Python hooks).

### 2. YAML Configuration Deserialization Safety
All declarative scene graphs and `DomainRandomizationSpec` configurations are parsed exclusively via `yaml.safe_load()` followed by strict Pydantic v2 schema validation (`extra="forbid"`). Arbitrary Python object instantiation tags (`!!python/object/apply`) are rejected at parse time.

### 3. Path Traversal and Asset Registry Containment
All asset paths referenced inside a scene configuration are canonicalized via `pathlib.Path.resolve()` and verified to reside within the configured `--asset-root` directory before loading. Symlinks pointing outside the asset root or relative path segments (`..`) are rejected with a `SecurityViolationError`.

### 4. OpenEXR Buffer Bounds and Resource Exhaustion Protection
To prevent denial-of-service or GPU/host heap exhaustion on 8 GB VRAM workstations:
- Maximum render resolution is bounded by `DomainRandomizationSpec` validation (`width <= 4096`, `height <= 4096`).
- Maximum physics rigid-body object count per scene is capped (`max_objects_per_scene <= 256`).
- Every generated dataset writes a cryptographic `manifest.json` containing SHA-256 hashes of all output files so downstream consumers ([R.A.P.T.O.R.](https://github.com/madhavmadupu/raptor) and [P.R.I.M.E.](https://github.com/madhavmadupu/prime)) can detect dataset tampering or bit-rot prior to model training.

## Reporting a Vulnerability

To report a security vulnerability in M.I.R.A.G.E. or any repository in the Physical AI portfolio:
- **Email**: `madhav.madupu@gmail.com`
- **Subject Line**: `[SECURITY][MIRAGE] Brief vulnerability description`
- Please include a minimal reproducible configuration file, stack trace, and affected version commit hash. Do not open public GitHub issues for unpatched security vulnerabilities. You will receive an acknowledgment within 48 hours.
