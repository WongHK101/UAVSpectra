# Evaluation

`tools/measured_geometry_v1/` contains the exporters and scoring implementations.
GCP and LiDAR use expected camera-z `M1/A` and the shared control-only transform.
Checkpoints are not used to fit that transform.

## GCP and LiDAR

From the repository root, use `evaluate.py`; `--scene` is `road` or `five_k`.
For LiDAR, supply an export-list JSON:

```json
[
  {"method": "umgs", "root": "/absolute/path/to/export", "manifest_sha256": "SHA256_OF_EXPORT_MANIFEST"}
]
```

```bash
python evaluate.py --metric lidar --scene five_k --data ../UAVSpectra-dataset \
  --exports /path/to/exports.json --output /path/to/new_lidar_scores
```

## Appearance

Run the following from `evaluation/` for the existing interfaces:

```bash
export PYTHONPATH="$(pwd)/../baseline:${PYTHONPATH:-}"
python -m tools.measured_geometry_v1.score_native_appearance --help
```

Appearance requires the rectified inputs, method renders, PyTorch/torchvision
and LPIPS weights. Pass `third_party/appearance` as the runtime root.
The prepared inputs are in `processed/<scene>/` after extraction.

## Proxy Geometry

```bash
python evaluate.py --metric proxy --scene road --data ../UAVSpectra-dataset \
  --predictions /path/to/predictions.json --output /path/to/new_proxy_scores
```

Supply all test targets and all methods in one predeclared comparison set:

```json
{
  "schema": "uavspectra_proxy_predictions_v1",
  "scene": "road",
  "methods": ["method_a", "method_b"],
  "formula": "M1/A",
  "depth_semantics": "camera_z",
  "depth_units": "source_common_SfM_reconstruction_units_not_metres",
  "alignment": "none",
  "depth_resize_used": false,
  "targets": [{
    "image_name": "DJI_20260602165038_0001_D.JPG",
    "packets": {
      "method_a": {"path": "method_a/000.npz", "sha256": "PACKET_SHA256", "camera_identity_sha256": "FROM_SCENE_JSON"},
      "method_b": {"path": "method_b/000.npz", "sha256": "PACKET_SHA256", "camera_identity_sha256": "FROM_SCENE_JSON"}
    }
  }]
}
```

The example shows one row; include every target from `proxy/<scene>/scene.json`.
Paths are relative to the prediction manifest. Use real renderer accumulators:
`accumulated_opacity`, `weighted_camera_z_sum`, `weighted_camera_z2_sum`,
`expected_camera_z`, `camera_z_variance`, `numeric_valid`. Do not fabricate
opacity or moments from a depth-only image. The proxy format is the frozen
six-array interface; measured GCP/LiDAR use the separate metric-packet v2.

The scorer reuses the original M1/A checks, common masks and metric functions.
It checks packet bytes and declared camera identity, but does not authenticate
the generating renderer: output is labeled `numerical_comparison_producer_unverified`.
The original method exporters and qualification gates remain in
`tools/measured_geometry_v1/` for producer verification.

Common support is recomputed across all declared methods. The paper's core
set is `umgs, jo, ms_splatting_neural, sig_mechanism, rgb_anchor`; its extension
set is the first three. A different set defines a different comparison, not
a score that can be appended to the paper's table. Missing targets never
produce a successful-only scene mean.

The two measured scenes use the fixed ROI proxy domain; the four vegetation
extensions use their common full-frame domain. Core results retain coverage
and the 4-by-4 ROI subregions. Aggregate the two scene groups separately;
do not average source-unit RMSE across differently scaled reconstructions.

The historical registry in `configs/proxy_registry.json` is retained as
provenance. The portable entry point does not require its old server paths.

`third_party/reference` is the unchanged numerical reference. Its source
manifest is a runtime dependency, not a separate reviewer checklist.
