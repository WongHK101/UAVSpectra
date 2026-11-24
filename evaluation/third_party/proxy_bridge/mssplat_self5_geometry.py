#!/usr/bin/env python3
"""Fail-closed self-five geometry bridge for MS-Splatting (common SfM).

This is a thin protocol wrapper around the already-audited external-five E3
expected-camera-z implementation.  It does not implement a second renderer or
weaken any packet gate.  The wrapper replaces only the E3 camera constructor:
self-five rendering is tied to the frozen support-lock camera fingerprints
(1200 x 868/869, znear=0.01, zfar=100.0), while the MS-Splatting checkpoint
remains in Nerfstudio's frozen dataparser frame.

Scientific invariants:

* source c2w is transformed forward into the dataparser frame;
* checkpoint geometry stays in the dataparser frame;
* M1 and expected-z are divided by dataparser scale s on return to source
  units, while M2 and variance are divided by s^2;
* complete-NaN log-scale sentinel rows are preserved and must have zero
  rasterizer radius for every rendered target;
* the audited packet validator, paired masks, metrics, and aggregation logic
  are reused without method-specific crop, Sim(3), scale, or shift fitting;
* every input/output file is SHA-256 bound and outputs are never overwritten.

The Road first-target qualification is intentionally a separate command.  A
passing qualification JSON is required before any scene-batch render or
evaluation command accepts work.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import math
import os
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


CONFIG_SCHEMA = "mmsplat_common_sfm_self5_geometry_registry_config_v1"
RUNTIME_PATHS_SCHEMA = "mmsplat_common_sfm_self5_runtime_paths_v1"
RUNTIME_SCHEMA = "mmsplat_common_sfm_self5_runtime_assets_v1"
REGISTRY_SCHEMA = "mmsplat_common_sfm_self5_geometry_frozen_registry_v1"
METHOD_MANIFEST_SCHEMA = "mmsplat_common_sfm_self5_method_scene_manifest_v1"
QUALIFICATION_SCHEMA = "mmsplat_common_sfm_self5_road_qualification_v1"
SCENE_EVALUATION_SCHEMA = "mmsplat_common_sfm_self5_scene_evaluation_v1"
AGGREGATE_SCHEMA = "mmsplat_common_sfm_self5_aggregate_v1"

SCENE_ORDER = ("road", "eucalyptus", "maize", "cassava", "papaya")
SCENE_IDS = {
    "road": "road_01_20260602_1648_40m",
    "eucalyptus": "eucalyptus_01_20260526_1053_pruned",
    "maize": "maize_02_20260526_1658",
    "cassava": "cassava_01_20260526_1603",
    "papaya": "papaya_01_20251217",
}
EXPECTED_TARGET_COUNTS = {
    "road": 12,
    "eucalyptus": 19,
    "maize": 11,
    "cassava": 15,
    "papaya": 26,
}
EXPECTED_HEIGHTS = {
    "road": 869,
    "eucalyptus": 868,
    "maize": 869,
    "cassava": 869,
    "papaya": 869,
}
TOTAL_TARGETS = 83
METHODS = ("umgs_full", "ms_splatting_common_sfm")
MASKS = ("paired_primary", "paired_opacity_ge_0_5", "paired_variance_valid")
OPENMVS_REFERENCE_ARRAYS = {"depth", "valid", "triangle_id", "barycentric"}
METRICS = (
    "mean_absrel",
    "median_absrel",
    "p90_absrel",
    "rmse",
    "spearman",
    "high_gradient_cosine",
)
LOWER_IS_BETTER = {"mean_absrel", "median_absrel", "p90_absrel", "rmse"}

AUDITED_E3_GIT_COMMIT = "7d6e5f63f731c42de22d08d4c3943a9c106eadec"
AUDITED_RUNTIME_SHA256 = {
    "tools/depth_reference_geometry_v2/external5_geometry_e3.py": "f510cd46ef9a2a8e4c12fb169e4eca75d2b5a2e6232be59a9c7ab7780f2e1bce",
    "tools/depth_reference_geometry_v2/external5_geometry_e1.py": "df19f3be321084e1946029f6228d2599f0d0c1290d6e61eadb71613a5b42c768",
    "tools/depth_reference_geometry_v2/export_ramp_expected_camera_z_packets.py": "580172aa91f4f18ef418363cae76a14befdfda3a42b78e3a1d4a45c6f36a254c",
    "tools/depth_reference_geometry_v2/export_umgs_expected_camera_z_packet.py": "a28c6488d55d30a836b85d6e8ffe7f20a4a33b3ae25325c588c3bed6426890b0",
    "tools/depth_reference_geometry_v2/validate_umgs_expected_camera_z_single_packet.py": "a8f1c0e883b8522e4402c68ac10fff5661ccc27e8d1a32c4b3318241f799007f",
    "tools/depth_reference_geometry_v2/openmvs_da3_overlap_corrected.py": "a99ad150a72410a87eb6a5cbba553ad9c65722b30790ae955ca469a524f37240",
    "tools/depth_reference_geometry_v2/evaluate_umgs_openmvs_camera_z_proxy_alignment.py": "9b2075a29d323d51c8aa35a4e3c58abeeadfb7ab584bad5b962bb6aa5dc4d138",
    "gaussian_renderer/__init__.py": "a15df36944df15d78305a9153ad7f87f20a985d63c6d0d7dd15135db924a84d8",
}

FORBIDDEN_ALIGNMENT_KEYS = {
    "method_scale",
    "method_shift",
    "scale_fit",
    "shift_fit",
    "crop_fit",
    "per_method_alignment",
    "sim3",
    "sim_3",
    "scale_shift",
    "umgs_alignment",
    "mss_alignment",
}


class Self5ContractError(RuntimeError):
    def __init__(self, counter: str, detail: str = ""):
        super().__init__(f"{counter}: {detail}" if detail else counter)
        self.counter = counter


def require(condition: bool, counter: str, detail: str = "") -> None:
    if not condition:
        raise Self5ContractError(counter, detail)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def write_json(path: str | Path, payload: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_json_bytes(payload))


def read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    require(isinstance(payload, dict), "json_root_not_object", str(path))
    return payload


def canonical_fingerprint_sha256(payload: Any) -> str:
    # The historical camera exporter intentionally omitted a terminal LF.
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def normalized_stem(value: str | Path) -> str:
    return Path(str(value).replace("\\", "/")).stem.casefold()


def utf8_sorted(values: Iterable[str]) -> list[str]:
    return sorted(values, key=lambda value: value.encode("utf-8"))


def normalized_name_set_sha256(names: Sequence[str]) -> str:
    ordered = utf8_sorted({normalized_stem(name) for name in names})
    return hashlib.sha256(("\n".join(ordered) + "\n").encode("utf-8")).hexdigest()


def file_record(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    require(source.is_file(), "missing_bound_file", str(source))
    return {"path": source.as_posix(), "size_bytes": int(source.stat().st_size), "sha256": sha256_file(source)}


def verify_record(record: Mapping[str, Any], label: str) -> Path:
    path = Path(str(record.get("path", "")))
    require(path.is_file(), f"missing_{label}", str(path))
    expected_sha = str(record.get("sha256", ""))
    require(len(expected_sha) == 64, f"{label}_sha256_missing", str(path))
    require(sha256_file(path) == expected_sha, f"{label}_sha256_mismatch", str(path))
    if "size_bytes" in record:
        require(path.stat().st_size == int(record["size_bytes"]), f"{label}_size_mismatch", str(path))
    return path


def assert_new_output(path: str | Path) -> Path:
    output = Path(path)
    require(not output.exists() or not any(output.iterdir()), "output_overwrite_forbidden", str(output))
    output.mkdir(parents=True, exist_ok=True)
    return output


def assert_output_external(output: Path, source_paths: Sequence[Path]) -> None:
    resolved = output.resolve()
    for source in source_paths:
        other = source.resolve()
        require(
            resolved != other and other not in resolved.parents and resolved not in other.parents,
            "output_source_path_overlap",
            f"{resolved} <-> {other}",
        )


def walk_keys(payload: Any):
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            yield str(key).casefold()
            yield from walk_keys(value)
    elif isinstance(payload, list):
        for value in payload:
            yield from walk_keys(value)


def require_no_alignment(payload: Any, label: str) -> None:
    present = set(walk_keys(payload)) & FORBIDDEN_ALIGNMENT_KEYS
    require(not present, "method_specific_alignment_forbidden", f"{label}:{sorted(present)}")


def _repo_root() -> Path:
    path = Path(__file__).resolve()
    # .../Multispectral/outputs/<campaign>/scripts/this.py
    return path.parents[3]


def audited_runtime_root() -> Path:
    configured = os.environ.get("UMGS_EXTERNAL5_E3_ROOT", "").strip()
    if configured:
        return Path(configured).resolve()
    local = _repo_root() / "worktrees" / "external5_geometry_protocol_20260717_worktree"
    return local.resolve()


def audited_runtime_bindings() -> dict[str, dict[str, Any]]:
    root = audited_runtime_root()
    bindings: dict[str, dict[str, Any]] = {}
    for relative, expected in AUDITED_RUNTIME_SHA256.items():
        path = root.joinpath(*relative.split("/"))
        require(path.is_file(), "audited_runtime_source_missing", str(path))
        actual = sha256_file(path)
        require(actual == expected, "audited_runtime_source_sha256_mismatch", f"{relative}:{actual}")
        bindings[relative] = file_record(path)
    return bindings


def audited_modules() -> SimpleNamespace:
    root = audited_runtime_root()
    audited_runtime_bindings()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    e3 = importlib.import_module("tools.depth_reference_geometry_v2.external5_geometry_e3")
    e1 = importlib.import_module("tools.depth_reference_geometry_v2.external5_geometry_e1")
    packets = importlib.import_module("tools.depth_reference_geometry_v2.export_ramp_expected_camera_z_packets")
    validator = importlib.import_module("tools.depth_reference_geometry_v2.validate_umgs_expected_camera_z_single_packet")
    require(
        Path(e3.__file__).resolve() == root / "tools" / "depth_reference_geometry_v2" / "external5_geometry_e3.py",
        "wrong_e3_module_imported",
        str(e3.__file__),
    )
    # SciPy/matplotlib metric modules remain lazy so CPU-only identity tests do
    # not initialize a second OpenMP runtime beside torch on Windows.
    return SimpleNamespace(e3=e3, e1=e1, packets=packets, validator=validator)


def audited_exporter():
    """Import the torch-dependent packet exporter only for an actual render."""
    audited_runtime_bindings()
    return importlib.import_module("tools.depth_reference_geometry_v2.export_umgs_expected_camera_z_packet")


def _projection_from_fingerprint(payload: Mapping[str, Any]) -> np.ndarray:
    fovx, fovy = float(payload["FoVx"]), float(payload["FoVy"])
    znear, zfar = float(payload["znear"]), float(payload["zfar"])
    p = np.zeros((4, 4), dtype=np.float32)
    p[0, 0] = np.float32(1.0 / math.tan(fovx / 2.0))
    p[1, 1] = np.float32(1.0 / math.tan(fovy / 2.0))
    p[3, 2] = np.float32(1.0)
    p[2, 2] = np.float32(zfar / (zfar - znear))
    p[2, 3] = np.float32(-(zfar * znear) / (zfar - znear))
    return p.T


def validate_camera_fingerprint(row: Mapping[str, Any], scene: str) -> dict[str, Any]:
    fingerprint = row.get("fingerprint")
    require(isinstance(fingerprint, Mapping), "camera_fingerprint_missing", f"{scene}:{row.get('image_name')}")
    payload = fingerprint.get("payload")
    require(isinstance(payload, Mapping), "camera_fingerprint_payload_missing", f"{scene}:{row.get('image_name')}")
    recorded = str(fingerprint.get("fingerprint_sha256", ""))
    actual = canonical_fingerprint_sha256(payload)
    require(recorded == actual, "camera_fingerprint_self_hash_mismatch", f"{scene}:{row.get('image_name')}")
    require(str(payload.get("image_name")) == str(row.get("image_name")), "camera_fingerprint_name_mismatch", scene)
    require(int(payload.get("image_width", -1)) == int(row.get("width", -2)) == 1200, "camera_width_mismatch", scene)
    require(
        int(payload.get("image_height", -1)) == int(row.get("height", -2)) == EXPECTED_HEIGHTS[scene],
        "camera_height_mismatch",
        scene,
    )
    require(float(payload.get("znear", math.nan)) == 0.01, "camera_znear_mismatch", scene)
    require(float(payload.get("zfar", math.nan)) == 100.0, "camera_zfar_mismatch", scene)
    require(math.isfinite(float(payload.get("FoVx", math.nan))), "camera_fov_nonfinite", scene)
    require(math.isfinite(float(payload.get("FoVy", math.nan))), "camera_fov_nonfinite", scene)

    world = np.asarray(payload.get("world_view_transform"), dtype=np.float32)
    projection = np.asarray(payload.get("projection_matrix"), dtype=np.float32)
    full = np.asarray(payload.get("full_proj_transform"), dtype=np.float32)
    require(world.shape == projection.shape == full.shape == (4, 4), "camera_matrix_shape_mismatch", scene)
    require(bool(np.isfinite(world).all() and np.isfinite(projection).all() and np.isfinite(full).all()), "camera_matrix_nonfinite", scene)
    expected_projection = _projection_from_fingerprint(payload)
    projection_error = float(np.max(np.abs(projection - expected_projection)))
    require(projection_error <= 1e-7, "camera_projection_mismatch", f"{scene}:{projection_error}")
    full_error = float(np.max(np.abs(full - (world @ projection).astype(np.float32))))
    require(full_error <= 1e-5, "camera_full_projection_mismatch", f"{scene}:{full_error}")

    # The stored renderer world-view matrix is authoritative.  Reconstruct the
    # source OpenCV/OpenGL c2w without fitting or changing units.
    w2c_opencv = world.T.astype(np.float64)
    c2w_opencv = np.linalg.inv(w2c_opencv)
    flip = np.diag([1.0, -1.0, -1.0, 1.0])
    c2w_opengl = c2w_opencv @ flip
    roundtrip = np.linalg.inv(c2w_opengl @ flip).astype(np.float64) - w2c_opencv
    roundtrip_error = float(np.max(np.abs(roundtrip)))
    require(roundtrip_error <= 1e-7, "fingerprint_camera_roundtrip_mismatch", f"{scene}:{roundtrip_error}")
    return {
        "image_name": str(row["image_name"]),
        "normalized_stem": normalized_stem(str(row["image_name"])),
        "width": int(row["width"]),
        "height": int(row["height"]),
        "fingerprint_sha256": actual,
        "fingerprint_payload": dict(payload),
        "source_c2w_opencv": c2w_opencv.tolist(),
        "source_c2w_opengl": c2w_opengl.tolist(),
        "projection_recompute_max_abs_error": projection_error,
        "full_projection_recompute_max_abs_error": full_error,
        "source_camera_roundtrip_max_abs_error": roundtrip_error,
    }


def _explicit_eval_names(split: Mapping[str, Any]) -> list[str]:
    rows = split.get("eval", split.get("test"))
    require(isinstance(rows, list), "explicit_split_eval_missing", "")
    names: list[str] = []
    for row in rows:
        if isinstance(row, Mapping):
            name = str(row.get("image_name", row.get("original_d_image_name", "")))
        else:
            name = str(row)
        require(bool(name), "explicit_split_name_missing", str(row))
        names.append(Path(name.replace("\\", "/")).name)
    return names


def _scene_static_evidence(evidence_root: Path, explicit_root: Path, scene: str) -> dict[str, Any]:
    scene_id = SCENE_IDS[scene]
    camera_path = evidence_root / "contracts_v2" / f"{scene}_CAMERA_INDEX.json"
    split_path = explicit_root / f"{scene_id}.json"
    audit_path = explicit_root / f"{scene_id}.audit.json"
    camera_index = read_json(camera_path)
    split = read_json(split_path)
    audit = read_json(audit_path)
    require(camera_index.get("schema") == "support_lock_geometry_camera_index_v1", "camera_index_schema_mismatch", scene)
    require(split.get("schema") == "mmsplat-common-sfm-original-d-split-v1", "explicit_split_schema_mismatch", scene)
    require(split.get("scene_id") == scene_id, "explicit_split_scene_mismatch", scene)
    require(audit.get("schema") == "mmsplat-common-sfm-original-d-split-audit-v1", "split_audit_schema_mismatch", scene)
    require(audit.get("status") == "pass", "split_audit_not_passed", scene)
    invariants = audit.get("invariants")
    require(isinstance(invariants, Mapping), "split_audit_invariants_missing", scene)
    for key, value in invariants.items():
        allowed = value is True or value == "pass"
        # Papaya's frozen split had no second protocol_split.json to request;
        # its generic split, explicit mapping, expected constraints, and camera
        # target set are still independently SHA/set checked below.
        if scene == "papaya" and key == "authoritative_protocol_crosscheck":
            allowed = value == "not_requested"
        require(allowed, "split_audit_invariant_failed", f"{scene}:{key}:{value}")

    raw_cameras = camera_index.get("cameras")
    require(isinstance(raw_cameras, list), "camera_index_rows_malformed", scene)
    cameras = [validate_camera_fingerprint(row, scene) for row in raw_cameras]
    expected_count = EXPECTED_TARGET_COUNTS[scene]
    require(len(cameras) == expected_count, "camera_target_count_mismatch", f"{scene}:{len(cameras)}")
    require(int(camera_index.get("test_camera_count", -1)) == expected_count, "camera_index_count_mismatch", scene)
    camera_stems = [row["normalized_stem"] for row in cameras]
    require(len(camera_stems) == len(set(camera_stems)), "duplicate_camera_target", scene)

    split_names = _explicit_eval_names(split)
    require(len(split_names) == expected_count, "explicit_eval_count_mismatch", scene)
    split_stems = [normalized_stem(name) for name in split_names]
    require(set(split_stems) == set(camera_stems), "camera_explicit_split_target_set_mismatch", scene)
    require(len(split_stems) == len(set(split_stems)), "duplicate_explicit_eval_target", scene)

    evaluation_dir = evidence_root / "evaluations" / scene
    evaluation_files = sorted(evaluation_dir.glob("*.json"), key=lambda path: path.name.encode("utf-8"))
    require(len(evaluation_files) == expected_count, "existing_evaluation_metadata_count_mismatch", scene)
    by_stem: dict[str, dict[str, Any]] = {}
    for path in evaluation_files:
        payload = read_json(path)
        require(payload.get("schema") == "support_lock_openmvs_paired_evaluation_v1", "existing_evaluation_schema_mismatch", str(path))
        require(payload.get("scene") == scene_id, "existing_evaluation_scene_mismatch", str(path))
        require(payload.get("method_specific_alignment_used") is False, "method_specific_alignment_forbidden", str(path))
        target = str(payload.get("target", ""))
        stem = normalized_stem(target)
        require(stem not in by_stem, "duplicate_existing_evaluation_target", f"{scene}:{stem}")
        for key in ("reference", "full_packet"):
            record = payload.get(key)
            require(isinstance(record, Mapping), "existing_asset_record_missing", f"{path}:{key}")
            require(len(str(record.get("sha256", ""))) == 64, "existing_asset_sha256_missing", f"{path}:{key}")
            require(bool(str(record.get("path", ""))), "existing_asset_path_missing", f"{path}:{key}")
        by_stem[stem] = {
            "target": target,
            "metadata": file_record(path),
            "reference_expected": dict(payload["reference"]),
            "umgs_full_packet_expected": dict(payload["full_packet"]),
        }
    require(set(by_stem) == set(camera_stems), "existing_evidence_target_set_mismatch", scene)
    for camera in cameras:
        camera["existing_evidence"] = by_stem[camera["normalized_stem"]]

    return {
        "scene": scene,
        "scene_id": scene_id,
        "camera_index": file_record(camera_path),
        "explicit_split": file_record(split_path),
        "explicit_split_audit": file_record(audit_path),
        "target_count": expected_count,
        "target_name_sha256": normalized_name_set_sha256([row["image_name"] for row in cameras]),
        "cameras": cameras,
    }


def audit_static_evidence(evidence_root: str | Path, explicit_root: str | Path) -> dict[str, Any]:
    evidence = Path(evidence_root)
    explicit = Path(explicit_root)
    scenes = [_scene_static_evidence(evidence, explicit, scene) for scene in SCENE_ORDER]
    total = sum(int(row["target_count"]) for row in scenes)
    require(total == TOTAL_TARGETS, "self5_target_total_mismatch", str(total))
    return {
        "schema": "mmsplat_common_sfm_self5_static_evidence_audit_v1",
        "status": "pass",
        "scene_order": list(SCENE_ORDER),
        "target_total": total,
        "method_specific_alignment_used": False,
        "camera_contract": {
            "width": 1200,
            "height_by_scene": EXPECTED_HEIGHTS,
            "znear": 0.01,
            "zfar": 100.0,
            "projection_authority": "embedded self5 fingerprint payload",
        },
        "scenes": [
            {
                "scene": row["scene"],
                "scene_id": row["scene_id"],
                "target_count": row["target_count"],
                "target_name_sha256": row["target_name_sha256"],
                "camera_index": row["camera_index"],
                "explicit_split": row["explicit_split"],
                "explicit_split_audit": row["explicit_split_audit"],
            }
            for row in scenes
        ],
    }


def build_registry_config(
    evidence_root: str | Path,
    explicit_root: str | Path,
    runtime_json: str | Path,
    output_json: str | Path,
) -> dict[str, Any]:
    output = Path(output_json)
    require(not output.exists(), "output_overwrite_forbidden", str(output))
    runtime_path = Path(runtime_json)
    runtime = read_json(runtime_path)
    require(runtime.get("schema") == RUNTIME_SCHEMA, "runtime_assets_schema_mismatch", str(runtime_path))
    if "runtime_paths_manifest" in runtime:
        verify_record(runtime["runtime_paths_manifest"], "runtime_paths_manifest")
    require(tuple(row.get("scene") for row in runtime.get("scenes", [])) == SCENE_ORDER, "runtime_scene_order_mismatch", "")
    runtime_by_scene = {row["scene"]: row for row in runtime["scenes"]}
    static = {
        row["scene"]: row
        for row in (
            _scene_static_evidence(Path(evidence_root), Path(explicit_root), scene)
            for scene in SCENE_ORDER
        )
    }
    scene_rows: list[dict[str, Any]] = []
    for scene in SCENE_ORDER:
        assets = runtime_by_scene[scene]
        require(assets.get("scene_id") == SCENE_IDS[scene], "runtime_scene_id_mismatch", scene)
        bound_runtime: dict[str, Any] = {}
        for key in ("checkpoint", "dataparser", "adapter_manifest", "training_config"):
            record = assets.get(key)
            require(isinstance(record, Mapping), "runtime_asset_missing", f"{scene}:{key}")
            verify_record(record, f"{scene}_{key}")
            bound_runtime[key] = dict(record)
        row = static[scene]
        scene_rows.append(
            {
                "scene": scene,
                "scene_id": row["scene_id"],
                "camera_index": row["camera_index"],
                "explicit_split": row["explicit_split"],
                "explicit_split_audit": row["explicit_split_audit"],
                "evidence_metadata": [camera["existing_evidence"]["metadata"] for camera in row["cameras"]],
                "runtime": bound_runtime,
            }
        )
    config = {
        "schema": CONFIG_SCHEMA,
        "claim_boundary": "validation-gated expected-camera-z engineering proxy; OpenMVS is not absolute geometry ground truth",
        "created_before_first_common_sfm_mss_geometry_render": True,
        "new_mss_metric_read_before_registry_freeze": False,
        "method_specific_scale_shift_crop_alignment": False,
        "camera_contract": {
            "authority": "existing self5 support-lock fingerprints",
            "width": 1200,
            "height_by_scene": EXPECTED_HEIGHTS,
            "znear": 0.01,
            "zfar": 100.0,
        },
        "unit_contract": {
            "checkpoint_frame": "frozen_dataparser",
            "source_c2w_operation": "forward_dataparser_transform",
            "M1_and_expected_z": "divide_by_dataparser_scale",
            "M2_and_variance": "divide_by_dataparser_scale_squared",
        },
        "runtime_assets_manifest": file_record(runtime_path),
        "scene_order": list(SCENE_ORDER),
        "scenes": scene_rows,
    }
    require_no_alignment(config, "registry_config")
    write_json(output, config)
    return {"config_path": output.as_posix(), "config_sha256": sha256_file(output), "scene_count": 5, "target_count": 83}


def freeze_runtime_assets(paths_json: str | Path, output_json: str | Path) -> dict[str, Any]:
    """Turn a human-readable path list into the SHA-bound runtime manifest."""
    source = Path(paths_json)
    output = Path(output_json)
    require(not output.exists(), "output_overwrite_forbidden", str(output))
    payload = read_json(source)
    require(payload.get("schema") == RUNTIME_PATHS_SCHEMA, "runtime_paths_schema_mismatch", str(source))
    require_no_alignment(payload, "runtime_paths")
    rows = payload.get("scenes")
    require(isinstance(rows, list) and tuple(row.get("scene") for row in rows) == SCENE_ORDER, "runtime_paths_scene_order_mismatch", "")
    frozen_rows: list[dict[str, Any]] = []
    for row in rows:
        scene = str(row["scene"])
        require(row.get("scene_id") == SCENE_IDS[scene], "runtime_paths_scene_id_mismatch", scene)
        frozen: dict[str, Any] = {"scene": scene, "scene_id": SCENE_IDS[scene]}
        for key in ("checkpoint", "dataparser", "adapter_manifest", "training_config"):
            path = row.get(key)
            require(isinstance(path, str) and bool(path), "runtime_path_missing", f"{scene}:{key}")
            frozen[key] = file_record(path)
        frozen_rows.append(frozen)
    result = {
        "schema": RUNTIME_SCHEMA,
        "runtime_paths_manifest": file_record(source),
        "scene_order": list(SCENE_ORDER),
        "scenes": frozen_rows,
    }
    write_json(output, result)
    return {"runtime_json": output.as_posix(), "runtime_sha256": sha256_file(output), "scene_count": 5}


def _dataparser_roundtrip(modules: SimpleNamespace, cameras: Sequence[Mapping[str, Any]], parsed: Mapping[str, Any]) -> float:
    maximum = 0.0
    transform = np.asarray(parsed["matrix"], dtype=np.float64)
    scale = float(parsed["scale"])
    for camera in cameras:
        source = np.asarray(camera["source_c2w_opengl"], dtype=np.float64)
        transformed = modules.e1.apply_dataparser_transform(source, transform, scale)
        unscaled = np.array(transformed, copy=True)
        unscaled[:3, 3] /= scale
        recovered = np.linalg.inv(transform) @ unscaled
        maximum = max(maximum, float(np.max(np.abs(recovered - source))))
    require(maximum <= 1e-7, "dataparser_camera_roundtrip_mismatch", str(maximum))
    return maximum


def _validate_adapter_manifest(path: Path, scene: str, split_binding: Mapping[str, Any]) -> dict[str, Any]:
    payload = read_json(path)
    require(payload.get("schema_version") == "mmsplat-common-sfm-adapter-v1", "adapter_manifest_schema_mismatch", scene)
    require(payload.get("scene_id") == SCENE_IDS[scene], "adapter_manifest_scene_mismatch", scene)
    protocol = payload.get("protocol")
    counts = payload.get("counts")
    source = payload.get("source")
    require(isinstance(protocol, Mapping) and isinstance(counts, Mapping) and isinstance(source, Mapping), "adapter_manifest_malformed", scene)
    require(float(protocol.get("pose_roundtrip_max_abs_error", math.inf)) <= 1e-7, "adapter_pose_roundtrip_mismatch", scene)
    require(protocol.get("five_channels_share_exact_pose_and_intrinsics") is True, "adapter_channel_pose_identity_failed", scene)
    require(protocol.get("trained_gaussian_ply_used") is False, "trained_geometry_initialization_forbidden", scene)
    require(protocol.get("training_masks_consumed_by_stock_ms_splatting") is False, "mss_training_mask_semantics_changed", scene)
    require(int(counts.get("eval_image_groups", -1)) == EXPECTED_TARGET_COUNTS[scene], "adapter_eval_count_mismatch", scene)
    frozen = source.get("frozen_split_json")
    require(isinstance(frozen, Mapping), "adapter_split_binding_missing", scene)
    require(str(frozen.get("sha256")) == str(split_binding["sha256"]), "adapter_split_sha256_mismatch", scene)
    parser_args = protocol.get("required_dataparser_arguments")
    require(parser_args == ["mmsplat-dataparser", "--downscale-factor", "0", "--eval-mode", "json-list"], "adapter_dataparser_arguments_mismatch", scene)
    parser_audit = protocol.get("stock_json_list_match_audit")
    require(isinstance(parser_audit, Mapping), "adapter_json_list_audit_missing", scene)
    require(int(parser_audit.get("overlap_matches", -1)) == 0, "adapter_split_overlap", scene)
    require(int(parser_audit.get("unmatched_frames", -1)) == 0, "adapter_unmatched_frames", scene)
    return payload


def freeze_registry(config_path: str | Path, output_json: str | Path) -> dict[str, Any]:
    output = Path(output_json)
    require(not output.exists(), "output_overwrite_forbidden", str(output))
    modules = audited_modules()
    config = read_json(config_path)
    require(config.get("schema") == CONFIG_SCHEMA, "registry_config_schema_mismatch", str(config_path))
    require_no_alignment(config, "registry_config")
    require(config.get("created_before_first_common_sfm_mss_geometry_render") is True, "registry_not_pre_render_frozen", "")
    require(config.get("new_mss_metric_read_before_registry_freeze") is False, "new_mss_metric_read_before_freeze", "")
    require(config.get("method_specific_scale_shift_crop_alignment") is False, "method_specific_alignment_forbidden", "")
    require(tuple(config.get("scene_order", [])) == SCENE_ORDER, "registry_config_scene_order_mismatch", "")
    verify_record(config["runtime_assets_manifest"], "runtime_assets_manifest")

    config_rows = config.get("scenes")
    require(isinstance(config_rows, list) and tuple(row.get("scene") for row in config_rows) == SCENE_ORDER, "registry_config_scene_set_mismatch", "")
    frozen_scenes: list[dict[str, Any]] = []
    for config_row in config_rows:
        scene = str(config_row["scene"])
        scene_id = SCENE_IDS[scene]
        require(config_row.get("scene_id") == scene_id, "registry_scene_id_mismatch", scene)
        camera_path = verify_record(config_row["camera_index"], f"{scene}_camera_index")
        split_path = verify_record(config_row["explicit_split"], f"{scene}_explicit_split")
        audit_path = verify_record(config_row["explicit_split_audit"], f"{scene}_split_audit")
        static = _scene_static_evidence(camera_path.parents[1], split_path.parent, scene)
        # Bind exactly the files selected by config, even if the static helper
        # discovered them through the same evidence directory.
        require(static["camera_index"]["sha256"] == config_row["camera_index"]["sha256"], "camera_index_binding_mismatch", scene)
        require(static["explicit_split"]["sha256"] == config_row["explicit_split"]["sha256"], "explicit_split_binding_mismatch", scene)
        require(static["explicit_split_audit"]["sha256"] == config_row["explicit_split_audit"]["sha256"], "split_audit_binding_mismatch", scene)

        runtime = config_row.get("runtime")
        require(isinstance(runtime, Mapping), "scene_runtime_missing", scene)
        runtime_paths = {key: verify_record(runtime[key], f"{scene}_{key}") for key in ("checkpoint", "dataparser", "adapter_manifest", "training_config")}
        adapter = _validate_adapter_manifest(runtime_paths["adapter_manifest"], scene, config_row["explicit_split"])
        parsed = modules.e1.load_dataparser_transform(runtime_paths["dataparser"])
        max_roundtrip = _dataparser_roundtrip(modules, static["cameras"], parsed)

        metadata_records = config_row.get("evidence_metadata")
        require(isinstance(metadata_records, list) and len(metadata_records) == EXPECTED_TARGET_COUNTS[scene], "evidence_metadata_count_mismatch", scene)
        metadata_by_sha = {str(row.get("sha256")): row for row in metadata_records}
        frozen_targets: list[dict[str, Any]] = []
        for camera in static["cameras"]:
            evidence = camera["existing_evidence"]
            metadata_expected = evidence["metadata"]
            require(metadata_expected["sha256"] in metadata_by_sha, "evidence_metadata_binding_missing", camera["image_name"])
            metadata_path = verify_record(metadata_by_sha[metadata_expected["sha256"]], f"{scene}_evidence_metadata")
            require(metadata_path.resolve() == Path(metadata_expected["path"]).resolve(), "evidence_metadata_path_mismatch", str(metadata_path))
            reference_path = verify_record(evidence["reference_expected"], f"{scene}_reference_packet")
            full_path = verify_record(evidence["umgs_full_packet_expected"], f"{scene}_umgs_full_packet")
            frozen_targets.append(
                {
                    **{key: value for key, value in camera.items() if key != "existing_evidence"},
                    "evidence_metadata": file_record(metadata_path),
                    "reference_packet": file_record(reference_path),
                    "umgs_full_packet": file_record(full_path),
                }
            )
        require(len(frozen_targets) == EXPECTED_TARGET_COUNTS[scene], "frozen_target_count_mismatch", scene)
        frozen_scenes.append(
            {
                "scene": scene,
                "scene_id": scene_id,
                "target_count": len(frozen_targets),
                "target_name_sha256": static["target_name_sha256"],
                "camera_index": static["camera_index"],
                "explicit_split": static["explicit_split"],
                "explicit_split_audit": static["explicit_split_audit"],
                "adapter_manifest": file_record(runtime_paths["adapter_manifest"]),
                "adapter_output_inventory_sha256": adapter["artifacts"]["r8_output_stable_inventory_sha256"],
                "training_config": file_record(runtime_paths["training_config"]),
                "checkpoint": file_record(runtime_paths["checkpoint"]),
                "dataparser": {
                    **file_record(runtime_paths["dataparser"]),
                    "transform": np.asarray(parsed["matrix"], dtype=np.float64).tolist(),
                    "scale": float(parsed["scale"]),
                    "orthonormal_max_abs_error": float(parsed["orthonormal_max_abs_error"]),
                    "rotation_determinant": float(parsed["rotation_determinant"]),
                    "camera_roundtrip_max_abs_error": max_roundtrip,
                },
                "camera_contract": {
                    "width": 1200,
                    "height": EXPECTED_HEIGHTS[scene],
                    "znear": 0.01,
                    "zfar": 100.0,
                    "projection_authority": "exact embedded self5 fingerprint projection matrix",
                },
                "targets": frozen_targets,
            }
        )

    total = sum(row["target_count"] for row in frozen_scenes)
    require(total == TOTAL_TARGETS, "frozen_self5_target_total_mismatch", str(total))
    registry = {
        "schema": REGISTRY_SCHEMA,
        "claim_boundary": config["claim_boundary"],
        "audited_external5_e3_git_commit": AUDITED_E3_GIT_COMMIT,
        "audited_runtime_sources": audited_runtime_bindings(),
        "registry_config": file_record(config_path),
        "runtime_assets_manifest": dict(config["runtime_assets_manifest"]),
        "created_before_first_common_sfm_mss_geometry_render": True,
        "new_mss_metric_read_before_registry_freeze": False,
        "method_specific_scale_shift_crop_alignment": False,
        "sim3_used": False,
        "crop_used": False,
        "scale_shift_fit_used": False,
        "checkpoint_frame": "frozen_dataparser",
        "source_camera_operation": "forward_dataparser_transform",
        "source_unit_return": {"M1": "divide_by_s", "expected_camera_z": "divide_by_s", "M2": "divide_by_s_squared", "variance": "divide_by_s_squared"},
        "target_weighting": "equal within scene",
        "scene_weighting": "equal across the five representative scenes",
        "scene_order": list(SCENE_ORDER),
        "target_total": total,
        "scenes": frozen_scenes,
    }
    require_no_alignment(registry, "frozen_registry")
    write_json(output, registry)
    return {"registry_path": output.as_posix(), "registry_sha256": sha256_file(output), "scene_count": 5, "target_count": total}


def load_registry(path: str | Path) -> dict[str, Any]:
    registry = read_json(path)
    require(registry.get("schema") == REGISTRY_SCHEMA, "registry_schema_mismatch", str(path))
    require_no_alignment(registry, "frozen_registry")
    require(tuple(registry.get("scene_order", [])) == SCENE_ORDER, "registry_scene_order_mismatch", "")
    require(int(registry.get("target_total", -1)) == TOTAL_TARGETS, "registry_target_total_mismatch", "")
    verify_record(registry["registry_config"], "registry_config")
    verify_record(registry["runtime_assets_manifest"], "runtime_assets_manifest")
    actual_runtime = audited_runtime_bindings()
    require(
        {key: value["sha256"] for key, value in registry["audited_runtime_sources"].items()}
        == {key: value["sha256"] for key, value in actual_runtime.items()},
        "audited_runtime_registry_binding_mismatch",
        "",
    )
    return registry


def scene_record(registry: Mapping[str, Any], scene: str) -> Mapping[str, Any]:
    matches = [row for row in registry["scenes"] if row["scene"] == scene]
    require(len(matches) == 1, "registry_scene_identity_mismatch", scene)
    return matches[0]


def target_record(scene: Mapping[str, Any], target: str) -> Mapping[str, Any]:
    stem = normalized_stem(target)
    matches = [row for row in scene["targets"] if row["normalized_stem"] == stem]
    require(len(matches) == 1, "registry_target_identity_mismatch", f"{scene['scene']}:{target}")
    return matches[0]


def _dp_camera_roundtrip(source: np.ndarray, transform: np.ndarray, scale: float, transformed: np.ndarray) -> float:
    unscaled = np.array(transformed, copy=True)
    unscaled[:3, 3] /= scale
    recovered = np.linalg.inv(transform) @ unscaled
    return float(np.max(np.abs(recovered - source)))


def build_renderer_camera(modules: SimpleNamespace, scene: Mapping[str, Any], target: Mapping[str, Any]):
    import torch
    from utils.graphics_utils import getWorld2View2

    payload = target["fingerprint_payload"]
    require(int(payload["image_width"]) == 1200, "render_camera_width_mismatch", target["image_name"])
    require(int(payload["image_height"]) == EXPECTED_HEIGHTS[str(scene["scene"])], "render_camera_height_mismatch", target["image_name"])
    require(float(payload["znear"]) == 0.01 and float(payload["zfar"]) == 100.0, "render_camera_clip_plane_mismatch", target["image_name"])
    source = np.asarray(target["source_c2w_opengl"], dtype=np.float64)
    transform = np.asarray(scene["dataparser"]["transform"], dtype=np.float64)
    scale = float(scene["dataparser"]["scale"])
    transformed = modules.e1.apply_dataparser_transform(source, transform, scale)
    roundtrip = _dp_camera_roundtrip(source, transform, scale, transformed)
    require(roundtrip <= 1e-7, "render_camera_dataparser_roundtrip_mismatch", f"{target['image_name']}:{roundtrip}")
    c2w_opencv = modules.e1.opengl_c2w_to_opencv_c2w(transformed)
    w2c = np.linalg.inv(c2w_opencv)
    r = w2c[:3, :3].T
    t = w2c[:3, 3]
    world = torch.tensor(getWorld2View2(r, t), dtype=torch.float32, device="cuda").transpose(0, 1)
    projection_np = np.asarray(payload["projection_matrix"], dtype=np.float32)
    require(float(np.max(np.abs(projection_np - _projection_from_fingerprint(payload)))) <= 1e-7, "render_projection_not_fingerprint_exact", target["image_name"])
    projection = torch.tensor(projection_np, dtype=torch.float32, device="cuda")
    full = world.unsqueeze(0).bmm(projection.unsqueeze(0)).squeeze(0)
    camera = SimpleNamespace(
        image_name=target["image_name"],
        colmap_id=int(payload["colmap_id"]),
        uid=int(payload["uid"]),
        image_width=int(payload["image_width"]),
        image_height=int(payload["image_height"]),
        FoVx=float(payload["FoVx"]),
        FoVy=float(payload["FoVy"]),
        znear=0.01,
        zfar=100.0,
        R=r,
        T=t,
        world_view_transform=world,
        projection_matrix=projection,
        full_proj_transform=full,
        camera_center=torch.inverse(world)[3, :3],
        c2w_opencv=c2w_opencv,
    )
    return camera, {
        "source_c2w_forward_dataparser_transform": True,
        "checkpoint_geometry_frame": "frozen_dataparser",
        "dataparser_scale": scale,
        "roundtrip_max_abs_error": roundtrip,
        "fingerprint_sha256": target["fingerprint_sha256"],
        "width": camera.image_width,
        "height": camera.image_height,
        "FoVx": camera.FoVx,
        "FoVy": camera.FoVy,
        "znear": camera.znear,
        "zfar": camera.zfar,
        "projection_matrix": projection_np.tolist(),
    }


def convert_packet_to_source_units(modules: SimpleNamespace, arrays: Mapping[str, np.ndarray], scale: float) -> dict[str, np.ndarray]:
    converted = modules.e1._convert_mss_packet_to_source_units(arrays, scale)
    linear = 1.0 / float(scale)
    quadratic = linear * linear
    for name in ("accumulated_opacity", "numeric_valid"):
        require(np.array_equal(np.asarray(converted[name]), np.asarray(arrays[name]), equal_nan=True), "packet_unit_conversion_changed_invariant", name)
    expected_factors = {
        "weighted_camera_z_sum": linear,
        "expected_camera_z": linear,
        "weighted_camera_z2_sum": quadratic,
        "camera_z_variance": quadratic,
    }
    for name, factor in expected_factors.items():
        expected = np.asarray(arrays[name]) * factor
        require(np.allclose(np.asarray(converted[name]), expected, rtol=0, atol=0, equal_nan=True), "packet_unit_conversion_mismatch", name)
    return converted


def m1_a_gate(modules: SimpleNamespace, arrays: Mapping[str, np.ndarray], shape: tuple[int, int]) -> dict[str, Any]:
    validation = modules.packets.validate_named_packet(arrays, shape)
    valid = np.asarray(arrays["numeric_valid"]).astype(bool)
    opacity = np.asarray(arrays["accumulated_opacity"])
    m1 = np.asarray(arrays["weighted_camera_z_sum"])
    expected = np.asarray(arrays["expected_camera_z"])
    residual = np.abs(expected[valid] - m1[valid] / opacity[valid])
    maximum = float(np.max(residual)) if residual.size else None
    return {
        **validation,
        "M1_over_A_identity_pass": True,
        "M1_over_A_max_abs_error": maximum,
        "validator_contract": "audited external5 E3 validate_named_packet / A1-Q tolerances",
    }


def _packet_relpath(target: str) -> str:
    return f"packets/{normalized_stem(target)}.npz"


def render_mss_targets(
    registry_path: str | Path,
    scene_name: str,
    targets: Sequence[str],
    output_path: str | Path,
    *,
    scope: str,
    opacity_epsilon: float = 1e-6,
    variance_clamp_tolerance: float = 1e-6,
) -> dict[str, Any]:
    modules = audited_modules()
    exporter = audited_exporter()
    registry = load_registry(registry_path)
    scene = scene_record(registry, scene_name)
    checkpoint = verify_record(scene["checkpoint"], f"{scene_name}_checkpoint")
    dataparser = verify_record(scene["dataparser"], f"{scene_name}_dataparser")
    adapter = verify_record(scene["adapter_manifest"], f"{scene_name}_adapter_manifest")
    training_config = verify_record(scene["training_config"], f"{scene_name}_training_config")
    requested = [target_record(scene, name) for name in targets]
    require(len(requested) == len(targets) == len({normalized_stem(name) for name in targets}), "render_target_list_malformed", scene_name)
    output = Path(output_path)
    assert_output_external(output, [checkpoint, dataparser, adapter, training_config, Path(registry_path)])
    output = assert_new_output(output)

    import torch
    import gaussian_renderer as gaussian_renderer_module

    expected_renderer = audited_runtime_root() / "gaussian_renderer" / "__init__.py"
    require(
        Path(gaussian_renderer_module.__file__).resolve() == expected_renderer.resolve(),
        "wrong_gaussian_renderer_imported",
        str(gaussian_renderer_module.__file__),
    )
    require(
        sha256_file(expected_renderer) == AUDITED_RUNTIME_SHA256["gaussian_renderer/__init__.py"],
        "gaussian_renderer_sha256_mismatch",
        str(expected_renderer),
    )
    render = gaussian_renderer_module.render

    checkpoint_sha_before = sha256_file(checkpoint)
    arrays, geometry = modules.e3.load_mss_checkpoint_geometry_e3(checkpoint)
    model = modules.e1.make_geometry_model(arrays)
    background = torch.zeros(3, dtype=torch.float32, device="cuda")
    override_color = torch.full((int(geometry["gaussian_count"]), 3), 0.5, dtype=torch.float32, device="cuda")
    pipeline = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False, antialiasing=False)
    nan_rows = np.asarray(geometry.get("raw_log_scaling_complete_nan_row_indices", []), dtype=np.int64)
    rows: list[dict[str, Any]] = []
    (output / "packets").mkdir()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for target in requested:
            camera, camera_audit = build_renderer_camera(modules, scene, target)
            rendered = render(
                camera,
                model,
                pipeline,
                background,
                override_color=override_color,
                use_trained_exp=False,
                separate_sh=False,
                return_expected_camera_z_packet=True,
                opacity_epsilon=float(opacity_epsilon),
                variance_clamp_tolerance=float(variance_clamp_tolerance),
            )
            sentinel_zero = True
            if nan_rows.size:
                radii = rendered.get("radii")
                require(radii is not None, "mss_rasterizer_radii_missing", target["image_name"])
                indices = torch.as_tensor(nan_rows, dtype=torch.long, device=radii.device)
                sentinel_zero = bool((radii.index_select(0, indices) == 0).all().item())
                require(sentinel_zero, "mss_nonfinite_scale_row_nonzero_radius", target["image_name"])
            named_dp = exporter.packet_tensor_to_named_arrays(
                rendered["expected_camera_z_packet"],
                opacity_epsilon=float(opacity_epsilon),
                variance_clamp_tolerance=float(variance_clamp_tolerance),
            )
            named = convert_packet_to_source_units(modules, named_dp, float(scene["dataparser"]["scale"]))
            shape = (int(target["height"]), int(target["width"]))
            validation = m1_a_gate(modules, named, shape)
            relative = _packet_relpath(target["image_name"])
            packet_path = output / relative
            modules.packets.deterministic_npz(packet_path, named)
            rows.append(
                {
                    "target": target["image_name"],
                    "normalized_stem": target["normalized_stem"],
                    "packet_file": relative,
                    "packet_sha256": sha256_file(packet_path),
                    "packet_array_sha256": modules.e1._packet_array_sha256(named),
                    "camera": camera_audit,
                    "validation": validation,
                    "complete_nan_log_scale_rows_zero_radius": sentinel_zero,
                }
            )
            del rendered, named_dp, named, camera
    torch.cuda.synchronize()
    checkpoint_sha_after = sha256_file(checkpoint)
    require(checkpoint_sha_before == checkpoint_sha_after == scene["checkpoint"]["sha256"], "checkpoint_modified", scene_name)
    manifest = {
        "schema": METHOD_MANIFEST_SCHEMA,
        "scope": scope,
        "scene": scene_name,
        "scene_id": scene["scene_id"],
        "method": "ms_splatting_common_sfm",
        "registry_sha256": sha256_file(registry_path),
        "checkpoint": scene["checkpoint"],
        "checkpoint_sha256_before": checkpoint_sha_before,
        "checkpoint_sha256_after": checkpoint_sha_after,
        "source_unchanged": True,
        "geometry": geometry,
        "nan_scale_sentinel_policy": "preserve complete-NaN rows and require zero rasterizer radius for every target",
        "nonfinite_scale_rows_zero_radius_all_targets": all(row["complete_nan_log_scale_rows_zero_radius"] for row in rows),
        "camera_contract": scene["camera_contract"],
        "camera_frame": "frozen_dataparser",
        "source_c2w_forward_dataparser_transform": True,
        "camera_roundtrip_max_abs_error": max(row["camera"]["roundtrip_max_abs_error"] for row in rows),
        "source_unit_conversion": {"M1": "divide_by_s", "expected_camera_z": "divide_by_s", "M2": "divide_by_s_squared", "variance": "divide_by_s_squared"},
        "packet_schema": exporter.packet_schema(),
        "depth_semantics": "sum(T_i*alpha_i*z_i)/sum(T_i*alpha_i)",
        "camera_z_output_units": "source common-SfM reconstruction units",
        "method_specific_alignment_used": False,
        "sim3_used": False,
        "crop_used": False,
        "scale_shift_fit_used": False,
        "constant_override_color_used": True,
        "fixed_rasterizer_contribution_cutoff": "alpha>=1/255",
        "fixed_rasterizer_early_termination": "remaining_transmittance<=1e-4",
        "target_count": len(rows),
        "target_name_sha256": normalized_name_set_sha256([row["target"] for row in rows]),
        "packets": rows,
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
        "runtime": modules.e1._renderer_source_sha256(),
    }
    manifest_path = output / "METHOD_SCENE_MANIFEST.json"
    write_json(manifest_path, manifest)
    return {"manifest_path": manifest_path.as_posix(), "manifest_sha256": sha256_file(manifest_path), "target_count": len(rows)}


def _load_npz_exact(modules: SimpleNamespace, record: Mapping[str, Any], keys: set[str], label: str) -> tuple[Path, dict[str, np.ndarray]]:
    path = verify_record(record, label)
    arrays = modules.e1._load_npz_exact(path, keys)
    return path, arrays


def evaluate_target_arrays(
    modules: SimpleNamespace,
    target: Mapping[str, Any],
    mss_packet_record: Mapping[str, Any],
) -> dict[str, Any]:
    gradient = importlib.import_module("tools.depth_reference_geometry_v2.openmvs_da3_overlap_corrected")
    expected_keys = set(modules.packets.EXPECTED_PACKET_ARRAYS)
    reference_path, reference = _load_npz_exact(
        modules,
        target["reference_packet"],
        OPENMVS_REFERENCE_ARRAYS,
        "reference_packet",
    )
    full_path, full = _load_npz_exact(modules, target["umgs_full_packet"], expected_keys, "umgs_full_packet")
    mss_path, mss = _load_npz_exact(modules, mss_packet_record, expected_keys, "mss_packet")
    shape = (int(target["height"]), int(target["width"]))
    depth = np.asarray(reference["depth"])
    valid = np.asarray(reference["valid"]).astype(bool)
    require(depth.shape == valid.shape == shape, "reference_packet_shape_mismatch", target["image_name"])
    require(bool(valid.any()), "reference_packet_has_no_valid_pixels", target["image_name"])
    require(bool(np.isfinite(depth[valid]).all() and (depth[valid] > 0).all()), "reference_depth_invalid", target["image_name"])
    full_gate = m1_a_gate(modules, full, shape)
    mss_gate = m1_a_gate(modules, mss, shape)
    masks = modules.packets.paired_masks(full, mss, depth, valid)
    metrics: dict[str, Any] = {}
    for mask_name in MASKS:
        mask = np.asarray(masks[mask_name]).astype(bool)
        require(bool(mask.any()), "paired_mask_has_no_pixels", f"{target['image_name']}:{mask_name}")
        domain = gradient.reference_high_gradient_domain(depth, mask)
        metrics[mask_name] = {
            "support_pixels": int(mask.sum()),
            "support_coverage": float(mask.mean()),
            "methods": {
                "umgs_full": modules.packets._candidate_metrics(depth, full["expected_camera_z"], mask, domain),
                "ms_splatting_common_sfm": modules.packets._candidate_metrics(depth, mss["expected_camera_z"], mask, domain),
            },
        }
    return {
        "target": target["image_name"],
        "reference_packet": {"path": reference_path.as_posix(), "sha256": sha256_file(reference_path)},
        "umgs_full_packet": {"path": full_path.as_posix(), "sha256": sha256_file(full_path), "M1_over_A_gate": full_gate},
        "ms_splatting_packet": {"path": mss_path.as_posix(), "sha256": sha256_file(mss_path), "M1_over_A_gate": mss_gate},
        "method_specific_alignment_used": False,
        "metrics": metrics,
    }


def _manifest_packet_map(manifest_path: str | Path, registry_path: str | Path, scene: str, expected_scope: str) -> tuple[dict[str, Any], dict[str, Mapping[str, Any]]]:
    manifest = read_json(manifest_path)
    require(manifest.get("schema") == METHOD_MANIFEST_SCHEMA, "method_manifest_schema_mismatch", str(manifest_path))
    require(manifest.get("scene") == scene, "method_manifest_scene_mismatch", str(manifest_path))
    require(manifest.get("scope") == expected_scope, "method_manifest_scope_mismatch", str(manifest_path))
    require(manifest.get("registry_sha256") == sha256_file(registry_path), "method_manifest_registry_mismatch", str(manifest_path))
    require(manifest.get("source_unchanged") is True, "method_source_modified", str(manifest_path))
    require(manifest.get("nonfinite_scale_rows_zero_radius_all_targets") is True, "nan_scale_sentinel_gate_failed", str(manifest_path))
    require(float(manifest.get("camera_roundtrip_max_abs_error", math.inf)) <= 1e-7, "method_manifest_roundtrip_mismatch", str(manifest_path))
    require(manifest.get("method_specific_alignment_used") is False, "method_specific_alignment_forbidden", str(manifest_path))
    rows = manifest.get("packets")
    require(isinstance(rows, list), "method_manifest_packets_malformed", str(manifest_path))
    root = Path(manifest_path).parent
    mapped: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        stem = normalized_stem(str(row.get("target", "")))
        require(stem not in mapped, "duplicate_method_packet", stem)
        packet = root / str(row["packet_file"])
        require(packet.is_file() and sha256_file(packet) == row["packet_sha256"], "method_packet_sha256_mismatch", str(packet))
        mapped[stem] = {"path": packet.as_posix(), "size_bytes": packet.stat().st_size, "sha256": row["packet_sha256"]}
    return manifest, mapped


def qualify_road_target(
    registry_path: str | Path,
    output_path: str | Path,
    target_name: str | None = None,
) -> dict[str, Any]:
    registry = load_registry(registry_path)
    road = scene_record(registry, "road")
    first = road["targets"][0]["image_name"]
    selected = target_name or first
    require(normalized_stem(selected) == normalized_stem(first), "road_qualification_target_not_frozen_first", f"{selected}!={first}")
    output = assert_new_output(output_path)
    render_root = output / "render"
    result = render_mss_targets(registry_path, "road", [first], render_root, scope="road_first_target_qualification")
    manifest_path = Path(result["manifest_path"])
    manifest, packets = _manifest_packet_map(manifest_path, registry_path, "road", "road_first_target_qualification")
    target = target_record(road, first)
    evaluation = evaluate_target_arrays(audited_modules(), target, packets[target["normalized_stem"]])
    qualification = {
        "schema": QUALIFICATION_SCHEMA,
        "status": "pass",
        "qualification_pass": True,
        "selection_rule": "first target in frozen Road self5 camera-index order",
        "scene": "road",
        "scene_id": road["scene_id"],
        "target": first,
        "registry_sha256": sha256_file(registry_path),
        "render_manifest": file_record(manifest_path),
        "camera_exact": {
            "fingerprint_sha256": target["fingerprint_sha256"],
            "width": target["width"],
            "height": target["height"],
            "znear": 0.01,
            "zfar": 100.0,
            "projection_recompute_max_abs_error": target["projection_recompute_max_abs_error"],
            "dataparser_roundtrip_max_abs_error": manifest["camera_roundtrip_max_abs_error"],
        },
        "nan_scale_sentinel_zero_radius": manifest["nonfinite_scale_rows_zero_radius_all_targets"],
        "source_unit_conversion": manifest["source_unit_conversion"],
        "M1_over_A_gate": evaluation["ms_splatting_packet"]["M1_over_A_gate"],
        "existing_reference_sha256": evaluation["reference_packet"]["sha256"],
        "existing_umgs_full_packet_sha256": evaluation["umgs_full_packet"]["sha256"],
        "new_mss_packet_sha256": evaluation["ms_splatting_packet"]["sha256"],
        "paired_mask_and_metrics": evaluation["metrics"],
        "method_specific_alignment_used": False,
        "sim3_used": False,
        "crop_used": False,
        "scale_shift_fit_used": False,
        "batch_authorized": True,
    }
    qualification_path = output / "ROAD_ONE_TARGET_QUALIFICATION.json"
    write_json(qualification_path, qualification)
    return {"qualification_path": qualification_path.as_posix(), "qualification_sha256": sha256_file(qualification_path), "target": first, "status": "pass"}


def validate_road_qualification(path: str | Path, registry_path: str | Path) -> dict[str, Any]:
    payload = read_json(path)
    require(payload.get("schema") == QUALIFICATION_SCHEMA, "road_qualification_schema_mismatch", str(path))
    require(payload.get("qualification_pass") is True and payload.get("batch_authorized") is True, "road_qualification_not_passed", str(path))
    require(payload.get("registry_sha256") == sha256_file(registry_path), "road_qualification_registry_mismatch", str(path))
    require(payload.get("nan_scale_sentinel_zero_radius") is True, "road_qualification_sentinel_failed", str(path))
    require(float(payload["camera_exact"]["dataparser_roundtrip_max_abs_error"]) <= 1e-7, "road_qualification_roundtrip_failed", str(path))
    require(payload.get("method_specific_alignment_used") is False, "method_specific_alignment_forbidden", str(path))
    return payload


def render_scene_batch(
    registry_path: str | Path,
    road_qualification: str | Path,
    scene_name: str,
    output_path: str | Path,
) -> dict[str, Any]:
    validate_road_qualification(road_qualification, registry_path)
    registry = load_registry(registry_path)
    scene = scene_record(registry, scene_name)
    targets = [row["image_name"] for row in scene["targets"]]
    return render_mss_targets(registry_path, scene_name, targets, output_path, scope="formal_scene_batch")


def _finite_mean(values: Sequence[float | None]) -> float | None:
    active = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(statistics.fmean(active)) if active else None


def _finite_median(values: Sequence[float | None]) -> float | None:
    active = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(statistics.median(active)) if active else None


def _direction(left: float | None, right: float | None, metric: str) -> str:
    if left is None or right is None:
        return "unavailable"
    if math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12):
        return "tie"
    umgs_better = float(left) < float(right) if metric in LOWER_IS_BETTER else float(left) > float(right)
    return "umgs_full" if umgs_better else "ms_splatting_common_sfm"


def aggregate_target_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    # Reuse E3's audited equal-target aggregation verbatim.  Only method labels
    # are translated at the boundary; numeric rows and mask names are unchanged.
    modules = audited_modules()
    to_e3 = {"umgs_full": "umgs_i", "ms_splatting_common_sfm": "ms_splatting"}
    mapped = [{**row, "method": to_e3[str(row["method"])]} for row in rows]
    audited = modules.e3._aggregate_target_rows(mapped)
    aggregate: dict[str, Any] = {}
    for mask in MASKS:
        directions: dict[str, Any] = {}
        for metric, counts in audited[mask]["target_direction_counts"].items():
            directions[metric] = {
                "umgs_full": counts["umgs_i"],
                "ms_splatting_common_sfm": counts["ms_splatting"],
                "tie": counts["tie"],
                "unavailable": counts["unavailable"],
            }
        aggregate[mask] = {
            "methods": {
                "umgs_full": audited[mask]["methods"]["umgs_i"],
                "ms_splatting_common_sfm": audited[mask]["methods"]["ms_splatting"],
            },
            "target_direction_counts": directions,
            "aggregation_implementation": "audited external5_geometry_e3._aggregate_target_rows; labels translated only",
        }
    return aggregate


def evaluate_scene(
    registry_path: str | Path,
    road_qualification: str | Path,
    scene_name: str,
    mss_manifest_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    validate_road_qualification(road_qualification, registry_path)
    modules = audited_modules()
    registry = load_registry(registry_path)
    scene = scene_record(registry, scene_name)
    manifest, packets = _manifest_packet_map(mss_manifest_path, registry_path, scene_name, "formal_scene_batch")
    expected_stems = [row["normalized_stem"] for row in scene["targets"]]
    require(list(packets) == expected_stems, "scene_packet_target_order_mismatch", scene_name)
    require(len(packets) == scene["target_count"], "scene_packet_target_count_mismatch", scene_name)
    output = assert_new_output(output_path)
    rows: list[dict[str, Any]] = []
    targets: list[dict[str, Any]] = []
    for target in scene["targets"]:
        detail = evaluate_target_arrays(modules, target, packets[target["normalized_stem"]])
        targets.append(detail)
        for mask_name, mask in detail["metrics"].items():
            for method in METHODS:
                metrics = mask["methods"][method]
                rows.append(
                    {
                        "scene": scene_name,
                        "scene_id": scene["scene_id"],
                        "target": target["image_name"],
                        "mask": mask_name,
                        "method": method,
                        "support_pixels": mask["support_pixels"],
                        "support_coverage": mask["support_coverage"],
                        **{metric: metrics.get(metric) for metric in METRICS},
                    }
                )
    metrics_csv = output / "PER_TARGET_METRICS.csv"
    with metrics_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {
        "schema": SCENE_EVALUATION_SCHEMA,
        "scene": scene_name,
        "scene_id": scene["scene_id"],
        "claim_boundary": registry["claim_boundary"],
        "registry_sha256": sha256_file(registry_path),
        "road_qualification_sha256": sha256_file(road_qualification),
        "mss_method_manifest_sha256": sha256_file(mss_manifest_path),
        "target_count": len(targets),
        "target_name_sha256": scene["target_name_sha256"],
        "method_specific_alignment_used": False,
        "paired_common_mask_used": True,
        "target_weighting": "equal",
        "output_artifacts": {"per_target_metrics_csv": file_record(metrics_csv)},
        "targets": targets,
        "aggregate": aggregate_target_rows(rows),
    }
    result_path = output / "SCENE_EVALUATION.json"
    write_json(result_path, result)
    return {"scene": scene_name, "target_count": len(targets), "result_path": result_path.as_posix(), "result_sha256": sha256_file(result_path)}


def _scene_metric(result: Mapping[str, Any], mask: str, method: str, metric: str) -> float | None:
    return result["aggregate"][mask]["methods"][method][metric]["equal_target_mean"]


def aggregate_scenes(
    registry_path: str | Path,
    road_qualification: str | Path,
    scene_results: Sequence[str | Path],
    output_path: str | Path,
) -> dict[str, Any]:
    validate_road_qualification(road_qualification, registry_path)
    modules = audited_modules()
    registry = load_registry(registry_path)
    results = [read_json(path) for path in scene_results]
    require(len(results) == 5, "aggregate_scene_result_count_mismatch", str(len(results)))
    require(tuple(row.get("scene") for row in results) == SCENE_ORDER, "aggregate_scene_order_mismatch", "")
    registry_sha = sha256_file(registry_path)
    require(all(row.get("schema") == SCENE_EVALUATION_SCHEMA for row in results), "aggregate_scene_schema_mismatch", "")
    require(all(row.get("registry_sha256") == registry_sha for row in results), "aggregate_registry_binding_mismatch", "")
    require(sum(int(row["target_count"]) for row in results) == TOTAL_TARGETS, "aggregate_target_total_mismatch", "")
    output = assert_new_output(output_path)

    metrics: dict[str, Any] = {}
    csv_rows: list[dict[str, Any]] = []
    for mask in MASKS:
        mask_record: dict[str, Any] = {}
        for method in METHODS:
            method_record: dict[str, Any] = {}
            for metric in METRICS:
                values = [_scene_metric(result, mask, method, metric) for result in results]
                record = {
                    "equal_scene_mean": modules.e3._finite_mean(values),
                    "equal_scene_median": modules.e3._finite_median(values),
                    "scene_value_count": len([value for value in values if value is not None]),
                }
                method_record[metric] = record
                csv_rows.append({"mask": mask, "method": method, "metric": metric, **record})
            mask_record[method] = method_record
        metrics[mask] = mask_record
    aggregate_csv = output / "SELF5_GEOMETRY_AGGREGATE.csv"
    with aggregate_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    scene_rows: list[dict[str, Any]] = []
    for scene, scene_result in zip(SCENE_ORDER, results):
        for method in METHODS:
            row: dict[str, Any] = {"scene": scene, "method": method}
            for metric in METRICS:
                row[f"paired_primary_{metric}"] = _scene_metric(scene_result, "paired_primary", method, metric)
            scene_rows.append(row)
    per_scene_csv = output / "SELF5_GEOMETRY_PER_SCENE.csv"
    with per_scene_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(scene_rows[0]))
        writer.writeheader()
        writer.writerows(scene_rows)
    result = {
        "schema": AGGREGATE_SCHEMA,
        "claim_boundary": registry["claim_boundary"],
        "registry_sha256": registry_sha,
        "road_qualification_sha256": sha256_file(road_qualification),
        "scene_result_sha256": {result["scene"]: sha256_file(path) for result, path in zip(results, scene_results)},
        "scene_order": list(SCENE_ORDER),
        "target_total": TOTAL_TARGETS,
        "target_weighting": "equal within scene",
        "scene_weighting": "equal across five representative scenes",
        "method_specific_alignment_used": False,
        "output_artifacts": {
            "aggregate_csv": file_record(aggregate_csv),
            "per_scene_csv": file_record(per_scene_csv),
        },
        "metrics": metrics,
    }
    aggregate_path = output / "SELF5_GEOMETRY_AGGREGATE.json"
    write_json(aggregate_path, result)
    return {"aggregate_path": aggregate_path.as_posix(), "aggregate_sha256": sha256_file(aggregate_path), "target_total": TOTAL_TARGETS}


def _cmd_audit(args: argparse.Namespace) -> None:
    result = audit_static_evidence(args.evidence_root, args.explicit_root)
    if args.output_json:
        output = Path(args.output_json)
        require(not output.exists(), "output_overwrite_forbidden", str(output))
        write_json(output, result)
        result = {**result, "output_json": output.as_posix(), "output_sha256": sha256_file(output)}
    print(json.dumps(result, sort_keys=True))


def _cmd_build_config(args: argparse.Namespace) -> None:
    print(json.dumps(build_registry_config(args.evidence_root, args.explicit_root, args.runtime_json, args.output_json), sort_keys=True))


def _cmd_freeze_runtime(args: argparse.Namespace) -> None:
    print(json.dumps(freeze_runtime_assets(args.paths_json, args.output_json), sort_keys=True))


def _cmd_freeze(args: argparse.Namespace) -> None:
    print(json.dumps(freeze_registry(args.config, args.output_json), sort_keys=True))


def _cmd_qualify(args: argparse.Namespace) -> None:
    print(json.dumps(qualify_road_target(args.registry, args.output, args.target), sort_keys=True))


def _cmd_render_scene(args: argparse.Namespace) -> None:
    print(json.dumps(render_scene_batch(args.registry, args.road_qualification, args.scene, args.output), sort_keys=True))


def _cmd_evaluate_scene(args: argparse.Namespace) -> None:
    print(json.dumps(evaluate_scene(args.registry, args.road_qualification, args.scene, args.mss_manifest, args.output), sort_keys=True))


def _cmd_aggregate(args: argparse.Namespace) -> None:
    print(json.dumps(aggregate_scenes(args.registry, args.road_qualification, args.scene_result, args.output), sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    audit = sub.add_parser("audit-static-evidence")
    audit.add_argument("--evidence-root", required=True)
    audit.add_argument("--explicit-root", required=True)
    audit.add_argument("--output-json")
    audit.set_defaults(func=_cmd_audit)
    config = sub.add_parser("build-registry-config")
    config.add_argument("--evidence-root", required=True)
    config.add_argument("--explicit-root", required=True)
    config.add_argument("--runtime-json", required=True)
    config.add_argument("--output-json", required=True)
    config.set_defaults(func=_cmd_build_config)
    runtime = sub.add_parser("freeze-runtime-assets")
    runtime.add_argument("--paths-json", required=True)
    runtime.add_argument("--output-json", required=True)
    runtime.set_defaults(func=_cmd_freeze_runtime)
    freeze = sub.add_parser("freeze-registry")
    freeze.add_argument("--config", required=True)
    freeze.add_argument("--output-json", required=True)
    freeze.set_defaults(func=_cmd_freeze)
    qualify = sub.add_parser("qualify-road-target")
    qualify.add_argument("--registry", required=True)
    qualify.add_argument("--output", required=True)
    qualify.add_argument("--target")
    qualify.set_defaults(func=_cmd_qualify)
    render_scene = sub.add_parser("render-scene")
    render_scene.add_argument("--registry", required=True)
    render_scene.add_argument("--road-qualification", required=True)
    render_scene.add_argument("--scene", required=True, choices=SCENE_ORDER)
    render_scene.add_argument("--output", required=True)
    render_scene.set_defaults(func=_cmd_render_scene)
    evaluate_scene_parser = sub.add_parser("evaluate-scene")
    evaluate_scene_parser.add_argument("--registry", required=True)
    evaluate_scene_parser.add_argument("--road-qualification", required=True)
    evaluate_scene_parser.add_argument("--scene", required=True, choices=SCENE_ORDER)
    evaluate_scene_parser.add_argument("--mss-manifest", required=True)
    evaluate_scene_parser.add_argument("--output", required=True)
    evaluate_scene_parser.set_defaults(func=_cmd_evaluate_scene)
    aggregate = sub.add_parser("aggregate")
    aggregate.add_argument("--registry", required=True)
    aggregate.add_argument("--road-qualification", required=True)
    aggregate.add_argument("--scene-result", action="append", required=True)
    aggregate.add_argument("--output", required=True)
    aggregate.set_defaults(func=_cmd_aggregate)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
        return 0
    except Self5ContractError as error:
        print(json.dumps({"status": "failed", "failure_counter": error.counter}, sort_keys=True))
        print(str(error), file=sys.stderr)
        return 2
    except Exception as error:  # audited helper errors are fail-closed here
        counter = getattr(error, "counter", "mmsplat_self5_internal_or_input_error")
        print(json.dumps({"status": "malformed", "failure_counter": counter}, sort_keys=True))
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
