#!/usr/bin/env python3
"""RAMP-GS standard-model expected-camera-z bridge.

The bridge is intentionally separate from the legacy checkpoint exporter. It
validates standard RAMP Model_G/R/RE/NIR exports on CPU, renders one canonical
Model_G geometry through the existing six-channel renderer, and evaluates M1
and M2 on a paired OpenMVS support mask. It does not alter checkpoints, PLYs,
training code, or the CUDA rasterizer.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import json
import sys
import unicodedata
import zipfile
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


BANDS = ("G", "R", "RE", "NIR")
CANONICAL_BAND = "G"
REQUIRED_ITERATION = 60000
BRIDGE_SCHEMA = "ramp_expected_camera_z_bridge_v1"
PREREGISTRATION_SCHEMA = "ramp_road_a0_preregistration_v1"
PAIRED_EVALUATION_SCHEMA = "ramp_openmvs_paired_proxy_evaluation_v1"
CANONICAL_CONTRACT_SCHEMA = "ramp_canonical_camera_resolution_contract_v1"
EXPECTED_PACKET_ARRAYS = (
    "accumulated_opacity",
    "weighted_camera_z_sum",
    "expected_camera_z",
    "numeric_valid",
    "weighted_camera_z2_sum",
    "camera_z_variance",
)
FORBIDDEN_ALIGNMENT_KEYS = {
    "method_scale",
    "method_shift",
    "scale_fit",
    "shift_fit",
    "per_method_alignment",
    "m1_alignment",
    "m2_alignment",
}


class BridgeContractError(RuntimeError):
    """Stable fail-fast error for a bridge contract violation."""

    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code


def require(condition: bool, code: str, message: str) -> None:
    if not condition:
        raise BridgeContractError(code, message)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def canonical_json_sha256(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def write_json(path: str | Path, payload: Any) -> None:
    Path(path).write_bytes(canonical_json_bytes(payload))


def normalized_stem(value: str | Path) -> str:
    name = Path(str(value).replace("\\", "/")).name
    return unicodedata.normalize("NFC", Path(name).stem).casefold()


def utf8_sorted(values: Iterable[str]) -> list[str]:
    return sorted(values, key=lambda value: value.encode("utf-8"))


def _numeric_field_key(name: str) -> tuple[str, int]:
    prefix, _, suffix = name.rpartition("_")
    return prefix, int(suffix) if suffix.isdigit() else -1


def _array_sha256(arrays: Sequence[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        canonical = np.ascontiguousarray(np.asarray(array, dtype="<f4"))
        digest.update(str(tuple(canonical.shape)).encode("ascii"))
        digest.update(b"\0<float32-le>\0")
        digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def parse_cfg_args(path: str | Path) -> dict[str, Any]:
    """Parse the repository's Namespace(...) cfg format without eval."""
    cfg_path = Path(path)
    require(cfg_path.is_file(), "missing_cfg_args", str(cfg_path))
    try:
        node = ast.parse(cfg_path.read_text(encoding="utf-8"), mode="eval").body
    except (SyntaxError, UnicodeError) as exc:
        raise BridgeContractError("malformed_cfg_args", str(cfg_path)) from exc
    require(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "Namespace"
        and not node.args,
        "malformed_cfg_args",
        "cfg_args must contain one keyword-only Namespace expression",
    )
    result: dict[str, Any] = {}
    for keyword in node.keywords:
        require(keyword.arg is not None, "malformed_cfg_args", "cfg_args may not use ** expansion")
        try:
            result[str(keyword.arg)] = ast.literal_eval(keyword.value)
        except (ValueError, TypeError) as exc:
            raise BridgeContractError("malformed_cfg_args", f"non-literal field {keyword.arg}") from exc
    return result


def read_name_file(path: str | Path) -> list[str]:
    names: list[str] = []
    for line in Path(path).read_text(encoding="utf-8", errors="strict").splitlines():
        text = line.strip()
        if text and not text.startswith("#"):
            names.append(Path(text.replace("\\", "/")).name)
    normalized = [normalized_stem(name) for name in names]
    require(len(normalized) == len(set(normalized)), "duplicate_split_view", str(path))
    return names


def locate_split_file(source_root: Path, name: str) -> Path:
    candidates = (
        source_root / name,
        source_root / "sparse" / name,
        source_root / "sparse" / "0" / name,
    )
    matches = [path for path in candidates if path.is_file()]
    require(len(matches) == 1, "split_file_identity_error", f"{name}: {matches}")
    return matches[0]


def camera_rows(path: str | Path) -> list[dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    require(isinstance(data, list) and bool(data), "malformed_camera_manifest", str(path))
    rows: list[dict[str, Any]] = []
    for raw in data:
        require(isinstance(raw, dict), "malformed_camera_manifest", str(path))
        name = str(raw.get("img_name", raw.get("image_name", "")))
        require(bool(name), "malformed_camera_manifest", "camera row missing image name")
        row = dict(raw)
        row["normalized_image_stem"] = normalized_stem(name)
        rows.append(row)
    rows.sort(key=lambda row: row["normalized_image_stem"].encode("utf-8"))
    stems = [row["normalized_image_stem"] for row in rows]
    require(len(stems) == len(set(stems)), "duplicate_camera_view", str(path))
    return rows


def ply_geometry_manifest(path: str | Path) -> dict[str, Any]:
    from plyfile import PlyData

    ply_path = Path(path)
    require(ply_path.is_file(), "missing_ramp_ply", str(ply_path))
    ply = PlyData.read(str(ply_path))
    require("vertex" in ply, "malformed_ramp_ply", "missing vertex element")
    vertex = ply["vertex"].data
    names = list(vertex.dtype.names or ())
    required = {"x", "y", "z", "opacity"}
    require(required.issubset(names), "malformed_ramp_ply", f"missing {sorted(required - set(names))}")
    scaling_names = sorted((name for name in names if name.startswith("scale_")), key=_numeric_field_key)
    rotation_names = sorted((name for name in names if name.startswith("rot_")), key=_numeric_field_key)
    require(len(scaling_names) == 3, "malformed_ramp_ply", f"scaling fields={scaling_names}")
    require(len(rotation_names) == 4, "malformed_ramp_ply", f"rotation fields={rotation_names}")
    groups = {
        "xyz": [np.asarray(vertex[name]) for name in ("x", "y", "z")],
        "scaling": [np.asarray(vertex[name]) for name in scaling_names],
        "rotation": [np.asarray(vertex[name]) for name in rotation_names],
        "opacity": [np.asarray(vertex["opacity"])],
    }
    group_sha = {name: _array_sha256(arrays) for name, arrays in groups.items()}
    identity = {
        "gaussian_count": int(len(vertex)),
        "group_sha256": group_sha,
    }
    return {
        **identity,
        "geometry_identity_sha256": canonical_json_sha256(identity),
        "ply_sha256": sha256_file(ply_path),
        "ply_size_bytes": ply_path.stat().st_size,
        "fields": names,
    }


def _source_scene_identity(source_path: Path, scene: str) -> dict[str, str]:
    require(source_path.name.endswith("_rectified"), "unexpected_band_source_root", str(source_path))
    require(scene in source_path.as_posix(), "scene_identity_mismatch", str(source_path))
    return {
        "source_parent": source_path.parent.as_posix(),
        "scene": scene,
    }


def inspect_band_model(method_dir: str | Path, band: str, scene: str, iteration: int) -> dict[str, Any]:
    require(iteration == REQUIRED_ITERATION, "iteration_contract_mismatch", str(iteration))
    model_dir = Path(method_dir) / f"Model_{band}"
    cfg_path = model_dir / "cfg_args"
    cameras_path = model_dir / "cameras.json"
    ply_path = model_dir / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
    cfg = parse_cfg_args(cfg_path)
    require(str(cfg.get("target_band")) == band, "band_cfg_identity_mismatch", str(model_dir))
    require(int(cfg.get("resolution", -1)) == 8, "training_resolution_mismatch", str(model_dir))
    require(str(cfg.get("stage2_mode")) == "joint_multispectral_export_view", "ramp_export_mode_mismatch", str(model_dir))
    source_path = Path(str(cfg.get("source_path", "")))
    source_identity = _source_scene_identity(source_path, scene)
    expected_suffix = f"{band}_rectified".casefold()
    require(source_path.name.casefold() == expected_suffix.casefold(), "band_source_root_mismatch", str(source_path))
    train_path = locate_split_file(source_path, "train.txt")
    test_path = locate_split_file(source_path, "test.txt")
    train_names = read_name_file(train_path)
    test_names = read_name_file(test_path)
    require(
        set(map(normalized_stem, train_names)).isdisjoint(map(normalized_stem, test_names)),
        "train_test_overlap",
        str(model_dir),
    )
    cameras = camera_rows(cameras_path)
    camera_names = [row["normalized_image_stem"] for row in cameras]
    split_names = [normalized_stem(name) for name in test_names + train_names]
    require(set(camera_names) == set(split_names), "camera_split_identity_mismatch", str(model_dir))
    return {
        "band": band,
        "model_dir": model_dir.as_posix(),
        "cfg_args_sha256": sha256_file(cfg_path),
        "cfg_contract": {
            key: cfg.get(key)
            for key in (
                "sh_degree",
                "resolution",
                "white_background",
                "train_test_exp",
                "eval",
                "input_dynamic_range",
                "radiometric_mode",
                "use_validity_mask",
                "stage2_mode",
            )
        },
        "source_identity": source_identity,
        "cameras_json_sha256": sha256_file(cameras_path),
        "camera_identity_sha256": canonical_json_sha256(cameras),
        "camera_names": camera_names,
        "train_names": [normalized_stem(name) for name in train_names],
        "test_names": [normalized_stem(name) for name in test_names],
        "train_file_sha256": sha256_file(train_path),
        "test_file_sha256": sha256_file(test_path),
        "geometry": ply_geometry_manifest(ply_path),
        "ply_path": ply_path.as_posix(),
    }


def preflight_ramp_method(
    method_dir: str | Path,
    scene: str,
    *,
    iteration: int = REQUIRED_ITERATION,
) -> dict[str, Any]:
    assets = [inspect_band_model(method_dir, band, scene, iteration) for band in BANDS]
    reference = assets[0]
    for asset in assets[1:]:
        require(asset["geometry"]["gaussian_count"] == reference["geometry"]["gaussian_count"], "four_band_gaussian_count_mismatch", asset["band"])
        require(asset["geometry"]["group_sha256"] == reference["geometry"]["group_sha256"], "four_band_geometry_mismatch", asset["band"])
        require(asset["camera_identity_sha256"] == reference["camera_identity_sha256"], "four_band_camera_identity_mismatch", asset["band"])
        require(asset["train_names"] == reference["train_names"], "four_band_train_split_mismatch", asset["band"])
        require(asset["test_names"] == reference["test_names"], "four_band_test_split_mismatch", asset["band"])
        require(asset["cfg_contract"] == reference["cfg_contract"], "four_band_cfg_contract_mismatch", asset["band"])
        require(asset["source_identity"] == reference["source_identity"], "four_band_source_identity_mismatch", asset["band"])
    return {
        "schema": BRIDGE_SCHEMA,
        "scene": scene,
        "method_dir": Path(method_dir).as_posix(),
        "iteration": iteration,
        "canonical_band": CANONICAL_BAND,
        "band_order": list(BANDS),
        "four_band_geometry_identical": True,
        "four_band_camera_identity_identical": True,
        "four_band_split_identical": True,
        "training_resolution": 8,
        "render_target_width": 1200,
        "assets": assets,
        "canonical_geometry_identity_sha256": reference["geometry"]["geometry_identity_sha256"],
        "canonical_ply_sha256": reference["geometry"]["ply_sha256"],
        "camera_identity_sha256": reference["camera_identity_sha256"],
        "train_names_sha256": canonical_json_sha256(reference["train_names"]),
        "test_names_sha256": canonical_json_sha256(reference["test_names"]),
    }


def heldout_targets(path: str | Path) -> list[str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = payload.get("heldout_images", []) if isinstance(payload, dict) else []
    require(isinstance(rows, list) and bool(rows), "malformed_heldout_manifest", str(path))
    targets: list[str] = []
    for row in rows:
        require(isinstance(row, dict), "malformed_heldout_manifest", str(path))
        name = str(row.get("image_name", ""))
        require(bool(name), "malformed_heldout_manifest", "missing image_name")
        targets.append(name)
    stems = [normalized_stem(name) for name in targets]
    require(len(stems) == len(set(stems)), "duplicate_heldout_target", str(path))
    return sorted(targets, key=lambda name: normalized_stem(name).encode("utf-8"))


def find_openmvs_packets(root: str | Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in sorted(Path(root).rglob("*.npz"), key=lambda item: item.as_posix().encode("utf-8")):
        stem = path.stem
        if stem.startswith("triangle_render_"):
            stem = stem[len("triangle_render_") :]
        key = normalized_stem(stem)
        require(key not in result, "duplicate_openmvs_packet", key)
        result[key] = path
    return result


def preregister_a0(
    heldout_manifest: str | Path,
    openmvs_root: str | Path,
    output_json: str | Path,
) -> dict[str, Any]:
    output = Path(output_json)
    require(not output.exists(), "output_overwrite_forbidden", str(output))
    targets = heldout_targets(heldout_manifest)
    packets = find_openmvs_packets(openmvs_root)
    valid = [name for name in targets if normalized_stem(name) in packets]
    require(bool(valid), "no_valid_openmvs_target", str(openmvs_root))
    selected = valid[0]
    payload = {
        "schema": PREREGISTRATION_SCHEMA,
        "selection_rule": "normalize held-out image stems, UTF-8 byte sort, select first target with exactly one OpenMVS packet",
        "heldout_manifest_sha256": sha256_file(heldout_manifest),
        "sorted_heldout_targets": targets,
        "sorted_heldout_targets_sha256": canonical_json_sha256(targets),
        "valid_openmvs_targets": valid,
        "a0_target": selected,
        "a0_target_normalized_stem": normalized_stem(selected),
        "a0_openmvs_packet": packets[normalized_stem(selected)].as_posix(),
        "a0_openmvs_packet_sha256": sha256_file(packets[normalized_stem(selected)]),
        "created_before_first_method_render": True,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, payload)
    return payload


def validate_openmvs_packet(path: str | Path) -> dict[str, Any]:
    packet_path = Path(path)
    with np.load(packet_path, allow_pickle=False) as data:
        require("depth" in data and "valid" in data, "malformed_openmvs_packet", str(packet_path))
        depth = np.asarray(data["depth"])
        valid = np.asarray(data["valid"])
    require(depth.ndim == 2 and depth.shape == valid.shape, "openmvs_packet_shape_mismatch", str(packet_path))
    require(depth.dtype == np.float32, "openmvs_packet_dtype_mismatch", f"depth={depth.dtype}")
    require(valid.dtype == np.uint8 or valid.dtype == np.bool_, "openmvs_packet_dtype_mismatch", f"valid={valid.dtype}")
    valid_bool = valid.astype(bool)
    require(bool(np.all(np.isfinite(depth[valid_bool]))), "openmvs_nonfinite_depth", str(packet_path))
    require(bool(np.all(depth[valid_bool] > 0)), "openmvs_nonpositive_depth", str(packet_path))
    return {
        "path": packet_path.as_posix(),
        "sha256": sha256_file(packet_path),
        "height": int(depth.shape[0]),
        "width": int(depth.shape[1]),
        "valid_pixels": int(valid_bool.sum()),
        "valid_coverage": float(valid_bool.mean()) if valid_bool.size else 0.0,
    }


def _walk_keys(payload: Any) -> Iterable[str]:
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            yield str(key)
            yield from _walk_keys(value)
    elif isinstance(payload, list):
        for value in payload:
            yield from _walk_keys(value)


def validate_frame_scale_decision(path: str | Path, scene: str, target: str) -> dict[str, Any]:
    decision_path = Path(path)
    payload = json.loads(decision_path.read_text(encoding="utf-8"))
    require(payload.get("status") == "correspondence_pass", "frame_scale_decision_not_passed", str(path))
    require(str(payload.get("scene")) == scene, "frame_scale_scene_mismatch", str(path))
    keys = {key.casefold() for key in _walk_keys(payload)}
    forbidden = sorted(keys & FORBIDDEN_ALIGNMENT_KEYS)
    require(not forbidden, "method_specific_alignment_forbidden", str(forbidden))
    transform = payload.get("transform_audit", {})
    text = str(transform.get("decision", "")) if isinstance(transform, dict) else ""
    require("identity_or_no_additional_transform" in text, "non_identity_frame_scale_decision", text)
    if payload.get("schema") == CANONICAL_CONTRACT_SCHEMA:
        require(payload.get("method_independent") is True, "method_specific_alignment_forbidden", str(path))
        require(payload.get("field_of_view_changed") is False, "camera_field_of_view_contract_mismatch", str(path))
        records = payload.get("camera_records", [])
        matches = [
            record
            for record in records
            if isinstance(record, dict) and normalized_stem(str(record.get("image_name", ""))) == normalized_stem(target)
        ]
        require(len(matches) == 1, "frame_scale_target_mismatch", f"{target}: {len(matches)}")
        camera = matches[0]
    else:
        require(normalized_stem(str(payload.get("target", ""))) == normalized_stem(target), "frame_scale_target_mismatch", str(path))
        camera = payload.get("camera_qualification", {})
        require(isinstance(camera, dict), "missing_camera_qualification", str(path))
    require(int(camera.get("width", -1)) == 1200, "camera_width_contract_mismatch", str(path))
    return {
        "path": decision_path.as_posix(),
        "sha256": sha256_file(decision_path),
        "schema": payload.get("schema"),
        "status": payload.get("status"),
        "scene": scene,
        "target": target,
        "width": int(camera["width"]),
        "height": int(camera.get("height", -1)),
        "method_specific_alignment_used": False,
        "frame_scale_mode": "identity_or_no_additional_transform_recorded",
    }


def validate_named_packet(arrays: Mapping[str, np.ndarray], expected_shape: tuple[int, int]) -> dict[str, Any]:
    from tools.depth_reference_geometry_v2.validate_umgs_expected_camera_z_single_packet import (
        DEFAULT_TOLERANCES,
        EXPECTED_ARRAYS,
    )

    require(set(arrays) == set(EXPECTED_ARRAYS), "packet_keyset_mismatch", str(sorted(arrays)))
    shapes = {tuple(np.asarray(arrays[name]).shape) for name in EXPECTED_PACKET_ARRAYS}
    require(shapes == {expected_shape}, "packet_shape_mismatch", str(shapes))
    opacity = np.asarray(arrays["accumulated_opacity"])
    weighted = np.asarray(arrays["weighted_camera_z_sum"])
    depth = np.asarray(arrays["expected_camera_z"])
    variance = np.asarray(arrays["camera_z_variance"])
    valid_raw = np.asarray(arrays["numeric_valid"])
    require(valid_raw.dtype == np.uint8 or valid_raw.dtype == np.bool_, "packet_valid_dtype_mismatch", str(valid_raw.dtype))
    require(set(np.unique(valid_raw).tolist()).issubset({0, 1, False, True}), "packet_valid_not_binary", "")
    valid = valid_raw.astype(bool)
    require(bool(valid.any()), "packet_has_no_valid_pixels", "")
    tol = DEFAULT_TOLERANCES
    require(bool(np.isfinite(opacity).all()), "packet_opacity_nonfinite", "")
    require(bool((opacity >= tol["opacity_min"]).all() and (opacity <= tol["opacity_max"]).all()), "packet_opacity_out_of_bounds", "")
    identity = np.isclose(depth[valid], weighted[valid] / opacity[valid], rtol=tol["identity_rtol"], atol=tol["identity_atol"])
    require(bool(identity.all()), "packet_expected_camera_z_identity_mismatch", "")
    require(bool(np.isfinite(depth[valid]).all() and (depth[valid] > 0).all()), "packet_depth_invalid", "")
    require(bool(np.isfinite(variance[valid]).all() and (variance[valid] >= tol["variance_min"]).all()), "packet_variance_invalid", "")
    require(bool(np.isnan(depth[~valid]).all()), "packet_invalid_depth_not_nan", "")
    require(bool(np.isnan(variance[~valid]).all()), "packet_invalid_variance_not_nan", "")
    return {
        "passed": True,
        "height": expected_shape[0],
        "width": expected_shape[1],
        "numeric_valid_pixels": int(valid.sum()),
        "numeric_valid_coverage": float(valid.mean()),
        "reused_validator_contract": "validate_umgs_expected_camera_z_single_packet.DEFAULT_TOLERANCES/EXPECTED_ARRAYS",
    }


def deterministic_npz(path: str | Path, arrays: Mapping[str, np.ndarray]) -> None:
    output = Path(path)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in utf8_sorted(arrays.keys()):
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, np.asarray(arrays[name]), allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, buffer.getvalue(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)


def _namespace_for_standard_scene(model_dir: Path, target_width: int) -> Namespace:
    cfg = parse_cfg_args(model_dir / "cfg_args")
    cfg.update(
        {
            "model_path": str(model_dir),
            "resolution": int(target_width),
            "data_device": "cuda",
            "eval": True,
        }
    )
    defaults = {
        "images": "images",
        "depths": "",
        "white_background": False,
        "train_test_exp": False,
        "sh_degree": 3,
    }
    for key, value in defaults.items():
        cfg.setdefault(key, value)
    return Namespace(**cfg)


def load_standard_ramp_scene(model_dir: str | Path, target_width: int, iteration: int = REQUIRED_ITERATION):
    """Load the canonical exported PLY through the repository Scene path."""
    require(iteration == REQUIRED_ITERATION, "iteration_contract_mismatch", str(iteration))
    from scene import Scene
    from scene.gaussian_model import GaussianModel

    root = Path(model_dir)
    args = _namespace_for_standard_scene(root, target_width)
    gaussians = GaussianModel(int(args.sh_degree))
    scene = Scene(args, gaussians, load_iteration=iteration, shuffle=False)
    return scene, gaussians, args


def _select_test_camera(scene: Any, target: str) -> Any:
    matches = [camera for camera in scene.getTestCameras() if normalized_stem(camera.image_name) == normalized_stem(target)]
    require(len(matches) == 1, "heldout_camera_identity_mismatch", f"{target}: {len(matches)}")
    return matches[0]


def render_packet(args: argparse.Namespace) -> dict[str, Any]:
    preflight = preflight_ramp_method(args.method_dir, args.scene, iteration=args.iteration)
    openmvs = validate_openmvs_packet(args.openmvs_packet)
    frame_scale = validate_frame_scale_decision(args.frame_scale_decision, args.scene, args.target)
    require(frame_scale["width"] == openmvs["width"] and frame_scale["height"] == openmvs["height"], "camera_openmvs_shape_mismatch", str(openmvs))
    targets = heldout_targets(args.heldout_manifest)
    require(normalized_stem(args.target) in {normalized_stem(name) for name in targets}, "target_not_official_heldout", args.target)
    output = Path(args.output)
    require(not output.exists() or not any(output.iterdir()), "output_overwrite_forbidden", str(output))

    # CUDA-dependent imports and model loading begin only after all CPU preflight.
    import torch
    from gaussian_renderer import render
    from tools.depth_reference_geometry_v2.export_umgs_expected_camera_z_packet import (
        camera_fingerprint,
        packet_schema,
        packet_tensor_to_named_arrays,
        verify_expected_camera_fingerprint,
    )

    model_dir = Path(args.method_dir) / f"Model_{CANONICAL_BAND}"
    scene, gaussians, dataset_args = load_standard_ramp_scene(model_dir, openmvs["width"], args.iteration)
    camera = _select_test_camera(scene, args.target)
    require(int(camera.image_width) == openmvs["width"] and int(camera.image_height) == openmvs["height"], "camera_openmvs_shape_mismatch", f"camera={camera.image_height}x{camera.image_width}, openmvs={openmvs['height']}x{openmvs['width']}")
    fingerprint = camera_fingerprint(camera)
    expected_fingerprint = Path(args.expected_camera_fingerprint)
    comparison = verify_expected_camera_fingerprint(
        fingerprint,
        SimpleNamespace(
            expected_camera_fingerprint=str(expected_fingerprint),
            expected_camera_fingerprint_sha256=sha256_file(expected_fingerprint),
            camera_fingerprint_atol=float(args.camera_fingerprint_atol),
        ),
    )
    background = torch.tensor(
        [1.0, 1.0, 1.0] if bool(dataset_args.white_background) else [0.0, 0.0, 0.0],
        dtype=torch.float32,
        device="cuda",
    )
    pipeline = SimpleNamespace(
        convert_SHs_python=False,
        compute_cov3D_python=False,
        debug=False,
        antialiasing=False,
    )
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        result = render(
            camera,
            gaussians,
            pipeline,
            background,
            use_trained_exp=False,
            separate_sh=False,
            return_expected_camera_z_packet=True,
            opacity_epsilon=float(args.opacity_epsilon),
            variance_clamp_tolerance=float(args.variance_clamp_tolerance),
        )
    arrays = packet_tensor_to_named_arrays(
        result["expected_camera_z_packet"],
        opacity_epsilon=float(args.opacity_epsilon),
        variance_clamp_tolerance=float(args.variance_clamp_tolerance),
    )
    validation = validate_named_packet(arrays, (openmvs["height"], openmvs["width"]))
    output.mkdir(parents=True, exist_ok=True)
    npz_path = output / f"{normalized_stem(args.target)}_{args.method_label}_expected_camera_z_packet.npz"
    deterministic_npz(npz_path, arrays)
    manifest = {
        "schema": packet_schema(),
        "bridge_schema": BRIDGE_SCHEMA,
        "scene": args.scene,
        "target": args.target,
        "method_label": args.method_label,
        "run_label": args.run_label,
        "iteration": args.iteration,
        "canonical_band": CANONICAL_BAND,
        "model_path": model_dir.as_posix(),
        "model_artifact_kind": "standard_ramp_model_g_ply",
        "checkpoint_sha256": preflight["canonical_ply_sha256"],
        "canonical_ply_sha256": preflight["canonical_ply_sha256"],
        "four_band_geometry_identity_sha256": preflight["canonical_geometry_identity_sha256"],
        "four_band_geometry_identical": True,
        "raster_height": openmvs["height"],
        "raster_width": openmvs["width"],
        "target_camera_fingerprint": fingerprint,
        "target_camera_fingerprint_comparison": comparison,
        "heldout_manifest_sha256": sha256_file(args.heldout_manifest),
        "openmvs_packet": openmvs,
        "frame_scale_decision": frame_scale,
        "depth_semantics": "sum(T_i*alpha_i*z_i)/sum(T_i*alpha_i)",
        "fixed_rasterizer_contribution_cutoff": "alpha>=1/255",
        "fixed_rasterizer_early_termination": "remaining_transmittance<=1e-4",
        "method_specific_alignment_used": False,
        "packet_validation": validation,
        "npz_path": npz_path.name,
        "npz_sha256": sha256_file(npz_path),
        "array_content_sha256": canonical_json_sha256({name: _array_sha256([arrays[name]]) for name in EXPECTED_PACKET_ARRAYS}),
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
    }
    write_json(output / "expected_camera_z_packet_manifest.json", manifest)
    return manifest


def load_packet(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        arrays = {name: np.asarray(data[name]) for name in data.files}
    require(set(arrays) == set(EXPECTED_PACKET_ARRAYS), "packet_keyset_mismatch", str(sorted(arrays)))
    return arrays


def paired_masks(
    m1: Mapping[str, np.ndarray],
    m2: Mapping[str, np.ndarray],
    openmvs_depth: np.ndarray,
    openmvs_valid: np.ndarray,
) -> dict[str, np.ndarray]:
    m1_z = np.asarray(m1["expected_camera_z"])
    m2_z = np.asarray(m2["expected_camera_z"])
    om_z = np.asarray(openmvs_depth)
    om_valid = np.asarray(openmvs_valid).astype(bool)
    require(m1_z.shape == m2_z.shape == om_z.shape == om_valid.shape, "paired_packet_shape_mismatch", "")
    primary = (
        om_valid
        & np.asarray(m1["numeric_valid"]).astype(bool)
        & np.asarray(m2["numeric_valid"]).astype(bool)
        & np.isfinite(om_z)
        & np.isfinite(m1_z)
        & np.isfinite(m2_z)
        & (om_z > 0)
        & (m1_z > 0)
        & (m2_z > 0)
    )
    opacity = primary & (np.asarray(m1["accumulated_opacity"]) >= 0.5) & (np.asarray(m2["accumulated_opacity"]) >= 0.5)
    m1_var = np.asarray(m1["camera_z_variance"])
    m2_var = np.asarray(m2["camera_z_variance"])
    variance = primary & np.isfinite(m1_var) & np.isfinite(m2_var) & (m1_var >= 0) & (m2_var >= 0)
    return {"paired_primary": primary, "paired_opacity_ge_0_5": opacity, "paired_variance_valid": variance}


def _candidate_metrics(reference: np.ndarray, candidate: np.ndarray, mask: np.ndarray, gradient_domain: Any) -> dict[str, Any]:
    from tools.depth_reference_geometry_v2.evaluate_umgs_openmvs_camera_z_proxy_alignment import descriptive_native_metrics
    from tools.depth_reference_geometry_v2.openmvs_da3_overlap_corrected import metrics_on_mask

    raw = metrics_on_mask(reference, candidate, mask, gradient_domain=gradient_domain)
    descriptive = descriptive_native_metrics(candidate, reference, mask)
    active = mask & np.isfinite(reference) & np.isfinite(candidate) & (reference > 0) & (candidate > 0)
    if active.any():
        ref = reference[active].astype(np.float64)
        cand = candidate[active].astype(np.float64)
        rel = np.abs(cand - ref) / np.maximum(np.abs(ref), 1e-6)
        mean_absrel = float(np.mean(rel))
    else:
        mean_absrel = None
    return {
        "pixels": raw["pixels"],
        "coverage": raw["coverage"],
        "mean_absrel": mean_absrel,
        "median_absrel": raw["absrel_median"],
        "p90_absrel": raw["absrel_p90"],
        "rmse": descriptive["rmse_camera_z_disagreement"],
        "median_relative_error": raw["absrel_median"],
        "spearman": raw["spearman"],
        "high_gradient_cosine": raw["high_gradient_cosine_median"],
        "high_gradient_pixels": raw["high_gradient_pixels"],
    }


def evaluate_paired_packets(
    m1_packet: str | Path,
    m2_packet: str | Path,
    openmvs_packet: str | Path,
    frame_scale_decision: str | Path,
    scene: str,
    target: str,
) -> dict[str, Any]:
    from tools.depth_reference_geometry_v2.openmvs_da3_overlap_corrected import reference_high_gradient_domain

    frame_scale = validate_frame_scale_decision(frame_scale_decision, scene, target)
    m1 = load_packet(m1_packet)
    m2 = load_packet(m2_packet)
    with np.load(openmvs_packet, allow_pickle=False) as data:
        reference = np.asarray(data["depth"])
        openmvs_valid = np.asarray(data["valid"])
    masks = paired_masks(m1, m2, reference, openmvs_valid)
    result_masks: dict[str, Any] = {}
    for mask_name, mask in masks.items():
        domain = reference_high_gradient_domain(reference, mask)
        m1_metrics = _candidate_metrics(reference, np.asarray(m1["expected_camera_z"]), mask, domain)
        m2_metrics = _candidate_metrics(reference, np.asarray(m2["expected_camera_z"]), mask, domain)
        deltas = {
            key: (None if m1_metrics[key] is None or m2_metrics[key] is None else float(m2_metrics[key] - m1_metrics[key]))
            for key in ("mean_absrel", "median_absrel", "rmse", "spearman", "high_gradient_cosine")
        }
        result_masks[mask_name] = {
            "support_pixels": int(mask.sum()),
            "support_coverage": float(mask.mean()) if mask.size else 0.0,
            "m1": m1_metrics,
            "m2": m2_metrics,
            "m2_minus_m1": deltas,
        }
    m1_native = (
        np.asarray(openmvs_valid).astype(bool)
        & np.asarray(m1["numeric_valid"]).astype(bool)
        & np.isfinite(reference)
        & np.isfinite(m1["expected_camera_z"])
        & (reference > 0)
        & (m1["expected_camera_z"] > 0)
    )
    m2_native = (
        np.asarray(openmvs_valid).astype(bool)
        & np.asarray(m2["numeric_valid"]).astype(bool)
        & np.isfinite(reference)
        & np.isfinite(m2["expected_camera_z"])
        & (reference > 0)
        & (m2["expected_camera_z"] > 0)
    )
    return {
        "schema": PAIRED_EVALUATION_SCHEMA,
        "scene": scene,
        "target": target,
        "depth_semantics": "opacity-composited expected camera-z",
        "frame_scale_decision": frame_scale,
        "method_specific_alignment_used": False,
        "paired_common_mask_used": True,
        "native_coverage": {
            "m1": float(m1_native.mean()),
            "m2": float(m2_native.mean()),
        },
        "masks": result_masks,
        "input_sha256": {
            "m1_packet": sha256_file(m1_packet),
            "m2_packet": sha256_file(m2_packet),
            "openmvs_packet": sha256_file(openmvs_packet),
        },
    }


def _cmd_preflight(args: argparse.Namespace) -> None:
    payload = preflight_ramp_method(args.method_dir, args.scene, iteration=args.iteration)
    if args.output_json:
        output = Path(args.output_json)
        require(not output.exists(), "output_overwrite_forbidden", str(output))
        output.parent.mkdir(parents=True, exist_ok=True)
        write_json(output, payload)
    print(json.dumps(payload, sort_keys=True))


def _cmd_preregister(args: argparse.Namespace) -> None:
    payload = preregister_a0(args.heldout_manifest, args.openmvs_root, args.output_json)
    print(json.dumps(payload, sort_keys=True))


def _cmd_render(args: argparse.Namespace) -> None:
    payload = render_packet(args)
    print(json.dumps(payload, sort_keys=True))


def _cmd_evaluate(args: argparse.Namespace) -> None:
    payload = evaluate_paired_packets(
        args.m1_packet,
        args.m2_packet,
        args.openmvs_packet,
        args.frame_scale_decision,
        args.scene,
        args.target,
    )
    output = Path(args.output_json)
    require(not output.exists(), "output_overwrite_forbidden", str(output))
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, payload)
    if args.output_csv:
        csv_path = Path(args.output_csv)
        require(not csv_path.exists(), "output_overwrite_forbidden", str(csv_path))
        rows: list[dict[str, Any]] = []
        for mask_name, record in payload["masks"].items():
            for method in ("m1", "m2"):
                rows.append({"scene": args.scene, "target": args.target, "mask": mask_name, "method": method, **record[method]})
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(payload, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RAMP expected-camera-z standard-model bridge")
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--method-dir", required=True)
    preflight.add_argument("--scene", required=True)
    preflight.add_argument("--iteration", type=int, default=REQUIRED_ITERATION)
    preflight.add_argument("--output-json")
    preflight.set_defaults(func=_cmd_preflight)

    preregister = subparsers.add_parser("preregister-a0")
    preregister.add_argument("--heldout-manifest", required=True)
    preregister.add_argument("--openmvs-root", required=True)
    preregister.add_argument("--output-json", required=True)
    preregister.set_defaults(func=_cmd_preregister)

    render = subparsers.add_parser("render")
    render.add_argument("--method-dir", required=True)
    render.add_argument("--scene", required=True)
    render.add_argument("--method-label", choices=("M1", "M2"), required=True)
    render.add_argument("--target", required=True)
    render.add_argument("--run-label", required=True)
    render.add_argument("--heldout-manifest", required=True)
    render.add_argument("--openmvs-packet", required=True)
    render.add_argument("--frame-scale-decision", required=True)
    render.add_argument("--expected-camera-fingerprint", required=True)
    render.add_argument("--camera-fingerprint-atol", type=float, default=1e-8)
    render.add_argument("--output", required=True)
    render.add_argument("--iteration", type=int, default=REQUIRED_ITERATION)
    render.add_argument("--opacity-epsilon", type=float, default=1e-6)
    render.add_argument("--variance-clamp-tolerance", type=float, default=1e-6)
    render.set_defaults(func=_cmd_render)

    evaluate = subparsers.add_parser("evaluate-paired")
    evaluate.add_argument("--m1-packet", required=True)
    evaluate.add_argument("--m2-packet", required=True)
    evaluate.add_argument("--openmvs-packet", required=True)
    evaluate.add_argument("--frame-scale-decision", required=True)
    evaluate.add_argument("--scene", required=True)
    evaluate.add_argument("--target", required=True)
    evaluate.add_argument("--output-json", required=True)
    evaluate.add_argument("--output-csv")
    evaluate.set_defaults(func=_cmd_evaluate)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except BridgeContractError as exc:
        print(json.dumps({"status": "failed", "failure_counter": exc.code}, sort_keys=True))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
