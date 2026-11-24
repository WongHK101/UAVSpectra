#!/usr/bin/env python3
"""Freeze and execute the external-five E3 geometry-proxy evaluation.

The registry is frozen before E3 rendering and binds the method-independent
OpenMVS references, held-out cameras, canonical image grids, and both method
assets.  Rendering never estimates a method-specific scale, shift, crop, or
alignment.  Results remain validation-gated engineering proxies rather than
absolute geometry ground truth.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from tools.depth_reference_geometry_v2.external5_geometry_e1 import (  # noqa: E402
    _array_sha256,
    _convert_mss_packet_to_source_units,
    _load_npz_exact,
    _packet_array_sha256,
    _renderer_source_sha256,
    apply_dataparser_transform,
    canonical_json_sha256,
    load_dataparser_transform,
    load_mss_checkpoint_geometry,
    load_umgs_ply_geometry,
    make_geometry_model,
    name_set_sha256,
    opengl_c2w_to_opencv_c2w,
    parse_undistorted_camera_row,
    sha256_file,
    validate_protocol_template,
    write_json,
)


REGISTRY_CONFIG_SCHEMA = "external5_geometry_e3_registry_config_v1"
REGISTRY_SCHEMA = "external5_geometry_e3_frozen_registry_v1"
REFERENCE_MANIFEST_SCHEMA = "external5_geometry_e3_reference_scene_manifest_v1"
METHOD_MANIFEST_SCHEMA = "external5_geometry_e3_method_scene_manifest_v1"
SCENE_EVALUATION_SCHEMA = "external5_geometry_e3_scene_evaluation_v1"
AGGREGATE_SCHEMA = "external5_geometry_e3_aggregate_v1"
SCENE_ORDER = (
    "aerial_golf",
    "aerial_lake",
    "aerial_solar",
    "vineyard_01",
    "vineyard_03",
)
STRATA = {
    "aerial_primary": ("aerial_golf", "aerial_lake", "aerial_solar"),
    "vineyard_matched_subset": ("vineyard_01", "vineyard_03"),
    "all_five_secondary": SCENE_ORDER,
}
METHODS = ("umgs_i", "ms_splatting")
MASKS = ("paired_primary", "paired_opacity_ge_0_5", "paired_variance_valid")
METRICS = (
    "mean_absrel",
    "median_absrel",
    "p90_absrel",
    "rmse",
    "spearman",
    "high_gradient_cosine",
)
LOWER_IS_BETTER = {"mean_absrel", "median_absrel", "p90_absrel", "rmse"}
FORBIDDEN_ALIGNMENT_KEYS = {
    "method_scale",
    "method_shift",
    "scale_fit",
    "shift_fit",
    "crop_fit",
    "per_method_alignment",
    "umgs_alignment",
    "mss_alignment",
}


class E3ContractError(RuntimeError):
    def __init__(self, counter: str, detail: str = ""):
        super().__init__(f"{counter}: {detail}" if detail else counter)
        self.counter = counter


def require(condition: bool, counter: str, detail: str = "") -> None:
    if not condition:
        raise E3ContractError(counter, detail)


def _utf8_sorted(values: Iterable[str]) -> list[str]:
    return sorted(values, key=lambda value: value.encode("utf-8"))


def _read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    require(isinstance(payload, dict), "json_root_not_object", str(path))
    return payload


def _binding(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    require(source.is_file(), "missing_bound_file", str(source))
    return {
        "path": source.as_posix(),
        "size_bytes": int(source.stat().st_size),
        "sha256": sha256_file(source),
    }


def _validate_binding(record: Mapping[str, Any], counter: str) -> Path:
    path = Path(str(record.get("path", "")))
    require(path.is_file(), counter, str(path))
    require(int(record.get("size_bytes", -1)) == path.stat().st_size, counter, str(path))
    require(str(record.get("sha256", "")) == sha256_file(path), counter, str(path))
    return path


def _require_expected_file(record: Mapping[str, Any], label: str) -> dict[str, Any]:
    path = Path(str(record.get("path", "")))
    require(path.is_file(), f"missing_{label}", str(path))
    actual = _binding(path)
    expected_sha = record.get("sha256")
    expected_size = record.get("size_bytes")
    if expected_sha is not None:
        require(str(expected_sha) == actual["sha256"], f"{label}_sha256_mismatch", str(path))
    if expected_size is not None:
        require(int(expected_size) == actual["size_bytes"], f"{label}_size_mismatch", str(path))
    return actual


def _assert_new_output(path: str | Path) -> Path:
    output = Path(path)
    require(not output.exists() or not any(output.iterdir()), "output_overwrite_forbidden", str(output))
    output.mkdir(parents=True, exist_ok=True)
    return output


def _assert_output_external(output: Path, source_paths: Sequence[Path]) -> None:
    resolved_output = output.resolve()
    for source in source_paths:
        resolved_source = source.resolve()
        require(
            resolved_output != resolved_source
            and resolved_source not in resolved_output.parents
            and resolved_output not in resolved_source.parents,
            "output_source_path_overlap",
            f"{resolved_output} <-> {resolved_source}",
        )


def _walk_keys(payload: Any):
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            yield str(key).casefold()
            yield from _walk_keys(value)
    elif isinstance(payload, list):
        for value in payload:
            yield from _walk_keys(value)


def _source_cameras(heldout: Mapping[str, Any]) -> list[dict[str, Any]]:
    from utils.read_write_model import qvec2rotmat

    cameras = heldout.get("cameras")
    require(isinstance(cameras, list) and bool(cameras), "heldout_camera_manifest_malformed", "")
    rows: list[dict[str, Any]] = []
    for row in cameras:
        name = str(row.get("image_name", ""))
        require(name and Path(name).name == name, "heldout_camera_name_malformed", name)
        qvec = np.asarray(row.get("qvec"), dtype=np.float64)
        tvec = np.asarray(row.get("tvec"), dtype=np.float64)
        require(qvec.shape == (4,) and tvec.shape == (3,), "heldout_camera_pose_malformed", name)
        require(np.isfinite(qvec).all() and np.isfinite(tvec).all(), "heldout_camera_pose_nonfinite", name)
        w2c = np.eye(4, dtype=np.float64)
        w2c[:3, :3] = qvec2rotmat(qvec)
        w2c[:3, 3] = tvec
        c2w_opencv = np.linalg.inv(w2c)
        flip = np.diag([1.0, -1.0, -1.0, 1.0])
        c2w_opengl = c2w_opencv @ flip
        rows.append(
            {
                "image_id": int(row.get("image_id", len(rows) + 1)),
                "image_name": name,
                "qvec": qvec.tolist(),
                "tvec": tvec.tolist(),
                "source_c2w_opencv": c2w_opencv.tolist(),
                "source_c2w_opengl": c2w_opengl.tolist(),
            }
        )
    rows.sort(key=lambda row: row["image_name"].encode("utf-8"))
    names = [row["image_name"] for row in rows]
    require(len(names) == len(set(name.casefold() for name in names)), "duplicate_heldout_camera", "")
    return rows


def _identity_hashes(identity: Mapping[str, Any]) -> tuple[str | None, str | None, str | None]:
    split = identity.get("split_hashes")
    if isinstance(split, Mapping):
        return str(split.get("train")), str(split.get("test")), str(split.get("all"))
    return (
        str(identity.get("train_name_sha256")) if identity.get("train_name_sha256") else None,
        str(identity.get("test_name_sha256")) if identity.get("test_name_sha256") else None,
        str(identity.get("all_name_sha256")) if identity.get("all_name_sha256") else None,
    )


def _validate_qualification(
    scene: str,
    protocol_row: Mapping[str, Any],
    config_row: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    qualification_path = Path(str(config_row["qualification_final_path"]))
    camera_stage_path = Path(str(config_row["camera_stage_path"]))
    heldout_path = Path(str(config_row["heldout_manifest_path"]))
    identity_path = Path(str(config_row["qualification_input_identity_path"]))
    qualification = _read_json(qualification_path)
    camera_stage = _read_json(camera_stage_path)
    heldout = _read_json(heldout_path)
    identity = _read_json(identity_path)

    require(qualification.get("qualification_pass") is True, "reference_qualification_not_passed", scene)
    require(qualification.get("heldout_isolation_pass") is True, "heldout_leakage", scene)
    require(qualification.get("method_independent_reference_build") is True, "reference_not_method_independent", scene)
    require(qualification.get("method_output_used_for_reference_build") is False, "reference_uses_method_output", scene)
    require(qualification.get("source_asset_unchanged") is True, "reference_source_modified", scene)
    require(int(qualification.get("train_view_count", -1)) == int(protocol_row["train_count"]), "train_count_mismatch", scene)
    require(int(qualification.get("heldout_view_count", -1)) == int(protocol_row["test_count"]), "test_count_mismatch", scene)
    require(int(qualification.get("heldout_view_count_used_for_neighbors", -1)) == 0, "heldout_neighbor_leakage", scene)
    require(camera_stage.get("camera_stage_pass") is True, "camera_stage_not_passed", scene)
    require(camera_stage.get("source_unchanged") is True, "camera_stage_source_modified", scene)
    require(int(camera_stage.get("heldout_image_record_in_undistorted_model_count", -1)) == 0, "heldout_camera_leakage", scene)
    require(int(camera_stage.get("heldout_rgb_in_undistorted_output_count", -1)) == 0, "heldout_rgb_leakage", scene)
    camera_rows = camera_stage.get("undistorted_camera_rows")
    require(isinstance(camera_rows, list) and len(camera_rows) == 1, "canonical_camera_row_count_mismatch", scene)
    canonical_camera = parse_undistorted_camera_row(str(camera_rows[0]))

    cameras = _source_cameras(heldout)
    target_names = [row["image_name"] for row in cameras]
    require(len(cameras) == int(protocol_row["test_count"]), "heldout_camera_count_mismatch", scene)
    target_hash = name_set_sha256(target_names)
    train_hash, identity_test_hash, all_hash = _identity_hashes(identity)
    require(int(identity.get("train_count", -1)) == int(protocol_row["train_count"]), "identity_train_count_mismatch", scene)
    require(int(identity.get("test_count", -1)) == int(protocol_row["test_count"]), "identity_test_count_mismatch", scene)
    require(identity_test_hash == target_hash, "heldout_manifest_hash_mismatch", scene)
    if protocol_row.get("train_name_sha256"):
        require(train_hash == protocol_row["train_name_sha256"], "train_name_hash_mismatch", scene)
    if protocol_row.get("test_name_sha256"):
        require(target_hash == protocol_row["test_name_sha256"], "test_name_hash_mismatch", scene)
    if protocol_row.get("all_name_sha256") and all_hash:
        require(all_hash == protocol_row["all_name_sha256"], "all_name_hash_mismatch", scene)

    qualification_binding = _binding(qualification_path)
    camera_stage_binding = _binding(camera_stage_path)
    heldout_binding = _binding(heldout_path)
    identity_binding = _binding(identity_path)
    return (
        {
            "qualification_final": qualification_binding,
            "camera_stage": camera_stage_binding,
            "heldout_manifest": heldout_binding,
            "qualification_input_identity": identity_binding,
            "qualification_schema": qualification.get("schema"),
            "claim_boundary": qualification.get("claim_boundary"),
            "source_inventory_sha256_before": qualification.get("source_inventory_sha256_before"),
            "source_inventory_sha256_after": qualification.get("source_inventory_sha256_after"),
            "source_unchanged": qualification.get("source_inventory_sha256_before")
            == qualification.get("source_inventory_sha256_after"),
        },
        canonical_camera,
        {"rows": cameras, "name_sha256": target_hash},
        {"train": train_hash, "test": target_hash, "all": all_hash},
    )


def _freeze_method_sources(scene: str, config_row: Mapping[str, Any], cameras: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    method_config = config_row.get("methods")
    require(isinstance(method_config, Mapping) and set(method_config) == set(METHODS), "method_source_set_mismatch", scene)
    frozen: dict[str, Any] = {}
    for method in METHODS:
        source = dict(method_config[method])
        kind = str(source.get("kind", ""))
        allowed = {"standard_gaussian_ply"} if method == "umgs_i" else {"standard_gaussian_ply", "mss_nerfstudio_checkpoint"}
        require(kind in allowed, "method_source_kind_mismatch", f"{scene}:{method}:{kind}")
        asset = _require_expected_file(source, f"{scene}_{method}_source")
        record: dict[str, Any] = {
            "kind": kind,
            "asset": asset,
            "camera_frame": "source",
            "camera_z_to_source_units": 1.0,
        }
        if kind == "mss_nerfstudio_checkpoint":
            dataparser_source = source.get("dataparser")
            require(isinstance(dataparser_source, Mapping), "missing_mss_dataparser", scene)
            dataparser_asset = _require_expected_file(dataparser_source, f"{scene}_mss_dataparser")
            parsed = load_dataparser_transform(dataparser_asset["path"])
            max_roundtrip = 0.0
            for camera in cameras:
                source_c2w = np.asarray(camera["source_c2w_opengl"], dtype=np.float64)
                transformed = apply_dataparser_transform(source_c2w, parsed["matrix"], parsed["scale"])
                unscaled = np.array(transformed, copy=True)
                unscaled[:3, 3] /= parsed["scale"]
                roundtrip = np.linalg.inv(parsed["matrix"]) @ unscaled
                max_roundtrip = max(max_roundtrip, float(np.max(np.abs(roundtrip - source_c2w))))
            require(max_roundtrip <= 1e-7, "dataparser_camera_roundtrip_mismatch", f"{scene}:{max_roundtrip}")
            record.update(
                {
                    "camera_frame": "frozen_dataparser",
                    "dataparser_asset": dataparser_asset,
                    "dataparser_transform": parsed["matrix"].tolist(),
                    "dataparser_scale": parsed["scale"],
                    "camera_z_to_source_units": 1.0 / parsed["scale"],
                    "camera_roundtrip_max_abs_error": max_roundtrip,
                    "orthonormal_max_abs_error": parsed["orthonormal_max_abs_error"],
                    "rotation_determinant": parsed["rotation_determinant"],
                }
            )
        frozen[method] = record
    return frozen


def freeze_registry(config_path: str | Path, output_json: str | Path) -> dict[str, Any]:
    output = Path(output_json)
    require(not output.exists(), "output_overwrite_forbidden", str(output))
    config = _read_json(config_path)
    require(config.get("schema") == REGISTRY_CONFIG_SCHEMA, "registry_config_schema_mismatch", str(config_path))
    keys = set(_walk_keys(config))
    require(not (keys & FORBIDDEN_ALIGNMENT_KEYS), "method_specific_alignment_forbidden", str(keys & FORBIDDEN_ALIGNMENT_KEYS))

    protocol_path = Path(str(config.get("protocol_path", "")))
    protocol = validate_protocol_template(protocol_path)
    protocol_sha = sha256_file(protocol_path)
    require(protocol_sha == config.get("protocol_sha256"), "protocol_sha256_mismatch", str(protocol_path))
    require(protocol.get("metric_read_before_freeze") is False, "registry_freeze_after_metric_read", "")
    require(protocol.get("created_before_first_method_render") is True, "protocol_not_pre_render_frozen", "")
    require(protocol["camera_contract"]["method_specific_scale_shift_crop_alignment"] is False, "method_specific_alignment_forbidden", "")

    runtime_config = config.get("runtime_sources")
    require(isinstance(runtime_config, Mapping) and bool(runtime_config), "runtime_source_registry_missing", "")
    runtime_sources = {
        name: _require_expected_file(record, f"runtime_source_{name}")
        for name, record in sorted(runtime_config.items(), key=lambda item: item[0].encode("utf-8"))
    }

    config_scenes = config.get("scenes")
    require(isinstance(config_scenes, list), "registry_scene_list_malformed", "")
    require(tuple(row.get("scene") for row in config_scenes) == SCENE_ORDER, "registry_scene_order_mismatch", "")
    protocol_rows = {row["scene"]: row for row in protocol["scenes"]}
    require(set(protocol_rows) == set(SCENE_ORDER), "protocol_scene_set_mismatch", "")

    frozen_scenes: list[dict[str, Any]] = []
    for config_row in config_scenes:
        scene = str(config_row["scene"])
        protocol_row = protocol_rows[scene]
        qualification, canonical_camera, heldout, split_hashes = _validate_qualification(scene, protocol_row, config_row)
        mesh = _require_expected_file(config_row["reference_mesh"], f"{scene}_reference_mesh")
        qualification_json = _read_json(config_row["qualification_final_path"])
        if qualification_json.get("mesh_sha256"):
            require(qualification_json["mesh_sha256"] == mesh["sha256"], "qualification_mesh_sha256_mismatch", scene)
        if qualification_json.get("mesh_path"):
            require(Path(qualification_json["mesh_path"]).resolve() == Path(mesh["path"]).resolve(), "qualification_mesh_path_mismatch", scene)
        methods = _freeze_method_sources(scene, config_row, heldout["rows"])
        frozen_scenes.append(
            {
                "scene": scene,
                "stratum": protocol_row["stratum"],
                "train_count": int(protocol_row["train_count"]),
                "test_count": int(protocol_row["test_count"]),
                "split_name_sha256": split_hashes,
                "canonical_camera": {
                    **canonical_camera,
                    "FoVx": float(2.0 * math.atan(canonical_camera["width"] / (2.0 * canonical_camera["fx"]))),
                    "FoVy": float(2.0 * math.atan(canonical_camera["height"] / (2.0 * canonical_camera["fy"]))),
                    "grid_authority": "method-independent qualified COLMAP undistorter output",
                },
                "heldout_cameras": heldout["rows"],
                "heldout_name_sha256": heldout["name_sha256"],
                "reference": {"mesh": mesh, "qualification": qualification},
                "methods": methods,
            }
        )

    registry = {
        "schema": REGISTRY_SCHEMA,
        "claim_boundary": "validation-gated source-only OpenMVS engineering proxy; not absolute geometry ground truth",
        "protocol": {**_binding(protocol_path), "schema": protocol.get("schema")},
        "registry_config": _binding(config_path),
        "runtime_sources": runtime_sources,
        "created_before_first_e3_method_render": True,
        "metric_read_before_registry_freeze": False,
        "reference_generation_uses_method_output": False,
        "method_specific_scale_shift_crop_alignment": False,
        "target_weighting": "equal within scene",
        "scene_weighting": "equal within stratum",
        "scene_order": list(SCENE_ORDER),
        "strata": {name: list(scenes) for name, scenes in STRATA.items()},
        "scenes": frozen_scenes,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, registry)
    return {"registry_path": output.as_posix(), "registry_sha256": sha256_file(output), "scene_count": len(frozen_scenes)}


def load_registry(path: str | Path) -> dict[str, Any]:
    registry = _read_json(path)
    require(registry.get("schema") == REGISTRY_SCHEMA, "registry_schema_mismatch", str(path))
    require(registry.get("method_specific_scale_shift_crop_alignment") is False, "method_specific_alignment_forbidden", str(path))
    require(registry.get("metric_read_before_registry_freeze") is False, "registry_freeze_after_metric_read", str(path))
    require(tuple(registry.get("scene_order", [])) == SCENE_ORDER, "registry_scene_order_mismatch", str(path))
    require(tuple(row.get("scene") for row in registry.get("scenes", [])) == SCENE_ORDER, "registry_scene_set_mismatch", str(path))
    _validate_binding(registry["protocol"], "registry_protocol_binding_mismatch")
    _validate_binding(registry["registry_config"], "registry_config_binding_mismatch")
    runtime_sources = registry.get("runtime_sources")
    require(isinstance(runtime_sources, Mapping) and bool(runtime_sources), "runtime_source_registry_missing", str(path))
    for name, record in runtime_sources.items():
        _validate_binding(record, f"runtime_source_{name}_binding_mismatch")
    return registry


def _scene_record(registry: Mapping[str, Any], scene: str) -> Mapping[str, Any]:
    matches = [row for row in registry["scenes"] if row["scene"] == scene]
    require(len(matches) == 1, "registry_scene_identity_mismatch", scene)
    return matches[0]


def _packet_relpath(target: str) -> str:
    require(Path(target).name == target and target not in {".", ".."}, "target_name_malformed", target)
    return f"packets/{target}.npz"


def _build_renderer_camera(scene: Mapping[str, Any], target: Mapping[str, Any], method: str):
    import torch
    from utils.graphics_utils import getProjectionMatrix, getWorld2View2

    c2w_opengl = np.asarray(target["source_c2w_opengl"], dtype=np.float64)
    source = scene["methods"][method]
    if source["camera_frame"] == "frozen_dataparser":
        c2w_opengl = apply_dataparser_transform(
            c2w_opengl,
            np.asarray(source["dataparser_transform"], dtype=np.float64),
            float(source["dataparser_scale"]),
        )
    else:
        require(source["camera_frame"] == "source", "unsupported_camera_frame", str(source["camera_frame"]))
    c2w_opencv = opengl_c2w_to_opencv_c2w(c2w_opengl)
    w2c = np.linalg.inv(c2w_opencv)
    r = w2c[:3, :3].T
    t = w2c[:3, 3]
    camera = scene["canonical_camera"]
    fovx, fovy = float(camera["FoVx"]), float(camera["FoVy"])
    znear, zfar = 0.01, 1000.0
    world = torch.tensor(getWorld2View2(r, t), dtype=torch.float32, device="cuda").transpose(0, 1)
    projection = getProjectionMatrix(znear=znear, zfar=zfar, fovX=fovx, fovY=fovy).to("cuda").transpose(0, 1)
    full = world.unsqueeze(0).bmm(projection.unsqueeze(0)).squeeze(0)
    return SimpleNamespace(
        image_name=target["image_name"],
        colmap_id=int(target["image_id"]),
        uid=int(target["image_id"]),
        image_width=int(camera["width"]),
        image_height=int(camera["height"]),
        FoVx=fovx,
        FoVy=fovy,
        znear=znear,
        zfar=zfar,
        R=r,
        T=t,
        world_view_transform=world,
        projection_matrix=projection,
        full_proj_transform=full,
        camera_center=torch.inverse(world)[3, :3],
        c2w_opencv=c2w_opencv,
    )


def render_reference_scene(registry_path: str | Path, scene_name: str, output_path: str | Path) -> dict[str, Any]:
    registry = load_registry(registry_path)
    scene = _scene_record(registry, scene_name)
    mesh_path = _validate_binding(scene["reference"]["mesh"], "reference_mesh_binding_mismatch")
    output = Path(output_path)
    _assert_output_external(output, [mesh_path, Path(registry_path)])
    output = _assert_new_output(output)

    from tools.depth_reference_geometry_v2.export_ramp_expected_camera_z_packets import deterministic_npz
    from tools.depth_reference_geometry_v2.openmvs_campaign_core import CameraView, load_ply_mesh, rasterize_mesh_camera_z

    before = sha256_file(mesh_path)
    vertices, faces, mesh_meta = load_ply_mesh(mesh_path)
    require(int(faces.shape[0]) > 0, "reference_mesh_has_no_faces", scene_name)
    packet_rows: list[dict[str, Any]] = []
    camera = scene["canonical_camera"]
    (output / "packets").mkdir()
    for target in scene["heldout_cameras"]:
        view = CameraView(
            image_id=int(target["image_id"]),
            image_name=target["image_name"],
            width=int(camera["width"]),
            height=int(camera["height"]),
            fx=float(camera["fx"]),
            fy=float(camera["fy"]),
            cx=float(camera["cx"]),
            cy=float(camera["cy"]),
            camera_to_world=np.asarray(target["source_c2w_opencv"], dtype=np.float64),
        )
        rendered = rasterize_mesh_camera_z(vertices, faces, view)
        depth = np.asarray(rendered.depth, dtype=np.float32)
        valid = np.asarray(rendered.valid, dtype=np.uint8)
        require(depth.shape == valid.shape == (view.height, view.width), "reference_packet_shape_mismatch", target["image_name"])
        require(bool(valid.any()), "reference_packet_has_no_valid_pixels", target["image_name"])
        require(bool(np.isfinite(depth[valid.astype(bool)]).all()), "reference_depth_nonfinite", target["image_name"])
        depth[~valid.astype(bool)] = np.nan
        relative = _packet_relpath(target["image_name"])
        packet_path = output / relative
        deterministic_npz(packet_path, {"depth": depth, "valid": valid})
        packet_rows.append(
            {
                "target": target["image_name"],
                "packet_file": relative,
                "packet_sha256": sha256_file(packet_path),
                "depth_array_sha256": _array_sha256(depth),
                "valid_pixels": int(valid.sum()),
                "valid_coverage": float(valid.mean()),
            }
        )
    after = sha256_file(mesh_path)
    require(before == after == scene["reference"]["mesh"]["sha256"], "reference_mesh_modified", scene_name)
    manifest = {
        "schema": REFERENCE_MANIFEST_SCHEMA,
        "scene": scene_name,
        "registry_sha256": sha256_file(registry_path),
        "claim_boundary": registry["claim_boundary"],
        "method_output_used": False,
        "method_specific_alignment_used": False,
        "target_count": len(packet_rows),
        "target_name_sha256": name_set_sha256([row["target"] for row in packet_rows]),
        "canonical_camera": camera,
        "mesh": mesh_meta,
        "mesh_sha256_before": before,
        "mesh_sha256_after": after,
        "source_unchanged": True,
        "packets": packet_rows,
    }
    manifest_path = output / "REFERENCE_SCENE_MANIFEST.json"
    write_json(manifest_path, manifest)
    return {"manifest_path": manifest_path.as_posix(), "manifest_sha256": sha256_file(manifest_path), "target_count": len(packet_rows)}


def load_mss_checkpoint_geometry_e3(path: str | Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load official MSS geometry while preserving complete-NaN scale sentinels.

    A complete NaN log-scale row is retained byte-for-byte and must later be
    shown to have zero rasterizer radius for every evaluated camera.  Partial
    NaNs, infinities, or non-finite values in any other field remain invalid.
    """
    import torch

    source = Path(path)
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    require(isinstance(checkpoint, Mapping) and "pipeline" in checkpoint, "malformed_mss_checkpoint", str(source))
    state = checkpoint["pipeline"]
    keys = {
        "xyz": "_model.gauss_params.means",
        "scaling": "_model.gauss_params.scales",
        "rotation": "_model.gauss_params.quats",
        "opacity": "_model.gauss_params.opacities",
    }
    missing = [key for key in keys.values() if key not in state]
    require(not missing, "mss_geometry_fields_missing", str(missing))
    arrays = {
        name: state[key].detach().cpu().to(torch.float32).contiguous().numpy()
        for name, key in keys.items()
    }
    count = int(arrays["xyz"].shape[0])
    expected_shapes = {
        "xyz": (count, 3),
        "scaling": (count, 3),
        "rotation": (count, 4),
        "opacity": (count, 1),
    }
    require(all(arrays[name].shape == shape for name, shape in expected_shapes.items()), "geometry_shape_mismatch", str({name: value.shape for name, value in arrays.items()}))
    for name in ("xyz", "rotation", "opacity"):
        require(bool(np.isfinite(arrays[name]).all()), "nonfinite_geometry", name)
    require(bool(np.all(np.linalg.norm(arrays["rotation"], axis=1) > 1e-12)), "zero_quaternion", str(source))
    scaling = arrays["scaling"]
    require(not bool(np.isinf(scaling).any()), "mss_log_scaling_inf_forbidden", str(source))
    nan_mask = np.isnan(scaling)
    complete_nan_rows = nan_mask.all(axis=1)
    require(bool(np.all(~nan_mask.any(axis=1) | complete_nan_rows)), "mss_partial_nan_log_scaling_forbidden", str(source))
    indices = np.flatnonzero(complete_nan_rows).astype(np.int64)
    finite_scaling = scaling[~complete_nan_rows]
    require(bool(np.isfinite(finite_scaling).all()), "nonfinite_geometry", "scaling")
    field_sha = {name: _array_sha256(arrays[name]) for name in ("xyz", "scaling", "rotation", "opacity")}
    identity = {
        "gaussian_count": count,
        "field_sha256": field_sha,
        "source_kind": "mss_nerfstudio_checkpoint",
        "raw_log_scaling_complete_nan_row_count": int(indices.size),
        "raw_log_scaling_complete_nan_row_indices_sha256": canonical_json_sha256(indices.tolist()),
    }
    manifest = {
        **identity,
        "geometry_identity_sha256": canonical_json_sha256(identity),
        "source_path": source.as_posix(),
        "source_size_bytes": source.stat().st_size,
        "source_sha256": sha256_file(source),
        "shapes": {name: list(shape) for name, shape in expected_shapes.items()},
        "checkpoint_step": int(checkpoint.get("step", -1)),
        "checkpoint_geometry_keys": keys,
        "raw_log_scaling_complete_nan_row_indices": indices.tolist(),
        "raw_log_scaling_policy": "preserve_exact_complete_nan_rows_and_require_zero_rasterizer_radius_for_every_target",
        "geometry_rows_filtered_or_rewritten": False,
    }
    return arrays, manifest


def _load_method_geometry(source: Mapping[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    path = _validate_binding(source["asset"], "method_source_binding_mismatch")
    if source["kind"] == "mss_nerfstudio_checkpoint":
        return load_mss_checkpoint_geometry_e3(path)
    arrays, manifest = load_umgs_ply_geometry(path)
    manifest = dict(manifest)
    manifest["source_kind"] = "standard_gaussian_ply"
    return arrays, manifest


def render_method_scene(
    registry_path: str | Path,
    scene_name: str,
    method: str,
    output_path: str | Path,
    opacity_epsilon: float = 1e-6,
    variance_clamp_tolerance: float = 1e-6,
) -> dict[str, Any]:
    require(method in METHODS, "unknown_method", method)
    registry = load_registry(registry_path)
    scene = _scene_record(registry, scene_name)
    source = scene["methods"][method]
    source_path = _validate_binding(source["asset"], "method_source_binding_mismatch")
    source_paths = [source_path, Path(registry_path)]
    if source.get("dataparser_asset"):
        source_paths.append(_validate_binding(source["dataparser_asset"], "dataparser_binding_mismatch"))
    output = Path(output_path)
    _assert_output_external(output, source_paths)
    output = _assert_new_output(output)

    import torch
    from gaussian_renderer import render
    from tools.depth_reference_geometry_v2.export_ramp_expected_camera_z_packets import deterministic_npz, validate_named_packet
    from tools.depth_reference_geometry_v2.export_umgs_expected_camera_z_packet import packet_schema, packet_tensor_to_named_arrays

    before = sha256_file(source_path)
    arrays, geometry = _load_method_geometry(source)
    model = make_geometry_model(arrays)
    background = torch.zeros(3, dtype=torch.float32, device="cuda")
    override_color = torch.full((int(geometry["gaussian_count"]), 3), 0.5, dtype=torch.float32, device="cuda")
    pipeline = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False, antialiasing=False)
    shape = (int(scene["canonical_camera"]["height"]), int(scene["canonical_camera"]["width"]))
    packet_rows: list[dict[str, Any]] = []
    nonfinite_scale_rows = np.asarray(
        geometry.get("raw_log_scaling_complete_nan_row_indices", []), dtype=np.int64
    )
    nonfinite_scale_rows_zero_radius_all_targets = True
    (output / "packets").mkdir()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for target in scene["heldout_cameras"]:
            camera = _build_renderer_camera(scene, target, method)
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
            if nonfinite_scale_rows.size:
                radii = rendered.get("radii")
                require(radii is not None, "mss_rasterizer_radii_missing", target["image_name"])
                row_indices = torch.as_tensor(
                    nonfinite_scale_rows,
                    dtype=torch.long,
                    device=radii.device,
                )
                bad_radii = radii.index_select(0, row_indices)
                zero_radius = bool((bad_radii == 0).all().item())
                require(zero_radius, "mss_nonfinite_scale_row_nonzero_radius", target["image_name"])
                nonfinite_scale_rows_zero_radius_all_targets &= zero_radius
            named = packet_tensor_to_named_arrays(
                rendered["expected_camera_z_packet"],
                opacity_epsilon=float(opacity_epsilon),
                variance_clamp_tolerance=float(variance_clamp_tolerance),
            )
            if source["camera_frame"] == "frozen_dataparser":
                named = _convert_mss_packet_to_source_units(named, float(source["dataparser_scale"]))
            validation = validate_named_packet(named, shape)
            relative = _packet_relpath(target["image_name"])
            packet_path = output / relative
            deterministic_npz(packet_path, named)
            packet_rows.append(
                {
                    "target": target["image_name"],
                    "packet_file": relative,
                    "packet_sha256": sha256_file(packet_path),
                    "packet_array_sha256": _packet_array_sha256(named),
                    "numeric_valid_pixels": validation["numeric_valid_pixels"],
                    "numeric_valid_coverage": validation["numeric_valid_coverage"],
                }
            )
            del rendered, named, camera
    torch.cuda.synchronize()
    after = sha256_file(source_path)
    require(before == after == source["asset"]["sha256"], "method_source_modified", f"{scene_name}:{method}")
    manifest = {
        "schema": METHOD_MANIFEST_SCHEMA,
        "scene": scene_name,
        "method": method,
        "registry_sha256": sha256_file(registry_path),
        "geometry": geometry,
        "source": source,
        "source_sha256_before": before,
        "source_sha256_after": after,
        "source_unchanged": True,
        "packet_schema": packet_schema(),
        "depth_semantics": "sum(T_i*alpha_i*z_i)/sum(T_i*alpha_i)",
        "camera_z_output_units": "source reconstruction units",
        "camera_z_unit_conversion": float(source["camera_z_to_source_units"]),
        "method_specific_alignment_used": False,
        "constant_override_color_used": True,
        "fixed_rasterizer_contribution_cutoff": "alpha>=1/255",
        "fixed_rasterizer_early_termination": "remaining_transmittance<=1e-4",
        "target_count": len(packet_rows),
        "target_name_sha256": name_set_sha256([row["target"] for row in packet_rows]),
        "packets": packet_rows,
        "nonfinite_scale_rows_zero_radius_all_targets": nonfinite_scale_rows_zero_radius_all_targets,
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
        "runtime": _renderer_source_sha256(),
    }
    manifest_path = output / "METHOD_SCENE_MANIFEST.json"
    write_json(manifest_path, manifest)
    return {"manifest_path": manifest_path.as_posix(), "manifest_sha256": sha256_file(manifest_path), "target_count": len(packet_rows)}


def _load_packet_manifest(
    root: Path,
    filename: str,
    schema: str,
    scene: Mapping[str, Any],
    registry_sha256: str,
) -> tuple[dict[str, Any], dict[str, Mapping[str, Any]]]:
    manifest_path = root / filename
    manifest = _read_json(manifest_path)
    require(manifest.get("schema") == schema, "packet_manifest_schema_mismatch", str(manifest_path))
    require(manifest.get("scene") == scene["scene"], "packet_manifest_scene_mismatch", str(manifest_path))
    require(manifest.get("registry_sha256") == registry_sha256, "packet_registry_binding_mismatch", str(manifest_path))
    require(manifest.get("method_specific_alignment_used") is False, "method_specific_alignment_forbidden", str(manifest_path))
    require(manifest.get("source_unchanged") is True, "packet_source_modified", str(manifest_path))
    rows = manifest.get("packets")
    require(isinstance(rows, list), "packet_manifest_rows_malformed", str(manifest_path))
    target_names = [row.get("target") for row in rows]
    expected_names = [row["image_name"] for row in scene["heldout_cameras"]]
    require(target_names == expected_names, "packet_target_set_mismatch", str(manifest_path))
    require(manifest.get("target_name_sha256") == scene["heldout_name_sha256"], "packet_target_hash_mismatch", str(manifest_path))
    expected_files = {_packet_relpath(name) for name in expected_names}
    actual_files = {
        path.relative_to(root).as_posix()
        for path in (root / "packets").rglob("*")
        if path.is_file()
    }
    require(actual_files == expected_files, "packet_file_set_mismatch", str(root))
    by_target: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        path = root / row["packet_file"]
        require(path.is_file() and sha256_file(path) == row["packet_sha256"], "packet_sha256_mismatch", str(path))
        by_target[str(row["target"])] = row
    return manifest, by_target


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
    return "umgs_i" if umgs_better else "ms_splatting"


def _aggregate_target_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    aggregate: dict[str, Any] = {}
    for mask in MASKS:
        mask_rows = [row for row in rows if row["mask"] == mask]
        methods: dict[str, Any] = {}
        for method in METHODS:
            method_rows = [row for row in mask_rows if row["method"] == method]
            methods[method] = {
                metric: {
                    "equal_target_mean": _finite_mean([row.get(metric) for row in method_rows]),
                    "equal_target_median": _finite_median([row.get(metric) for row in method_rows]),
                    "target_value_count": len([row for row in method_rows if row.get(metric) is not None]),
                }
                for metric in METRICS
            }
            methods[method]["native_coverage"] = {
                "equal_target_mean": _finite_mean([row.get("native_coverage") for row in method_rows]),
                "equal_target_median": _finite_median([row.get("native_coverage") for row in method_rows]),
                "target_value_count": len(method_rows),
            }
        directions: dict[str, Any] = {}
        for metric in METRICS:
            counts = {"umgs_i": 0, "ms_splatting": 0, "tie": 0, "unavailable": 0}
            by_target: dict[str, dict[str, Mapping[str, Any]]] = {}
            for row in mask_rows:
                by_target.setdefault(row["target"], {})[row["method"]] = row
            for target in _utf8_sorted(by_target):
                pair = by_target[target]
                outcome = _direction(pair.get("umgs_i", {}).get(metric), pair.get("ms_splatting", {}).get(metric), metric)
                counts[outcome] += 1
            directions[metric] = counts
        aggregate[mask] = {"methods": methods, "target_direction_counts": directions}
    return aggregate


def evaluate_scene(
    registry_path: str | Path,
    scene_name: str,
    reference_root: str | Path,
    umgs_root: str | Path,
    mss_root: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    registry = load_registry(registry_path)
    scene = _scene_record(registry, scene_name)
    output = _assert_new_output(output_path)
    reference_root = Path(reference_root)
    method_roots = {"umgs_i": Path(umgs_root), "ms_splatting": Path(mss_root)}
    registry_sha = sha256_file(registry_path)
    _validate_binding(scene["reference"]["mesh"], "reference_mesh_binding_mismatch")
    for source in scene["methods"].values():
        _validate_binding(source["asset"], "method_source_binding_mismatch")
        if source.get("dataparser_asset"):
            _validate_binding(source["dataparser_asset"], "dataparser_binding_mismatch")
    reference_manifest, reference_rows = _load_packet_manifest(
        reference_root,
        "REFERENCE_SCENE_MANIFEST.json",
        REFERENCE_MANIFEST_SCHEMA,
        scene,
        registry_sha,
    )
    method_manifests: dict[str, Any] = {}
    method_rows: dict[str, dict[str, Mapping[str, Any]]] = {}
    for method, root in method_roots.items():
        manifest, rows = _load_packet_manifest(
            root,
            "METHOD_SCENE_MANIFEST.json",
            METHOD_MANIFEST_SCHEMA,
            scene,
            registry_sha,
        )
        require(manifest.get("method") == method, "method_manifest_identity_mismatch", method)
        require(manifest.get("source_sha256_before") == scene["methods"][method]["asset"]["sha256"], "method_manifest_source_mismatch", method)
        require(manifest.get("source_sha256_after") == scene["methods"][method]["asset"]["sha256"], "method_manifest_source_mismatch", method)
        method_manifests[method] = manifest
        method_rows[method] = rows

    from tools.depth_reference_geometry_v2.export_ramp_expected_camera_z_packets import (
        EXPECTED_PACKET_ARRAYS,
        _candidate_metrics,
        paired_masks,
        validate_named_packet,
    )
    from tools.depth_reference_geometry_v2.openmvs_da3_overlap_corrected import reference_high_gradient_domain

    shape = (int(scene["canonical_camera"]["height"]), int(scene["canonical_camera"]["width"]))
    rows: list[dict[str, Any]] = []
    target_details: list[dict[str, Any]] = []
    for target in [row["image_name"] for row in scene["heldout_cameras"]]:
        reference = _load_npz_exact(reference_root / reference_rows[target]["packet_file"], {"depth", "valid"})
        packets = {
            method: _load_npz_exact(method_roots[method] / method_rows[method][target]["packet_file"], set(EXPECTED_PACKET_ARRAYS))
            for method in METHODS
        }
        depth = np.asarray(reference["depth"])
        valid = np.asarray(reference["valid"]).astype(bool)
        require(depth.shape == valid.shape == shape, "reference_packet_shape_mismatch", target)
        require(bool(np.isfinite(depth[valid]).all()), "reference_depth_nonfinite", target)
        for packet in packets.values():
            validate_named_packet(packet, shape)
        masks = paired_masks(packets["umgs_i"], packets["ms_splatting"], depth, valid)
        native: dict[str, float] = {}
        for method, packet in packets.items():
            native_mask = (
                valid
                & packet["numeric_valid"].astype(bool)
                & np.isfinite(depth)
                & np.isfinite(packet["expected_camera_z"])
                & (depth > 0)
                & (packet["expected_camera_z"] > 0)
            )
            native[method] = float(native_mask.mean())
        detail_masks: dict[str, Any] = {}
        for mask_name in MASKS:
            mask = masks[mask_name]
            domain = reference_high_gradient_domain(depth, mask)
            method_metrics: dict[str, Any] = {}
            for method in METHODS:
                metrics = _candidate_metrics(depth, packets[method]["expected_camera_z"], mask, domain)
                method_metrics[method] = metrics
                rows.append(
                    {
                        "scene": scene_name,
                        "stratum": scene["stratum"],
                        "target": target,
                        "mask": mask_name,
                        "method": method,
                        "support_pixels": int(mask.sum()),
                        "support_coverage": float(mask.mean()),
                        "native_coverage": native[method],
                        **{metric: metrics.get(metric) for metric in METRICS},
                    }
                )
            detail_masks[mask_name] = {
                "support_pixels": int(mask.sum()),
                "support_coverage": float(mask.mean()),
                "methods": method_metrics,
                "direction": {
                    metric: _direction(method_metrics["umgs_i"].get(metric), method_metrics["ms_splatting"].get(metric), metric)
                    for metric in METRICS
                },
            }
        target_details.append(
            {
                "target": target,
                "reference_packet_sha256": reference_rows[target]["packet_sha256"],
                "method_packet_sha256": {method: method_rows[method][target]["packet_sha256"] for method in METHODS},
                "native_coverage": native,
                "masks": detail_masks,
            }
        )

    aggregate = _aggregate_target_rows(rows)
    result = {
        "schema": SCENE_EVALUATION_SCHEMA,
        "scene": scene_name,
        "stratum": scene["stratum"],
        "claim_boundary": registry["claim_boundary"],
        "method_specific_alignment_used": False,
        "paired_common_mask_used": True,
        "target_weighting": "equal",
        "target_count": len(target_details),
        "target_name_sha256": scene["heldout_name_sha256"],
        "registry_sha256": sha256_file(registry_path),
        "input_manifest_sha256": {
            "reference": sha256_file(reference_root / "REFERENCE_SCENE_MANIFEST.json"),
            "umgs_i": sha256_file(method_roots["umgs_i"] / "METHOD_SCENE_MANIFEST.json"),
            "ms_splatting": sha256_file(method_roots["ms_splatting"] / "METHOD_SCENE_MANIFEST.json"),
        },
        "targets": target_details,
        "aggregate": aggregate,
    }
    write_json(output / "SCENE_EVALUATION.json", result)
    fieldnames = list(rows[0])
    with (output / "PER_TARGET_METRICS.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return {"scene": scene_name, "target_count": len(target_details), "result_sha256": sha256_file(output / "SCENE_EVALUATION.json")}


def _scene_metric(scene_result: Mapping[str, Any], mask: str, method: str, metric: str) -> float | None:
    return scene_result["aggregate"][mask]["methods"][method][metric]["equal_target_mean"]


def aggregate_scenes(registry_path: str | Path, scene_results: Sequence[str | Path], output_path: str | Path) -> dict[str, Any]:
    registry = load_registry(registry_path)
    output = _assert_new_output(output_path)
    results = [_read_json(path) for path in scene_results]
    require(all(row.get("schema") == SCENE_EVALUATION_SCHEMA for row in results), "scene_evaluation_schema_mismatch", "")
    require(tuple(row.get("scene") for row in results) == SCENE_ORDER, "scene_evaluation_order_mismatch", "")
    require(all(row.get("registry_sha256") == sha256_file(registry_path) for row in results), "scene_registry_binding_mismatch", "")
    by_scene = {row["scene"]: row for row in results}
    strata: dict[str, Any] = {}
    csv_rows: list[dict[str, Any]] = []
    for stratum, scenes in STRATA.items():
        metrics: dict[str, Any] = {}
        for mask in MASKS:
            mask_record: dict[str, Any] = {}
            for method in METHODS:
                method_record: dict[str, Any] = {}
                for metric in METRICS:
                    values = [_scene_metric(by_scene[scene], mask, method, metric) for scene in scenes]
                    record = {
                        "equal_scene_mean": _finite_mean(values),
                        "equal_scene_median": _finite_median(values),
                        "scene_value_count": len([value for value in values if value is not None]),
                    }
                    method_record[metric] = record
                    csv_rows.append(
                        {
                            "stratum": stratum,
                            "coverage_disclosure": "3/3" if stratum == "aerial_primary" else ("2/6" if stratum == "vineyard_matched_subset" else "5 scenes secondary"),
                            "mask": mask,
                            "method": method,
                            "metric": metric,
                            **record,
                        }
                    )
                mask_record[method] = method_record
            metrics[mask] = mask_record
        strata[stratum] = {
            "scenes": list(scenes),
            "scene_weighting": "equal",
            "coverage_disclosure": "3/3 official Aerial scenes" if stratum == "aerial_primary" else ("2/6 matched Vineyard scenes" if stratum == "vineyard_matched_subset" else "secondary all-five aggregate"),
            "metrics": metrics,
        }
    result = {
        "schema": AGGREGATE_SCHEMA,
        "claim_boundary": registry["claim_boundary"],
        "primary_stratum": "aerial_primary",
        "vineyard_subset_reported_separately": True,
        "all_five_secondary_only": True,
        "method_specific_alignment_used": False,
        "target_weighting": "equal within scene",
        "scene_weighting": "equal within stratum",
        "scientific_interpretation_status": "metrics_complete_pending_independent_evidence_synthesis",
        "registry_sha256": sha256_file(registry_path),
        "scene_result_sha256": {row["scene"]: sha256_file(path) for row, path in zip(results, scene_results)},
        "strata": strata,
    }
    write_json(output / "EXTERNAL5_E3_AGGREGATE.json", result)
    with (output / "EXTERNAL5_E3_STRATUM_METRICS.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    scene_rows: list[dict[str, Any]] = []
    for scene in SCENE_ORDER:
        for method in METHODS:
            row = {"scene": scene, "stratum": by_scene[scene]["stratum"], "method": method}
            for metric in METRICS:
                row[f"paired_primary_{metric}"] = _scene_metric(by_scene[scene], "paired_primary", method, metric)
            scene_rows.append(row)
    with (output / "EXTERNAL5_E3_SCENE_METRICS.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(scene_rows[0]))
        writer.writeheader()
        writer.writerows(scene_rows)
    return {"aggregate_path": (output / "EXTERNAL5_E3_AGGREGATE.json").as_posix(), "aggregate_sha256": sha256_file(output / "EXTERNAL5_E3_AGGREGATE.json")}


def _cmd_freeze(args: argparse.Namespace) -> None:
    print(json.dumps(freeze_registry(args.config, args.output_json), sort_keys=True))


def _cmd_reference(args: argparse.Namespace) -> None:
    print(json.dumps(render_reference_scene(args.registry, args.scene, args.output), sort_keys=True))


def _cmd_method(args: argparse.Namespace) -> None:
    print(
        json.dumps(
            render_method_scene(
                args.registry,
                args.scene,
                args.method,
                args.output,
                args.opacity_epsilon,
                args.variance_clamp_tolerance,
            ),
            sort_keys=True,
        )
    )


def _cmd_evaluate(args: argparse.Namespace) -> None:
    print(
        json.dumps(
            evaluate_scene(args.registry, args.scene, args.reference_root, args.umgs_root, args.mss_root, args.output),
            sort_keys=True,
        )
    )


def _cmd_aggregate(args: argparse.Namespace) -> None:
    print(json.dumps(aggregate_scenes(args.registry, args.scene_result, args.output), sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze-registry")
    freeze.add_argument("--config", required=True)
    freeze.add_argument("--output-json", required=True)
    freeze.set_defaults(func=_cmd_freeze)
    reference = subparsers.add_parser("render-reference-scene")
    reference.add_argument("--registry", required=True)
    reference.add_argument("--scene", required=True, choices=SCENE_ORDER)
    reference.add_argument("--output", required=True)
    reference.set_defaults(func=_cmd_reference)
    method = subparsers.add_parser("render-method-scene")
    method.add_argument("--registry", required=True)
    method.add_argument("--scene", required=True, choices=SCENE_ORDER)
    method.add_argument("--method", required=True, choices=METHODS)
    method.add_argument("--output", required=True)
    method.add_argument("--opacity-epsilon", type=float, default=1e-6)
    method.add_argument("--variance-clamp-tolerance", type=float, default=1e-6)
    method.set_defaults(func=_cmd_method)
    evaluate = subparsers.add_parser("evaluate-scene")
    evaluate.add_argument("--registry", required=True)
    evaluate.add_argument("--scene", required=True, choices=SCENE_ORDER)
    evaluate.add_argument("--reference-root", required=True)
    evaluate.add_argument("--umgs-root", required=True)
    evaluate.add_argument("--mss-root", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.set_defaults(func=_cmd_evaluate)
    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--registry", required=True)
    aggregate.add_argument("--scene-result", action="append", required=True)
    aggregate.add_argument("--output", required=True)
    aggregate.set_defaults(func=_cmd_aggregate)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
        return 0
    except E3ContractError as error:
        print(json.dumps({"status": "failed", "failure_counter": error.counter}, sort_keys=True))
        print(str(error), file=sys.stderr)
        return 2
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        print(json.dumps({"status": "malformed", "failure_counter": "external5_e3_internal_or_input_error"}, sort_keys=True))
        print(type(error).__name__, file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
