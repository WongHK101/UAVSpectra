#!/usr/bin/env python3
"""Freeze and qualify the external-five geometry proxy protocol.

The E1 qualification renders one Aerial-Golf held-out camera against a
source-only OpenMVS mesh, the UMGS-I standard Gaussian PLY, and the official
MS-Splatting Nerfstudio checkpoint.  Camera and model transforms are taken
only from dataset/training metadata; no metric-driven alignment is allowed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


PROTOCOL_SCHEMA = "external5_geometry_protocol_v1"
E1_CONTRACT_SCHEMA = "external5_aerial_golf_e1_contract_v1"
METHOD_PACKET_SCHEMA = "external5_expected_camera_z_packet_v1"
REFERENCE_PACKET_SCHEMA = "external5_openmvs_reference_packet_v1"
DEFAULT_PROTOCOL_TEMPLATE = Path(__file__).with_name("external5_geometry_protocol_v1.json")
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


class ProtocolError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code


def require(condition: bool, code: str, message: str) -> None:
    if not condition:
        raise ProtocolError(code, message)


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
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_json_bytes(payload))


def normalized_name(value: str | Path) -> str:
    return Path(str(value).replace("\\", "/")).name.casefold()


def name_set_sha256(names: Sequence[str]) -> str:
    ordered = sorted((Path(name).name for name in names), key=lambda item: item.encode("utf-8"))
    return hashlib.sha256(("\n".join(ordered) + "\n").encode("utf-8")).hexdigest()


def assert_output_empty(path: str | Path) -> Path:
    output = Path(path)
    require(not output.exists() or not any(output.iterdir()), "output_overwrite_forbidden", str(output))
    output.mkdir(parents=True, exist_ok=True)
    return output


def _walk_keys(payload: Any):
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            yield str(key).casefold()
            yield from _walk_keys(value)
    elif isinstance(payload, list):
        for value in payload:
            yield from _walk_keys(value)


def validate_protocol_template(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    require(payload.get("schema") == PROTOCOL_SCHEMA, "protocol_schema_mismatch", str(path))
    keys = set(_walk_keys(payload))
    require(not (keys & FORBIDDEN_ALIGNMENT_KEYS), "method_specific_alignment_forbidden", str(keys & FORBIDDEN_ALIGNMENT_KEYS))
    require(
        payload.get("camera_contract", {}).get("method_specific_scale_shift_crop_alignment") is False,
        "method_specific_alignment_forbidden",
        str(path),
    )
    scenes = payload.get("scenes", [])
    require(len(scenes) == 5, "external5_scene_count_mismatch", str(len(scenes)))
    require(len({row["scene"] for row in scenes}) == 5, "duplicate_scene", str(path))
    return payload


def load_aerial_split(path: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    frames = payload.get("frames")
    require(isinstance(frames, list) and bool(frames), "malformed_aerial_transforms", str(path))
    rows: list[dict[str, Any]] = []
    for frame in frames:
        name = Path(str(frame.get("file_path", "")).replace("\\", "/")).name
        matrix = np.asarray(frame.get("transform_matrix"), dtype=np.float64)
        require(bool(name) and matrix.shape == (4, 4), "malformed_aerial_frame", str(frame))
        require(np.all(np.isfinite(matrix)), "nonfinite_aerial_camera", name)
        require(np.allclose(matrix[3], [0, 0, 0, 1], rtol=0, atol=1e-8), "malformed_aerial_camera", name)
        rows.append({"image_name": name, "c2w_opengl": matrix})
    names = [row["image_name"] for row in rows]
    require(len(names) == len(set(map(str.casefold, names))), "duplicate_aerial_view", str(path))
    return payload, rows


def find_target(rows: Sequence[Mapping[str, Any]], target: str) -> Mapping[str, Any]:
    matches = [row for row in rows if normalized_name(str(row["image_name"])) == normalized_name(target)]
    require(len(matches) == 1, "target_camera_identity_mismatch", f"{target}: {len(matches)}")
    return matches[0]


def load_dataparser_transform(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    raw = np.asarray(payload.get("transform"), dtype=np.float64)
    scale = float(payload.get("scale", math.nan))
    require(raw.shape == (3, 4), "malformed_dataparser_transform", str(path))
    require(np.all(np.isfinite(raw)) and math.isfinite(scale) and scale > 0, "malformed_dataparser_transform", str(path))
    rotation = raw[:, :3]
    ortho_error = float(np.max(np.abs(rotation @ rotation.T - np.eye(3))))
    determinant = float(np.linalg.det(rotation))
    require(ortho_error <= 1e-5 and abs(determinant - 1.0) <= 1e-5, "nonrigid_dataparser_transform", str(path))
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :4] = raw
    return {
        "matrix": matrix,
        "scale": scale,
        "orthonormal_max_abs_error": ortho_error,
        "rotation_determinant": determinant,
        "file_sha256": sha256_file(path),
    }


def apply_dataparser_transform(c2w_opengl: np.ndarray, transform: np.ndarray, scale: float) -> np.ndarray:
    source = np.asarray(c2w_opengl, dtype=np.float64)
    matrix = np.asarray(transform, dtype=np.float64)
    require(source.shape == (4, 4) and matrix.shape == (4, 4), "camera_transform_shape_mismatch", "")
    result = matrix @ source
    result[:3, 3] *= float(scale)
    result[3] = [0, 0, 0, 1]
    return result


def opengl_c2w_to_opencv_c2w(c2w_opengl: np.ndarray) -> np.ndarray:
    flip = np.diag([1.0, -1.0, -1.0, 1.0])
    return np.asarray(c2w_opengl, dtype=np.float64) @ flip


def parse_undistorted_camera_row(row: str) -> dict[str, Any]:
    parts = row.strip().split()
    require(len(parts) == 8 and parts[1] == "PINHOLE", "malformed_undistorted_camera_row", row)
    width, height = int(parts[2]), int(parts[3])
    fx, fy, cx, cy = map(float, parts[4:8])
    require(width > 0 and height > 0 and fx > 0 and fy > 0, "invalid_undistorted_camera", row)
    require(abs(cx - width / 2.0) <= 1e-8 and abs(cy - height / 2.0) <= 1e-8, "offcenter_canonical_camera", row)
    return {"width": width, "height": height, "fx": fx, "fy": fy, "cx": cx, "cy": cy}


def _array_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(np.asarray(array, dtype="<f4"))
    digest = hashlib.sha256()
    digest.update(str(tuple(value.shape)).encode("ascii"))
    digest.update(b"\0float32-le\0")
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def geometry_manifest(arrays: Mapping[str, np.ndarray], source_path: str | Path, source_kind: str) -> dict[str, Any]:
    required = ("xyz", "scaling", "rotation", "opacity")
    require(set(required).issubset(arrays), "geometry_fields_missing", source_kind)
    count = int(np.asarray(arrays["xyz"]).shape[0])
    shapes = {name: list(np.asarray(arrays[name]).shape) for name in required}
    require(shapes["xyz"] == [count, 3], "geometry_shape_mismatch", str(shapes))
    require(shapes["scaling"] == [count, 3], "geometry_shape_mismatch", str(shapes))
    require(shapes["rotation"] == [count, 4], "geometry_shape_mismatch", str(shapes))
    require(shapes["opacity"] == [count, 1], "geometry_shape_mismatch", str(shapes))
    for name in required:
        require(bool(np.all(np.isfinite(arrays[name]))), "nonfinite_geometry", name)
    require(bool(np.all(np.linalg.norm(arrays["rotation"], axis=1) > 1e-12)), "zero_quaternion", source_kind)
    field_sha = {name: _array_sha256(arrays[name]) for name in required}
    identity = {"gaussian_count": count, "field_sha256": field_sha, "source_kind": source_kind}
    path = Path(source_path)
    return {
        **identity,
        "geometry_identity_sha256": canonical_json_sha256(identity),
        "source_path": path.as_posix(),
        "source_size_bytes": path.stat().st_size,
        "source_sha256": sha256_file(path),
        "shapes": shapes,
    }


def load_umgs_ply_geometry(path: str | Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    from plyfile import PlyData

    ply = PlyData.read(str(path), mmap=True)
    vertex = ply["vertex"].data
    names = set(vertex.dtype.names or ())
    required = {"x", "y", "z", "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"}
    require(required.issubset(names), "malformed_umgs_ply", str(sorted(required - names)))
    arrays = {
        "xyz": np.column_stack([np.asarray(vertex[name], dtype=np.float32) for name in ("x", "y", "z")]),
        "scaling": np.column_stack([np.asarray(vertex[f"scale_{index}"], dtype=np.float32) for index in range(3)]),
        "rotation": np.column_stack([np.asarray(vertex[f"rot_{index}"], dtype=np.float32) for index in range(4)]),
        "opacity": np.asarray(vertex["opacity"], dtype=np.float32)[:, None],
    }
    return arrays, geometry_manifest(arrays, path, "umgs_standard_gaussian_ply")


def load_mss_checkpoint_geometry(path: str | Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    import torch

    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    require(isinstance(checkpoint, Mapping) and "pipeline" in checkpoint, "malformed_mss_checkpoint", str(path))
    state = checkpoint["pipeline"]
    keys = {
        "xyz": "_model.gauss_params.means",
        "scaling": "_model.gauss_params.scales",
        "rotation": "_model.gauss_params.quats",
        "opacity": "_model.gauss_params.opacities",
    }
    missing = [key for key in keys.values() if key not in state]
    require(not missing, "mss_geometry_fields_missing", str(missing))
    arrays = {name: state[key].detach().cpu().to(torch.float32).contiguous().numpy() for name, key in keys.items()}
    manifest = geometry_manifest(arrays, path, "mss_nerfstudio_checkpoint")
    manifest["checkpoint_step"] = int(checkpoint.get("step", -1))
    manifest["checkpoint_geometry_keys"] = keys
    return arrays, manifest


def freeze_e0_e1_contract(args: argparse.Namespace) -> dict[str, Any]:
    output = assert_output_empty(args.output)
    template = validate_protocol_template(args.protocol_template)
    template_scene = next(row for row in template["scenes"] if row["scene"] == "aerial_golf")

    _, train_rows = load_aerial_split(args.transforms_train)
    _, test_rows = load_aerial_split(args.transforms_test)
    train_names = [row["image_name"] for row in train_rows]
    test_names = [row["image_name"] for row in test_rows]
    require(not ({name.casefold() for name in train_names} & {name.casefold() for name in test_names}), "train_test_overlap", "aerial_golf")
    require(len(train_names) == int(template_scene["train_count"]), "train_count_mismatch", str(len(train_names)))
    require(len(test_names) == int(template_scene["test_count"]), "test_count_mismatch", str(len(test_names)))
    require(name_set_sha256(train_names) == template_scene["train_name_sha256"], "train_name_hash_mismatch", "aerial_golf")
    require(name_set_sha256(test_names) == template_scene["test_name_sha256"], "test_name_hash_mismatch", "aerial_golf")

    qualification = json.loads(Path(args.qualification_final).read_text(encoding="utf-8"))
    camera_stage = json.loads(Path(args.camera_stage).read_text(encoding="utf-8"))
    heldout = json.loads(Path(args.heldout_manifest).read_text(encoding="utf-8"))
    require(qualification.get("qualification_pass") is True, "reference_qualification_not_passed", str(args.qualification_final))
    require(qualification.get("heldout_isolation_pass") is True, "heldout_leakage", str(args.qualification_final))
    require(qualification.get("method_output_used_for_reference_build") is False, "reference_uses_method_output", str(args.qualification_final))
    require(camera_stage.get("camera_stage_pass") is True, "camera_stage_not_passed", str(args.camera_stage))
    rows = camera_stage.get("undistorted_camera_rows", [])
    require(isinstance(rows, list) and len(rows) == 1, "canonical_camera_row_count_mismatch", str(rows))
    intrinsics = parse_undistorted_camera_row(rows[0])
    heldout_rows = heldout.get("cameras", heldout.get("heldout_images", []))
    heldout_names = [str(row.get("image_name", "")) for row in heldout_rows]
    require(name_set_sha256(heldout_names) == template_scene["test_name_sha256"], "heldout_manifest_hash_mismatch", "aerial_golf")
    target = args.target or sorted(test_names, key=lambda name: name.encode("utf-8"))[0]
    target_frame = find_target(test_rows, target)
    target_heldout = find_target(heldout_rows, target)

    # Independently cross-check the official transforms camera against the
    # qualification manifest's COLMAP camera conversion.
    from utils.read_write_model import qvec2rotmat

    source_opencv_c2w = opengl_c2w_to_opencv_c2w(target_frame["c2w_opengl"])
    source_w2c = np.linalg.inv(source_opencv_c2w)
    manifest_w2c = np.eye(4, dtype=np.float64)
    manifest_w2c[:3, :3] = qvec2rotmat(np.asarray(target_heldout["qvec"], dtype=np.float64))
    manifest_w2c[:3, 3] = np.asarray(target_heldout["tvec"], dtype=np.float64)
    pose_max_abs_delta = float(np.max(np.abs(source_w2c - manifest_w2c)))
    require(pose_max_abs_delta <= 1e-8, "heldout_camera_pose_mismatch", str(pose_max_abs_delta))

    dataparser = load_dataparser_transform(args.mss_dataparser_transform)
    mss_c2w = apply_dataparser_transform(
        target_frame["c2w_opengl"], dataparser["matrix"], dataparser["scale"]
    )
    mss_unscaled_c2w = np.array(mss_c2w, copy=True)
    mss_unscaled_c2w[:3, 3] /= dataparser["scale"]
    mss_source_roundtrip = np.linalg.inv(dataparser["matrix"]) @ mss_unscaled_c2w
    roundtrip_error = float(np.max(np.abs(mss_source_roundtrip - target_frame["c2w_opengl"])))
    require(roundtrip_error <= 1e-7, "dataparser_camera_roundtrip_mismatch", str(roundtrip_error))

    mesh = Path(args.mesh_ply)
    umgs = Path(args.umgs_ply)
    mss = Path(args.mss_checkpoint)
    for path, label in ((mesh, "mesh"), (umgs, "umgs"), (mss, "mss")):
        require(path.is_file(), f"missing_{label}_source", str(path))

    protocol = dict(template)
    protocol.update(
        {
            "protocol_template_sha256": sha256_file(args.protocol_template),
            "metric_read_before_freeze": False,
            "created_before_first_method_render": True,
            "execution_scope": "E0 protocol freeze and E1 one-target qualification only",
        }
    )
    protocol_path = output / "EXTERNAL5_GEOMETRY_PROTOCOL.json"
    write_json(protocol_path, protocol)

    contract = {
        "schema": E1_CONTRACT_SCHEMA,
        "scene": "aerial_golf",
        "target": target,
        "selection_rule": "UTF-8 byte-sorted first official held-out target unless explicitly predeclared",
        "protocol_path": protocol_path.as_posix(),
        "protocol_sha256": sha256_file(protocol_path),
        "canonical_camera": {
            **intrinsics,
            "FoVx": float(2.0 * math.atan(intrinsics["width"] / (2.0 * intrinsics["fx"]))),
            "FoVy": float(2.0 * math.atan(intrinsics["height"] / (2.0 * intrinsics["fy"]))),
            "source_c2w_opengl": np.asarray(target_frame["c2w_opengl"]).tolist(),
            "source_c2w_opencv": source_opencv_c2w.tolist(),
            "qualification_manifest_pose_max_abs_delta": pose_max_abs_delta,
            "grid_authority": "method-independent COLMAP undistorter output from train-only qualification",
            "centered_projection": True,
        },
        "split_binding": {
            "train_count": len(train_names),
            "test_count": len(test_names),
            "train_name_sha256": name_set_sha256(train_names),
            "test_name_sha256": name_set_sha256(test_names),
            "transforms_train_sha256": sha256_file(args.transforms_train),
            "transforms_test_sha256": sha256_file(args.transforms_test),
            "heldout_manifest_sha256": sha256_file(args.heldout_manifest),
            "target_in_train": target.casefold() in {name.casefold() for name in train_names},
            "target_in_test": target.casefold() in {name.casefold() for name in test_names},
        },
        "reference_binding": {
            "mesh_path": mesh.as_posix(),
            "mesh_sha256": sha256_file(mesh),
            "mesh_size_bytes": mesh.stat().st_size,
            "qualification_final_sha256": sha256_file(args.qualification_final),
            "camera_stage_sha256": sha256_file(args.camera_stage),
            "source_only": True,
            "heldout_used_for_reconstruction": False,
            "method_output_used": False,
        },
        "method_sources": {
            "umgs_i": {
                "kind": "umgs_standard_gaussian_ply",
                "path": umgs.as_posix(),
                "sha256": sha256_file(umgs),
                "size_bytes": umgs.stat().st_size,
                "camera_frame": "source",
            },
            "ms_splatting": {
                "kind": "mss_nerfstudio_checkpoint",
                "path": mss.as_posix(),
                "sha256": sha256_file(mss),
                "size_bytes": mss.stat().st_size,
                "camera_frame": "frozen_dataparser",
                "dataparser_transform_path": Path(args.mss_dataparser_transform).as_posix(),
                "dataparser_transform_sha256": dataparser["file_sha256"],
                "dataparser_transform": dataparser["matrix"].tolist(),
                "dataparser_scale": dataparser["scale"],
                "orthonormal_max_abs_error": dataparser["orthonormal_max_abs_error"],
                "rotation_determinant": dataparser["rotation_determinant"],
                "camera_roundtrip_max_abs_error": roundtrip_error,
                "camera_z_to_source_units": 1.0 / dataparser["scale"],
            },
        },
        "evaluation_contract": {
            "depth_semantics": "opacity-composited expected camera-z",
            "paired_common_mask": True,
            "method_specific_alignment_used": False,
            "metric_interpretation_authorized_by_e1": False,
        },
    }
    contract_path = output / "AERIAL_GOLF_E1_CONTRACT.json"
    write_json(contract_path, contract)
    return {
        "protocol_path": protocol_path.as_posix(),
        "protocol_sha256": sha256_file(protocol_path),
        "contract_path": contract_path.as_posix(),
        "contract_sha256": sha256_file(contract_path),
        "target": target,
    }


def load_contract(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    require(payload.get("schema") == E1_CONTRACT_SCHEMA, "e1_contract_schema_mismatch", str(path))
    require(payload.get("evaluation_contract", {}).get("method_specific_alignment_used") is False, "method_specific_alignment_forbidden", str(path))
    return payload


def build_renderer_camera(contract: Mapping[str, Any], method: str):
    import torch
    from utils.graphics_utils import getProjectionMatrix, getWorld2View2

    record = contract["canonical_camera"]
    c2w_opengl = np.asarray(record["source_c2w_opengl"], dtype=np.float64)
    if method == "ms_splatting":
        source = contract["method_sources"]["ms_splatting"]
        c2w_opengl = apply_dataparser_transform(
            c2w_opengl,
            np.asarray(source["dataparser_transform"], dtype=np.float64),
            float(source["dataparser_scale"]),
        )
    elif method != "umgs_i":
        raise ProtocolError("unknown_method", method)
    c2w_opencv = opengl_c2w_to_opencv_c2w(c2w_opengl)
    w2c = np.linalg.inv(c2w_opencv)
    r = w2c[:3, :3].T
    t = w2c[:3, 3]
    fovx = float(record["FoVx"])
    fovy = float(record["FoVy"])
    znear, zfar = 0.01, 1000.0
    world = torch.tensor(getWorld2View2(r, t), dtype=torch.float32, device="cuda").transpose(0, 1)
    projection = getProjectionMatrix(znear=znear, zfar=zfar, fovX=fovx, fovY=fovy).to("cuda").transpose(0, 1)
    full = world.unsqueeze(0).bmm(projection.unsqueeze(0)).squeeze(0)
    return SimpleNamespace(
        image_name=contract["target"],
        colmap_id=1,
        uid=0,
        image_width=int(record["width"]),
        image_height=int(record["height"]),
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


def make_geometry_model(arrays: Mapping[str, np.ndarray]):
    import torch
    import torch.nn.functional as functional

    class GeometryModel:
        active_sh_degree = 0
        max_sh_degree = 0

        def __init__(self):
            self._xyz = torch.as_tensor(np.ascontiguousarray(arrays["xyz"]), dtype=torch.float32, device="cuda")
            self._scaling = torch.as_tensor(np.ascontiguousarray(arrays["scaling"]), dtype=torch.float32, device="cuda")
            self._rotation = torch.as_tensor(np.ascontiguousarray(arrays["rotation"]), dtype=torch.float32, device="cuda")
            self._opacity = torch.as_tensor(np.ascontiguousarray(arrays["opacity"]), dtype=torch.float32, device="cuda")

        @property
        def get_xyz(self):
            return self._xyz

        @property
        def get_scaling(self):
            return torch.exp(self._scaling)

        @property
        def get_rotation(self):
            return functional.normalize(self._rotation)

        @property
        def get_opacity(self):
            return torch.sigmoid(self._opacity)

    return GeometryModel()


def _renderer_source_sha256() -> dict[str, str]:
    import diff_gaussian_rasterization
    import gaussian_renderer

    extension = Path(diff_gaussian_rasterization._C.__file__).resolve()
    renderer = Path(gaussian_renderer.__file__).resolve()
    return {
        "renderer_path": renderer.as_posix(),
        "renderer_sha256": sha256_file(renderer),
        "rasterizer_extension_path": extension.as_posix(),
        "rasterizer_extension_sha256": sha256_file(extension),
    }


def _convert_mss_packet_to_source_units(
    arrays: Mapping[str, np.ndarray], scale: float
) -> dict[str, np.ndarray]:
    require(math.isfinite(scale) and scale > 0, "invalid_dataparser_scale", str(scale))
    converted = {name: np.array(value, copy=True) for name, value in arrays.items()}
    linear = 1.0 / float(scale)
    quadratic = linear * linear
    converted["weighted_camera_z_sum"] *= linear
    converted["expected_camera_z"] *= linear
    converted["weighted_camera_z2_sum"] *= quadratic
    converted["camera_z_variance"] *= quadratic
    return converted


def _packet_array_sha256(arrays: Mapping[str, np.ndarray]) -> dict[str, str]:
    return {name: _array_sha256(np.asarray(arrays[name])) for name in sorted(arrays)}


def render_method_packet(args: argparse.Namespace) -> dict[str, Any]:
    contract = load_contract(args.contract)
    method = str(args.method)
    require(method in {"umgs_i", "ms_splatting"}, "unknown_method", method)
    source = contract["method_sources"][method]
    source_path = Path(source["path"])
    require(source_path.is_file(), "missing_method_source", str(source_path))
    source_sha_before = sha256_file(source_path)
    require(source_sha_before == source["sha256"], "method_source_sha256_mismatch", method)

    output = assert_output_empty(args.output)
    if method == "umgs_i":
        arrays, geometry = load_umgs_ply_geometry(source_path)
    else:
        arrays, geometry = load_mss_checkpoint_geometry(source_path)

    import torch
    from gaussian_renderer import render
    from tools.depth_reference_geometry_v2.export_ramp_expected_camera_z_packets import (
        deterministic_npz,
        validate_named_packet,
    )
    from tools.depth_reference_geometry_v2.export_umgs_expected_camera_z_packet import (
        packet_schema,
        packet_tensor_to_named_arrays,
    )

    camera = build_renderer_camera(contract, method)
    model = make_geometry_model(arrays)
    background = torch.zeros(3, dtype=torch.float32, device="cuda")
    override_color = torch.full(
        (int(geometry["gaussian_count"]), 3),
        0.5,
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
        rendered = render(
            camera,
            model,
            pipeline,
            background,
            override_color=override_color,
            use_trained_exp=False,
            separate_sh=False,
            return_expected_camera_z_packet=True,
            opacity_epsilon=float(args.opacity_epsilon),
            variance_clamp_tolerance=float(args.variance_clamp_tolerance),
        )
    torch.cuda.synchronize()
    named = packet_tensor_to_named_arrays(
        rendered["expected_camera_z_packet"],
        opacity_epsilon=float(args.opacity_epsilon),
        variance_clamp_tolerance=float(args.variance_clamp_tolerance),
    )
    if method == "ms_splatting":
        named = _convert_mss_packet_to_source_units(
            named, float(source["dataparser_scale"])
        )
    shape = (int(contract["canonical_camera"]["height"]), int(contract["canonical_camera"]["width"]))
    validation = validate_named_packet(named, shape)
    packet_path = output / f"{method}_expected_camera_z_packet.npz"
    deterministic_npz(packet_path, named)
    source_sha_after = sha256_file(source_path)
    require(source_sha_after == source_sha_before, "method_source_modified", method)
    manifest = {
        "schema": METHOD_PACKET_SCHEMA,
        "packet_schema": packet_schema(),
        "scene": contract["scene"],
        "target": contract["target"],
        "method": method,
        "qualification_only": True,
        "scientific_interpretation_authorized": False,
        "geometry": geometry,
        "camera": {
            "frame": source["camera_frame"],
            "width": camera.image_width,
            "height": camera.image_height,
            "FoVx": camera.FoVx,
            "FoVy": camera.FoVy,
            "c2w_opencv": np.asarray(camera.c2w_opencv).tolist(),
        },
        "depth_semantics": "sum(T_i*alpha_i*z_i)/sum(T_i*alpha_i)",
        "camera_z_output_units": "source reconstruction units",
        "camera_z_unit_conversion": (
            1.0 if method == "umgs_i" else float(source["camera_z_to_source_units"])
        ),
        "method_specific_alignment_used": False,
        "fixed_rasterizer_contribution_cutoff": "alpha>=1/255",
        "fixed_rasterizer_early_termination": "remaining_transmittance<=1e-4",
        "constant_override_color_used": True,
        "packet_validation": validation,
        "packet_file": packet_path.name,
        "packet_sha256": sha256_file(packet_path),
        "packet_array_sha256": _packet_array_sha256(named),
        "source_sha256_before": source_sha_before,
        "source_sha256_after": source_sha_after,
        "source_unchanged": source_sha_before == source_sha_after,
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
        "runtime": _renderer_source_sha256(),
    }
    manifest_path = output / "METHOD_PACKET_MANIFEST.json"
    write_json(manifest_path, manifest)
    return {
        "manifest_path": manifest_path.as_posix(),
        "manifest_sha256": sha256_file(manifest_path),
        "packet_path": packet_path.as_posix(),
        "packet_sha256": sha256_file(packet_path),
        "numeric_valid_coverage": validation["numeric_valid_coverage"],
    }


def render_reference_packet(args: argparse.Namespace) -> dict[str, Any]:
    contract = load_contract(args.contract)
    binding = contract["reference_binding"]
    mesh_path = Path(binding["mesh_path"])
    require(mesh_path.is_file(), "missing_reference_mesh", str(mesh_path))
    mesh_sha_before = sha256_file(mesh_path)
    require(mesh_sha_before == binding["mesh_sha256"], "reference_mesh_sha256_mismatch", str(mesh_path))
    output = assert_output_empty(args.output)

    from tools.depth_reference_geometry_v2.export_ramp_expected_camera_z_packets import deterministic_npz
    from tools.depth_reference_geometry_v2.openmvs_campaign_core import (
        CameraView,
        load_ply_mesh,
        rasterize_mesh_camera_z,
    )

    camera = contract["canonical_camera"]
    view = CameraView(
        image_id=1,
        image_name=contract["target"],
        width=int(camera["width"]),
        height=int(camera["height"]),
        fx=float(camera["fx"]),
        fy=float(camera["fy"]),
        cx=float(camera["cx"]),
        cy=float(camera["cy"]),
        camera_to_world=np.asarray(camera["source_c2w_opencv"], dtype=np.float64),
    )
    vertices, faces, mesh_meta = load_ply_mesh(mesh_path)
    require(int(faces.shape[0]) > 0, "reference_mesh_has_no_faces", str(mesh_path))
    rendered = rasterize_mesh_camera_z(vertices, faces, view)
    depth = np.asarray(rendered.depth, dtype=np.float32)
    valid = np.asarray(rendered.valid, dtype=np.uint8)
    require(depth.shape == (view.height, view.width), "reference_packet_shape_mismatch", str(depth.shape))
    require(bool(valid.any()), "reference_packet_has_no_valid_pixels", contract["target"])
    require(bool(np.isfinite(depth[valid.astype(bool)]).all()), "reference_depth_nonfinite", contract["target"])
    depth[~valid.astype(bool)] = np.nan
    packet_path = output / "openmvs_reference_camera_z_packet.npz"
    deterministic_npz(packet_path, {"depth": depth, "valid": valid})
    mesh_sha_after = sha256_file(mesh_path)
    require(mesh_sha_before == mesh_sha_after, "reference_mesh_modified", str(mesh_path))
    manifest = {
        "schema": REFERENCE_PACKET_SCHEMA,
        "scene": contract["scene"],
        "target": contract["target"],
        "qualification_only": True,
        "scientific_interpretation_authorized": False,
        "reference_claim": "source-only OpenMVS proxy; not absolute geometry ground truth",
        "mesh": mesh_meta,
        "camera": {
            "width": view.width,
            "height": view.height,
            "fx": view.fx,
            "fy": view.fy,
            "cx": view.cx,
            "cy": view.cy,
            "c2w_opencv": np.asarray(view.camera_to_world).tolist(),
        },
        "valid_pixels": int(valid.sum()),
        "valid_coverage": float(valid.mean()),
        "packet_file": packet_path.name,
        "packet_sha256": sha256_file(packet_path),
        "depth_array_sha256": _array_sha256(depth),
        "valid_array_sha256": hashlib.sha256(np.ascontiguousarray(valid).tobytes()).hexdigest(),
        "mesh_sha256_before": mesh_sha_before,
        "mesh_sha256_after": mesh_sha_after,
        "source_unchanged": mesh_sha_before == mesh_sha_after,
        "method_output_used": False,
    }
    manifest_path = output / "REFERENCE_PACKET_MANIFEST.json"
    write_json(manifest_path, manifest)
    return {
        "manifest_path": manifest_path.as_posix(),
        "manifest_sha256": sha256_file(manifest_path),
        "packet_path": packet_path.as_posix(),
        "packet_sha256": sha256_file(packet_path),
        "valid_coverage": manifest["valid_coverage"],
    }


def _load_npz_exact(path: str | Path, keys: set[str]) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        require(set(archive.files) == keys, "packet_keyset_mismatch", str(archive.files))
        return {name: np.asarray(archive[name]) for name in archive.files}


def evaluate_e1(args: argparse.Namespace) -> dict[str, Any]:
    contract = load_contract(args.contract)
    from tools.depth_reference_geometry_v2.export_ramp_expected_camera_z_packets import (
        EXPECTED_PACKET_ARRAYS,
        _candidate_metrics,
        paired_masks,
        validate_named_packet,
    )
    from tools.depth_reference_geometry_v2.openmvs_da3_overlap_corrected import (
        reference_high_gradient_domain,
    )

    expected_method_keys = set(EXPECTED_PACKET_ARRAYS)
    umgs = _load_npz_exact(args.umgs_packet, expected_method_keys)
    mss = _load_npz_exact(args.mss_packet, expected_method_keys)
    reference_packet = _load_npz_exact(args.reference_packet, {"depth", "valid"})
    reference = np.asarray(reference_packet["depth"])
    reference_valid = np.asarray(reference_packet["valid"]).astype(bool)
    shape = (int(contract["canonical_camera"]["height"]), int(contract["canonical_camera"]["width"]))
    validate_named_packet(umgs, shape)
    validate_named_packet(mss, shape)
    require(reference.shape == reference_valid.shape == shape, "reference_packet_shape_mismatch", str(reference.shape))
    require(bool(np.isfinite(reference[reference_valid]).all()), "reference_depth_nonfinite", "")
    masks = paired_masks(umgs, mss, reference, reference_valid)
    evaluated_masks: dict[str, Any] = {}
    for mask_name, mask in masks.items():
        domain = reference_high_gradient_domain(reference, mask)
        method_metrics = {
            "umgs_i": _candidate_metrics(reference, umgs["expected_camera_z"], mask, domain),
            "ms_splatting": _candidate_metrics(reference, mss["expected_camera_z"], mask, domain),
        }
        deltas = {}
        for metric in ("mean_absrel", "median_absrel", "p90_absrel", "rmse", "spearman", "high_gradient_cosine"):
            left = method_metrics["umgs_i"].get(metric)
            right = method_metrics["ms_splatting"].get(metric)
            deltas[metric] = None if left is None or right is None else float(left - right)
        evaluated_masks[mask_name] = {
            "support_pixels": int(mask.sum()),
            "support_coverage": float(mask.mean()),
            "methods": method_metrics,
            "umgs_i_minus_ms_splatting": deltas,
        }

    native = {}
    for name, packet in (("umgs_i", umgs), ("ms_splatting", mss)):
        mask = (
            reference_valid
            & packet["numeric_valid"].astype(bool)
            & np.isfinite(reference)
            & np.isfinite(packet["expected_camera_z"])
            & (reference > 0)
            & (packet["expected_camera_z"] > 0)
        )
        native[name] = {"pixels": int(mask.sum()), "coverage": float(mask.mean())}
    result = {
        "schema": "external5_aerial_golf_e1_evaluation_v1",
        "scene": contract["scene"],
        "target": contract["target"],
        "qualification_only": True,
        "scientific_interpretation_authorized": False,
        "e2_authorization_requires_engineering_gate_only": True,
        "claim_boundary": "single-target adapter qualification; not a method ranking",
        "depth_semantics": "opacity-composited expected camera-z in source reconstruction units",
        "method_specific_alignment_used": False,
        "paired_common_mask_used": True,
        "native_coverage": native,
        "masks": evaluated_masks,
        "input_sha256": {
            "contract": sha256_file(args.contract),
            "reference_packet": sha256_file(args.reference_packet),
            "umgs_packet": sha256_file(args.umgs_packet),
            "ms_splatting_packet": sha256_file(args.mss_packet),
        },
    }
    output = Path(args.output_json)
    require(not output.exists(), "output_overwrite_forbidden", str(output))
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, result)
    if args.output_csv:
        csv_path = Path(args.output_csv)
        require(not csv_path.exists(), "output_overwrite_forbidden", str(csv_path))
        rows = []
        for mask_name, mask_record in evaluated_masks.items():
            for method, metrics in mask_record["methods"].items():
                rows.append(
                    {
                        "scene": contract["scene"],
                        "target": contract["target"],
                        "mask": mask_name,
                        "method": method,
                        **metrics,
                    }
                )
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return result


def _cmd_freeze(args: argparse.Namespace) -> None:
    print(json.dumps(freeze_e0_e1_contract(args), sort_keys=True))


def _cmd_render_reference(args: argparse.Namespace) -> None:
    print(json.dumps(render_reference_packet(args), sort_keys=True))


def _cmd_render_method(args: argparse.Namespace) -> None:
    print(json.dumps(render_method_packet(args), sort_keys=True))


def _cmd_evaluate(args: argparse.Namespace) -> None:
    payload = evaluate_e1(args)
    print(json.dumps(payload, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    freeze = subparsers.add_parser("freeze", help="Freeze E0 and the Aerial-Golf E1 contract")
    freeze.add_argument("--protocol-template", default=str(DEFAULT_PROTOCOL_TEMPLATE))
    freeze.add_argument("--transforms-train", required=True)
    freeze.add_argument("--transforms-test", required=True)
    freeze.add_argument("--qualification-final", required=True)
    freeze.add_argument("--camera-stage", required=True)
    freeze.add_argument("--heldout-manifest", required=True)
    freeze.add_argument("--mesh-ply", required=True)
    freeze.add_argument("--umgs-ply", required=True)
    freeze.add_argument("--mss-checkpoint", required=True)
    freeze.add_argument("--mss-dataparser-transform", required=True)
    freeze.add_argument("--target")
    freeze.add_argument("--output", required=True)
    freeze.set_defaults(func=_cmd_freeze)

    reference = subparsers.add_parser("render-reference")
    reference.add_argument("--contract", required=True)
    reference.add_argument("--output", required=True)
    reference.set_defaults(func=_cmd_render_reference)

    method = subparsers.add_parser("render-method")
    method.add_argument("--contract", required=True)
    method.add_argument("--method", required=True, choices=("umgs_i", "ms_splatting"))
    method.add_argument("--output", required=True)
    method.add_argument("--opacity-epsilon", type=float, default=1e-6)
    method.add_argument("--variance-clamp-tolerance", type=float, default=1e-6)
    method.set_defaults(func=_cmd_render_method)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--contract", required=True)
    evaluate.add_argument("--reference-packet", required=True)
    evaluate.add_argument("--umgs-packet", required=True)
    evaluate.add_argument("--mss-packet", required=True)
    evaluate.add_argument("--output-json", required=True)
    evaluate.add_argument("--output-csv")
    evaluate.set_defaults(func=_cmd_evaluate)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
        return 0
    except ProtocolError as error:
        print(json.dumps({"status": "failed", "failure_counter": error.code}, sort_keys=True))
        print(str(error), file=sys.stderr)
        return 2
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(json.dumps({"status": "malformed", "failure_counter": "external5_e1_internal_or_input_error"}, sort_keys=True))
        print(type(error).__name__, file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
