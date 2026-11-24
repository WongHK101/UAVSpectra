"""Relative-path inputs for the unchanged proxy formulas and common masks."""
import argparse
import csv
import importlib
import importlib.util
import json
import os
from pathlib import Path

import numpy as np

from tools.measured_geometry_v1.contracts import read_json, safe_member, sha256, verify_sha
from tools.measured_geometry_v1.proxy_checkpoint_export import LEGACY_NAMES, WRAPPER_SHA
from tools.measured_geometry_v1.proxy_common_score import (
    METRICS, common_masks, complete_mean, load_packet, mask_digest, unique_rows,
)
from tools.measured_geometry_v1.proxy_roi import roi_common_masks, validate_camera, validate_reference

ROOT = Path(__file__).resolve().parent


def metric_runtime():
    os.environ['UMGS_EXTERNAL5_E3_ROOT'] = str(ROOT / 'third_party/proxy')
    path = ROOT / 'third_party/proxy_bridge/mssplat_self5_geometry.py'
    verify_sha(path, WRAPPER_SHA)
    spec = importlib.util.spec_from_file_location('uavspectra_frozen_proxy', path)
    wrapper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wrapper)
    modules = wrapper.audited_modules()
    gradient = importlib.import_module('tools.depth_reference_geometry_v2.openmvs_da3_overlap_corrected')
    return wrapper, modules, gradient


def load_reference(root, scene):
    settings = read_json(ROOT / 'configs/proxy_data.json')[scene]
    path = root / 'scene.json'
    verify_sha(path, settings['scene_sha256'])
    data = read_json(path)
    if (data['schema'] != 'uavspectra_proxy_reference_v1' or data['scene'] != scene
            or data['depth_semantics'] != 'camera_z'
            or data['depth_units'] != 'source_common_SfM_reconstruction_units_not_metres'
            or data['alignment'] != 'none' or data['depth_resize_allowed']):
        raise ValueError('Reference contract mismatch')
    names = [r['image_name'] for r in data['targets']]
    unique_rows(data['targets'], 'image_name', names)
    if len(names) != settings['target_count'] or len(names) != data['target_count']:
        raise ValueError('Reference target count mismatch')
    return data


def local_record(root, record):
    return dict(path=str(safe_member(root, record['path'])), sha256=record['sha256'])


def score_target(root, data, target, prediction_root, predictions, methods, runtime):
    wrapper, modules, gradient = runtime
    camera = target['camera']
    reference = load_packet(local_record(root, target['reference']), {'depth', 'valid', 'triangle_id', 'barycentric'})
    if data['support'] == 'reference_visible_roi':
        validate_camera(camera)
        valid = validate_reference(reference, camera)
        if target['camera_identity_sha256'] != camera['record_sha256']:
            raise ValueError('Camera record mismatch')
    elif data['support'] == 'reference_valid_full_frame':
        wrapper.validate_camera_fingerprint(dict(image_name=target['image_name'], width=camera['width'],
            height=camera['height'], fingerprint=dict(payload=camera['fingerprint_payload'],
            fingerprint_sha256=target['camera_identity_sha256'])), data['scene'])
        valid = reference['valid'].astype(bool)
        if (reference['depth'].shape != (camera['height'], camera['width']) or valid.shape != reference['depth'].shape
                or not valid.any() or not (np.isfinite(reference['depth'][valid]) & (reference['depth'][valid] > 0)).all()):
            raise ValueError('Invalid frozen reference')
    else:
        raise ValueError('Unknown reference support')
    packets = {}
    for method in methods:
        record = predictions['packets'][method]
        if record['camera_identity_sha256'] != target['camera_identity_sha256']:
            raise ValueError('Prediction/reference camera identity mismatch')
        packets[method] = load_packet(local_record(prediction_root, record), LEGACY_NAMES)
        wrapper.m1_a_gate(modules, packets[method], reference['depth'].shape)
    roi = None
    regions = None
    if data['support'] == 'reference_visible_roi':
        path = safe_member(root, target['mask']['path'])
        verify_sha(path, target['mask']['sha256'])
        roi = np.load(path, allow_pickle=False)
        if mask_digest(roi) != target['mask']['matrix_sha256'] or int(roi.sum()) != target['reference_roi_pixels']:
            raise ValueError('ROI matrix identity mismatch')
        region_path = safe_member(root, target['subregions']['path'])
        verify_sha(region_path, target['subregions']['sha256'])
        regions = np.load(region_path, allow_pickle=False)
        if regions.shape != roi.shape or not np.array_equal(regions < 16, roi.astype(bool)):
            raise ValueError('ROI subregion identity mismatch')
        masks, coverage = roi_common_masks(reference['depth'], valid, packets, methods, roi)
    else:
        masks = common_masks(reference['depth'], valid, packets, methods)
        coverage = {name: dict(valid_pixels=int(mask.sum()), support_coverage=float(mask.mean()))
                    for name, mask in masks.items()}
    result = []
    details = dict(image_name=target['image_name'], masks={})
    for name, mask in masks.items():
        domain = gradient.reference_high_gradient_domain(reference['depth'], mask)
        details['masks'][name] = dict(coverage[name], mask_sha256=mask_digest(mask),
            support_coverage=float(mask.mean()), reference_gradient_mask_sha256=mask_digest(domain.high_mask),
            reference_gradient_threshold=domain.threshold)
        if regions is not None:
            details['masks'][name]['subregions'] = [dict(region=i,
                reference_roi_pixels=int(np.count_nonzero(regions == i)),
                supported_pixels=int(np.count_nonzero(mask & (regions == i))),
                coverage=float(np.count_nonzero(mask & (regions == i)) / np.count_nonzero(regions == i))
                if np.any(regions == i) else None) for i in range(16)]
        for method in methods:
            values = modules.packets._candidate_metrics(reference['depth'], packets[method]['expected_camera_z'], mask, domain) if mask.any() else {}
            result.append(dict(scene=data['scene'], image_name=target['image_name'], method=method, mask=name,
                mask_sha256=mask_digest(mask), **coverage[name],
                **{key: float(values[key]) if values.get(key) is not None and np.isfinite(values[key]) else None for key in METRICS}))
    return result, details


def score(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    data = load_reference(args.reference, args.scene)
    job = read_json(args.predictions)
    methods = job['methods']
    if (job['schema'] != 'uavspectra_proxy_predictions_v1' or job['scene'] != args.scene
            or not methods or len(set(methods)) != len(methods)
            or job['formula'] != 'M1/A' or job['depth_semantics'] != data['depth_semantics']
            or job['depth_units'] != data['depth_units'] or job['alignment'] != 'none'
            or job['depth_resize_used']):
        raise ValueError('Prediction contract mismatch')
    names = [t['image_name'] for t in data['targets']]
    predictions = unique_rows(job['targets'], 'image_name', names)
    if any(set(p['packets']) != set(methods) for p in predictions.values()):
        raise ValueError('All declared methods are required on every target')
    runtime = metric_runtime()
    rows, details = [], []
    for target in data['targets']:
        values, detail = score_target(args.reference, data, target, args.predictions.parent,
                                     predictions[target['image_name']], methods, runtime)
        rows.extend(values)
        details.append(detail)
    aggregate = {mask: {method: {metric: complete_mean([
        r[metric] for r in rows if r['mask'] == mask and r['method'] == method], len(names))
        for metric in METRICS} for method in methods} for mask in data['masks']}
    args.output.mkdir(parents=True)
    with (args.output / 'per_target_metrics.csv').open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = dict(scene=args.scene, methods=methods, target_count=len(names), aggregate=aggregate,
        targets=details, status='numerical_comparison_producer_unverified',
        prediction_camera_check='declared_identity_matches_frozen_reference',
        original_export_producer_chain_verified=False,
        reference_sha256=sha256(args.reference / 'scene.json'), predictions_sha256=sha256(args.predictions),
        paper_method_set_matches=methods == data['paper_methods'],
        support=data['support'], depth_units=data['depth_units'], alignment='none', gpu_used=False,
        note='Common support is recomputed across the declared methods; compare results only within the same set. '
             'This file checks arrays and declared camera identities, not the authenticity of their renderer/exporter. '
             'Use the original export qualification before claiming a method result.')
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    score(parser.parse_args())


if __name__ == '__main__':
    main()
