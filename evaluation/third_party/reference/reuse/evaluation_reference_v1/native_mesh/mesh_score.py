"""Independent native-Mesh track; reuses the frozen rectangular-ROI NN core.

Run this on 901. Importing the module does not load any model or LiDAR file.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / 'python-deps'))
import numpy as np


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def load_rect_core(repo):
    repo = Path(repo)
    sys.path.insert(0, str(repo / 'code/gcp'))
    import evaluate_uavgs_lidar_rectangular_roi_v1 as rect
    return rect, rect.load_core(repo)


def to_metric(vertices, sim3, normalization=None):
    from pyproj import Transformer
    xyz = np.asarray(vertices, dtype=np.float64)
    if normalization is not None:
        # x_norm = (x_colmap + translate) * scale; invert BOTH terms.
        xyz = xyz * normalization['original_colmap_units_per_normalized_unit']
        xyz = xyz - np.asarray(normalization['translate_original_colmap'])
    t = sim3['transform']
    target = t['scale'] * (xyz @ np.asarray(t['rotation']).T) + np.asarray(t['translation'])
    transformer = Transformer.from_crs(4545, 32649, always_xy=True)
    e, n = transformer.transform(target[:, 0], target[:, 1])
    # Common Sim(3) already predicts normal height. Only LiDAR needed the
    # ellipsoid-to-normal offset; applying it again here would be an error.
    return np.column_stack((e, n, target[:, 2])).astype(np.float64)


def clip_polygon(poly, axis, bound, lower):
    out = []
    for a, b in zip(poly, np.roll(poly, -1, axis=0)):
        ina = a[axis] >= bound if lower else a[axis] <= bound
        inb = b[axis] >= bound if lower else b[axis] <= bound
        if ina:
            out.append(a)
        if ina != inb:
            t = (bound - a[axis]) / (b[axis] - a[axis])
            p = a + t * (b - a)
            p[axis] = bound
            out.append(p)
    return np.asarray(out, dtype=np.float64).reshape(-1, 3)


def clip_triangles(triangles, bounds):
    """Exact rectangle-prism clipping, including triangles crossing its edge.

    Never closes holes or caps the cut boundary. There is no vertical crop.
    """
    tri = np.asarray(triangles, dtype=np.float64)
    if not np.isfinite(tri).all():
        raise ValueError('non-finite mesh vertices')
    xmin, ymin, xmax, ymax = bounds
    lo, hi = tri.min(axis=1), tri.max(axis=1)
    outside = (hi[:, 0] < xmin) | (lo[:, 0] > xmax) | (hi[:, 1] < ymin) | (lo[:, 1] > ymax)
    inside = (lo[:, 0] >= xmin) & (hi[:, 0] <= xmax) & (lo[:, 1] >= ymin) & (hi[:, 1] <= ymax)
    pieces = [tri[inside]]
    boundary = []
    for polygon in tri[~outside & ~inside]:
        for axis, value, lower in [(0, xmin, True), (0, xmax, False), (1, ymin, True), (1, ymax, False)]:
            polygon = clip_polygon(polygon, axis, value, lower)
            if len(polygon) < 3:
                break
        for k in range(1, len(polygon) - 1):
            boundary.append([polygon[0], polygon[k], polygon[k + 1]])
    if boundary:
        pieces.append(np.asarray(boundary))
    clipped = np.concatenate(pieces)
    areas = triangle_areas(clipped)
    return clipped[areas > 0]


def triangle_areas(triangles):
    return 0.5 * np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1)


def sample_batches(triangles, density, seed=20260912, chunk=250000):
    """Uniform area-weighted IID samples; same stream at higher density.

    Three random numbers per point, fixed chunk size, no dependence on method.
    Sample count follows physical triangle area, never vertex count.
    """
    areas = triangle_areas(triangles)
    total = float(areas.sum())
    if not math.isfinite(total) or total <= 0 or density <= 0:
        raise ValueError('empty surface or invalid sample density')
    n = math.ceil(total * density)
    cdf = np.cumsum(areas)
    rng = np.random.default_rng(seed)
    for start in range(0, n, chunk):
        u = rng.random((min(chunk, n - start), 3))
        indices = np.searchsorted(cdf, u[:, 0] * total, side='right')
        t = triangles[np.minimum(indices, len(triangles) - 1)]
        r = np.sqrt(u[:, 1:2])
        yield (1 - r) * t[:, 0] + r * (1 - u[:, 2:3]) * t[:, 1] + r * u[:, 2:3] * t[:, 2]


def voxel_sample(triangles, core, density, seed):
    ids = np.empty(0, dtype=np.uint64)
    pending = []
    count = 0
    for xyz in sample_batches(triangles, density, seed):
        pending.append(core.voxel_batch_ids(xyz, .05, np.zeros(3, dtype=np.float64)))
        count += len(xyz)
        if len(pending) == 16:
            ids = np.union1d(ids, np.concatenate(pending))
            pending.clear()
            print(f'sampled={count} unique_voxels={len(ids)}', flush=True)
    if pending:
        ids = np.union1d(ids, np.concatenate(pending))
    return core.voxel_centers_local(ids, .05), count


def evaluate(plan_path, method_id, mesh_path, output, density_override=None):
    import trimesh
    plan_path = Path(plan_path)
    plan = json.loads(plan_path.read_text())
    spec = next(m for m in plan['methods'] if m['method_id'] == method_id)
    out = Path(output)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'refusing to overwrite: {out}')
    out.mkdir(parents=True, exist_ok=True)
    cache_path = Path(plan['reference_cache_manifest']['path'])
    if sha(cache_path) != plan['reference_cache_manifest']['sha256']:
        raise ValueError('reference cache binding changed')
    cache = json.loads(cache_path.read_text())
    for item in [cache['points'], cache['binding']['sim3'], cache['binding']['roi_configuration']]:
        if sha(item['path']) != item['sha256']:
            raise ValueError(f'identity mismatch: {item["path"]}')
    mesh = trimesh.load(str(mesh_path), force='mesh', process=False)
    if not hasattr(mesh, 'faces') or mesh.faces.ndim != 2 or mesh.faces.shape[1] != 3 or not len(mesh.faces):
        raise ValueError('mesh must contain triangles, not Gaussian centres')
    if (mesh.faces < 0).any() or (mesh.faces >= len(mesh.vertices)).any():
        raise ValueError('invalid triangle vertex index')
    sim3 = json.loads(Path(cache['binding']['sim3']['path']).read_text())
    vertices = to_metric(mesh.vertices, sim3, spec.get('normalization'))
    original_triangles = vertices[mesh.faces]
    degenerate = int((triangle_areas(original_triangles) == 0).sum())
    # Shift first so clipping/areas use local coordinates and preserve precision.
    origin = np.asarray(cache['binding']['local_origin_utm49n_normal_height_m'], dtype=np.float64)
    bounds = np.asarray(cache['binding']['roi_bounds_utm49n']) - origin[[0, 1, 0, 1]]
    clipped = clip_triangles(original_triangles - origin, bounds)
    density = density_override or plan['scoring']['samples_per_m2']
    if density not in (plan['scoring']['samples_per_m2'], plan['scoring']['convergence_samples_per_m2']):
        raise ValueError('sample density is not frozen in the plan')
    rect, core = load_rect_core(plan['scoring']['benchmark_repo'])
    surface, sample_count = voxel_sample(clipped, core, density, plan['scoring']['seed'])
    reference = np.load(cache['points']['path'], mmap_mode='r', allow_pickle=False)
    metrics, d_rec, d_ref = rect.summarize_exact(core, surface, reference, 250000, -1)
    metrics.update({
        'schema': 'uavgs_native_mesh_core3_extension_metrics_v1',
        'status': 'COMPLETE_PILOT_SCORED', 'method_id': method_id,
        'representation': spec['representation'], 'scene': plan['scene'],
        'plan_sha256': sha(plan_path), 'mesh_sha256': sha(mesh_path),
        'mesh_path': str(mesh_path), 'scoring_voxel_m': .05,
        'samples_per_m2': density, 'seed': plan['scoring']['seed'],
        'mesh_faces_before_crop': len(mesh.faces), 'degenerate_faces_removed': degenerate,
        'triangles_after_exact_rectangle_clip': len(clipped),
        'mesh_area_inside_roi_m2': float(triangle_areas(clipped).sum()),
        'sample_count': sample_count, 'reconstruction_voxel_count': len(surface),
        'reference_voxel_count': len(reference), 'reference_binding': cache['binding']['canonical_sha256'],
        'local_origin_utm49n_normal_height_m': origin.tolist(),
        'old_results_modified': False, 'registration_refit': False,
        'camera_set': 'frozen_train' + str(len(plan['cameras'])), 'interpretation': 'native postprocessed Mesh within the frozen image-defined rectangular ROI',
    })
    np.savez_compressed(out/'surface_voxel_centres.npz', points_local_m=surface, local_origin_utm49n_normal_height_m=origin)
    np.save(out/'reconstruction_to_lidar.npy', d_rec)
    np.save(out/'lidar_to_reconstruction.npy', d_ref)
    write_json(out/'metrics.json', metrics)
    print(json.dumps({'status': metrics['status'], 'method_id': method_id, 'voxels': len(surface)}))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--plan', required=True)
    p.add_argument('--method', required=True)
    p.add_argument('--mesh', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--density', type=int)
    a = p.parse_args()
    evaluate(a.plan, a.method, a.mesh, a.output, a.density)
