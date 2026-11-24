"""Integrity/import/small synthetic checks only; no real datasets, CUDA, network or experiments."""
import argparse
import contextlib
import hashlib
import importlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest


def verify(root):
    manifest=json.loads((root/'REFERENCE_MANIFEST.json').read_text(encoding='utf8'))
    seen=set()
    for row in manifest['files']:
        rel=Path(row['path']);p=root/rel
        if rel.is_absolute() or '..' in rel.parts or root.resolve() not in p.resolve().parents or p.is_symlink():
            raise ValueError('Unsafe source member')
        if row['path'] in seen:raise ValueError('Duplicate manifest entry')
        seen.add(row['path']);data=p.read_bytes()
        if len(data)!=row['bytes'] or hashlib.sha256(data).hexdigest()!=row['sha256']:
            raise ValueError('Source mismatch: '+row['path'])
    actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    if actual!=seen|{'REFERENCE_MANIFEST.json'}:
        raise ValueError('Unexpected or missing files in source snapshot')
    kit=root/'reuse/evaluation_reference_v1'
    provenance=json.loads((kit/'PROVENANCE.json').read_text())
    for name,expected in provenance['unchanged_frozen_files'].items():
        if hashlib.sha256((kit/name).read_bytes()).hexdigest()!=expected:
            raise ValueError('Frozen Mesh source modified')
    if hashlib.sha256((root/'code/gcp/evaluate_uavgs_lidar_rectangular_roi_v1.py').read_bytes()).hexdigest()!=provenance['rectangular_core_sha256']:
        raise ValueError('Rectangular metric core changed')
    return {'status':'PASS_SOURCE_SHA_VERIFICATION','verified_files':len(seen),
            'manifest_sha256':hashlib.sha256((root/'REFERENCE_MANIFEST.json').read_bytes()).hexdigest()}


def smoke(root):
    sys.path[:0]=[str(root/'code/gcp'),str(root/'reuse/evaluation_reference_v1'),
                  str(root/'reuse/evaluation_reference_v1/native_mesh')]
    import numpy as np
    modules={n:importlib.import_module(n) for n in ['numpy','scipy','pyproj','shapely','laspy','trimesh','PIL']}
    import trimesh.exchange.ply
    import evaluate_m3m_native_quarter_geometry
    import evaluate_m3m_native_quarter_rgb_quality
    import evaluate_uavgs_lidar_rectangular_roi_v1 as rect
    import evaluate_m3m_gcp_lidar_formal_v1 as core
    import mesh_score as mesh
    import exact_voxel_union as union
    import score_native_mesh as entry
    import test_metric_depth_packet as packet_tests
    import test_evaluate_uavgs_lidar_rectangular_roi_v1 as roi_tests
    for module,path in [(rect,root/'code/gcp'/Path(rect.__file__).name),(core,root/'code/gcp'/Path(core.__file__).name),
                        (mesh,root/'reuse/evaluation_reference_v1/native_mesh/mesh_score.py')]:
        if Path(module.__file__).resolve()!=path.resolve():raise ValueError('Imported an external code copy')
    checks=[]
    def check(name,ok):
        if not ok:raise AssertionError(name)
        checks.append(name)
    check('real_dependency_and_evaluator_imports_no_torch','torch' not in sys.modules)
    check('float64_voxel_numeric_core',core.run_numeric_self_tests(.05)['status']=='PASS')
    tri=np.array([[[-1.,0.,0.],[2.,0.,0.],[.5,2.,0.]]])
    clipped=mesh.clip_triangles(tri,[0.,0.,1.,1.])
    check('exact_rectangle_clipping_area',np.isclose(mesh.triangle_areas(clipped).sum(),1.))
    samples=np.concatenate(list(mesh.sample_batches(clipped,200,seed=1)))
    repeated=np.concatenate(list(mesh.sample_batches(clipped,200,seed=1)))
    check('area_sample_count_and_determinism',len(samples)==200 and np.array_equal(samples,repeated))
    with contextlib.redirect_stdout(io.StringIO()):
        baseline,count=mesh.voxel_sample(clipped,core,200,1)
        original=mesh.voxel_sample
        try:
            union.install(mesh);optimized,count2=mesh.voxel_sample(clipped,core,200,1)
        finally:mesh.voxel_sample=original
    check('exact_union_preserves_sampled_voxel_centres',count==count2 and np.array_equal(baseline,optimized))
    u=union.ExactUnion();rng=np.random.default_rng(3);all_ids=[]
    for _ in range(35):
        ids=rng.integers(0,2000,size=100,dtype=np.uint64);all_ids.append(ids);u.add(ids)
    check('multilevel_union_exact_sorted_unique',np.array_equal(u.finish(),np.unique(np.concatenate(all_ids))))
    a=np.array([[0.,0.,0.],[.2,0.,0.]],dtype=np.float64)
    b=np.array([[0.,0.,0.],[.1,0.,0.]],dtype=np.float64)
    metrics,_,_=rect.summarize_exact(core,a,b,100,1)
    check('precision_recall_f5_f10_boundary',metrics['fscore_5cm']==.5 and metrics['fscore_10cm']==1.)
    sim={'transform':{'scale':2.,'rotation':np.eye(3).tolist(),'translation':[400000.,2800000.,50.]}}
    normal={'original_colmap_units_per_normalized_unit':2.,'translate_original_colmap':[3.,4.,5.]}
    x=np.array([[1.,2.,3.]])
    expected=mesh.to_metric(x,sim)
    actual=mesh.to_metric((x+np.array([3.,4.,5.]))/2.,sim,normal)
    check('normalization_inverse_and_no_double_height_shift',np.allclose(actual,expected) and actual[0,2]==56.)
    args=['--plan','example.json','--method','example','--mesh','example.ply','--output','unused']
    with contextlib.redirect_stdout(io.StringIO()) as output:entry.main(args)
    check('mesh_wrapper_default_is_dry_run',json.loads(output.getvalue())['status']=='DRY_RUN_NO_SCORING')
    try:
        with contextlib.redirect_stderr(io.StringIO()):entry.main(args+['--execute'])
    except SystemExit as error:check('mesh_wrapper_requires_coordinate_ack',error.code==2)
    else:raise AssertionError('Missing coordinate acknowledgment allowed')
    with tempfile.TemporaryDirectory(prefix='reference-synthetic-') as tmp:
        plan=Path(tmp)/'plan.json';plan.write_text(json.dumps({'methods':[{'method_id':'example','official_mesh':'post.ply'}]}))
        args[1]=str(plan)
        try:
            with contextlib.redirect_stderr(io.StringIO()):entry.main(args+['--execute','--acknowledge-benchmark-specific-coordinate-contract'])
        except SystemExit as error:check('raw_mesh_not_mislabeled_as_post',error.code==2)
        else:raise AssertionError('Raw/post identity check missing')
    packet=packet_tests.run_tests()
    check('all_cpu_metric_packet_cases',all(row['status']=='PASS' for row in packet))
    result=unittest.TextTestRunner(stream=io.StringIO()).run(unittest.defaultTestLoader.loadTestsFromModule(roi_tests))
    check('frozen_roi_configuration_regressions',result.wasSuccessful() and result.testsRun==3)
    check('no_torch_or_cuda_import','torch' not in sys.modules)
    return {'status':'PASS_REAL_IMPORTS_AND_SYNTHETIC_ONLY','check_groups':checks,
        'metric_packet_cpu_cases':len(packet),'roi_unit_tests':result.testsRun,
        'versions':{n:getattr(m,'__version__','unspecified') for n,m in modules.items()},
        'real_model_lidar_scoring_performed':False,'formal_rgb_metrics_executed':False,'torch_imported':False}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2])
    g=p.add_mutually_exclusive_group(required=True);g.add_argument('--verify-only',action='store_true');g.add_argument('--smoke',action='store_true')
    a=p.parse_args();r={'integrity':verify(a.root.resolve())}
    if a.smoke:r['synthetic']=smoke(a.root.resolve())
    print(json.dumps(r,indent=2))
