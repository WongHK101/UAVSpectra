"""Explicit entry to the frozen benchmark Mesh scorer, NOT a generic CRS adapter."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['plan','method','mesh','output']:
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--acknowledge-benchmark-specific-coordinate-contract',action='store_true')
    args=parser.parse_args(argv)
    if not args.execute:
        print(json.dumps({'status':'DRY_RUN_NO_SCORING','plan':args.plan,'method':args.method,
            'mesh':args.mesh,'output':args.output,'coordinate_contract':'EPSG4545 to EPSG32649; normal height',
            'requires_new_project_port_before_reuse':True}))
        return 0
    if not args.acknowledge_benchmark_specific_coordinate_contract:
        parser.error('Read README: this preserved scorer has benchmark-specific coordinates, ROI and datum')
    plan=json.loads(Path(args.plan).read_text(encoding='utf-8-sig'))
    method=next(m for m in plan['methods'] if m['method_id']==args.method)
    if Path(args.mesh).resolve()!=Path(method['official_mesh']).resolve():
        parser.error('This frozen output schema labels postprocessed Mesh; raw Mesh needs a separately labeled track')
    folder=Path(__file__).resolve().parent/'native_mesh'
    sys.path.insert(0,str(folder))
    spec=importlib.util.spec_from_file_location('reference_frozen_mesh',folder/'mesh_score.py')
    scorer=importlib.util.module_from_spec(spec);spec.loader.exec_module(scorer)
    from exact_voxel_union import install
    install(scorer)
    scorer.evaluate(args.plan,args.method,args.mesh,args.output)
    return 0


if __name__=='__main__':raise SystemExit(main())
