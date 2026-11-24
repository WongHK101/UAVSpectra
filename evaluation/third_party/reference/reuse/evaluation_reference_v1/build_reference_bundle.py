"""Build a source-only evaluation reference snapshot; no SSH, data or experiment calls."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

MODULES=['evaluate_m3m_native_quarter_geometry','evaluate_m3m_native_quarter_rgb_quality',
 'evaluate_uavgs_lidar_rectangular_roi_v1','evaluate_m3m_gcp_lidar_formal_v1','metric_depth_packet',
 'test_evaluate_uavgs_lidar_rectangular_roi_v1','test_evaluate_m3m_gcp_lidar_formal_v1_numeric',
 'test_metric_depth_packet']
CONFIGS=['uavgs_image_defined_rectangular_roi_v1.json','m3m_gcp_native_quarter_current.json',
 'm3m_gcp_native_quarter_protocol_release_v2.json','m3m_gcp_native_quarter_rgb_quality_v1.json',
 'm3m_gcp_lidar_formal_v1.json','m3m_gcp_lidar_formal_artifact_schema_v1.json']
DOCS=['GS_GCP_NATIVE_QUARTER_ACTIVE_PROTOCOL_V2.md','M3M_GCP_NATIVE_QUARTER_RGB_QUALITY_V1.md',
 'M3M_GCP_LIDAR_FORMAL_PROTOCOL_V1.md','metric_depth_packet/METRIC_DEPTH_PACKET_SEMANTICS_20260626.md']


def digest(data):return hashlib.sha256(data).hexdigest()


def selected_files(repo):
    lookup={p.stem:p for folder in [repo/'code/gcp',repo/'code/colmap/utils'] for p in folder.glob('*.py')}
    pending=list(MODULES);seen=set()
    while pending:
        name=pending.pop()
        if name in seen:continue
        seen.add(name)
        for node in ast.walk(ast.parse(lookup[name].read_text(encoding='utf-8-sig'))):
            names=([a.name.split('.')[0] for a in node.names] if isinstance(node,ast.Import) else
                   [node.module.split('.')[0]] if isinstance(node,ast.ImportFrom) and node.module else [])
            pending.extend(n for n in names if n in lookup)
    files=[lookup[n] for n in sorted(seen)]
    files += [repo/'configs'/n for n in CONFIGS]+[repo/'docs'/n for n in DOCS]
    kit=repo/'reuse/evaluation_reference_v1'
    files += [p for p in kit.rglob('*') if p.is_file() and p.suffix in ('.py','.md','.json','.txt') and '__pycache__' not in p.parts]
    if any(p.name in ('continuation.py','runtime_queue.py','receive_terminal.py','legacy_terminal.py') for p in files):
        raise RuntimeError('Operational controller is not part of a reference bundle')
    return sorted(set(files))


def build(repo,output):
    repo=repo.resolve();output=output.resolve()
    if repo==output or repo in output.parents:
        raise ValueError('Build outside the source repository')
    output.mkdir(parents=True,exist_ok=False)
    snapshot=output/'GS-GCP-evaluation-reference';snapshot.mkdir()
    files=[]
    for p in selected_files(repo):
        data=p.read_bytes();rel=p.relative_to(repo).as_posix()
        dest=snapshot/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
        files.append({'path':rel,'bytes':len(data),'sha256':digest(data)})
    head=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    guide=(repo/'reuse/evaluation_reference_v1/README.md').read_bytes()
    (snapshot/'README.md').write_bytes(guide)
    files.append({'path':'README.md','bytes':len(guide),'sha256':digest(guide),
                  'source':'reuse/evaluation_reference_v1/README.md'})
    start=b'Read reuse/evaluation_reference_v1/README.md first.\nRun its verify-only command before importing code. This is a source reference, not an authorized experiment.\n'
    (snapshot/'START_HERE.md').write_bytes(start)
    files.append({'path':'START_HERE.md','bytes':len(start),'sha256':digest(start)})
    manifest={'schema':'gs_gcp_evaluation_reference_source_bundle_v1','source_git_head':head,
        'snapshot_sha256_is_authoritative_git_alone_is_not':True,'files':files,
        'scope':'reference source and synthetic checks; no datasets, credentials, weights or experiment controllers',
        'scientific_core_modified':False,'portable_source_reference_not_generic_dataset_cli':True}
    (snapshot/'REFERENCE_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf8')
    archive=output/'GS-GCP-evaluation-reference-20260922.zip'
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(snapshot.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(output).as_posix())
    receipt={'status':'BUILT_SOURCE_REFERENCE_PENDING_VALIDATION','archive':str(archive),
        'sha256':digest(archive.read_bytes()),'bytes':archive.stat().st_size,
        'manifest_sha256':digest((snapshot/'REFERENCE_MANIFEST.json').read_bytes()),'files':len(files)}
    (output/'build_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf8')
    print(json.dumps(receipt,indent=2));return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[2])
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();build(a.repo,a.output)
