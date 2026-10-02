"""Capture immutable review evidence and an additive tests-only patch.

Run from this staging directory after tests; never edits root sources or parallel.
"""
import ast
import difflib
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from replay_empirical import definitions,datahash


def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    stage=Path(__file__).resolve().parent
    repo=stage.parents[2]
    parallel=stage.parent/'parallel'
    candidate_manifest=json.loads((parallel/'manifest.json').read_text())
    captured=[]
    for entry in candidate_manifest['files']:
        source=parallel/'candidate'/entry['path']
        target=stage/'review-snapshot'/entry['path']
        target.parent.mkdir(parents=True,exist_ok=True)
        raw=source.read_bytes();observed=hashlib.sha256(raw).hexdigest()
        target.write_bytes(raw)
        captured.append(dict(path=entry['path'],snapshot_sha256=observed,
                             matches_owner_manifest=observed==entry['candidate_sha256']))
    for name in ('manifest.json','HANDOFF.md','affine-vjp-results.json'):
        shutil.copyfile(parallel/name,stage/'review-snapshot'/name)
    (stage/'review-snapshot'/'snapshot.json').write_text(json.dumps(captured,indent=2)+'\n')
    source=repo/'tests/test_mamba2_vs_mamba3_real.py'
    ctx=definitions(source,None)
    tree=ast.parse(source.read_text(encoding='utf-8'))
    main_node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    nodes=[]
    for node in main_node.body:
        if isinstance(node,ast.Assign) and any(isinstance(child,ast.Call) and
            isinstance(child.func,ast.Name) and child.func.id=='build_mamba2_model' for child in ast.walk(node)): break
        nodes.append(node)
    main_node.name='datasets_only'
    main_node.body=nodes+[ast.Return(value=ast.Tuple(elts=[ast.Name(id=n,ctx=ast.Load()) for n in
        ('flappy_dataset','kv_train','kv_test','rev_train','rev_test')],ctx=ast.Load()))]
    ast.fix_missing_locations(main_node)
    exec(compile(ast.Module(body=[main_node],type_ignores=[]),str(source),'exec'),ctx)
    data=ctx['datasets_only']()
    datasets={name:dict(examples=len(values),sha256=datahash(values)) for name,values in
              zip(('flappy','recall_train','recall_test','last_token_train','last_token_test'),data)}
    (stage/'empirical-datasets.json').write_text(json.dumps(datasets,indent=2)+'\n')
    estimates=[]
    # Mirror only tape.workspace() allocations observed in the candidate source.
    for S in (1,32,256,1024):
      for R in (1,4):
        B,H,P,N,A=1,4,64,128,32
        for mode in ('dense_reference','parallel_fp32_v1','flash_fp32_v1'):
            mimo=R>1;norm=True
            local=2*N+2+2*R*N+(3*R*P if mimo else 0)+(P if norm else 0)
            total=2*N+2*H+2*H*R*N+(3*H*R*P if mimo else 0)+(H*P if norm else 0)
            history=B*H*(S+1)*P*N
            if mode=='flash_fp32_v1': history=B*H*((S+31)//32+1)*P*N
            rotation=B*H*S*R*N;phase=B*H*S*A;readout=B*H*S*R*P
            scratch=B*(P*N+3*R*N+2*R*P+A)
            elems=history+2*rotation+phase+readout+B*total+scratch
            if mode!='dense_reference':
                tokens=B*H*S
                elems+=readout+tokens*local+tokens*2*R*N+phase+history+2*tokens*P+3*tokens
            estimates.append(dict(B=B,H=H,P=P,N=N,S=S,R=R,mimo=mimo,out_norm=norm,
                                  provider=mode,reported_tape_workspace_bytes=4*elems,
                                  ssm_history_bytes=4*history))
    (stage/'workspace-estimates.json').write_text(json.dumps(dict(
        measured=False,scope='source formula for tape.workspace; excludes projections, weights, outputs, BLAS, allocator peaks',
        details=estimates),indent=2)+'\n')
    additions=('mamba3_scan_oracle.py','test_scan_recompute.py','replay_empirical.py',
               'probe_current_math.cpp','test_current_source.py','package_validation.py')
    patch=''
    for name in additions:
        lines=(stage/name).read_text(encoding='utf-8').splitlines(keepends=True)
        patch+=''.join(difflib.unified_diff([],lines,fromfile='/dev/null',
            tofile='b/OXN/nsos/tests/mamba3_validation/'+name))
    (stage/'vela-validation-tests.patch').write_text(patch,encoding='utf-8',newline='\n')
    check=subprocess.run(['git','apply','--check',str(stage/'vela-validation-tests.patch')],cwd=repo,capture_output=True,text=True)
    (stage/'patch-check.log').write_text(check.stdout+check.stderr+f'\nexit_code={check.returncode}\n')
    if check.returncode: raise RuntimeError('additive patch check failed')
    current_paths=['OXN/nsos/src/cuda/mamba3_layer_kernels.cu','OXN/nsos/include/mamba3_layer_math.h',
        'OXN/nsos/include/cuda/mamba3_layer_kernels.cuh','OXN/nsos/src/mamba3_layer.cpp',
        'OXN/nsos/src/jamba.cpp','OXN/nsos/src/bindings.cpp','OXN/nsos/src/runtime_execution_identity.cpp',
        'tests/test_mamba2_vs_mamba3_real.py']
    root_hashes={}
    for path in current_paths:
        raw=(repo/path).read_bytes();root_hashes[path]=hashlib.sha256(raw).hexdigest()
        if path.endswith(('runtime_execution_identity.cpp','mamba3_layer.cpp')):
            target=stage/'root-observed-snapshot'/path
            target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
    files={str(path.relative_to(stage)).replace('\\','/'):digest(path) for path in sorted(stage.rglob('*'))
        if path.is_file() and '__pycache__' not in path.parts and path not in (stage/'manifest.json',stage/'SHA256SUMS.txt')}
    info=dict(owner='Vela independent validation; root owns integration/GPU; root/mamba3_parallel owns kernels',
              source_root=str(repo),source_hashes=root_hashes,files=files,
              source_hash_scope='observed at packaging; CPU probe used source-snapshot headers and empirical used isolated pre-integration binary',
              current_cpu_probe_exe_sha256=digest(stage/'probe_current_math.exe'),
              prebuilt_cpu_module_sha256=digest(stage/'native-cpu/nsos_ext.cp312-win_amd64.pyd'),
              gpu_executed=False,shared_sources_written=False,shared_build_executed=False,
              patch_check_passed=True,reviewed_candidate_files=captured,
              missing_connected_note='NSOS-GPU-Mamba3-20261001 not exposed in maestri list; controls taken from AGENTS and current user instruction')
    (stage/'manifest.json').write_text(json.dumps(info,indent=2)+'\n')
    sums={**files,'manifest.json':digest(stage/'manifest.json')}
    (stage/'SHA256SUMS.txt').write_text(''.join(f'{value}  {key}\n' for key,value in sorted(sums.items())))
    print('VELA HANDOFF READY: patch SHA256='+digest(stage/'vela-validation-tests.patch'))
    print('VELA manifest SHA256='+digest(stage/'manifest.json'))


if __name__=='__main__': main()
