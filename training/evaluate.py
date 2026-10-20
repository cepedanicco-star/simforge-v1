"""Fresh native episode: pure privileged BC, BC+servo/filter, or separate expert."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import yaml

ROOT=Path(__file__).resolve().parents[1]


def load(path):return yaml.safe_load(Path(path).read_text(encoding='utf8'))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()


def command_for(config,mode,directory):
    args=dict(load(config['arguments_file']));args['out']=str(directory/'physics')
    if mode!='expert':args['bc_checkpoint']=config['checkpoint']
    if mode in ('safe','preflight'):args['safe_parameters']=str(directory/'frozen-parameters.json')
    if mode=='preflight':
        args.update(scenario_version='v2',scenario_preflight_only=True,cup_offset_x=0.,cup_offset_y=.12,start_offset_y=0.)
    for key in ('scene','robot_config','asset_root','policy','walking_policy','cup_asset','transfer_plan','placement_plan','navigation_policy'):
        if args.get(key):args[key]=str(Path(args[key]).expanduser().resolve())
    worker='probe_apartment_g1.py' if mode=='expert' else 'probe_apartment_g1_privileged.py' if mode=='bc' else 'probe_apartment_g1_safe.py'
    command=[config['native_python'],str(ROOT/'pipeline/native'/worker)]
    for key,value in args.items():
        if value is None or value is False:continue
        command.append('--'+key.replace('_','-'))
        if value is not True:command.append(str(value))
    return command,args


def run_worker(command,out,environment,timeout):
    started=time.monotonic();ready=None;timeout_hit=False
    receipt=out/'physics/probe-report.json'
    with (out/'native.console.log').open('wb') as log:
        child=subprocess.Popen(command,env=environment,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        try:
            while child.poll() is None:
                if receipt.is_file():
                    try:terminal=load(receipt).get('status') in ('completed','failed','incomplete')
                    except (ValueError,yaml.YAMLError):terminal=False
                    if terminal:
                        ready=ready or time.monotonic()
                        if time.monotonic()-ready>5:child.terminate();break
                if time.monotonic()-started>timeout:timeout_hit=True;child.terminate();break
                time.sleep(.25)
            try:child.wait(timeout=15)
            except subprocess.TimeoutExpired:child.kill();child.wait(timeout=15)
        finally:
            if child.poll() is None:child.kill();child.wait(timeout=15)
    return dict(native=load(receipt) if receipt.exists() else None,harness_timeout=timeout_hit,exit_code=child.returncode)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True)
    p.add_argument('--mode',choices=['bc','safe','expert','preflight'],required=True);p.add_argument('--dry-run',action='store_true')
    p.add_argument('--seed',type=int);args=p.parse_args();config=load(args.config)
    for key in ('arguments_file','run_root','output','checkpoint','execution_parameters'):
        if key in config:config[key]=str(Path(config[key]).expanduser().resolve())
    native=config['native_python']
    config['native_python']=shutil.which(native) or str(Path(native).expanduser().resolve())
    out,root=Path(config['output']),Path(config['run_root'])
    if out==root or not out.is_relative_to(root):raise ValueError('output must be a fresh child of run_root')
    if out.exists():raise FileExistsError('Prior episode outputs are read-only')
    command,episode_args=command_for(config,args.mode,out)
    if any('<LOCAL_' in str(v) for v in [*episode_args.values(),config['native_python']]):
        raise ValueError('Fill local asset/runtime placeholders before launching')
    if args.dry_run:
        print(json.dumps(dict(will_launch=False,mode=args.mode,seed=args.seed,command=command),indent=2));return
    environment=dict(os.environ,SIMFORGE_RUN_ROOT=str(root),PYTHONPATH=str(ROOT))
    environment.update(config.get('environment',{}))
    if any('<LOCAL_' in str(v) for v in config.get('environment',{}).values()):raise ValueError('Unset optional overlays or fill their paths')
    protected=[config['arguments_file'],episode_args['scene']]
    if args.mode!='expert':protected.append(config['checkpoint'])
    if args.mode in ('safe','preflight'):protected.append(config['execution_parameters'])
    protected += [episode_args[k] for k in ('robot_config','policy','walking_policy','cup_asset','transfer_plan','placement_plan','navigation_policy') if episode_args.get(k)]
    hashes={str(Path(path).resolve(strict=True)):sha(path) for path in protected}
    out.mkdir(parents=True)
    if args.mode in ('safe','preflight'):
        (out/'frozen-parameters.json').write_text(json.dumps(load(config['execution_parameters']),indent=2),encoding='utf8')
    if args.seed is not None:
        from pipeline.simforge.eval.batch import prepare
        sample=prepare(config,args.seed,out/'initialization',episode_args)
        episode_args=sample['episode_arguments'];episode_args['out']=str(out/'physics')
        # Preserve exactly the same native options after measured seed authoring.
        command=command[:2]
        for k,v in episode_args.items():
            if v is None or v is False:continue
            command.append('--'+k.replace('_','-'))
            if v is not True:command.append(str(v))
    result=run_worker(command,out,environment,config.get('timeout_seconds',1200))
    result['inputs_unchanged']=all(sha(p)==h for p,h in hashes.items())
    ledger=out/'physics/bc-observations/decisions.jsonl'
    calls=len(ledger.read_text(encoding='utf8').splitlines()) if ledger.exists() else 0
    result.update(mode=args.mode,seed=args.seed,BC_calls=calls)
    native=result['native'] or {}
    if args.mode=='preflight':
        scene=out/'physics/scenario-v2.json';measured=load(scene) if scene.exists() else None
        passed=bool(measured and measured.get('G0_pass') is True and calls==0)
        result.update(scenario=measured,policy_success_rate=None)
    else:
        passed=native.get('status')=='completed' and native.get('full_transfer_completed') is True
        result.update(physical_success=passed,policy_success=passed if calls>0 else None,
                      expert_not_BC=args.mode=='expert')
    result['status']='PASS' if passed and not result['harness_timeout'] and result['inputs_unchanged'] else 'FAIL_STOP'
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(result['status'])
    if result['status']!='PASS':raise SystemExit(1)


if __name__=='__main__':main()
