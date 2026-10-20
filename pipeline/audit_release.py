"""Scan publishable source and all reachable Git objects without exposing matches."""
import argparse
from pathlib import Path
import re
import subprocess

EXCLUDE={'.git','.venv','__pycache__','.pytest_cache','.ruff_cache','dist','build','runs','output','local'}
BINARY={'.ckpt','.pt','.pth','.onnx','.usd','.usda','.usdc','.usdz','.npz','.npy','.h5','.hdf5',
        '.mp4','.mov','.avi','.exr','.exe','.dll','.pem','.key','.jsonl','.zip','.bundle'}


def inspect_text(text):
    patterns={
        'absolute_drive_path':r'(?<![A-Za-z])[A-Za-z]:(?:\\+|/)',
        'user_home_path':'/'+'(?:Us'+'ers|home)/[^\\s<>]+|/[a-z]/Us'+'ers/',
        'github_credential':r'gh'+r'[pousr]_[A-Za-z0-9]{20,}|github_'+r'pat_[A-Za-z0-9_]{20,}',
        'model_service_credential':r'sk-'+r'[A-Za-z0-9_-]{24,}',
        'cloud_credential':r'AK'+r'IA[0-9A-Z]{16}',
        'private_key':'-----BEGIN '+r'(?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
    }
    errors=[name for name,pattern in patterns.items() if re.search(pattern,text,re.I)]
    # Keywords such as USD Token are legitimate. Only suspicious literal assignments fail.
    assignment=re.compile(r'''(?i)(?:api_key|password|secret|access_token)\s*[=:]\s*["']([^"']{8,})["']''')
    for value in assignment.findall(text):
        if not value.startswith(('<LOCAL_','${','<YOUR_')):errors.append('credential_literal')
    return sorted(set(errors))


def inspect_file(name,data):
    p=Path(name);errors=[]
    if p.suffix.lower() in BINARY:errors.append('excluded_binary_or_ledger')
    if p.name.startswith(('bonus-report-','auto-run-')) or any(x in ('plan24-out','plan25-out') for x in p.parts):
        errors.append('raw_experiment_artifact')
    if len(data)>100*1024*1024:errors.append('over_100MiB')
    if p.parts and p.parts[0]=='results' and p.suffix not in ('.md','.csv','.txt'):
        errors.append('results_must_be_sanitized_text_or_metrics')
    try:text=data.decode('utf8')
    except UnicodeDecodeError:errors.append('non_text_source');return errors
    return sorted(set(errors+inspect_text(text)))


def git_objects(root):
    lines=subprocess.check_output(['git','rev-list','--objects','--all'],cwd=root).decode().splitlines()
    ids=[s.split(' ',1)[0] for s in lines]
    labels={s.split(' ',1)[0]:s.split(' ',1)[1] if ' ' in s else '(commit)' for s in lines}
    child=subprocess.Popen(['git','cat-file','--batch'],cwd=root,stdin=subprocess.PIPE,stdout=subprocess.PIPE)
    stream,_=child.communicate(('\n'.join(ids)+'\n').encode())
    if child.returncode:raise RuntimeError('Git object read failed')
    offset=0
    for oid in ids:
        end=stream.index(b'\n',offset);header=stream[offset:end].decode().split();size=int(header[2])
        data=stream[end+1:end+1+size];offset=end+size+2
        yield header[1],labels[oid],data


def audit(root,history=True):
    root=Path(root).resolve();errors=[];files=[]
    for p in sorted(root.rglob('*')):
        relative=p.relative_to(root)
        if p.is_file() and not any(part in EXCLUDE for part in relative.parts):
            files.append(relative.as_posix())
            for reason in inspect_file(relative.as_posix(),p.read_bytes()):errors.append(dict(file=relative.as_posix(),reason=reason))
    objects=0;commits=None;tags=[]
    if history and (root/'.git').exists():
        for kind,name,data in git_objects(root):
            objects+=1
            reasons=inspect_file(name,data) if kind=='blob' else inspect_text(data.decode('utf8')) if kind in ('commit','tag') else []
            for reason in reasons:errors.append(dict(history_object=name,reason=reason))
        commits=int(subprocess.check_output(['git','rev-list','--count','--all'],cwd=root))
        tags=subprocess.check_output(['git','tag','--list'],cwd=root).decode().splitlines()
    return dict(status='PASS' if not errors else 'FAIL',source_files=len(files),
        source_bytes=sum((root/n).stat().st_size for n in files),reachable_objects=objects,
        reachable_commits=commits,tags=tags,errors=errors,
        credential_review='No detected credentials; SDK Token keywords are not authentication secrets')


def main():
    import json
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',default='.')
    p.add_argument('--source-only',action='store_true');a=p.parse_args();r=audit(a.root,not a.source_only)
    print(json.dumps(r,indent=2))
    if r['status']!='PASS':raise SystemExit(1)


if __name__=='__main__':main()
