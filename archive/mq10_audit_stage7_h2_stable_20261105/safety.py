"""Independent read-only protected-tree verifier; writes only its new output directory."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time
BASE=Path(__file__).resolve().parent
ROOT=BASE.parent/'mq10'
def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()
def snapshot(destination):
    destination=Path(destination).resolve()
    if not destination.is_relative_to(BASE): raise PermissionError(destination)
    if destination.exists(): raise FileExistsError(destination)
    roots=[ROOT]+sorted(p for p in BASE.parent.glob('mq10_audit_*') if p.is_dir() and p.resolve()!=BASE)
    out={}
    for root in roots:
        entries={}
        for parent,dirs,names in os.walk(root):
            for name in sorted(names):
                p=Path(parent)/name
                if p.is_symlink(): raise RuntimeError('Unexpected symlink '+str(p))
                a=p.stat(); sha=digest(p); b=p.stat()
                if (a.st_size,a.st_mtime_ns)!=(b.st_size,b.st_mtime_ns): raise RuntimeError('Changed while hashing '+str(p))
                entries[str(p.relative_to(root))]=[a.st_size,a.st_mtime_ns,sha]
        out[root.name]=entries
        print('protected',root.name,len(entries),flush=True)
    env=dict(os.environ,GIT_OPTIONAL_LOCKS='0')
    out['git']={k:subprocess.check_output(['git','-c','safe.directory='+str(ROOT),'-C',str(ROOT),*args],env=env).decode('utf-8') for k,args in [('head',['rev-parse','HEAD']),('status',['status','--porcelain=v1','--untracked-files=all'])]}
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(out,ensure_ascii=False,indent=1),encoding='utf-8')
    return out
def compare(a,b,out):
    a,b,out=[Path(p).resolve() for p in (a,b,out)]
    if not all(p.is_relative_to(BASE) for p in (a,b,out)):raise PermissionError('SAFETY_PATH_VIOLATION')
    if out.exists():raise FileExistsError(out)
    a=json.loads(Path(a).read_text(encoding='utf-8'));b=json.loads(Path(b).read_text(encoding='utf-8'))
    result={}
    for root in sorted((a.keys()|b.keys())-{'git'}):
        x,y=a.get(root,{}),b.get(root,{})
        result[root]={'added':sorted(y.keys()-x.keys()),'removed':sorted(x.keys()-y.keys()),'changed':sorted(k for k in x.keys()&y.keys() if x[k]!=y[k])}
    result['git_same']=a['git']==b['git']
    result['unchanged']=result['git_same'] and all(not any(v.values()) for v in result.values() if isinstance(v,dict))
    Path(out).write_text(json.dumps(result,indent=2),encoding='utf-8')
    if not result['unchanged']: raise RuntimeError('Protected tree changed')
    return result
if __name__=='__main__':
    if sys.argv[1]=='snapshot': snapshot(sys.argv[2])
    else: compare(*sys.argv[2:5])
