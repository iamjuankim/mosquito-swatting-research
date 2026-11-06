"""Standalone runner. No production is started by preflight."""
import sys
from pathlib import Path
BASE=Path(__file__).resolve().parent
# Snapshot helper intentionally runs before runtime's subprocess guard; Git is read-only.
if len(sys.argv)>1 and sys.argv[1]=='--snapshot':
    if len(sys.argv)!=3:raise SystemExit('Invalid snapshot arguments')
    from safety import snapshot
    snapshot(sys.argv[2])
    raise SystemExit(0)
if len(sys.argv)>1 and sys.argv[1]=='--compare-safety':
    if len(sys.argv)!=5:raise SystemExit('Invalid comparison arguments')
    from safety import compare
    compare(*sys.argv[2:5])
    raise SystemExit(0)
import common as C
from common import *
import argparse,contextlib,msvcrt,threading,traceback
@contextlib.contextmanager
def exclusive(name="execution.lock"):
    p=BASE/name
    with p.open('a+b') as f:
        if f.tell()==0:f.write(b'0');f.flush()
        f.seek(0)
        try:msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
        except OSError:raise RuntimeError('DUPLICATE_EXECUTION')
        try:yield
        finally:f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)
def peak_rss():
    import ctypes
    from ctypes import wintypes as W
    class PMC(ctypes.Structure):
        _fields_=[('cb',W.DWORD),('PageFaultCount',W.DWORD),('PeakWorkingSetSize',ctypes.c_size_t),('WorkingSetSize',ctypes.c_size_t),('QuotaPeakPagedPoolUsage',ctypes.c_size_t),('QuotaPagedPoolUsage',ctypes.c_size_t),('QuotaPeakNonPagedPoolUsage',ctypes.c_size_t),('QuotaNonPagedPoolUsage',ctypes.c_size_t),('PagefileUsage',ctypes.c_size_t),('PeakPagefileUsage',ctypes.c_size_t)]
    ctypes.windll.psapi.GetProcessMemoryInfo.argtypes=[W.HANDLE,ctypes.POINTER(PMC),W.DWORD]
    p=PMC();p.cb=ctypes.sizeof(p)
    k=ctypes.windll.kernel32;k.GetCurrentProcess.restype=W.HANDLE
    if not ctypes.windll.psapi.GetProcessMemoryInfo(k.GetCurrentProcess(),ctypes.byref(p),p.cb):raise ctypes.WinError()
    return p.PeakWorkingSetSize
def emit_heartbeat(path,state):
    atomic_json(path,dict(state,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),monotonic_s=time.monotonic(),rss_peak_bytes=peak_rss()))
def start_heartbeat(path,state,interval=10):
    stop=threading.Event()
    def beat():
        while not stop.wait(interval):emit_heartbeat(path,state)
    thread=threading.Thread(target=beat,daemon=True);thread.start()
    return stop,thread
def invoke(*args,log=None):
    import subprocess
    return subprocess.run([sys.executable,'-B','-u',str(BASE/'runner.py'),*map(str,args)],cwd=BASE,stdout=log,stderr=subprocess.STDOUT,check=True)
def postflight(sig):
    specs=read(BASE/'conditions.json');records=[];paired=[]
    for s in specs:
        with np.load(s['legacy_path'],allow_pickle=False) as old:
            if not np.array_equal(old['j'],np.arange(200)):raise RuntimeError('Legacy j mismatch')
            for j in range(200):
                z,m=load_checkpoint(s['id'],j,sig)
                d=z['captured'].astype(np.int8)-old['captured'][j].astype(np.int8)
                paired.append({'condition_id':s['id'],'j':j,'old_capture':int(old['captured'][j].sum()),'new_capture':int(z['captured'].sum()),'gain':int((d==1).sum()),'loss':int((d==-1).sum()),'delta':float(d.mean()),'paired_mcse':float(d.std(ddof=1)/np.sqrt(4000))})
                p=checkpoint_path(s['id'],j);records.append({'path':str(p.relative_to(BASE)),'sha256':digest(p),'bytes':p.stat().st_size})
    atomic_json(BASE/'production'/'output_manifest.json',records)
    atomic_json(BASE/'production'/'paired_capture_by_draw.json',paired)
    atomic_json(BASE/'production'/'postflight.json',{'passed':True,'conditions':48,'batches':9600,'trajectories':38400000,'signature':sig})
def production(args):
    g=read(BASE/'GO.json')
    if not g.get('go') or g['signature']!=signature():raise RuntimeError('NO_GO or code/environment drift')
    verify_inputs();sig=signature()
    if (BASE/'production'/'COMPLETE.json').exists():raise RuntimeError('Already COMPLETE; no additional simulations')
    outdir=BASE/'production';outdir.mkdir(exist_ok=True)
    if (outdir/'status.json').exists() and read(outdir/'status.json').get('phase')=='FAILED':raise RuntimeError('Previous failure requires manual review; automatic retry forbidden')
    session=time.strftime('%Y%m%d_%H%M%S')+'_'+str(os.getpid())
    before=outdir/f'safety_{session}_start.json';after=outdir/f'safety_{session}_end.json'
    # Always collect fresh start/end protection evidence for every production/restart session.
    invoke('--snapshot',before)
    state={'phase':'running','complete':False,'signature':sig,'pid':os.getpid(),'session':session,'completed':0,'current':None}
    stop,thread=start_heartbeat(outdir/'heartbeat.json',state)
    start=time.monotonic();failure=None
    try:
        ctx=context();count=0;stopped=False
        for s in read(BASE/'conditions.json'):
            for j in range(200):
                p=checkpoint_path(s['id'],j)
                if p.exists() or p.with_suffix('.json').exists():
                    load_checkpoint(s['id'],j,sig);count+=1;state['completed']=count;continue
                if (BASE/'STOP_REQUESTED').exists() or time.monotonic()-start>=args.max_hours*3600:
                    stopped=True;break
                state['current']={'condition_id':s['id'],'j':j}
                print(time.strftime('%Y-%m-%d %H:%M:%S'),'START',s['id'],j,flush=True)
                z,wall=compute(ctx,s,j,'stable');p=save_checkpoint(s['id'],j,z,sig,wall)
                count+=1;state['completed']=count
                print('DONE',s['id'],j,'wall_s',wall,'bytes',p.stat().st_size,'peak_rss_bytes',peak_rss(),flush=True)
                atomic_json(outdir/'status.json',state)
            if stopped:break
        if count==9600:
            state['phase']='postflight';postflight(sig)
        else:state['phase']='stopped_resumable'
    except BaseException as e:
        failure=e;state.update(phase='FAILED',error=repr(e),traceback=traceback.format_exc())
    finally:
        stop.set();thread.join()
        try:
            invoke('--snapshot',after)
            invoke('--compare-safety',before,after,outdir/f'safety_{session}_comparison.json')
        except BaseException as e:
            state.update(phase='FAILED',protection_error=repr(e));failure=failure or e
        if failure is None and state['completed']==9600:
            state.update(phase='COMPLETE',complete=True)
            atomic_json(outdir/'COMPLETE.json',{'complete':True,'signature':sig,'batches':9600,'trajectories':38400000,'postflight_sha256':digest(outdir/'postflight.json'),'safety_comparison':f'safety_{session}_comparison.json'})
        atomic_json(outdir/'status.json',state);atomic_json(outdir/'heartbeat.json',dict(state,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
    if failure:raise failure
def main():
    a=argparse.ArgumentParser()
    a.add_argument('--production',action='store_true');a.add_argument('--approve-full-48',action='store_true')
    a.add_argument('--max-hours',type=float,default=12)
    a.add_argument('--probe',choices=['lock','fail','partial','resume','isolation','interrupt']);a.add_argument('--signature')
    args=a.parse_args()
    with exclusive("execution.lock" if not args.probe or args.probe=="lock" else "probe.lock"):
        if args.probe:
            try:
                if args.probe=='interrupt':
                    p=checkpoint_path('_interruptprobe',0).with_name('j000.npz.partial')
                    p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'OS interruption probe')
                    atomic_json(BASE/'probe_interrupt_ready.json',{'complete':False,'pid':os.getpid()})
                    time.sleep(60)
                    raise RuntimeError('Parent did not interrupt probe')
                if args.probe=='isolation':
                    import socket,subprocess
                    checks=[]
                    def denied(label,fn):
                        try:fn()
                        except PermissionError:checks.append(label)
                        else:raise RuntimeError('Child accepted '+label)
                    denied('external_write',lambda:(BASE.parent/'stage7_forbidden_child.tmp').write_bytes(b'forbidden'))
                    denied('process',lambda:subprocess.run([sys.executable,'-c','pass'],cwd=BASE))
                    with socket.socket() as sock:denied('network',lambda:sock.connect(('192.0.2.1',9)))
                    atomic_json(BASE/'probe_isolation.json',{'passed':True,'checks':checks,'no_simulations':True})
                if args.probe=='fail':raise RuntimeError('EXPECTED_PROBE_FAILURE')
                if args.probe=='partial':
                    p=checkpoint_path('_resumeprobe',0).with_name('j000.npz.partial');p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'interrupted write')
                    raise RuntimeError('EXPECTED_INTERRUPTION')
                if args.probe=='resume':
                    z,m=load_checkpoint('_resumeprobe',0,args.signature)
                    atomic_json(BASE/'probe_resume.json',{'passed':True,'no_simulations':True,'arrays':npzhash_arrays(z)})
                return
            except BaseException as e:
                atomic_json(BASE/'probe_failure.json',{'complete':False,'phase':'FAILED','error':repr(e)});raise
        if not args.production or not args.approve_full_48:a.error('Explicit --production --approve-full-48 required')
        production(args)
if __name__=='__main__':
    try:main()
    except BaseException:
        traceback.print_exc();sys.exit(1)
