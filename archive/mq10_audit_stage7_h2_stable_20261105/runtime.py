from pathlib import Path
import os, sys
BASE=Path(__file__).resolve().parent
ROOT=BASE.parent/'mq10'
for k,v in {'TEMP':'tmp','TMP':'tmp','TMPDIR':'tmp','XDG_CACHE_HOME':'cache','MPLCONFIGDIR':'cache/mpl','NUMBA_CACHE_DIR':'cache/numba','JOBLIB_TEMP_FOLDER':'tmp','PYTHONPYCACHEPREFIX':'cache/pycache'}.items():
    os.environ[k]=str(BASE/v)
os.environ['PYTHONDONTWRITEBYTECODE']='1'
os.environ['PYTENSOR_FLAGS']='compiledir='+str(BASE/'cache/pytensor')
os.environ['MQ10_FLOW_MODEL']=str(ROOT/'data/interim/p1d/flow_model.joblib')
sys.dont_write_bytecode=True
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0,str(BASE/'source'))
def inside(p):
    return isinstance(p,int) or Path(os.fsdecode(p)).resolve().is_relative_to(BASE)

def process_allowed(exe, argv, cwd, env):
    # Windows audits the final command line, and executable may be None.
    # Accept only canonical list2cmdline encodings of the exact helper grammar.
    import ctypes, subprocess
    if isinstance(argv, str):
        from ctypes import wintypes as W
        shell32=ctypes.WinDLL('shell32',use_last_error=True)
        kernel32=ctypes.WinDLL('kernel32',use_last_error=True)
        shell32.CommandLineToArgvW.argtypes=[W.LPCWSTR,ctypes.POINTER(ctypes.c_int)]
        shell32.CommandLineToArgvW.restype=ctypes.POINTER(W.LPWSTR)
        kernel32.LocalFree.argtypes=[ctypes.c_void_p]
        kernel32.LocalFree.restype=ctypes.c_void_p
        n=ctypes.c_int(); ptr=shell32.CommandLineToArgvW(argv,ctypes.byref(n))
        if not ptr:return False
        try: words=[ptr[i] for i in range(n.value)]
        finally:kernel32.LocalFree(ptr)
        if subprocess.list2cmdline(words)!=argv:return False
    elif isinstance(argv,(tuple,list)) and all(isinstance(a,str) for a in argv):
        words=list(argv)
    else:return False
    if not words or words[0]!=str(Path(sys.executable).resolve()):return False
    if exe is not None and (not Path(exe).is_absolute() or Path(exe).resolve()!=Path(sys.executable).resolve()):return False
    if cwd is None or Path(cwd).resolve()!=BASE:return False
    if env is not None and env!=dict(os.environ):return False
    tail=words[1:]
    if not tail or tail.pop(0)!='-B':return False
    if tail and tail[0]=='-u':tail.pop(0)
    if not tail or tail.pop(0)!=str(BASE/'runner.py'):return False
    if tail in [['--probe',k] for k in ('lock','fail','partial','isolation','interrupt')]:return True
    if len(tail)==4 and tail[:3]==['--probe','resume','--signature']:
        return len(tail[3])==64 and all(c in '0123456789abcdef' for c in tail[3])
    count={'--snapshot':1,'--compare-safety':3}.get(tail[0] if tail else '')
    if count is None or len(tail)!=count+1:return False
    paths=[Path(p) for p in tail[1:]]
    return all(p.is_absolute() and p.resolve().is_relative_to(BASE) for p in paths)

def guard(event,args):
    if event=='open':
        p,m,f=args
        write=(isinstance(m,str) and any(c in m for c in 'wax+')) or (isinstance(f,int) and bool(f&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND)))
        if write and not inside(p): raise PermissionError('WRITE_PATH_VIOLATION '+str(p))
    if event in ('os.mkdir','os.remove','os.rmdir','os.chmod','os.utime') and not inside(args[0]): raise PermissionError('WRITE_PATH_VIOLATION')
    if event in ('os.rename','os.link','os.symlink') and not (inside(args[0]) and inside(args[1])): raise PermissionError('WRITE_PATH_VIOLATION')
    if event=='socket.connect':
        frame=sys._getframe(); internal=False
        while frame:
            if frame.f_code.co_name=='_fallback_socketpair' and frame.f_globals.get('__name__')=='socket': internal=True
            frame=frame.f_back
        if not (internal and args[1][0] in ('127.0.0.1','::1')): raise PermissionError('NETWORK_FORBIDDEN')
    if event=='os.system': raise PermissionError('SHELL_FORBIDDEN')
    if event=='subprocess.Popen':
        exe,argv,cwd,env=args
        if not process_allowed(exe,argv,cwd,env): raise PermissionError('PROCESS_FORBIDDEN')
sys.addaudithook(guard)
