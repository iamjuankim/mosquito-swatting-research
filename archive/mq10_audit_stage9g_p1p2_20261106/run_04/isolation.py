from pathlib import Path
import os,sys
BASE=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
sys.stdout.reconfigure(encoding='utf-8')
for k,v in {'TEMP':'tmp','TMP':'tmp','TMPDIR':'tmp','XDG_CACHE_HOME':'cache','MPLCONFIGDIR':'cache/mpl','NUMBA_CACHE_DIR':'cache/numba','JOBLIB_TEMP_FOLDER':'tmp','PYTHONPYCACHEPREFIX':'cache/pycache'}.items():os.environ[k]=str(BASE/v)
os.environ['PYTHONDONTWRITEBYTECODE']='1'
os.environ['PYTENSOR_FLAGS']='compiledir='+str(BASE/'cache/pytensor')
os.environ['MQ10_FLOW_MODEL']=str(BASE.parent.parent/'mq10/data/interim/p1d/flow_model.joblib')
sys.path.insert(0,str(BASE/'source'))
# Avoid platform.py shell-based `ver` probe; native version information only.
import platform
_windows_version=sys.getwindowsversion()
platform._syscmd_ver=lambda *args,**kwargs: ('Windows',str(_windows_version.major),'.'.join(map(str,(_windows_version.major,_windows_version.minor,_windows_version.build))))
READS=set(); VIOLATIONS=[]; NULL_OPENS=[]
def inside(p):return isinstance(p,int) or Path(os.fsdecode(p)).resolve().is_relative_to(BASE)
def deny(msg):
    VIOLATIONS.append(msg)
    raise PermissionError(msg)
def guard(event,args):
    if event=='open':
        p,m,f=args
        if isinstance(p,(str,bytes)) and os.fsdecode(p).upper()=="NUL":
            NULL_OPENS.append({"device":os.fsdecode(p),"mode":m,"flags":f}); return  # bare Windows discard device; no persistent file
        write=(isinstance(m,str) and any(c in m for c in 'wax+')) or (isinstance(f,int) and bool(f&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND)))
        if write and not inside(p):deny('EXTERNAL_WRITE '+str(p))
        if not write and not isinstance(p,int):READS.add(str(Path(os.fsdecode(p)).resolve()))
    if event in ('os.mkdir','os.remove','os.rmdir','os.chmod','os.utime') and not inside(args[0]):deny('EXTERNAL_MUTATION '+event+' '+str(args[0]))
    if event in ('os.rename','os.link','os.symlink') and not (inside(args[0]) and inside(args[1])):deny('EXTERNAL_MUTATION '+event)
    if event=='socket.connect':
        frame=sys._getframe(); internal=False
        while frame:
            if frame.f_code.co_name=='_fallback_socketpair' and frame.f_globals.get('__name__')=='socket':internal=True
            frame=frame.f_back
        if not (internal and args[1][0] in ('127.0.0.1','::1')):deny('NETWORK_FORBIDDEN')
    if event in ('os.system','subprocess.Popen'):deny('PROCESS_FORBIDDEN '+event)
sys.addaudithook(guard)
