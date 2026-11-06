"""One-shot Stage9G CFI0-4 and authorized sequential P1/P2; never resumes."""
import isolation as ISO
from isolation import BASE as B
from pathlib import Path
import os,sys,time,json,hashlib,ast,difflib,itertools,traceback,shutil,csv,ctypes
from datetime import datetime,timezone
import importlib.metadata as IM
T0=time.perf_counter(); CPU0=time.process_time()
S7=B.parent.parent/'mq10_audit_stage7_h2_stable_20261105'; ROOT=B.parent.parent/'mq10'
SCIENCE_DEADLINE=datetime(2026,11,6,7,0,tzinfo=timezone.utc)
REPORT_DEADLINE=datetime(2026,11,6,9,0,tzinfo=timezone.utc)
RES=[]; G={f'CFI{i}':{'status':'BLOCKED','reason':'not reached'} for i in range(5)}
STATE={'status':'RUNNING','attempted_batches':0,'completed_batches':0,'attempted_trajectories':0,'completed_trajectories':0,'P1_complete':False,'P2_complete':False,'automatic_retry':False,'P3_executed':False}
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(n,x):
    p=B/n;p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def code_hashes(cm):
    # Normalize existing keys only; never invent entries or bypass missing hashes.
    result={}
    for key,value in cm['code_sha256'].items():
        normalized=key.replace(chr(92),'/')
        check(normalized not in result,'MANIFEST_NORMALIZED_KEY_COLLISION '+normalized)
        result[normalized]=value
    return result

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def ah(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def check(ok,msg):
    if not ok:raise RuntimeError(msg)
def flush():
    save('execution_state.json',STATE);save('cfi_gate_results.json',G)
    if RES:
        keys=list(dict.fromkeys(k for r in RES for k in r))
        with (B/'resource_usage.csv').open('w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(RES)
class PMC(ctypes.Structure):
    _fields_=[('cb',ctypes.c_ulong),('PageFaultCount',ctypes.c_ulong)]+[(n,ctypes.c_size_t) for n in ['PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage']]
class IOC(ctypes.Structure):
    _fields_=[(n,ctypes.c_ulonglong) for n in ['ReadOperationCount','WriteOperationCount','OtherOperationCount','ReadTransferCount','WriteTransferCount','OtherTransferCount']]
K=ctypes.WinDLL('kernel32',use_last_error=True); PS=ctypes.WinDLL('psapi',use_last_error=True)
K.GetCurrentProcess.restype=ctypes.c_void_p
PS.GetProcessMemoryInfo.argtypes=[ctypes.c_void_p,ctypes.POINTER(PMC),ctypes.c_ulong]
K.GetProcessIoCounters.argtypes=[ctypes.c_void_p,ctypes.POINTER(IOC)]
def counters():
    m=PMC();m.cb=ctypes.sizeof(m);io=IOC();h=K.GetCurrentProcess()
    check(PS.GetProcessMemoryInfo(h,ctypes.byref(m),m.cb),'GetProcessMemoryInfo failed')
    check(K.GetProcessIoCounters(h,ctypes.byref(io)),'GetProcessIoCounters failed')
    return {'rss_bytes':m.WorkingSetSize,'peak_rss_bytes':m.PeakWorkingSetSize,'read_bytes':io.ReadTransferCount,'write_bytes':io.WriteTransferCount,'read_operations':io.ReadOperationCount,'write_operations':io.WriteOperationCount}
def limits(science=False):
    c=counters();check(c['peak_rss_bytes']<1024**3,'RSS_1GiB_STOP')
    check(shutil.disk_usage(B).free>=5*10**9,'DISK_5GB_STOP')
    check(not ISO.VIOLATIONS,'ISOLATION_VIOLATION')
    check(datetime.now(timezone.utc)<(SCIENCE_DEADLINE if science else REPORT_DEADLINE),'DEADLINE_STOP')
    return c
def begin():return time.perf_counter(),time.process_time(),counters()
def metric(stage,st,**extra):
    t,c,a=st;b=counters()
    RES.append({'stage':stage,'wall_s':time.perf_counter()-t,'cpu_s':time.process_time()-c,'peak_rss_bytes':b['peak_rss_bytes'],'rss_bytes':b['rss_bytes'],'read_bytes':b['read_bytes']-a['read_bytes'],'write_bytes':b['write_bytes']-a['write_bytes'],'read_operations':b['read_operations']-a['read_operations'],'write_operations':b['write_operations']-a['write_operations'],**extra})
def exact(a,b,label):
    a,b=np.asarray(a),np.asarray(b)
    check(a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes(),'EXACT_MISMATCH '+label)
def array_map_equal(a,b,label):
    check(a.keys()==b.keys(),'SCHEMA_MISMATCH '+label)
    for k in a:exact(a[k],b[k],label+'/'+k)
def manifest_check(p):
    p=Path(p).resolve();r=INPUT.get(str(p))
    check(r is not None,'MISSING_STAGE7_INPUT_LINK '+str(p))
    h=sha(p);check(h==r['sha256'],'INPUT_CHANGED '+str(p))
    VERIFIED[str(p)]={'sha256':h,'bytes':p.stat().st_size}
def saved(spec,j):
    p=S7/'checkpoints'/spec['id']/f'j{j:03d}.npz'; side=p.with_suffix('.json')
    rel=str(p.relative_to(S7));check(rel in OUTPUT,'MISSING_STAGE7_OUTPUT_LINK '+rel)
    check(sha(p)==OUTPUT[rel]['sha256'],'STAGE7_OUTPUT_CHANGED '+rel)
    # Production manifest lists NPZ files. Sidecars are anchored on first use and rechecked on subsequent reads and at completion.
    h=sha(side)
    if str(side) in SIDECARS:check(h==SIDECARS[str(side)],'STAGE7_SIDECAR_CHANGED')
    else:SIDECARS[str(side)]=h
    marker=read(side);check(marker['complete'] and marker['signature']==SIGNATURE,'CHECKPOINT_SIGNATURE')
    check(sha(p)==marker['sha256'],'CHECKPOINT_SIDECAR_HASH')
    with np.load(p,allow_pickle=False) as f:
        z={k:f[k] for k in f.files if k!='metadata'};meta=json.loads(str(f['metadata']))
    check(meta['signature']==SIGNATURE and meta['condition_id']==spec['id'] and meta['j']==j and meta['complete'],'CHECKPOINT_METADATA')
    check(meta['arrays']=={k:ah(v) for k,v in z.items()},'CHECKPOINT_ARRAY_HASH')
    check(set(z)==SCHEMA,'UNEXPECTED_STAGE7_SCHEMA '+str(sorted(z)))
    EX.validate_arrays(z)
    return z
def small_lag_gate():
    h=np.array([[2.,3.,5.,7.,11.,13.],[17.,19.,23.,29.,31.,37.],[41.,43.,47.,53.,59.,61.]])
    i0=np.array([0,2,4]);out=[]
    for i in [4,5]:
        for f in [0.,1.,2.,0.25,1.5,4.75,7.25]:
            for mod,name in [(X,'B'),(L,'L')]:
                got=mod._lagged(h,i,i0,f);ref=[]
                q=int(np.floor(f));w=f-q
                for row,start in enumerate(i0):
                    vals=[]
                    for k in [i-q,i-q-1]:
                        vals.append(0. if name=='L' and k<start else h[row,max(k,start)])
                    ref.append((1-w)*vals[0]+w*vals[1])
                ref=np.array(ref)
                check(np.all(np.abs(got-ref)<=1e-14+1e-12*np.abs(ref)),'LAG_ALGEBRA_FAIL')
                out.append({'i':i,'f':f,'variant':name,'value':got.tolist(),'reference':ref.tolist()})
    save('lag_fixed_array_checks.json',{'status':'PASS','cases':out,'zero_integer_fractional_straddling_and_different_row_starts':True})
def independent_drive(loc,idx,variant):
    # Scalar references independently implement the stated two-reference formula.
    P=loc['P'];i=loc['i'];dt=loc['dt'];i0=loc['i0'];answer=[]
    for r in idx:
        def lag(h,tau):
            f=tau/dt;q=int(np.floor(f));w=f-q
            def at(k):return 0. if variant=='L' and k<i0[r] else float(h[r,max(k,int(i0[r]))])
            return (1-w)*at(i-q)+w*at(i-q-1)
        th=lag(loc['hist_thA'],P['tau_v'])
        sz=det.C2_MV*np.exp(-(np.log(max(th,1e-6))-np.log(P['theta_pk']))**2/(2*P['sigma_theta']**2))
        answer.append(loc['c_vis'][r]*P['w_v']*(lag(loc['hist_V'],P['tau_v'])+sz/P['rho_vs'])+P['w_v']*P['rho_av']*lag(loc['hist_air'],P['tau_a']))
    ref=np.asarray(answer);got=loc['D'][idx]
    check(np.all(np.abs(got-ref)<=1e-14+1e-12*np.abs(ref)),'DRIVE_ALGEBRA_FAIL')
    return float(np.max(np.abs(got-ref),initial=0))
def batch_args(ctx,spec,j):
    P,raw=PARAMS[j];batch=EX.mc_batch(ctx,j,4000,'a',raw);c=X.Condition(**spec['condition']);esc,b=EX.draw_escape(ctx.boot,j);k=EX.draw_k(j,c)
    rng=np.random.default_rng([20261105,j,0]);pick=rng.integers(0,len(ctx.init),4000);u,za=X.common_randoms(rng,4000);z=rng.normal(size=4000)
    exact(batch.u,u,'full_uniform');exact(batch.z_a,za,'z_a');exact(batch.eta,raw*z,'MC_eta')
    fields={k:v for k,v in vars(batch).items() if v is not None}
    fields.update(pick=pick,z=z,raw_sigma_sess=np.asarray(raw),escape_b=np.asarray(b),k=np.asarray(k))
    fields.update({'P/'+name:np.asarray(v) for name,v in P.items()})
    for disk,e in esc.items():fields.update({'escape/'+disk+'/'+name:np.asarray(v) for name,v in vars(e).items()})
    fields.update({'condition/'+name:np.asarray(v if v is not None else 'None') for name,v in vars(c).items()})
    return batch,c,P,esc,k,fields
def compute(ctx,spec,j,variant,args):
    batch,c,P,esc,k,fields=args
    check(STATE['attempted_batches']<40,'BATCH_CAP')
    limits(True)
    check(time.perf_counter()-SCI_START<30*60,'SCIENCE_30MIN_BUDGET')
    STATE['attempted_batches']+=1;STATE['attempted_trajectories']+=4000
    STATE['current']={'variant':variant,'condition_id':spec['id'],'j':j};flush()
    diag={'M0':np.full(4000,np.nan),'theta':np.full(4000,np.nan),'i0':np.full(4000,-1,np.int64),'trigger_i':np.full(4000,-1,np.int64),'stimulus':np.zeros(4000,bool),'spontaneous':np.zeros(4000,bool),'u_threshold':batch.u[:,0].copy()}
    extra={'eta':batch.eta+c.theta_shift,'F0':np.full(4000,np.nan),'logZ':np.full(4000,np.nan),'log_theta':np.full(4000,np.nan),'final_logS':np.full(4000,np.nan)}
    prevM=np.zeros(4000);prevS=np.ones(4000);started=0;obs_time=0.;max_drive_error=0.;max_u_error=0.;max_tail_scaled=0.;max_S_increase=0.
    def observe(event,loc):
        nonlocal started,obs_time,max_drive_error,max_u_error,max_tail_scaled,max_S_increase
        tt=time.perf_counter()
        if event=='start':
            m=loc['new'];started+=int(m.sum())
            for d,s in [('M0','M'),('theta','Theta'),('i0','i0')]:diag[d][m]=loc[s][m]
            result=loc['result'];check(not result.unresolved.any(),'STRICT_UNRESOLVED')
            extra['F0'][m]=ndtr((np.log(loc['M'][m]+1e-12)-loc['eta'][m])/P['sigma'])
            extra['logZ'][m]=result.log_survival0;extra['log_theta'][m]=result.log_theta
            check(np.isfinite(result.log_survival0).all(),'NONFINITE_LOGZ')
            recovered=-np.expm1(log_ndtr(-result.z)-result.log_survival0)
            err=np.max(np.abs(recovered-batch.u[m,0]),initial=0);check(err<=2e-11,'UNIFORM_RECOVERY_FAIL');max_u_error=max(max_u_error,float(err))
            tol=64*np.finfo(float).eps*np.maximum(1.,np.abs(result.log_survival_target))
            ratio=float(np.max(np.abs(result.log_tail_residual)/tol,initial=0));check(ratio<=1,'TAIL_TOL_FAIL');max_tail_scaled=max(max_tail_scaled,ratio)
            check(np.where(batch.u[m,0]==0,loc['Theta'][m]>=loc['M'][m],loc['Theta'][m]>loc['M'][m]).all(),'CONDITIONING_FAIL')
            if j==0 and c.n_exp==1.:max_drive_error=max(max_drive_error,independent_drive(loc,np.flatnonzero(m),variant))
        if event=='trigger':
            m=loc['trig'];diag['trigger_i'][m]=loc['i'];diag['stimulus'][m]=(loc['M']>=loc['Theta'])[m];diag['spontaneous'][m]=(loc['t']-(loc['t0']+loc['i0']*loc['dt'])>=loc['t_sp'])[m]
        if event=='frame':
            limits(True)
            for key in ['X','V','M','D','u_air','Xn','Vn']:
                check(np.isfinite(loc[key]).all(),'NONFINITE '+key)
            m=loc['act'];check((loc['D'][m]>=0).all() and (loc['M'][m]>=prevM[m]).all(),'DRIVE_STATE_FAIL')
            logS=-P['lam0']*(loc['t']-(loc['t0']+loc['i0'][m]*loc['dt']))+log_ndtr(-(np.log(loc['M'][m]+1e-12)-loc['eta'][m])/P['sigma'])-extra['logZ'][m]
            ss=np.exp(logS);check(np.isfinite(logS).all() and np.isfinite(ss).all(),'SURVIVAL_NONFINITE')
            increase=float(np.max(ss-prevS[m],initial=0));max_S_increase=max(max_S_increase,increase)
            check((ss>=-1e-12).all() and (ss<=1+1e-12).all() and increase<=1e-12,'SURVIVAL_RANGE_MONOTONIC_FAIL')
            prevS[m]=ss;prevM[m]=loc['M'][m];extra['final_logS'][m]=logS
            if j==0 and c.n_exp==1.:
                ids=np.flatnonzero(m)[:3]
                if len(ids):max_drive_error=max(max_drive_error,independent_drive(loc,ids,variant))
        obs_time+=time.perf_counter()-tt
    st=begin()
    mod=X if variant=='B' else L
    out={k:np.asarray(v) for k,v in mod.simulate(batch,c,P,esc,k,ctx.kin,ctx.grid,ctx.U,ctx.Aref,threshold_mode='stable',observer=observe).items()}
    out.update(diag);out.update(j=np.asarray(j,np.int64),trajectory_id=np.arange(4000,dtype=np.int64))
    check(started==4000,'INITIALIZATION_INCOMPLETE');EX.validate_arrays(out);check(set(out)==SCHEMA,'OUTPUT_SCHEMA')
    check(all(np.isfinite(v).all() for v in extra.values()),'CONDITIONING_DIAGNOSTIC_NONFINITE')
    metric('simulation',st,phase='P1' if not STATE['P1_complete'] else 'P2',variant=variant,condition_id=spec['id'],j=j,N=4000,observer_validation_wall_s=obs_time)
    st=begin();p=B/'checkpoints'/variant/spec['id']/f'j{j:03d}.npz';p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('xb') as f:np.savez_compressed(f,**out)
    d=p.with_name(p.stem+'_conditioning.npz')
    with d.open('xb') as f:np.savez_compressed(f,**extra)
    save(str(p.with_suffix('.json').relative_to(B)),{'complete':True,'sha256':sha(p),'arrays':{k:ah(v) for k,v in out.items()},'conditioning_sha256':sha(d),'input_fingerprint':{k:{'shape':list(np.asarray(v).shape),'dtype':str(np.asarray(v).dtype),'sha256':ah(v)} for k,v in fields.items()},'numeric':{'max_uniform_recovery_error':max_u_error,'max_tail_residual_fraction_of_existing_tolerance':max_tail_scaled,'max_independent_drive_absolute_error':max_drive_error,'max_survival_increase':max_S_increase,'logZ_lt_log1e_8_count':int((extra['logZ']<np.log(1e-8)).sum())}})
    with np.load(p,allow_pickle=False) as z:array_map_equal(out,{k:z[k] for k in z.files},'checkpoint_roundtrip')
    metric('checkpoint_write_verify',st,variant=variant,condition_id=spec['id'],j=j,checkpoint_bytes=p.stat().st_size+d.stat().st_size)
    STATE['completed_batches']+=1;STATE['completed_trajectories']+=4000;flush()
    print('COMPLETE',STATE['completed_batches'],variant,spec['id'],j,flush=True)
    return out,extra
def preflight():
    global np,pd,ndtr,log_ndtr,det,flow,X,L,EX,INPUT,OUTPUT,VERIFIED,SIGNATURE,PARAMS,SCHEMA,SIDECARS
    st=begin()
    import numpy as np
    import pandas as pd
    from scipy.special import ndtr,log_ndtr
    import baseline as X
    import variant_L as L
    from swat import detection as det,flow,escape as E
    import extracted as EX
    from types import SimpleNamespace
    EX.np=np;EX.S=X;EX.E=E;EX.SEED_MC=20261105;EX.SEED_K=20261103;EX.SEED_ESC=20261104;EX.MC_SET_ID={'a':0,'b':1}
    SIDECARS={}
    VERIFIED={};INPUT={str(Path(r['path']).resolve()):r for r in read(S7/'input_manifest.json')};OUTPUT={r['path']:r for r in read(S7/'production/output_manifest.json')}
    cm=read(S7/'code_manifest.json');SIGNATURE=cm['signature']
    used=['executor.py','stable_threshold.py','common.py','source/run_original.py']+['source/swat/'+n+'.py' for n in ['__init__','detection','flow','escape','porous']]
    for name in used:check(sha(S7/name)==code_hashes(cm)[name],'STAGE7_CODE_CHANGED '+name)
    for name in ['__init__','detection','flow','escape','porous']:
        check((B/'source/swat'/(name+'.py')).read_text(encoding='utf-8')==(S7/'source/swat'/(name+'.py')).read_text(encoding='utf-8-sig'),'DEPENDENCY_COPY')
    check((B/'baseline.py').read_text(encoding='utf-8')==(S7/'executor.py').read_text(encoding='utf-8-sig'),'BASELINE_SOURCE_COPY')
    check((B/'stable_threshold.py').read_text(encoding='utf-8')==(S7/'stable_threshold.py').read_text(encoding='utf-8-sig'),'SAMPLER_SOURCE_COPY')
    oldenv=read(S7/'gate_results.json')['gates']['G1_inputs_code']['environment']
    env={'python':sys.version,'executable':str(Path(sys.executable).resolve()),**{k:IM.version(k) for k in ['numpy','scipy','pandas','xarray']},'platform':__import__('platform').platform(),'threads':{k:os.environ.get(k) for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']}}
    save('environment_comparison.json',{'historical':oldenv,'current':env,'differences':{k:{'historical':oldenv.get(k),'current':env.get(k)} for k in set(env)|set(oldenv) if env.get(k)!=oldenv.get(k)},'policy':'record operational differences; scientific identity established by input checks and exact sentinels'})
    conditions=read(S7/'conditions.json')
    specs=[s for s in conditions if s['model'] in ['original','calibrated'] and s['condition']['v_mps'] in [8.,12.] and s['condition']['n_exp'] in [.8,1.,1.2]]
    check(len(specs)==12,'CONDITION_SELECTION')
    specs.sort(key=lambda s:(['original','calibrated'].index(s['model']),s['condition']['n_exp'],s['condition']['v_mps']))
    for s in specs:
        c=s['condition'];check(not c['clap'] and c['phi']==0 and c['mc_set']=='a' and s['N']==4000 and s['prior']=='P1','CONDITION_CONTRACT')
        check(c['theta_shift']==(0. if s['model']=='original' else -.87) and c['k_fixed']==(None if s['model']=='original' else 8.2),'K_SHIFT_CONTRACT')
    save('frozen_selection_and_P3_rules.json',{'frozen_utc':datetime.now(timezone.utc).isoformat(),'normative_plan_sha256':sha(B.parent.parent/'mq10_audit_stage9d_reviewed_20261105/minimal_compute_plan_v2.md'),'P2_results_seen':False,'selection':[{'condition_id':s['id'],'j':j,'N':4000,'MC':'a','variant':'L'} for s in specs for j in [0,73,199]],'claim_target':'Whether all six model x n 8-to-12 decreases hold under L for the existing empirical P1 200 coordinates / MC a. Three draws cannot answer this target.','current_reporting_need':'Authorized P1/P2 case screening is sufficient for this task. P3 is not necessary to complete this task; a request for the stated 200-coordinate claim would require separate review and authorization.','expansion_not_based_on':['large effect','zero effect','favorable G','selected blocks or draws'],'P3_required_conditions':['explicit scientific need for stated claim, recorded before P2 viewing','all six model x n blocks, both speeds, all existing 200 coordinates, L only','report every block/failure/incomplete regardless of direction','CFI0-4 PASS; P2 all 36 L complete; no unresolved blocking numerical/definition/conditioning issue','200-coordinate parameter/chunk Gate under separate read authority','conservative measured execution+input+QA+protection+interpretation+report budget fits deadline','separate user approval for P3 and local unchanged G analysis'],'P3_new_batches_if_authorized':2364,'P3_reserved_batches':0,'P3_authorized':False})
    for rel in ['results/p3b/draws_P1.csv','results/p3b/draws_P1_meta.json','results/p2d_r/mcmc_P0_d8000/mcmc_exp2.json','data/interim/p3b/init_states.pkl','data/interim/p3b/flow_grid.npz','data/interim/p1d/flow_model.joblib','results/p2e/p2e_bootstrap.csv','data/interim/2024_bowwave/cfd/Data S3 - CFD/_CFD/full/piston_kinematic_corrected.csv']:manifest_check(ROOT/rel)
    dr=pd.read_csv(ROOT/'results/p3b/draws_P1.csv');check(len(dr)==200 and np.array_equal(dr.j,np.arange(200)),'DRAW_TABLE')
    check(np.all(dr.flat==dr.chain*8000+dr.draw),'FLAT_COORDINATES')
    meta=read(ROOT/'results/p2d_r/mcmc_P0_d8000/mcmc_exp2.json');zp=ROOT/meta['posterior_file']
    selmeta=read(ROOT/'results/p3b/draws_P1_meta.json')
    # The metadata file hash, rather than a newly generated mapping, anchors the original selection.
    coords={j:(int(dr.iloc[j].chain),int(dr.iloc[j].draw)) for j in [0,73,199]}
    variables=det.PARAM_NAMES+['a_sp','b_sp','sigma_sess','z_sess']
    access={};zmeta={}
    for group in [zp,zp/'posterior']:
        manifest_check(group/'zarr.json')
    for name in variables:
        p=zp/'posterior'/name/'zarr.json';manifest_check(p);m=read(p);zmeta[name]=m
        check(m['dimension_names'][:2]==['chain','draw'] and m['shape'][:2]==[4,8000] and m['data_type']=='float64','ZARR_DIMENSION_TYPE '+name)
        chunks=m['chunk_grid']['configuration']['chunk_shape'];check(m['chunk_key_encoding']=={'name':'default','configuration':{'separator':'/'}},'ZARR_CHUNK_ENCODING')
        paths=set()
        for chain,draw in coords.values():
            tails=[range((m['shape'][d]+chunks[d]-1)//chunks[d]) for d in range(2,len(chunks))]
            for tail in itertools.product(*tails):
                key=(chain//chunks[0],draw//chunks[1],*tail);q=zp/'posterior'/name/'c'/Path(*map(str,key));manifest_check(q);paths.add(str(q))
        access[name]={'dimensions':m['dimension_names'],'shape':m['shape'],'chunks':chunks,'verified_chunks':sorted(paths)}
    G['CFI0']={'status':'PASS','environment_recorded':env,'source_manifest_hashes_verified':len(used),'input_files_verified':len(VERIFIED),'Zarr_access_scope':access,'protection_scope':'actually used inputs/code and selected checkpoints only; application audit guard plus workspace sandbox; no ACL probes or full tree scan','historical_P0_lineage':'unproven; disclosed, not reintroduced as automatic blocker','final_protection_pending':True};flush()
    limits();metric('imports_and_integrity',st)
    st=begin()
    import zarr
    zarr.config.set({'async.concurrency':1,'threading.max_workers':1})
    post=zarr.open_group(str(zp),mode='r',zarr_format=3,use_consolidated=False)['posterior']
    PARAMS={};records={}
    class Vals:
        def __init__(self,name,coord,cache):self.name=name;self.coord=coord;self.cache=cache
        def __getitem__(self,coord):
            check(coord==self.coord,'UNAPPROVED_COORDINATE')
            if self.name not in self.cache:self.cache[self.name]=np.asarray(post[self.name][coord])
            return self.cache[self.name]
    class Adapter:
        def __init__(self,coord):self.coord=coord;self.cache={}
        def __getitem__(self,name):
            check(name in variables,'UNAPPROVED_VARIABLE')
            return SimpleNamespace(values=Vals(name,self.coord,self.cache))
    init=pd.read_pickle(ROOT/'data/interim/p3b/init_states.pkl');check(len(init)==841,'INITIAL_STATES')
    ns=int(init.session_idx.max())+1
    for j,coord in coords.items():
        adapter=Adapter(coord);P=det.params_from_draw(adapter,*coord,1,ns);raw=float(adapter['sigma_sess'].values[coord]);PARAMS[j]=(P,raw)
        check(all(np.isfinite(v).all() for v in P.values()) and np.isfinite(raw),'PARAMETER_NONFINITE')
        check(P['sigma']>0 and P['tau_v']>=0 and P['tau_a']>=0 and P['lam0']>0 and raw>=0,'PARAMETER_DOMAIN')
        check(P['a_sp'].shape==(1,) and P['b_sp'].shape==(1,) and P['eta_sess'].shape==(ns,),'PARAMETER_SHAPE')
        exact(P['sigma'],float(adapter.cache['sigma'])+1e-3,'sigma_epsilon')
        records[str(j)]={'chain':coord[0],'draw':coord[1],'n_sessions':ns,'raw_sigma_sess':raw,'P':{k:np.asarray(v).tolist() for k,v in P.items()},'materialized_variable_shapes':{k:list(v.shape) for k,v in adapter.cache.items()},'materialized_bytes':sum(v.nbytes for v in adapter.cache.values()),'transform':'Exact existing det.params_from_draw via guarded .values[chain,draw] adapter, index-first Zarr __getitem__; no full variable materialization. No stored full P copy exists for direct comparison.'}
    kin=pd.read_csv(ROOT/'data/interim/2024_bowwave/cfd/Data S3 - CFD/_CFD/full/piston_kinematic_corrected.csv');kin=(kin.time_s.to_numpy(float),np.maximum.accumulate(np.maximum(kin.y_pos.to_numpy(float),0.)))
    fm=flow.load_model(ROOT/'data/interim/p1d/flow_model.joblib')
    ctx=SimpleNamespace(init=init,kin=kin,U=fm.U_ref_cfd_mps,Aref=det.reference_airflow(fm),grid=X.FlowGrid.load(ROOT/'data/interim/p3b/flow_grid.npz'),boot=pd.read_csv(ROOT/'results/p2e/p2e_bootstrap.csv'))
    accessed_zarr=sorted(p for p in ISO.READS if Path(p).is_relative_to(zp) and Path(p).is_file())
    check(all(p in VERIFIED for p in accessed_zarr),'UNVERIFIED_ZARR_FILE_READ')
    save('actual_zarr_read_access.json',{'paths':accessed_zarr,'all_observed_paths_manifest_verified':True,'selected_coordinates':{str(j):list(c) for j,c in coords.items()},'zarr_async_concurrency':1,'zarr_max_workers':1})
    save('selected_parameter_provenance.json',records);save('verified_input_manifest.json',VERIFIED)
    G['CFI1']={'status':'PASS','coordinates':records,'scope':'Only j0/73/199; same file bytes plus exact deterministic original transform; not direct comparison against an absent saved full P copy','MC_eta':'raw sigma_sess * new MC standard normal; observed eta_sess not substituted'};flush();limits();metric('limited_input_loading_and_parameters',st)
    st=begin();small_lag_gate()
    ba=ast.parse((B/'baseline.py').read_text(encoding='utf-8'));la=ast.parse((B/'variant_L.py').read_text(encoding='utf-8'))
    bn=[ast.dump(n,include_attributes=False) for n in ba.body if not isinstance(n,ast.FunctionDef) or n.name!='_lagged'];ln=[ast.dump(n,include_attributes=False) for n in la.body if not isinstance(n,ast.FunctionDef) or n.name!='_lagged'];check(bn==ln,'L_NONLOCAL_DIFF')
    for node in ast.parse((B/'extracted.py').read_text(encoding='utf-8')).body:
        if isinstance(node,ast.FunctionDef):
            origin=S7/('common.py' if node.name=='validate_arrays' else 'source/run_original.py')
            ref=next(n for n in ast.parse(origin.read_text(encoding='utf-8-sig')).body if isinstance(n,ast.FunctionDef) and n.name==node.name)
            check(ast.dump(node,include_attributes=False)==ast.dump(ref,include_attributes=False),'EXTRACTED_FUNCTION_CHANGED '+node.name)
    SCHEMA={'captured','t_contact_s','t_det_s','t_end_obs_s','dt_s','n_steps','M0','theta','i0','trigger_i','stimulus','spontaneous','u_threshold','j','trajectory_id'}
    sent=[s for s in specs if s['condition']['n_exp']==1.]
    for s in sent:
        z=saved(s,0)
        for key in ['t_det_s','t_contact_s','t_end_obs_s','M0','theta','u_threshold','dt_s']:check(z[key].dtype==np.float64,'STAGE7_FLOAT64_SCHEMA '+key)
    save('fixed_source_and_contract_review.json',{'baseline':'normalized source text exact Stage7 executor','L':'sole AST change _lagged; independently zero both pre-start references; unchanged interpolation precision','feature_definitions':'Stage7 theta, C1 max(backward derivative,0), scaled air; same dt and row i0; no future reference for active rows','size_term':'max(theta,1e-6), no forced total D/F0/Z','conditioning':'same strict log-tail sampler, finite exp(logTheta), same M>=Theta; sigma epsilon +1e-3; spontaneous uniform separate','RNG':'default_rng([20261105,j,0]) -> pick -> uniform(N,6) -> normal z_a -> normal z; original k seed20261103; escape seed20261104','actual_stage7_end_time_key':'t_end_obs_s (+inf for these MC batches); no t_end_s key exists in actual saved schema','not_claimed':'Stage4 full padding/fitting equivalence; historical full RNG direct match; historical P0 full lineage'})
    G['CFI2']={'status':'PASS','lag_cases':28,'baseline_text_exact':True,'L_only_lag_AST_diff':True,'original_extracted_function_ASTs_exact':True,'sampler_text_exact':True,'contract_review':'fixed_source_and_contract_review.json','schema':sorted(SCHEMA)};flush();metric('local_source_lag_schema_validation',st)
    check(time.perf_counter()-T0<2*3600,'PREPARATION_2H_BUDGET');return ctx,specs,sent
def summary_pair(spec,j,base,alt,ce):
    P,raw=PARAMS[j];batch=EX.mc_batch(CTX,j,4000,'a',raw);eta=batch.eta+spec['condition']['theta_shift']
    a=(np.log(base['M0']+1e-12)-eta)/P['sigma'];f0=ndtr(a);lz=log_ndtr(-a)
    r={'condition_id':spec['id'],'model':spec['model'],'n':spec['condition']['n_exp'],'v_mps':spec['condition']['v_mps'],'j':j,'N':4000,'MC':'a'}
    for name,z in [('B',base),('L',alt)]:
        cap,detected,censor=X.outcomes(z)
        r.update({f'{name}_captured':int(cap.sum()),f'{name}_capture_rate':float(cap.mean()),f'{name}_triggered':int(np.isfinite(z['t_det_s']).sum()),f'{name}_detected_before_contact_or_end':int(detected.sum()),f'{name}_no_trigger':int(np.isposinf(z['t_det_s']).sum()),f'{name}_no_capture':int((~cap).sum()),f'{name}_administratively_censored_detection':int((~cap&~detected).sum()),f'{name}_contact_censored_detection':int((cap&~detected).sum()),f'{name}_stimulus_trigger':int(z['stimulus'].sum()),f'{name}_spontaneous_trigger':int(z['spontaneous'].sum()),f'{name}_both_trigger':int((z['stimulus']&z['spontaneous']).sum())})
    bc,bd,_=X.outcomes(base);lc,ld,_=X.outcomes(alt)
    r.update(capture_B0_L1=int((~bc&lc).sum()),capture_B1_L0=int((bc&~lc).sum()),capture_paired_delta_L_minus_B=float(lc.mean()-bc.mean()),detection_B0_L1=int((~bd&ld).sum()),detection_B1_L0=int((bd&~ld).sum()),detection_paired_delta_L_minus_B=float(ld.mean()-bd.mean()),M0_changed=int((base['M0']!=alt['M0']).sum()),M0_delta_mean=float((alt['M0']-base['M0']).mean()),B_F0_mean=float(f0.mean()),L_F0_mean=float(ce['F0'].mean()),F0_delta_mean=float((ce['F0']-f0).mean()),B_logZ_min=float(lz.min()),L_logZ_min=float(ce['logZ'].min()),logZ_delta_mean=float((ce['logZ']-lz).mean()),B_logZ_sensitive_count=int((lz<np.log(1e-8)).sum()),L_logZ_sensitive_count=int((ce['logZ']<np.log(1e-8)).sum()),threshold_changed=int((base['theta']!=alt['theta']).sum()))
    return r
def main():
    global CTX,SCI_START,used_code_hashes
    if (B/'execution_once.lock').exists():raise RuntimeError('NO_RESUME_OR_RETRY')
    with (B/'execution_once.lock').open('x') as f:f.write(datetime.now(timezone.utc).isoformat())
    cm=read(S7/'code_manifest.json');names=['executor.py','stable_threshold.py','common.py','source/run_original.py']+['source/swat/'+n+'.py' for n in ['__init__','detection','flow','escape','porous']]
    used_code_hashes={n:code_hashes(cm)[n] for n in names}
    CTX,specs,sent=preflight();SCI_START=time.perf_counter();pairs=[];new_baseline={};sentinel_matches=[]
    for s in sent:
        argsB=batch_args(CTX,s,0);argsL=batch_args(CTX,s,0)
        st=begin();array_map_equal(argsB[-1],argsL[-1],'B_L_full_batch');metric('full_batch_bit_comparison',st,condition_id=s['id'],j=0)
        base,be=compute(CTX,s,0,'B',argsB)
        st=begin();old=saved(s,0);array_map_equal(base,old,'Stage7_baseline/'+s['id']);metric('saved_baseline_exact_comparison',st,condition_id=s['id'],j=0)
        sentinel_matches.append({'condition_id':s['id'],'j':0,'arrays':{k:{'dtype':str(v.dtype),'shape':list(v.shape),'bit_exact':True} for k,v in base.items()},'new_B_L_full_input_randoms_bit_exact':True})
        alt,ce=compute(CTX,s,0,'L',argsL)
        pairs.append(summary_pair(s,0,old,alt,ce))
    G['CFI3']={'status':'PASS','sentinels':sentinel_matches,'past_full_rng_direct_comparison':'impossible: historical full pick/u/z arrays not stored; only saved u_threshold directly compared; connected via source/manifest/generation contract and four exact output sentinels','new_B_L_full_batch_direct_bit_comparison':True}
    G['CFI4']={'status':'PASS','strict_original_tolerance_unchanged':True,'all_P1_numeric_and_operating_checks_passed':True,'P2_checks':'required continuously per frame and checkpoint','RSS_stop_bytes':1024**3,'disk_min_bytes':5*10**9,'survival_tolerance':1e-12}
    STATE['P1_complete']=True;save('sentinel_exact_comparison.json',G['CFI3']);flush()
    sims=[r for r in RES if r['stage']=='simulation'];maxwall=max(r['wall_s'] for r in sims);remaining=32
    conservative=2*maxwall*remaining+90*60+30*60
    check((REPORT_DEADLINE-datetime.now(timezone.utc)).total_seconds()>conservative,'P2_TOTAL_BUDGET_NO_GO')
    check((SCIENCE_DEADLINE-datetime.now(timezone.utc)).total_seconds()>2*maxwall*remaining,'P2_SCIENCE_DEADLINE_NO_GO')
    check(30*60-(time.perf_counter()-SCI_START)>2*maxwall*remaining,'P2_30MIN_BUDGET_NO_GO')
    save('P1_measured_P2_feasibility.json',{'P1_batches':8,'P1_trajectories':32000,'max_observed_batch_wall_s_including_observer_validation':maxwall,'remaining_batches':32,'conservative_remaining_compute_s':2*maxwall*remaining,'reserved_postvalidation_protection_interpretation_reporting_s':90*60,'reserved_stop_cleanup_s':30*60,'total_feasible':True,'peak_RSS_bytes':counters()['peak_rss_bytes'],'disk_free_bytes':shutil.disk_usage(B).free,'extrapolation':'measured maximum x remaining x2; uncertain, sequential, no new parameter/seed/draw/N or tolerance'})
    # Frozen selection and P3 necessity/report rules already committed before any scientific output.
    for s in specs:
        for j in [0,73,199]:
            if j==0 and s['condition']['n_exp']==1.:continue
            check(all(G[f'CFI{i}']['status']=='PASS' for i in range(5)),'P2_GATE_BLOCK')
            args=batch_args(CTX,s,j);alt,ce=compute(CTX,s,j,'L',args)
            st=begin();base=saved(s,j);pairs.append(summary_pair(s,j,base,alt,ce));metric('paired_saved_baseline_read_and_summary',st,condition_id=s['id'],j=j)
    check(len(pairs)==36 and STATE['completed_batches']==40,'INCOMPLETE_P2')
    pairs.sort(key=lambda r:(r['model'],r['n'],r['v_mps'],r['j']))
    with (B/'paired_screening_results.csv').open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(pairs[0]));w.writeheader();w.writerows(pairs)
    speed=[]
    for model,n,j in itertools.product(['original','calibrated'],[.8,1.,1.2],[0,73,199]):
        a,b=[next(r for r in pairs if r['model']==model and r['n']==n and r['j']==j and r['v_mps']==v) for v in [8.,12.]]
        za={};
        for v in [8.,12.]:
            spec=next(s for s in specs if s['model']==model and s['condition']['n_exp']==n and s['condition']['v_mps']==v)
            with np.load(B/'checkpoints/L'/spec['id']/f'j{j:03d}.npz',allow_pickle=False) as f:lcap=f['captured']
            za[v]={'B':saved(spec,j)['captured'],'L':lcap}
        row={'model':model,'n':n,'j':j,'N':4000}
        for variant in ['B','L']:
            d=za[12.][variant].astype(int)-za[8.][variant].astype(int)
            row.update({variant+'_capture_rate_v8':a[variant+'_capture_rate'],variant+'_capture_rate_v12':b[variant+'_capture_rate'],variant+'_paired_delta_v12_minus_v8':float(d.mean()),variant+'_v8_0_v12_1':int((d==1).sum()),variant+'_v8_1_v12_0':int((d==-1).sum())})
        row['difference_of_paired_speed_deltas_L_minus_B']=row['L_paired_delta_v12_minus_v8']-row['B_paired_delta_v12_minus_v8'];speed.append(row)
    with (B/'paired_speed_contrasts.csv').open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(speed[0]));w.writeheader();w.writerows(speed)
    STATE.update(status='SCIENCE_COMPLETE',P2_complete=True,science_completed_utc=datetime.now(timezone.utc).isoformat(),new_baseline_batches=4,new_L_batches=36,P1_L_reused_in_P2=4)
    G['CFI4']['P2_all_36_L_numeric_and_operating_checks_passed']=True
    save('read_access_inventory.json',{'read_paths':sorted(ISO.READS),'violations':ISO.VIOLATIONS,'Zarr_read_scope':'metadata and selected-coordinate chunks only; physical decompression may span co-stored coordinates, materialized arrays index-first only'})
    check(not ISO.VIOLATIONS,'EXTERNAL_WRITE_VIOLATION')
    st=begin()
    for path,r in VERIFIED.items():check(sha(path)==r['sha256'],'FINAL_INPUT_CHANGED '+path)
    for path,h in SIDECARS.items():check(sha(path)==h,'FINAL_SIDECAR_CHANGED '+path)
    for name in used_code_hashes:check(sha(S7/name)==used_code_hashes[name],'FINAL_SOURCE_CHANGED '+name)
    for s in specs:
        for j in [0,73,199]:saved(s,j)
    save('targeted_preservation.json',{'status':'PASS','inputs':len(VERIFIED),'sidecars':len(SIDECARS),'used_code':len(used_code_hashes),'selected_stored_baselines':36,'full_tree_scan':False,'scope_limit':'Only consumed scientific inputs, source and selected checkpoints checked; no claim of whole-tree protection validation'})
    G['CFI0']['final_protection_pending']=False
    metric('targeted_final_preservation',st);flush()
    print('SCIENCE_COMPLETE 40 batches 160000 trajectories',flush=True)
if __name__=='__main__':
    try:main()
    except BaseException as e:
        save('isolation_failure_diagnostics.json',{'violations':ISO.VIOLATIONS,'null_opens':ISO.NULL_OPENS})
        STATE.update(status='STOPPED',error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc(),stopped_utc=datetime.now(timezone.utc).isoformat())
        # No rerun, repair, alternate seed, tolerance change or automatic continuation.
        active=next((f'CFI{i}' for i in range(5) if G[f'CFI{i}']['status']!='PASS'),None)
        if active:G[active]={'status':'FAIL' if not isinstance(e,(FileNotFoundError,PermissionError)) else 'BLOCKED','reason':str(e)}
        else:G['CFI4']={'status':'FAIL','reason':str(e)}
        save('failure_record.json',STATE);flush();print(traceback.format_exc(),flush=True);sys.exit(1)
