"""One-shot Stage9G CFI0-4 and authorized sequential P1/P2; never resumes."""
import isolation as ISO
from isolation import BASE as B
from pathlib import Path
import os,sys,time,json,hashlib,ast,difflib,itertools,traceback,shutil,csv,ctypes
from datetime import datetime,timezone
import importlib.metadata as IM
T0=time.perf_counter(); CPU0=time.process_time()
S7=B.parent/'mq10_audit_stage7_h2_stable_20261105'; ROOT=B.parent/'mq10'
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
    check(STATE['attempted_batches']<2364,'BATCH_CAP')
    limits(True)
    check(time.perf_counter()-SCI_START<SCI_BUDGET,'P3_MEASURED_SCIENCE_BUDGET')
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
    metric('simulation',st,phase='P3',variant=variant,condition_id=spec['id'],j=j,N=4000,observer_validation_wall_s=obs_time)
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
