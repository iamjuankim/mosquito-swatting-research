import runtime
from runtime import BASE,ROOT
from pathlib import Path
import ast,hashlib,json,os,time,platform,sys
import numpy as np
import pandas as pd
import scipy
import xarray as xr
from types import SimpleNamespace
from swat import simulate as S, detection as det, flow, escape as E
import executor as X
SEED_MC,SEED_K,SEED_ESC=20261105,20261103,20261104
MC_SET_ID={'a':0,'b':1}
SCAN=BASE.parent/'mq10_audit_stage5_threshold_scan_20261104'
STAGE6=BASE.parent/'mq10_audit_stage6_66traj_20261105'
def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)
def objhash(x):return hashlib.sha256(canonical(x).encode()).hexdigest()
def atomic_json(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp')
    with tmp.open('w',encoding='utf-8') as f:
        f.write(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False));f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def code_hashes():
    paths=[BASE/n for n in ['runtime.py','common.py','executor.py','stable_threshold.py','runner.py','preflight.py','safety.py','conditions.json','preregistered_gates.json']]
    paths+=sorted((BASE/'source').rglob('*.py'))
    return {str(p.relative_to(BASE)):digest(p) for p in paths}
def environment():
    return {'python':sys.version,'executable':str(Path(sys.executable).resolve()),'numpy':np.__version__,'scipy':scipy.__version__,'pandas':pd.__version__,'xarray':xr.__version__,'platform':platform.platform(),'threads':{k:os.environ.get(k) for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']}}
def extract(path,names):
    nodes=[n for n in ast.parse(Path(path).read_text(encoding='utf-8-sig')).body if isinstance(n,ast.FunctionDef) and n.name in names]
    if len(nodes)!=len(names):raise RuntimeError('Extraction mismatch')
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),globals())
extract(BASE/'source/run_original.py',['mc_batch','draw_params','draw_k','draw_escape'])
def context():
    init=pd.read_pickle(ROOT/'data/interim/p3b/init_states.pkl')
    if len(init)!=841:raise RuntimeError('Initial states mismatch')
    meta=read(ROOT/'results/p2d_r/mcmc_P0_d8000/mcmc_exp2.json')
    post=xr.open_datatree(ROOT/meta['posterior_file'],engine='zarr')['posterior'].to_dataset()
    dr=pd.read_csv(ROOT/'results/p3b/draws_P1.csv')
    if len(dr)!=200 or not np.array_equal(dr.j,np.arange(200)):raise RuntimeError('Draw order mismatch')
    kin=pd.read_csv(ROOT/'data/interim/2024_bowwave/cfd/Data S3 - CFD/_CFD/full/piston_kinematic_corrected.csv')
    kin=(kin.time_s.to_numpy(float),np.maximum.accumulate(np.maximum(kin.y_pos.to_numpy(float),0.)))
    fm=flow.load_model(ROOT/'data/interim/p1d/flow_model.joblib')
    return SimpleNamespace(init=init,post=post,dr=dr,kin=kin,U=fm.U_ref_cfd_mps,Aref=det.reference_airflow(fm),grid=S.FlowGrid.load(ROOT/'data/interim/p3b/flow_grid.npz'),boot=pd.read_csv(ROOT/'results/p2e/p2e_bootstrap.csv'),n_sess=int(init.session_idx.max())+1)
def condition(spec):
    return S.Condition(**spec['condition'])
def batch_args(ctx,spec,j):
    row=ctx.dr.iloc[j]
    if int(row.j)!=j or int(row.flat)!=int(row.chain)*ctx.post.sizes['draw']+int(row.draw):raise RuntimeError('Draw mismatch')
    P=draw_params(ctx.post,row,ctx.n_sess)
    batch=mc_batch(ctx,j,4000,'a',float(ctx.post.sigma_sess.values[int(row.chain),int(row.draw)]))
    esc,b=draw_escape(ctx.boot,j)
    c=condition(spec)
    return batch,c,P,esc,draw_k(j,c)
def npzhash_arrays(z):
    return {k:hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest() for k,v in z.items()}
def compute(ctx,spec,j,mode):
    batch,c,P,esc,k=batch_args(ctx,spec,j)
    n=4000
    diag={'M0':np.full(n,np.nan),'theta':np.full(n,np.nan),'i0':np.full(n,-1,np.int64),'trigger_i':np.full(n,-1,np.int64),'stimulus':np.zeros(n,bool),'spontaneous':np.zeros(n,bool),'u_threshold':batch.u[:,0].copy()}
    started=0
    def observe(event,L):
        nonlocal started
        if event=='start':
            m=L['new'];started+=int(m.sum());diag['M0'][m]=L['M'][m];diag['theta'][m]=L['Theta'][m];diag['i0'][m]=L['i0'][m]
            if mode=='stable':
                if not np.isfinite(L['Theta'][m]).all() or not np.where(L['u_rand'][m,0]==0,L['Theta'][m]>=L['M'][m],L['Theta'][m]>L['M'][m]).all():raise RuntimeError('Conditioning failed')
        if event=='trigger':
            m=L['trig'];diag['trigger_i'][m]=L['i'];diag['stimulus'][m]=(L['M']>=L['Theta'])[m];diag['spontaneous'][m]=(L['t']-(L['t0']+L['i0']*L['dt'])>=L['t_sp'])[m]
        if event=='frame':
            for key in ['X','V','M','D','u_air','Xn','Vn']:
                if not np.isfinite(L[key]).all():raise FloatingPointError('Nonfinite '+key)
    t=time.perf_counter()
    r=X.simulate(batch,c,P,esc,k,ctx.kin,ctx.grid,ctx.U,ctx.Aref,threshold_mode=mode,observer=observe)
    wall=time.perf_counter()-t
    if started!=4000:raise RuntimeError('Not all trajectories initialized')
    out={key:np.asarray(value) for key,value in r.items()}
    out.update(diag);out['j']=np.asarray(j,np.int64);out['trajectory_id']=np.arange(n,dtype=np.int64)
    validate_arrays(out)
    return out,wall
def validate_arrays(z):
    for key in ['captured','t_det_s','t_contact_s','M0','theta','i0','trigger_i','stimulus','spontaneous','trajectory_id']:
        if z[key].shape!=(4000,):raise RuntimeError('Bad shape '+key)
    if z['captured'].dtype!=np.bool_ or not np.array_equal(z['trajectory_id'],np.arange(4000)):raise RuntimeError('Bad IDs/dtype')
    for key in ['t_det_s','t_contact_s']:
        if not np.all(np.isfinite(z[key])|np.isposinf(z[key])):raise RuntimeError('Bad time state')
    if not np.array_equal(z['captured'],np.isfinite(z['t_contact_s'])):raise RuntimeError('Solid capture/contact mismatch')
    finite=np.isfinite(z['t_det_s'])
    if not np.array_equal(finite,z['trigger_i']>=0) or not np.all(z['trigger_i'][finite]>z['i0'][finite]):raise RuntimeError('Illegal trigger')
    if not np.all((z['stimulus']|z['spontaneous'])[finite]):raise RuntimeError('Trigger condition missing')
    if not np.isfinite(z['theta']).all() or not np.isfinite(z['M0']).all():raise RuntimeError('Bad thresholds')
def compare_saved(out,spec,j):
    with np.load(spec['legacy_path'],allow_pickle=False) as z:
        if z['captured'].shape!=(200,4000) or not np.array_equal(z['j'],np.arange(200)):raise RuntimeError('Legacy schema mismatch')
        if not np.array_equal(out['captured'],z['captured'][j]):raise RuntimeError('Legacy capture mismatch')
        for key in ['t_det_s','t_contact_s']:
            if z[key].dtype!=np.float32 or not np.array_equal(out[key].astype(np.float32).view(np.uint32),z[key][j].view(np.uint32)):raise RuntimeError('Legacy bit mismatch '+key)
    return {'captured_exact':True,'time_float32_bits_and_states_exact':True,'ids':4000}
def signature():
    m=read(BASE/'input_manifest.json')
    return objhash({'code':code_hashes(),'inputs':m,'environment':environment()})
def verify_inputs():
    for r in read(BASE/'input_manifest.json'):
        if digest(r['path'])!=r['sha256']:raise RuntimeError('Input hash mismatch '+r['path'])
def checkpoint_path(cid,j):return BASE/'checkpoints'/cid/f'j{j:03d}.npz'
def save_checkpoint(cid,j,z,sig,wall):
    p=checkpoint_path(cid,j);p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():raise FileExistsError(p)
    meta={'signature':sig,'condition_id':cid,'j':j,'wall_s':wall,'arrays':npzhash_arrays(z),'complete':True}
    tmp=p.with_name(p.name+'.partial')
    with tmp.open('wb') as f:
        np.savez_compressed(f,**z,metadata=np.asarray(canonical(meta)));f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)
    # Sidecar is the final commit marker. A crash between renames is detected as incomplete.
    atomic_json(p.with_suffix('.json'),{'sha256':digest(p),'signature':sig,'complete':True})
    load_checkpoint(cid,j,sig)
    return p
def load_checkpoint(cid,j,sig):
    p=checkpoint_path(cid,j);marker=p.with_suffix('.json')
    if not p.exists() or not marker.exists():raise RuntimeError('INCOMPLETE_CHECKPOINT '+str(p))
    m=read(marker)
    if m['signature']!=sig or not m['complete'] or digest(p)!=m['sha256']:raise RuntimeError('Checkpoint hash/signature mismatch')
    with np.load(p,allow_pickle=False) as f:z={k:f[k] for k in f.files if k!='metadata'};meta=json.loads(str(f['metadata']))
    if meta['signature']!=sig or meta['condition_id']!=cid or meta['j']!=j or meta['arrays']!=npzhash_arrays(z) or not meta['complete']:raise RuntimeError('Checkpoint metadata mismatch')
    if int(z['j'])!=j:raise RuntimeError('Checkpoint j mismatch')
    validate_arrays(z)
    if not np.where(z['u_threshold']==0,z['theta']>=z['M0'],z['theta']>z['M0']).all():raise RuntimeError('Stable checkpoint conditioning')
    return z,meta
