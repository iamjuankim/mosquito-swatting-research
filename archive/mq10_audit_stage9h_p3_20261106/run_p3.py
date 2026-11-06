"""One authorized exploratory P3, sequential; never resumes or retries science."""
import p3_core as C
from p3_core import B,read,save,sha,check,exact,array_map_equal,begin,metric,limits
from pathlib import Path
from datetime import datetime,timezone
from types import SimpleNamespace
import os,sys,time,json,ast,itertools,traceback,shutil
G9=B.parent/'mq10_audit_stage9g_p1p2_20261106/run_04'
S8=B.parent/'mq10_audit_stage8_paired_20261105'
ANCHORS={}
def anchor(p,h=None):
    p=Path(p).resolve();v=sha(p)
    check(h is None or v==h,'INTEGRITY '+str(p))
    if str(p) in ANCHORS:check(ANCHORS[str(p)]==v,'ANCHOR_CHANGED '+str(p))
    ANCHORS[str(p)]=v
    return v
GM={}
for key,value in read(G9/'final_output_manifest.json')['files'].items():
    normalized=key.replace(chr(92),'/')
    check(normalized not in GM,'G9_MANIFEST_KEY_COLLISION '+normalized)
    GM[normalized]=value
def gcheck(name): return anchor(G9/name,GM[name]['sha256'])
def load_g(spec,j,args=None):
    stem=f"checkpoints/L/{spec['id']}/j{j:03d}"
    for ext in ['.npz','.json','_conditioning.npz']:gcheck(stem+ext)
    marker=read(G9/(stem+'.json'));p=G9/(stem+'.npz');d=G9/(stem+'_conditioning.npz')
    check(marker['complete'] and sha(p)==marker['sha256'] and sha(d)==marker['conditioning_sha256'],'G9_CHECKPOINT_LINK')
    with C.np.load(p,allow_pickle=False) as f:z={k:f[k] for k in f.files}
    check(marker['arrays']=={k:C.ah(v) for k,v in z.items()},'G9_ARRAY_HASH')
    C.EX.validate_arrays(z);check(set(z)==C.SCHEMA,'G9_SCHEMA')
    exact(z['j'],C.np.asarray(j,C.np.int64),'G9_j')
    if args is not None:
        fp={k:{'shape':list(C.np.asarray(v).shape),'dtype':str(C.np.asarray(v).dtype),'sha256':C.ah(v)} for k,v in args[-1].items()}
        check(fp==marker['input_fingerprint'],'G9_INPUT_FINGERPRINT')
    with C.np.load(d,allow_pickle=False) as f:ce={k:f[k] for k in f.files}
    check(all(C.np.isfinite(v).all() for v in ce.values()),'G9_DIAG_NONFINITE')
    return z,ce

def setup():
    st=begin()
    import numpy as np
    import pandas as pd
    from scipy.special import ndtr,log_ndtr
    import baseline as X
    import variant_L as L
    import extracted as EX
    import stable_threshold as ST
    from swat import detection as det,flow,escape as E
    import importlib.metadata as IM
    for k,v in locals().copy().items():
        if k in ['np','pd','ndtr','log_ndtr','X','L','EX','det']:setattr(C,k,v)
    EX.np=np;EX.S=X;EX.E=E;EX.SEED_MC=20261105;EX.SEED_K=20261103;EX.SEED_ESC=20261104;EX.MC_SET_ID={'a':0,'b':1}
    C.SIDECARS={};C.VERIFIED={}
    C.INPUT={str(Path(r['path']).resolve()):r for r in read(C.S7/'input_manifest.json')}
    C.OUTPUT={r['path']:r for r in read(C.S7/'production/output_manifest.json')}
    cm=read(C.S7/'code_manifest.json');C.SIGNATURE=cm['signature']
    for n,h in read(G9/'run_code_manifest.json').items():anchor(G9/n,h)
    for n in ['baseline.py','variant_L.py','stable_threshold.py','extracted.py']+[f'source/swat/{v}.py' for v in ['__init__','detection','flow','escape','porous']]:
        check((B/n).read_bytes()==(G9/n).read_bytes(),'LOCAL_SCIENCE_CODE '+n)
    used=['executor.py','stable_threshold.py','common.py','source/run_original.py']+[f'source/swat/{v}.py' for v in ['__init__','detection','flow','escape','porous']]
    for n in used:anchor(C.S7/n,C.code_hashes(cm)[n])
    for n in ['gate_summary.json','cfi_gate_results.json','sentinel_exact_comparison.json','frozen_selection_and_P3_rules.json','selected_parameter_provenance.json','resource_summary.json','resource_usage.csv','completion_status.json','targeted_preservation.json']:gcheck(n)
    check(all(v['status']=='PASS' for v in read(G9/'gate_summary.json').values()),'G9_GATES')
    check(read(G9/'execution_state.json')['completed_batches']==40,'G9_COMPLETION')
    env={'python':sys.version,'executable':str(Path(sys.executable).resolve()),**{k:IM.version(k) for k in ['numpy','scipy','pandas','xarray']},'platform':__import__('platform').platform(),'threads':{k:os.environ.get(k) for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']}}
    previous=read(G9/'environment_comparison.json')['current']
    check(env==previous,'ENVIRONMENT_MISMATCH_FROM_VERIFIED_RUN04')
    save('environment_identity.json',{'status':'PASS','current':env,'comparison':'single comparison with run_04; no repeated environment audit'})
    specs=[s for s in read(C.S7/'conditions.json') if s['model'] in ['original','calibrated'] and s['condition']['v_mps'] in [8.,12.] and s['condition']['n_exp'] in [.8,1.,1.2]]
    specs.sort(key=lambda s:(['original','calibrated'].index(s['model']),s['condition']['n_exp'],s['condition']['v_mps']))
    check(len(specs)==12,'CONDITIONS')
    for s in specs:
        c=s['condition'];check(not c['clap'] and c['phi']==0 and c['mc_set']=='a' and s['N']==4000 and s['prior']=='P1','CONDITION_CONTRACT')
        check(c['theta_shift']==(0. if s['model']=='original' else -.87) and c['k_fixed']==(None if s['model']=='original' else 8.2),'K_SHIFT_CONTRACT')
    frozen=read(G9/'frozen_selection_and_P3_rules.json')
    save('p3_analysis_contract.json',{'frozen_utc':datetime.now(timezone.utc).isoformat(),'analysis_type':'exploratory robustness analysis','prior_P3_necessity_condition_met':False,'evidence':frozen['current_reporting_need'],'P2_results_already_available':True,'authorization':'2026-11-06 user explicitly authorizes P3 and unchanged local Rule G despite absence of prior necessity commitment','selection':[{'condition_id':s['id'],'j':list(range(200))} for s in specs],'N':4000,'MC':'a','new_batch_cap':2364,'reuse_L_batches':36,'new_trajectory_cap':9456000,'rule_G_alpha':.05/35,'delta':.02,'scope':'local Delta_7 only; no overall H2/E/other adjacent interval recomputation','conditioning':'B and L define different start-survival conditional populations; common uniforms do not identify same-individual causal effects','plan_sha256':anchor(B.parent/'mq10_audit_stage9d_reviewed_20261105/minimal_compute_plan_v2.md')})
    # Anchor the existing Rule G functions against the already stored Stage8 input manifest.
    im8=read(S8/'input_manifest.json')['files']; judge=C.ROOT/'phases/p3b/judge.py'
    anchor(judge,im8[str(judge)]['sha256'])
    original=ast.parse(judge.read_text(encoding='utf-8-sig'));local=ast.parse((B/'local_judge.py').read_text(encoding='utf-8'))
    for n in ['interval','classify','effect_row']:
        get=lambda t:next(x for x in t.body if isinstance(x,ast.FunctionDef) and x.name==n)
        check(ast.dump(get(original),include_attributes=False)==ast.dump(get(local),include_attributes=False),'JUDGE_AST '+n)
    om8={str(Path(r['path']).resolve()):r for r in read(S8/'output_manifest.json')}
    for n in ['h2_effects_rule_G.csv','h2_effects_by_draw.csv']:
        p=S8/n;anchor(p,om8[str(p)]['sha256'])
    for rel in ['results/p3b/draws_P1.csv','results/p3b/draws_P1_meta.json','results/p2d_r/mcmc_P0_d8000/mcmc_exp2.json','data/interim/p3b/init_states.pkl','data/interim/p3b/flow_grid.npz','data/interim/p1d/flow_model.joblib','results/p2e/p2e_bootstrap.csv','data/interim/2024_bowwave/cfd/Data S3 - CFD/_CFD/full/piston_kinematic_corrected.csv']:C.manifest_check(C.ROOT/rel)
    dr=pd.read_csv(C.ROOT/'results/p3b/draws_P1.csv')
    check(len(dr)==200 and np.array_equal(dr.j,np.arange(200)) and np.all(dr.flat==dr.chain*8000+dr.draw),'ORIGINAL_SELECTION')
    coords={j:(int(dr.iloc[j].chain),int(dr.iloc[j].draw)) for j in range(200)}
    zp=C.ROOT/read(C.ROOT/'results/p2d_r/mcmc_P0_d8000/mcmc_exp2.json')['posterior_file']
    variables=det.PARAM_NAMES+['a_sp','b_sp','sigma_sess','z_sess'];access={}
    for group in [zp,zp/'posterior']:C.manifest_check(group/'zarr.json')
    for name in variables:
        p=zp/'posterior'/name/'zarr.json';C.manifest_check(p);m=read(p)
        check(m['dimension_names'][:2]==['chain','draw'] and m['shape'][:2]==[4,8000] and m['data_type']=='float64','ZARR_DIMENSION_TYPE '+name)
        chunks=m['chunk_grid']['configuration']['chunk_shape'];check(m['chunk_key_encoding']=={'name':'default','configuration':{'separator':'/'}},'ZARR_CHUNK_ENCODING')
        paths=set()
        for chain,draw in coords.values():
            tails=[range((m['shape'][d]+chunks[d]-1)//chunks[d]) for d in range(2,len(chunks))]
            for tail in itertools.product(*tails):paths.add(zp/'posterior'/name/'c'/Path(*map(str,(chain//chunks[0],draw//chunks[1],*tail))))
        for p in sorted(paths):C.manifest_check(p)
        access[name]={'dimensions':m['dimension_names'],'shape':m['shape'],'chunks':chunks,'verified_chunks':[str(p) for p in sorted(paths)]}
    C.G['CFI0']={'status':'PASS','source':'run_04 input/code/sentinel Gates reused; current targeted input/chunk and copied code integrity confirmed','verified_input_files':len(C.VERIFIED),'no_tree_scan':True,'no_WRITE_DAC':True}
    metric('imports_and_targeted_integrity',st);st=begin()
    import zarr
    zarr.config.set({'async.concurrency':1,'threading.max_workers':1})
    post=zarr.open_group(str(zp),mode='r',zarr_format=3,use_consolidated=False)['posterior']
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
    init=pd.read_pickle(C.ROOT/'data/interim/p3b/init_states.pkl');check(len(init)==841,'INIT_STATES');ns=int(init.session_idx.max())+1
    C.PARAMS={};records={};prior=read(G9/'selected_parameter_provenance.json')
    for j,coord in coords.items():
        adapter=Adapter(coord);P=det.params_from_draw(adapter,*coord,1,ns);raw=float(adapter['sigma_sess'].values[coord]);C.PARAMS[j]=(P,raw)
        check(all(np.isfinite(v).all() for v in P.values()) and np.isfinite(raw),'PARAMETER_NONFINITE')
        check(P['sigma']>0 and P['tau_v']>=0 and P['tau_a']>=0 and P['lam0']>0 and raw>=0,'PARAMETER_DOMAIN')
        check(P['a_sp'].shape==(1,) and P['b_sp'].shape==(1,) and P['eta_sess'].shape==(ns,),'PARAMETER_SHAPE')
        # Independent deterministic mapping, same dtype, dimension order and scalar conversion.
        manual={k:float(adapter.cache[k]) for k in det.PARAM_NAMES}
        manual['sigma']+=1e-3
        manual.update(a_sp=np.asarray(adapter.cache['a_sp']).reshape(1),b_sp=np.asarray(adapter.cache['b_sp']).reshape(1),eta_sess=raw*np.asarray(adapter.cache['z_sess']).reshape(ns))
        array_map_equal(P,manual,'parameter_transform_j'+str(j))
        if str(j) in prior:
            for k,v in P.items():exact(v,np.asarray(prior[str(j)]['P'][k]),'G9_P '+k)
            exact(raw,prior[str(j)]['raw_sigma_sess'],'G9_raw_sigma_sess')
        records[str(j)]={'chain':coord[0],'draw':coord[1],'n_sessions':ns,'raw_sigma_sess':raw,'P':{k:np.asarray(v).tolist() for k,v in P.items()},'materialized_shapes':{k:list(v.shape) for k,v in adapter.cache.items()},'materialized_bytes':sum(v.nbytes for v in adapter.cache.values()),'manual_mapping_bit_exact':True,'prior_run04_value_comparison':j in [0,73,199]}
    kin=pd.read_csv(C.ROOT/'data/interim/2024_bowwave/cfd/Data S3 - CFD/_CFD/full/piston_kinematic_corrected.csv');kin=(kin.time_s.to_numpy(float),np.maximum.accumulate(np.maximum(kin.y_pos.to_numpy(float),0.)))
    fm=flow.load_model(C.ROOT/'data/interim/p1d/flow_model.joblib')
    C.CTX=SimpleNamespace(init=init,kin=kin,U=fm.U_ref_cfd_mps,Aref=det.reference_airflow(fm),grid=X.FlowGrid.load(C.ROOT/'data/interim/p3b/flow_grid.npz'),boot=pd.read_csv(C.ROOT/'results/p2e/p2e_bootstrap.csv'))
    actual=sorted(p for p in C.ISO.READS if Path(p).is_relative_to(zp) and Path(p).is_file())
    check(all(p in C.VERIFIED for p in actual),'UNVERIFIED_ZARR_READ')
    save('selected_parameter_provenance.json',{'records':records,'reading':'Only fixed 200 chain/draw coordinates materialized, variables index-first float64; physical chunks may contain other coordinates','validation':'Stage7 manifest bytes + exact existing params_from_draw AST + independent mapping bit check; no saved full-P copy comparison claimed; observed session eta not substituted for raw-sigma MC eta'})
    save('actual_zarr_read_access.json',{'paths':actual,'selected_coordinates':{str(j):list(c) for j,c in coords.items()},'all_observed_paths_manifest_verified':True,'variables':access})
    save('verified_input_manifest.json',C.VERIFIED)
    C.G['CFI1']={'status':'PASS','fixed_coordinates':200,'n_sessions':ns,'index_first':True,'manual_mapping_bit_exact':True,'run04_three_draw_values_exact':True,'new_estimation_or_calibration':False}
    # Existing validated path and strict checks are byte identical; no repeated dynamic fixtures/sentinels.
    C.SCHEMA=set(read(G9/'cfi_gate_results.json')['CFI2']['schema'])
    C.G['CFI2']={'status':'PASS','reused':'run_04 CFI2 and CFI3; baseline/L/sampler/extracted/dependency bytes exact','new_diff':'P3 orchestration only; p3_core_changes.diff'}
    metric('limited_200_coordinate_loading',st);C.flush();st=begin()
    reuse=[];baseline_checks=0
    for s in specs:
        for j in range(200):
            limits();base=C.saved(s,j);args=C.batch_args(C.CTX,s,j);batch,c,P,esc,k,fields=args
            exact(base['j'],np.asarray(j,np.int64),'B_j')
            exact(base['u_threshold'],batch.u[:,0],'B_saved_u_threshold')
            sr=ST.sample_log_threshold(base['M0'],batch.eta+c.theta_shift,P['sigma'],batch.u[:,0],strict=True)
            exact(base['theta'],np.exp(sr.log_theta),'B_saved_threshold_reconstruction')
            baseline_checks+=1
            if j in [0,73,199]:
                alt,ce=load_g(s,j,args);exact(alt['u_threshold'],base['u_threshold'],'reuse_u')
                reuse.append({'condition_id':s['id'],'j':j,'source':str(G9/'checkpoints/L'/s['id']/f'j{j:03d}.npz'),'input_fingerprint_exact':True})
    check(len(reuse)==36 and baseline_checks==2400,'PREFLIGHT_COUNTS')
    save('reuse_and_baseline_gate.json',{'status':'PASS','reused_L':reuse,'stored_B_batches_checked':2400,'all_schema_array_hash_metadata_checks':True,'all_B_u_threshold_and_strict_theta_reconstruction_bit_exact':True,'new_baseline_simulations':0,'new_L_simulations':0,'historical_full_random_arrays':'not stored; same original generator and run_04 exact four sentinels reused; no historical full-RNG direct comparison claimed'})
    C.G['CFI3']={'status':'PASS','reused_L':36,'stored_B_checked':2400,'stored_B_threshold_reconstruction':'strict, float64 bit exact for all 9.6M stored trajectories','run04_input_fingerprints':'all 36 exact','new_baseline_simulations':0}
    C.G['CFI4']={'status':'PASS','preflight':'run_04 numerical Gates reused; all 2400 stored B strict thresholds reconstruct','production':'unchanged observer requires finite drives, strict original tail tolerance, u recovery 2e-11, survival atol1e-12 every frame; ongoing checks required','RSS_limit_bytes':1024**3,'disk_min_bytes':5*10**9}
    metric('stored_B_and_reused_L_connection_gate',st)
    rr=C.pd.read_csv(G9/'resource_usage.csv')
    maxsim=float(rr.loc[rr.stage=='simulation','wall_s'].max());maxwrite=float(rr.loc[rr.stage=='checkpoint_write_verify','wall_s'].max());maxpair=float(rr.loc[rr.stage=='paired_saved_baseline_read_and_summary','wall_s'].max())
    base_est=(maxsim+maxwrite+maxpair)*2364;C.SCI_BUDGET=2*base_est+600
    now=datetime.now(timezone.utc);available=(C.SCIENCE_DEADLINE-now).total_seconds();totalavailable=(C.REPORT_DEADLINE-now).total_seconds()
    check(available>C.SCI_BUDGET and totalavailable>C.SCI_BUDGET+7200,'P3_TIME_NO_GO')
    limits()
    save('p3_resource_gate.json',{'status':'PASS','checked_utc':now.isoformat(),'source':'run_04 actual maximum costs','max_simulation_s':maxsim,'max_checkpoint_s':maxwrite,'max_pair_read_summary_s':maxpair,'remaining_new_batches':2364,'single_pass_estimate_s':base_est,'two_times_estimate_plus_600s_IO_startup_budget_s':C.SCI_BUDGET,'postvalidation_report_reserve_s':5400,'stop_cleanup_reserve_s':1800,'science_time_available_s':available,'report_time_available_s':totalavailable,'peak_RSS_bytes':C.counters()['peak_rss_bytes'],'disk_free_bytes':shutil.disk_usage(B).free,'expected_output_upper_bytes':maxwrite and 2*400000*2364,'execution':'one process, one batch, sequential; extrapolation uncertain, stop on budget/limits/failure'})
    save('input_and_reuse_anchors.json',ANCHORS);C.flush()
    print('P3_PREFLIGHT_PASS 200 coordinates; 2400 stored B validated; 36 L reused; budget_s',C.SCI_BUDGET,flush=True)
    return specs

def main():
    if (B/'execution_once.lock').exists():raise RuntimeError('NO_SCIENCE_RESUME_OR_RETRY')
    with (B/'execution_once.lock').open('x') as f:f.write(datetime.now(timezone.utc).isoformat())
    C.STATE={'status':'PREPARING','attempted_batches':0,'completed_batches':0,'attempted_trajectories':0,'completed_trajectories':0,'automatic_retry':False,'P3_executed':False,'new_baseline_batches':0,'reused_L_batches':36,'total_L_target':2400}
    specs=setup();C.SCI_START=time.perf_counter();C.STATE.update(status='RUNNING',science_started_utc=datetime.now(timezone.utc).isoformat(),P3_executed=True);C.flush()
    pairs=[]
    for s in specs:
        for j in range(200):
            check(all(v['status']=='PASS' for v in C.G.values()),'P3_GATE_BLOCK')
            base=C.saved(s,j);args=C.batch_args(C.CTX,s,j)
            if j in [0,73,199]:alt,ce=load_g(s,j,args);origin='run_04_reused'
            else:alt,ce=C.compute(C.CTX,s,j,'L',args);origin='Stage9H_new'
            st=begin();row=C.summary_pair(s,j,base,alt,ce);row['L_source']=origin
            x=alt['captured'].astype(int)-base['captured'].astype(int)
            row['capture_L_minus_B_mcse']=float(x.std(ddof=1)/(4000**.5))
            row['B_capture_mcse']=float(base['captured'].std(ddof=1)/(4000**.5));row['L_capture_mcse']=float(alt['captured'].std(ddof=1)/(4000**.5))
            pairs.append(row);metric('paired_saved_baseline_read_and_summary',st,condition_id=s['id'],j=j)
        C.pd.DataFrame(pairs).to_csv(B/'p3_condition_draw_results.csv',index=False)
    check(len(pairs)==2400 and C.STATE['completed_batches']==2364 and C.STATE['attempted_batches']==2364,'P3_INCOMPLETE')
    C.STATE.update(status='SCIENCE_COMPLETE',science_completed_utc=datetime.now(timezone.utc).isoformat(),complete_L_batches_including_reuse=2400)
    C.G['CFI4']['all_2364_new_L_numeric_checks_passed']=True
    C.flush();save('scientific_input_anchors.json',{**ANCHORS,**{p:r['sha256'] for p,r in C.VERIFIED.items()},**C.SIDECARS})
    print('SCIENCE_COMPLETE 2364 new L, 9456000 new trajectories; 36 reused L; zero baseline recomputation',flush=True)
if __name__=='__main__':
    try:main()
    except BaseException as e:
        C.STATE.update(status='STOPPED',error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc(),stopped_utc=datetime.now(timezone.utc).isoformat())
        save('failure_record.json',C.STATE);save('isolation_failure_diagnostics.json',{'violations':C.ISO.VIOLATIONS,'null_opens':C.ISO.NULL_OPENS});C.flush();print(traceback.format_exc(),flush=True);sys.exit(1)
