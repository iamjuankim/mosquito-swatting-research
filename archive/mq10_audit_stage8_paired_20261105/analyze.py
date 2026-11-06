from pathlib import Path
import json, hashlib, csv, sys, platform, time
import numpy as np
import pandas as pd
from scipy.stats import norm
from pypdf import PdfReader
sys.stdout.reconfigure(encoding='utf-8')
OUT=Path(__file__).resolve().parent
BASE=OUT.parent
S7=BASE/'mq10_audit_stage7_h2_stable_20261105'
S8=BASE/'mq10_audit_stage8_completion_20261105'
ROOT=BASE/'mq10'
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,o): (OUT/p).write_text(json.dumps(o,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''): h.update(b)
 return h.hexdigest()
def inventory():
 files=sorted(list(S7.rglob('*'))+list(S8.rglob('*')))
 files=[p for p in files if p.is_file()]+[Path(c['legacy_path']) for c in conds]
 return {str(p):{'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns,'sha256':sha(p)} for p in files}
def summ(x):
 x=np.asarray(x,float)
 return dict(mean=float(x.mean()),median=float(np.median(x)),sd=float(x.std(ddof=1)),q025=float(np.quantile(x,.025)),q975=float(np.quantile(x,.975)),q05=float(np.quantile(x,.05)),q95=float(np.quantile(x,.95)))
def se(x): return x.std(axis=-1,ddof=1)/np.sqrt(x.shape[-1])
def cls(l,u,m):
 return '증가' if l>0 and m>=.02 else '감소' if u<0 else '배제' if u<.02 else '불확실'
def interval(x,alpha):
 q=np.quantile(x,[alpha/2,1-alpha/2]); z=norm.ppf(1-alpha/2)
 return min(q[0],x.mean()-z*x.std(ddof=1)),max(q[1],x.mean()+z*x.std(ddof=1))
conds=read(S7/'conditions.json'); assert len(conds)==48 and len({c['id'] for c in conds})==48
assert {(c['model'],c['condition']['v_mps'],c['condition']['n_exp']) for c in conds}=={(m,v,n) for m in ['original','calibrated'] for v in [.5,1,2,3,4,6,8,12] for n in [.8,1,1.2]}
# Analysis conventions are frozen before reading outcomes. This does not replace the historical rule.
write('analysis_plan.json',{'scope':'48 fixed conditions x 200 draws x N4000; no simulator imports','join':['condition_id','j','trajectory_id'],'posterior_summary':'equal empirical weights of stored P1 selected draws, no resampling or reweighting','time':'native float32 legacy / float64 stable; deltas only both finite; state transitions on full denominator','intervals':'descriptive equal-tail 95% and 90%; historical G: hull of quantile and mean +/- z SD, alpha E=.01, adjacent=.05/35, delta=.02','MC':'paired sample SD/sqrt(4000); sensitivity intervals shifted by +/-2 SE, diagnostic only; no new decision rule','precision':'float32 bit equality after casting stable times is a precision-compatible diagnostic, not proof of identical path'})
print('Snapshot start',flush=True)
start=inventory(); write('protected_inputs_before.json',start)
manifest=dict(start)
def record(p):
 p=Path(p); manifest[str(p)]={'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns,'sha256':sha(p)}
for p in [ROOT/'results/p3b/draws_P1.csv',ROOT/'results/p3b/draws_P1_meta.json',ROOT/'logs/decisions.md',ROOT/'phases/p3b/judge.py',ROOT/'results/p3b_judge/h2_rule_g.csv',ROOT/'results/p3b_judge/h2_speed_table.csv',ROOT/'results/p3b/validate.json',ROOT/'results/p3b_cal/calibrate.json',ROOT/'logs/phase_reports/p3b.md']:
 record(p)
draws=pd.read_csv(ROOT/'results/p3b/draws_P1.csv'); assert np.array_equal(draws.j,np.arange(200))
oldstats={(r['condition_id'],r['j']):r for r in read(S7/'production/paired_capture_by_draw.json')}
audit={(r['condition_id'],r['j']):r for r in read(S8/'checkpoint_audit.json')}
input_expected={r['path']:r for r in read(S7/'input_manifest.json')}
rows=[]; trows=[]; prows=[]; diag=[]; checks=[]
(OUT/'trajectory_differences').mkdir(exist_ok=True)
for ic,c in enumerate(conds):
 cid=c['id']; lp=Path(c['legacy_path']); assert sha(lp)==input_expected[str(lp)]['sha256']
 with np.load(lp,allow_pickle=False) as z: L={k:z[k] for k in z.files}
 assert L['captured'].shape==(200,4000) and L['captured'].dtype==np.bool_ and np.array_equal(L['j'],np.arange(200))
 for k in ['t_det_s','t_contact_s']: assert L[k].dtype==np.float32
 S=np.zeros((200,4000),bool); D=np.zeros_like(S,dtype=np.int8)
 delta={k:np.full((200,4000),np.nan) for k in ['t_det_s','t_contact_s']}
 states={k:np.zeros((200,4000),np.uint8) for k in delta}
 for j in range(200):
  p=S7/'checkpoints'/cid/f'j{j:03d}.npz'; h=sha(p); assert h==audit[cid,j]['sha256'] and h==read(p.with_suffix('.json'))['sha256']
  with np.load(p,allow_pickle=False) as z: Z={k:z[k] for k in z.files}
  meta=json.loads(str(Z.pop('metadata')))
  assert meta['condition_id']==cid and meta['j']==j and int(Z['j'])==j and np.array_equal(Z['trajectory_id'],np.arange(4000))
  assert all(hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()==meta['arrays'][k] for k,v in Z.items())
  assert np.where(Z['u_threshold']==0,Z['theta']>=Z['M0'],Z['theta']>Z['M0']).all()
  A=L['captured'][j]; B=Z['captured']; S[j]=B; x=B.astype(np.int8)-A; D[j]=x
  r=dict(condition_id=cid,model=c['model'],v_mps=c['condition']['v_mps'],n_exp=c['condition']['n_exp'],j=j,N=4000,legacy_p=float(A.mean()),stable_p=float(B.mean()),delta=float(x.mean()),n01=int(((~A)&B).sum()),n10=int((A&(~B)).sum()),n00=int(((~A)&(~B)).sum()),n11=int((A&B).sum()),paired_mcse=float(se(x)),legacy_mcse=float(se(A)),stable_mcse=float(se(B)),legacy_path=str(lp),stable_path=str(p))
  q=oldstats[cid,j]; assert (q['old_capture'],q['new_capture'],q['gain'],q['loss'])==(int(A.sum()),int(B.sum()),r['n01'],r['n10'])
  rows.append(r)
  for k in delta:
   a=L[k][j]; b=Z[k]; assert b.dtype==np.float64
   assert (np.isfinite(a)|np.isposinf(a)).all() and (np.isfinite(b)|np.isposinf(b)).all()
   fa=np.isfinite(a); fb=np.isfinite(b); both=fa&fb
   states[k][j]=fa.astype(np.uint8)+2*fb.astype(np.uint8)
   y=b[both]-a[both].astype(np.float64); delta[k][j,both]=y
   cast=b.astype(np.float32); bit_same=cast.view(np.uint32)==a.view(np.uint32)
   tr=dict(condition_id=cid,j=j,metric=k,N=4000,legacy_finite=int(fa.sum()),stable_finite=int(fb.sum()),inf_inf=int((~fa&~fb).sum()),inf_finite=int((~fa&fb).sum()),finite_inf=int((fa&~fb).sum()),both_finite=int(both.sum()),finite_mean_delta=float(y.mean()) if len(y) else None,finite_median_delta=float(np.median(y)) if len(y) else None,finite_q025=float(np.quantile(y,.025)) if len(y) else None,finite_q975=float(np.quantile(y,.975)) if len(y) else None,finite_maxabs_delta=float(abs(y).max()) if len(y) else None,finite_exact_equal=int((y==0).sum()),float32_bit_equal_finite=int((bit_same&both).sum()),float32_bit_different_finite=int((~bit_same&both).sum()),finite_delta_mcse=float(se(y)) if len(y)>1 else None)
   trows.append(tr)
  assert np.array_equal(B,np.isfinite(Z['t_contact_s'])) and np.array_equal(A,np.isfinite(L['t_contact_s'][j]))
  f=np.isfinite(Z['t_det_s']); assert np.array_equal(f,Z['trigger_i']>=0) and (Z['trigger_i'][f]>Z['i0'][f]).all()
  diag.append(dict(condition_id=cid,j=j,theta_equal_M0=int((Z['theta']==Z['M0']).sum()),theta_min_margin=float((Z['theta']-Z['M0']).min()),stimulus_only=int((f&Z['stimulus']&~Z['spontaneous']).sum()),spontaneous_only=int((f&~Z['stimulus']&Z['spontaneous']).sum()),both_trigger=int((f&Z['stimulus']&Z['spontaneous']).sum()),detected=int(f.sum()),changed_capture_with_detection_state_change=int(((x!=0)&(np.isfinite(L['t_det_s'][j])!=f)).sum()),dt_s=float(Z['dt_s']),n_steps=int(Z['n_steps'])))
 np.savez_compressed(OUT/'trajectory_differences'/f'{cid}.npz',j=np.arange(200),trajectory_id=np.arange(4000),capture_delta=D,legacy_capture=L['captured'],stable_capture=S,det_delta_s=delta['t_det_s'],contact_delta_s=delta['t_contact_s'],det_state=states['t_det_s'],contact_state=states['t_contact_s'])
 cr=rows[-200:]
 for key in ['legacy_p','stable_p','delta']:
  vals=np.array([r[key] for r in cr]); ses=np.array([r[{'legacy_p':'legacy_mcse','stable_p':'stable_mcse','delta':'paired_mcse'}[key]] for r in cr])
  prows.append(dict(condition_id=cid,model=c['model'],v_mps=c['condition']['v_mps'],n_exp=c['condition']['n_exp'],quantity=key,**summ(vals),mcse_median=float(np.median(ses)),mcse_q90=float(np.quantile(ses,.9)),mean_conditional_mc_variance=float(np.mean(ses**2)),between_draw_variance=float(vals.var(ddof=1)),variance_minus_mc_diagnostic=float(max(0,vals.var(ddof=1)-np.mean(ses**2))),fraction_abs_le_2mcse=float(np.mean(abs(vals)<=2*ses))))
 print(f'{ic+1}/48 {cid}',flush=True)
pd.DataFrame(rows).to_csv(OUT/'paired_capture_by_condition_draw.csv',index=False)
pd.DataFrame(trows).to_csv(OUT/'paired_time_by_condition_draw.csv',index=False)
pd.DataFrame(prows).to_csv(OUT/'posterior_by_speed_n_model.csv',index=False)
pd.DataFrame(diag).to_csv(OUT/'stable_diagnostics_by_draw.csv',index=False)
# Speed covariance justified by condition-independent mc_batch seed/j/set; also verify saved u_threshold across all 8 speeds for each model,n,j.
speeds=[.5,1,2,3,4,6,8,12]; erows=[]; edraw=[]; modelrows=[]; covchecks=0
for model in ['original','calibrated']:
 for n in [.8,1,1.2]:
  cs=[next(c for c in conds if c['model']==model and c['condition']['n_exp']==n and c['condition']['v_mps']==v) for v in speeds]
  arrays=[]
  for c in cs:
   with np.load(OUT/'trajectory_differences'/f"{c['id']}.npz") as z: arrays.append((z['legacy_capture'],z['stable_capture']))
  for j in range(200):
   u0=None
   for c in cs:
    with np.load(S7/'checkpoints'/c['id']/f'j{j:03d}.npz') as z: u=z['u_threshold']
    if u0 is None: u0=u
    else: assert np.array_equal(u,u0); covchecks+=1
  for version,idx in [('legacy',0),('stable',1)]:
   local=[]
   for name,a,b in [('E',0,7)]+[(f'Delta_{i+1}',i,i+1) for i in range(7)]:
    x=arrays[b][idx].astype(np.int8)-arrays[a][idx].astype(np.int8); d=x.mean(1); s=se(x); alpha=.01 if name=='E' else .05/35
    l,u=interval(d,alpha); cl=cls(l,u,float(np.median(d)))
    # Deterministic perturbations are sensitivity envelopes, not posterior draws or a new significance test.
    lm,um=interval(d-2*s,alpha); lp,up=interval(d+2*s,alpha)
    r=dict(model=model,n_exp=n,version=version,effect=name,v_from=speeds[a],v_to=speeds[b],alpha=alpha,**summ(d),L=float(l),U=float(u),classification=cl,mcse_median=float(np.median(s)),mcse_q90=float(np.quantile(s,.9)),fraction_abs_le_2mcse=float(np.mean(abs(d)<=2*s)),minus2se_class=cls(lm,um,float(np.median(d-2*s))),plus2se_class=cls(lp,up,float(np.median(d+2*s))))
    local.append(r); erows.append(r)
    for j in range(200): edraw.append(dict(model=model,n_exp=n,version=version,effect=name,j=j,delta=float(d[j]),paired_mcse=float(s[j])))
   E=local[0]['classification']; inc=any(r['classification']=='증가' for r in local[1:]); dec=any(r['classification']=='감소' for r in local[1:])
   verdict=('부분 기각' if dec else '지지') if E=='증가' else '부분 기각' if inc else '기각' if E in ['배제','감소'] else '판정 불가'
   for r in local: r['mechanical_rule_G_n']=verdict
pd.DataFrame(erows).to_csv(OUT/'h2_effects_rule_G.csv',index=False)
pd.DataFrame(edraw).to_csv(OUT/'h2_effects_by_draw.csv',index=False)
# Same draws and common randoms across models; this contrast combines the historical k and shift changes.
for n in [.8,1,1.2]:
 for v in speeds:
  cs=[next(c for c in conds if c['model']==m and c['condition']['n_exp']==n and c['condition']['v_mps']==v) for m in ['original','calibrated']]
  with np.load(OUT/'trajectory_differences'/f"{cs[0]['id']}.npz") as z: a=z['stable_capture']
  with np.load(OUT/'trajectory_differences'/f"{cs[1]['id']}.npz") as z: b=z['stable_capture']
  x=b.astype(np.int8)-a.astype(np.int8); s=se(x)
  modelrows.append(dict(n_exp=n,v_mps=v,contrast='calibrated_stable-original_stable',**summ(x.mean(1)),mcse_median=float(np.median(s)),mcse_q90=float(np.quantile(s,.9))))
pd.DataFrame(modelrows).to_csv(OUT/'model_contrasts.csv',index=False)
pdf=Path('/PRIVATE_INPUT/manuscript.pdf'); record(pdf)
r=PdfReader(pdf)
(OUT/'manuscript_extracted_text.txt').write_text('\n\n'.join(f'PAGE {i+1}\n'+p.extract_text() for i,p in enumerate(r.pages)),encoding='utf-8')
# Historical results mapped only to the two authorized branches.
hist=pd.read_csv(ROOT/'results/p3b_judge/h2_speed_table.csv'); hist=hist[hist.branch.isin(['①','②']) & (hist.prior=='P1')]
compar=[]
for _,h in hist.iterrows():
 model={'①':'original','②':'calibrated'}[h.branch]
 pp=next(p for p in prows if p['model']==model and p['n_exp']==h.n_exp and p['v_mps']==h.v_mps and p['quantity']=='stable_p')
 ll=next(p for p in prows if p['condition_id']==pp['condition_id'] and p['quantity']=='legacy_p')
 compar.append(dict(model=model,n_exp=h.n_exp,v_mps=h.v_mps,previous_median=h.p_cap_median,recomputed_legacy_median=ll['median'],stable_median=pp['median'],stable_minus_previous=pp['median']-h.p_cap_median,legacy_matches=bool(np.isclose(ll['median'],h.p_cap_median,atol=1e-12)),source=str(ROOT/'results/p3b_judge/h2_speed_table.csv')))
pd.DataFrame(compar).to_csv(OUT/'previous_speed_comparison.csv',index=False)
hist=pd.read_csv(ROOT/'results/p3b_judge/h2_rule_g.csv'); ec=[]
for e in erows:
 if e['version']!='legacy': continue
 b={'original':'①','calibrated':'②'}[e['model']]
 hs=hist[(hist.branch==b)&(hist.n_exp==e['n_exp'])&(hist.v_from==e['v_from'])&(hist.v_to==e['v_to'])]
 assert len(hs)==1
 h=hs.iloc[0]; st=next(s for s in erows if s['model']==e['model'] and s['n_exp']==e['n_exp'] and s['effect']==e['effect'] and s['version']=='stable')
 ec.append(dict(model=e['model'],n_exp=e['n_exp'],effect=e['effect'],previous_median=h['median'],legacy_median=e['median'],stable_median=st['median'],legacy_minus_previous=e['median']-h['median'],previous_cls=h['cls'],legacy_cls=e['classification'],stable_cls=st['classification'],source=str(ROOT/'results/p3b_judge/h2_rule_g.csv')))
pd.DataFrame(ec).to_csv(OUT/'previous_effect_comparison.csv',index=False)
print('Snapshot end',flush=True)
end=inventory(); write('protected_inputs_after.json',end)
changes=[p for p in set(start)|set(end) if start.get(p)!=end.get(p)]
write('input_manifest.json',{'files':manifest,'draw_order':str(ROOT/'results/p3b/draws_P1.csv'),'python':sys.version,'numpy':np.__version__,'platform':platform.platform(),'analysis_code_sha256':sha(__file__)})
write('validation_results.json',{'conditions':48,'draw_pairs':len(rows),'trajectory_pairs':len(rows)*4000,'checkpoint_hash_metadata_array_validation':'PASS','Stage7_summary_reproduction':'PASS','legacy_manifest_hashes':'PASS','time_precision_and_infinity_validation':'PASS','cross_speed_u_threshold_equal_comparisons':covchecks,'protected_files':len(start),'protected_changes':changes,'previous_speed_all_match':all(x['legacy_matches'] for x in compar),'simulator_imported':False,'final_scientific_H2_verdict':'DEFERRED'})
assert not changes
print('Analysis complete',flush=True)
