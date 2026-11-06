"""Verify completed P3 and analyze stored paired outcomes, no simulator calls."""
import p3_core as C
from p3_core import B,read,save,sha,check,exact,begin,metric,limits
from pathlib import Path
from datetime import datetime,timezone
import json,time,sys,itertools,traceback,shutil
import numpy as np
import pandas as pd
from scipy.special import ndtr,log_ndtr
import extracted as EX
import local_judge as J
G9=B.parent/'mq10_audit_stage9g_p1p2_20261106/run_04'
S8=B.parent/'mq10_audit_stage8_paired_20261105'
EX.np=np
C.np=np;C.pd=pd
N=4000
NUM=[];OUTPUTS={}
def load(spec,j,variant):
    if variant=='B':
        p=C.S7/'checkpoints'/spec['id']/f'j{j:03d}.npz';marker=read(p.with_suffix('.json'))
        check(sha(p)==marker['sha256'],'B_HASH')
        with np.load(p,allow_pickle=False) as f:
            z={k:f[k] for k in f.files if k!='metadata'};meta=json.loads(str(f['metadata']))
        check(meta['condition_id']==spec['id'] and meta['j']==j and meta['complete'],'B_METADATA')
        check(meta['arrays']=={k:C.ah(v) for k,v in z.items()},'B_ARRAY_HASH')
    else:
        parent=G9 if j in [0,73,199] else B
        p=parent/'checkpoints/L'/spec['id']/f'j{j:03d}.npz';marker=read(p.with_suffix('.json'));cp=p.with_name(p.stem+'_conditioning.npz')
        check(marker['complete'] and sha(p)==marker['sha256'] and sha(cp)==marker['conditioning_sha256'],'L_HASH')
        with np.load(p,allow_pickle=False) as f:z={k:f[k] for k in f.files}
        check(marker['arrays']=={k:C.ah(v) for k,v in z.items()},'L_ARRAY_HASH')
        with np.load(cp,allow_pickle=False) as f:ce={k:f[k] for k in f.files}
        check(all(np.isfinite(v).all() for v in ce.values()),'L_CE_FINITE')
        exact(z['theta'],np.exp(ce['log_theta']),'L_theta_log_link')
        P=PARAMS[str(j)]['P'];a=(np.log(z['M0']+1e-12)-ce['eta'])/P['sigma']
        exact(ce['F0'],ndtr(a),'L_F0');exact(ce['logZ'],log_ndtr(-a),'L_logZ')
        NUM.append(marker['numeric'])
        for q in [p,p.with_suffix('.json'),cp]:OUTPUTS[str(q.relative_to(parent)) if parent==B else 'reused_run04/'+str(q.relative_to(parent))]={'sha256':sha(q),'bytes':q.stat().st_size}
    EX.validate_arrays(z);exact(z['j'],np.asarray(j,np.int64),'checkpoint_j')
    return z
def mse(x):return float(x.std(ddof=1)/np.sqrt(len(x)))
def stat(x,s):
    x=np.asarray(x,float);s=np.asarray(s,float)
    return {'mean':float(x.mean()),'median':float(np.median(x)),'sd':float(x.std(ddof=1)),'q025':float(np.quantile(x,.025)),'q975':float(np.quantile(x,.975)),'q05':float(np.quantile(x,.05)),'q95':float(np.quantile(x,.95)),'min':float(x.min()),'max':float(x.max()),'negative_draws':int((x<0).sum()),'zero_draws':int((x==0).sum()),'positive_draws':int((x>0).sum()),'mcse_median':float(np.median(s)),'mcse_q90':float(np.quantile(s,.9)),'mcse_max':float(s.max()),'fraction_abs_le_2mcse':float((np.abs(x)<=2*s).mean()),'mean_conditional_mc_variance':float(np.mean(s**2)),'between_draw_variance':float(x.var(ddof=1)),'variance_minus_mc_diagnostic':float(max(0,x.var(ddof=1)-np.mean(s**2)))}
def main():
    global PARAMS
    state=read(B/'execution_state.json');check(state['status']=='SCIENCE_COMPLETE' and state['completed_batches']==2364 and state['completed_trajectories']==9456000,'SCIENCE_NOT_COMPLETE')
    PARAMS=read(B/'selected_parameter_provenance.json')['records']
    cond=pd.read_csv(B/'p3_condition_draw_results.csv');check(len(cond)==2400 and not cond.duplicated(['condition_id','j']).any(),'CONDITION_RESULT_KEYS')
    cond=cond.set_index(['condition_id','j'])
    specs=[s for s in read(C.S7/'conditions.json') if s['model'] in ['original','calibrated'] and s['condition']['v_mps'] in [8.,12.] and s['condition']['n_exp'] in [.8,1.,1.2]]
    draws=[];st=begin();pair_checks=0
    for model,n in itertools.product(['original','calibrated'],[.8,1.,1.2]):
        ss={v:next(s for s in specs if s['model']==model and s['condition']['n_exp']==n and s['condition']['v_mps']==v) for v in [8.,12.]}
        for j in range(200):
            limits();z={(variant,v):load(ss[v],j,variant) for variant,v in itertools.product(['B','L'],[8.,12.])}
            row={'model':model,'n':n,'j':j,'chain':PARAMS[str(j)]['chain'],'draw':PARAMS[str(j)]['draw'],'N':N,'MC':'a','L_source':'run_04_reused' if j in [0,73,199] else 'Stage9H_new'}
            for (var,v),a in z.items():
                exact(a['u_threshold'],z['B',8.]['u_threshold'],'paired_threshold_uniform')
                exact(a['trajectory_id'],z['B',8.]['trajectory_id'],'paired_trajectory_ids')
                cap=a['captured'];p=float(cap.mean());row[f'{var}_capture_rate_v{v:g}']=p
                check(int(cap.sum())==int(cond.loc[(ss[v]['id'],j)][f'{var}_captured']),'CAPTURE_CSV_RECONCILIATION');pair_checks+=1
            vectors={}
            for var in ['B','L']:
                x=z[var,12.]['captured'].astype(int)-z[var,8.]['captured'].astype(int);vectors[f'{var}_speed_delta']=x
                row[f'{var}_v8_0_v12_1']=int((x==1).sum());row[f'{var}_v8_1_v12_0']=int((x==-1).sum())
            for v in [8.,12.]:
                x=z['L',v]['captured'].astype(int)-z['B',v]['captured'].astype(int);vectors[f'L_minus_B_v{v:g}']=x
                row[f'v{v:g}_B0_L1']=int((x==1).sum());row[f'v{v:g}_B1_L0']=int((x==-1).sum())
            vectors['speed_delta_L_minus_B']=vectors['L_speed_delta']-vectors['B_speed_delta']
            for name,x in vectors.items():row[name]=float(x.mean());row[name+'_mcse']=mse(x)
            check(abs(row['speed_delta_L_minus_B']-(row['L_minus_B_v12']-row['L_minus_B_v8']))<1e-16,'DIFFERENCE_OF_DIFFERENCES_ALGEBRA')
            draws.append(row)
        print('ANALYSIS_BLOCK',model,n,flush=True)
    check(pair_checks==4800 and len(NUM)==2400,'ANALYSIS_COUNTS')
    df=pd.DataFrame(draws);df.to_csv(B/'p3_draw_results.csv',index=False)
    summaries=[];rules=[]
    priorrules=pd.read_csv(S8/'h2_effects_rule_G.csv');priordraw=pd.read_csv(S8/'h2_effects_by_draw.csv')
    for (model,n),g in df.groupby(['model','n'],sort=False):
        check(np.array_equal(g.j,np.arange(200)),'DRAW_ORDER')
        for name in ['B_speed_delta','L_speed_delta','L_minus_B_v8','L_minus_B_v12','speed_delta_L_minus_B']:
            summaries.append({'model':model,'n':n,'quantity':name,'n_draws':200,**stat(g[name],g[name+'_mcse'])})
        row={'model':model,'n':n,'effect':'Delta_7','v_from':8,'v_to':12,'alpha':.05/35,'delta':.02,'n_draws':200,'N':4000,'MC':'a'}
        for var in ['B','L']:
            d=g[var+'_speed_delta'].to_numpy();s=g[var+'_speed_delta_mcse'].to_numpy()
            r=J.effect_row('Delta_7',d,s,.05/35,.02)
            for k in ['L','U','median','mean','cls','mcse_med','mcse_q90']:row[var+'_'+k]=r[k]
            row[var+'_q025']=float(np.quantile(d,.025));row[var+'_q975']=float(np.quantile(d,.975))
            for sign,label in [(-1,'minus2se'),(1,'plus2se')]:
                pert=d+sign*2*s;lo,up,med=J.interval(pert,.05/35);row[var+'_'+label+'_cls']=J.classify(lo,up,med,.02)
                row[var+'_'+label+'_L']=lo;row[var+'_'+label+'_U']=up
            row[var+'_negative_draws']=int((d<0).sum());row[var+'_fraction_abs_le_2mcse']=float((np.abs(d)<=2*s).mean())
            if var=='B':
                old=priorrules[(priorrules.model==model)&(priorrules.n_exp==n)&(priorrules.version=='stable')&(priorrules.effect=='Delta_7')]
                check(len(old)==1,'PRIOR_RULE_KEY');old=old.iloc[0]
                for k,nk in [('L','L'),('U','U'),('median','median'),('mean','mean')]:check(abs(r[k]-float(old[nk]))<=1e-14,'STORED_B_RULE_RECONCILIATION '+k)
                check(r['cls']==old.classification,'STORED_B_CLASSIFICATION')
                od=priordraw[(priordraw.model==model)&(priordraw.n_exp==n)&(priordraw.version=='stable')&(priordraw.effect=='Delta_7')].sort_values('j')
                check(len(od)==200 and np.max(np.abs(d-od.delta.to_numpy()))<=1e-15,'STORED_B_DRAW_RECONCILIATION')
        row['decrease_classification_retained']=row['B_cls']=='감소' and row['L_cls']=='감소'
        row['L_MC_perturbation_classification_stable']=row['L_cls']==row['L_minus2se_cls']==row['L_plus2se_cls']
        rules.append(row)
    pd.DataFrame(summaries).to_csv(B/'p3_contrast_summary.csv',index=False)
    pd.DataFrame(rules).to_csv(B/'p3_rule_g_summary.csv',index=False)
    metric('postcompute_checkpoint_and_paired_analysis',st)
    # One targeted preservation check on consumed files; no environment re-audit or tree hashes.
    st=begin();anchors=read(B/'scientific_input_anchors.json')
    for p,h in anchors.items():check(sha(p)==h,'FINAL_CONSUMED_INPUT_CHANGED '+p)
    check(not C.ISO.VIOLATIONS,'EXTERNAL_WRITE_ATTEMPT')
    metric('targeted_consumed_input_preservation',st)
    numeric={k:max(r[k] for r in NUM) for k in NUM[0] if k!='logZ_lt_log1e_8_count'}
    numeric['L_logZ_sensitive_total']=sum(r['logZ_lt_log1e_8_count'] for r in NUM)
    save('p3_verification.json',{'status':'PASS','checked_utc':datetime.now(timezone.utc).isoformat(),'draw_rows':len(df),'condition_draw_rows':len(cond),'local_rule_G_rows':len(rules),'capture_counts_reconciled':pair_checks,'new_checkpoints':2364,'reused_checkpoints':36,'strict_tolerances_unchanged':True,'numeric_maxima':numeric,'B_matches_existing_Stage8_Delta7_draws_intervals_and_classification':True,'consumed_input_preservation_files':len(anchors),'no_external_write_attempts':not C.ISO.VIOLATIONS,'no_new_simulations_in_analysis':True,'overall_H2_rejudged':False})
    save('checkpoint_output_manifest.json',OUTPUTS)
    with (B/'postvalidation_resource_usage.csv').open('w',encoding='utf-8',newline='') as f:pd.DataFrame(C.RES).to_csv(f,index=False)
    print('VERIFICATION_PASS 1200 draw rows; 6 local Rule G blocks',flush=True)
if __name__=='__main__':
    try:main()
    except BaseException as e:
        save('postvalidation_failure.json',{'status':'STOPPED','error':str(e),'traceback':traceback.format_exc()});print(traceback.format_exc(),flush=True);sys.exit(1)
