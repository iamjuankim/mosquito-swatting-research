from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
O=Path(__file__).resolve().parent; B=O.parent; S=B/'mq10_audit_stage7_h2_stable_20261105'; R=B/'mq10'
conds=json.loads((S/'conditions.json').read_text(encoding='utf-8'))
rows=[]; post=[]
for c in conds:
 with np.load(c['legacy_path'],allow_pickle=False) as z: ld=z['t_det_s']; lc=z['t_contact_s']
 for j in range(200):
  with np.load(S/'checkpoints'/c['id']/f'j{j:03d}.npz',allow_pickle=False) as z: sd=z['t_det_s']; sc=z['t_contact_s']
  # Exact static transcription of simulate.outcomes(window=False); no module import.
  a=ld[j]<np.minimum(lc[j],0.0); b=sd<np.minimum(sc,0.0); x=b.astype(np.int8)-a.astype(np.int8)
  rows.append(dict(condition_id=c['id'],model=c['model'],v_mps=c['condition']['v_mps'],n_exp=c['condition']['n_exp'],j=j,legacy_p=float(a.mean()),stable_p=float(b.mean()),delta=float(x.mean()),n01=int((~a&b).sum()),n10=int((a&~b).sum()),n00=int((~a&~b).sum()),n11=int((a&b).sum()),paired_mcse=float(x.std(ddof=1)/np.sqrt(4000)),legacy_mcse=float(a.std(ddof=1)/np.sqrt(4000)),stable_mcse=float(b.std(ddof=1)/np.sqrt(4000)),raw_finite_legacy_but_not_precontact=int((np.isfinite(ld[j])&~a).sum()),raw_finite_stable_but_not_precontact=int((np.isfinite(sd)&~b).sum())))
 print('precontact',c['id'],flush=True)
f=pd.DataFrame(rows); f.to_csv(O/'precontact_detection_by_draw.csv',index=False)
for (cid,model,v,n),g in f.groupby(['condition_id','model','v_mps','n_exp'],sort=False):
 for k,sek in [('legacy_p','legacy_mcse'),('stable_p','stable_mcse'),('delta','paired_mcse')]:
  x=g[k].to_numpy(); se=g[sek].to_numpy()
  post.append(dict(condition_id=cid,model=model,v_mps=v,n_exp=n,quantity=k,mean=x.mean(),median=np.median(x),q025=np.quantile(x,.025),q975=np.quantile(x,.975),mcse_median=np.median(se),mcse_q90=np.quantile(se,.9)))
pd.DataFrame(post).to_csv(O/'precontact_detection_posterior.csv',index=False)
# Formal manuscript table1 labels and historical numeric statements.
e=pd.read_csv(O/'h2_effects_rule_G.csv'); cmp=pd.read_csv(O/'previous_effect_comparison.csv'); manuscript=pd.read_csv(O/'manuscript_comparison.csv')
extra=[]
for model in ['original','calibrated']:
 for n in [.8,1,1.2]:
  g=e[(e.model==model)&(e.n_exp==n)&(e.version=='stable')]; E=g[g.effect=='E'].iloc[0]
  extra.append(dict(item='Table1 E',model=model,v_mps=None,n_exp=n,manuscript_or_previous_value=None,stable_value=E['median'],stable_minus_old=None,status='Table1 range +.61 to +.66 compatible at stated precision; mechanical n verdict '+E.mechanical_rule_G_n,evidence='manuscript_page3_X73.png; h2_effects_rule_G.csv'))
manuscript['replacement_guidance']='historical only / not revalidated'
for i,h in manuscript.iterrows():
 if pd.notna(h.stable_value): manuscript.loc[i,'replacement_guidance']='retain at 3-decimal precision' if round(h.stable_value,3)==round(h.manuscript_or_previous_value,3) else 'refresh displayed value from stable; original source rounding may differ'
manuscript=pd.concat([manuscript,pd.DataFrame(extra)],ignore_index=True); manuscript.to_csv(O/'manuscript_comparison.csv',index=False)
g=f.groupby('model')[['n01','n10','raw_finite_legacy_but_not_precontact','raw_finite_stable_but_not_precontact']].sum()
text='''\n\n## 최종 검토 보충 — 검출 정의와 결과 해석\n\n`detection_contact_posterior.csv`의 t_det_s는 저장된 raw finite-trigger 여부다. 기존 연구의 p_det은 `src/swat/simulate.py:342-349`의 `t_det < min(t_contact,0)`(window=False)라는 접촉 전 감지 정의다. 두 지표를 혼동하지 않도록 precontact_detection_by_draw.csv(9600행), precontact_detection_posterior.csv(144행)에 그 정의를 정적 전사해 전체 48조건을 별도 계산했다. 동일 시각 접촉/감지는 strict < 규칙으로 처리했고 원래 시간 정밀도를 유지했다. 새 가설/판정 규칙이 아니며 기존 정의를 보완 확인한 것이다. 분석 코드는 supplement.py이며 생산 모듈 import는 없다.\n\n'''
text+=g.to_string()+'\n\n'
text+='''포획 상태가 바뀐 궤적은 총 421개(0→1 389, 1→0 32), 순증 357개다. raw 검출 소실은 original 73,677, calibrated 194,901개다. 따라서 이번 수치 수정의 영향은 포획보다 감지 경로에서 크게 나타나며, 포획의 거의 불변을 감지 모형 전체의 수치적 불변으로 읽을 수 없다. 다만 소실 원인의 궤적별 분해는 legacy threshold/trigger 진단 부재로 UNKNOWN이다.\n\n48조건 모두 stable−legacy 포획확률 draw 차이의 중앙값은 0이다. 조건별 평균 차이는 original에서 0~0.00000875, calibrated에서 -0.000005~0.00012625(확률 단위)다. 모든 조건을 똑같이 가중한 차이는 original +0.0001354%p, calibrated +0.001724%p이며 일반적인 모집단 효과는 아니다. 포획의 가장 큰 draw 차이는 각각 0.05%p, 0.175%p다.\n\n기존 H2 E/인접구간의 **48개 분류는 모두 동일**했다(previous_effect_comparison.csv). original은 n 세 값 모두 6→8·8→12 감소, calibrated는 n=.8/1의 두 구간 및 n=1.2의 8→12 감소가 유지됐다. calibrated n=1.2 6→8의 기존 분류는 '배제'이며 '감소'는 아니었다. 이는 저장된 기존 분류와 일치한다. 두 모형 모두 n별 G 기계적 출력은 부분 기각이며, 세 지수 결합의 기계적 출력도 각각 부분 기각이다. 이 결과가 세 갈래 종합을 포함하는 최종 과학적 H2 판정을 확정하지는 않는다.\n\nE의 paired MC SE 중앙값은 original .00870~.00888, calibrated .00808~.00850으로 기존 N 목표 delta/4=.005보다 크다. 기존 N4000을 유지했으며 목표 충족으로 표시하지 않았다. E 증가와 original n=.8/1의 고속 감소 및 calibrated 8→12 감소는 ±2SE 진단에서 유지됐다. 그 밖 일부 작은 구간 분류는 민감했다. 고속 감소가 전부 MC 오차라고 설명되지는 않지만, 오차가 작다는 단일 문구도 부적절하다.\n\n표1·그림1의 PDF 내장 이미지를 직접 열어 확인했다(manuscript_page3_X73.png, manuscript_page4_X81.png). ①②의 표1 기계적 라벨과 그림1의 대체적인 형태는 유지된다. 포획·E·인접효과의 정밀 표는 stable 산출물로 교체해야 하며, 본문에 반올림된 값은 manuscript_comparison.csv의 replacement_guidance를 따른다. 그림1의 ③ 곡선·표1의 ③ 및 종합은 이번 범위에서 재검증하지 않았다. 검출률·시간 및 그에 의존하는 해석은 stable 진단과 접촉 전 정의로 다시 확인해야 한다. 원고는 변경하지 않았다.\n\nH/Z라는 표기의 원문상 정확한 기호 정의는 현재 확인한 근거만으로 특정하지 못해 UNKNOWN으로 남긴다. Stage3에서 확인한 누적/조건부 생존 정규화, soft/hard 누적 및 지연 경계 차이를 그 표기와 동일하다고 단정하지 않는다. 수치 문턱 수정이 그 구조 차이를 해결하지 않았다는 제한은 유지한다.\n'''
report=O/'stage8_paired_analysis_report.md'; report.write_text(report.read_text(encoding='utf-8')+text,encoding='utf-8')
# Record every additional source consulted and refresh analysis provenance after final code is written.
m=json.loads((O/'input_manifest.json').read_text(encoding='utf-8'))
for p in [R/'src/swat/simulate.py',B/'mq10_audit_stage3_20261104/audit_report.md',B/'mq10_audit_stage6_66traj_20261105/stage6_66traj_report.md']:
 m['files'][str(p)]={'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
m['analysis_code_sha256']=hashlib.sha256((O/'analyze.py').read_bytes()).hexdigest(); m['supplement_code_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
(O/'input_manifest.json').write_text(json.dumps(m,ensure_ascii=False,indent=2),encoding='utf-8')
v=json.loads((O/'validation_results.json').read_text(encoding='utf-8')); v['precontact_detection_rows']=len(f); v['precontact_state_partition']='PASS'; v['historical_H2_effect_classifications_unchanged']=bool((cmp.previous_cls==cmp.stable_cls).all());v['all_condition_capture_delta_medians_zero']=bool((pd.read_csv(O/'posterior_by_speed_n_model.csv').query('quantity == "delta"')['median']==0).all())
(O/'validation_results.json').write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
assert ((f.n01+f.n10+f.n00+f.n11)==4000).all()
print(g.to_string(),flush=True)
