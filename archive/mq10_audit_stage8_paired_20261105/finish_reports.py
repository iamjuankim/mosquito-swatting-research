from pathlib import Path
import json,csv,hashlib,sys
import numpy as np
import pandas as pd
from pypdf import PdfReader
O=Path(__file__).resolve().parent; B=O.parent; R=B/'mq10'; S=B/'mq10_audit_stage7_h2_stable_20261105'
sys.stdout.reconfigure(encoding='utf-8')
def md(df):
 return '| '+' | '.join(map(str,df.columns))+' |\n|'+ '|'.join(['---']*len(df.columns))+'|\n'+'\n'.join('| '+' | '.join(f'{v:.7g}' if isinstance(v,(float,np.floating)) else str(v) for v in row)+' |' for row in df.itertuples(index=False,name=None))
def put(n,s): (O/n).write_text(s,encoding='utf-8')
p=pd.read_csv(O/'paired_capture_by_condition_draw.csv'); t=pd.read_csv(O/'paired_time_by_condition_draw.csv'); d=pd.read_csv(O/'stable_diagnostics_by_draw.csv'); post=pd.read_csv(O/'posterior_by_speed_n_model.csv'); e=pd.read_csv(O/'h2_effects_rule_G.csv'); val=json.loads((O/'validation_results.json').read_text(encoding='utf-8')); conds=json.loads((S/'conditions.json').read_text(encoding='utf-8'))
# All-condition time-state summary; counts retain complete denominators.
ts=t.groupby(['condition_id','metric']).agg(N=('N','sum'),legacy_finite=('legacy_finite','sum'),stable_finite=('stable_finite','sum'),inf_inf=('inf_inf','sum'),inf_finite=('inf_finite','sum'),finite_inf=('finite_inf','sum'),both_finite=('both_finite','sum'),finite_exact_equal=('finite_exact_equal','sum'),float32_bit_equal_finite=('float32_bit_equal_finite','sum'),float32_bit_different_finite=('float32_bit_different_finite','sum'),finite_maxabs_delta=('finite_maxabs_delta','max')).reset_index(); ts.to_csv(O/'time_state_summary.csv',index=False)
timepost=[]
for (cid,metric),g in t.groupby(['condition_id','metric'],sort=False):
 c=next(c for c in conds if c['id']==cid)
 for version in ['legacy','stable']:
  x=g[version+'_finite'].to_numpy()/4000
  timepost.append(dict(condition_id=cid,model=c['model'],v_mps=c['condition']['v_mps'],n_exp=c['condition']['n_exp'],metric=metric,version=version,mean=x.mean(),median=np.median(x),q025=np.quantile(x,.025),q975=np.quantile(x,.975),mcse_median=np.median(np.sqrt(x*(1-x)/3999))))
pd.DataFrame(timepost).to_csv(O/'detection_contact_posterior.csv',index=False)
# Exact pooled finite-pair descriptive quantiles are condition-wise, not survival estimates.
ft=[]
for c in conds:
 with np.load(O/'trajectory_differences'/f"{c['id']}.npz") as z:
  for name,key in [('t_det_s','det_delta_s'),('t_contact_s','contact_delta_s')]:
   x=z[key]; x=x[np.isfinite(x)]
   ft.append(dict(condition_id=c['id'],metric=name,finite_pairs=len(x),mean=x.mean() if len(x) else None,median=np.median(x) if len(x) else None,q025=np.quantile(x,.025) if len(x) else None,q975=np.quantile(x,.975) if len(x) else None,maxabs=np.abs(x).max() if len(x) else None))
pd.DataFrame(ft).to_csv(O/'finite_time_delta_summary.csv',index=False)
pdf=Path('/PRIVATE_INPUT/manuscript.pdf'); reader=PdfReader(pdf)
for i in [2,3]:
 for im in reader.pages[i].images: (O/f'manuscript_page{i+1}_{im.name}').write_bytes(im.data)
# Manuscript n=1 numerical statements, explicitly located on page 4; rounded to 3 decimals.
mr=[]
for model,vs in [('original',{.5:.071,1:.279,4:.756,6:.756,8:.734,12:.694}),('calibrated',{.5:.038,1:.170,4:.702,12:.683})]:
 for v,old in vs.items():
  st=post[(post.model==model)&(post.n_exp==1)&(post.v_mps==v)&(post.quantity=='stable_p')].iloc[0]
  mr.append(dict(item='Figure1 / H2 n=1 capture',model=model,v_mps=v,manuscript_or_previous_value=old,stable_value=st['median'],stable_minus_old=st['median']-old,status='compare rounded values; figure points anchored by historical speed CSV',evidence=str(pdf)+' pages 3-4; '+str(R/'results/p3b_judge/h2_speed_table.csv')))
for model in ['original','calibrated']:
 st=post[(post.model==model)&(post.n_exp==1)&(post.v_mps==1)&(post.quantity=='stable_p')].iloc[0]
 mr.append(dict(item='1m/s hand prediction (NOT disk validation)',model=model,v_mps=1,manuscript_or_previous_value=.279 if model=='original' else .170,stable_value=st['median'],stable_minus_old=st['median']-(.279 if model=='original' else .170),status='hand resampling, opaque; not 848-trial replication',evidence=str(R/'logs/phase_reports/p3b.md')+':194-197'))
for label,value in [('opaque disk capture original',.311),('transparent disk response original',.781),('transparent disk capture calibrated',.178),('observed opaque capture',72/417),('observed transparent capture',110/431)]:
 mr.append(dict(item=label,model='historical disk/observation',v_mps=1,manuscript_or_previous_value=value,stable_value=None,stable_minus_old=None,status='NOT REVALIDATED: fixed 48 hand conditions do not include disk validation trials',evidence=str(pdf)+' page 4; '+str(R/'results/p3b/validate.json')+'; '+str(R/'results/p3b_cal/calibrate.json')))
pd.DataFrame(mr).to_csv(O/'manuscript_comparison.csv',index=False)
agg=p.groupby('model').agg(n01=('n01','sum'),n10=('n10','sum'),n00=('n00','sum'),n11=('n11','sum'),mean_delta=('delta','mean'),max_abs_draw_delta=('delta',lambda x:abs(x).max())).reset_index()
st=e[e.version=='stable']; stable95=post[(post.quantity=='stable_p')&(post.n_exp==1)][['model','v_mps','mean','median','q025','q975','mcse_median']]
mechanical=st[['model','n_exp','mechanical_rule_G_n']].drop_duplicates()
mcchanged=st[(st.classification!=st.minus2se_class)|(st.classification!=st.plus2se_class)]
put('mc_uncertainty_validation_report.md',f'''# Stage 8B MC SE 및 불확실성 검증

근거: `{O/'analyze.py'}`, `{S/'conditions.json'}`, `{R/'logs/decisions.md'}` (규칙 G/H2), `{R/'phases/p3b/judge.py'}` (interval/h2), `{S/'source/run_original.py'}` (draws/mc_batch), `{R/'results/p3b/draws_P1.csv'}`. 원자료 경로는 paired_capture/time CSV 각 행 및 input_manifest.json.

## 계산

같은 draw j에서 x_i=stable_capture_i-legacy_capture_i, d_j=sum(x_i)/N. MC SE_j=sqrt([sum(x_i²)-N*d_j²]/[N*(N-1)]). 0→1/1→0/0→0/1→1은 N4000 전체를 분할한다. 이는 짝차이 표본분산이며 독립 두 비율 식을 쓰지 않는다. 단일 포획률은 이항 표본분산 p(1-p)/(N-1)의 제곱근을 병기했다. H2 속도 차이도 x_i=C_to-C_from의 같은 식이다.

조건별 200개 d_j의 평균/중앙값/SD/95% equal-tail 구간 및 90% 구간은 경험적 posterior 예측 분포의 요약이다. 200개는 실험 반복이 아니고 SD/sqrt(200)를 실험 오차로 보고하지 않았다. 조건별 posterior 구간에는 유한 N 오차가 섞인다. mean(SE_j²)를 별도 출력하고 var(d_j)-mean(SE_j²)의 양수 부분도 진단으로만 출력한다. 후자는 독립·무편향 조건 아래 분산 분리의 근사이며, 정제된 posterior credible interval이나 정확한 분산 분해가 아니다. draw 수 200의 극단 분위수 및 평균의 적분 오차/ESS는 별도로 추정하지 못했다.

P1 draw는 기존 PSIS 확률 비례 비복원 선택 200개이다. 저장 CSV에는 잔여 확률 가중치 열이 없다. 기존 선택을 그대로 균등 경험분포로 요약했으며 재가중·재표집하지 않았다. 이를 전체 재가중 posterior의 정확한 표본으로 과장하지 않는다.

## covariance 및 가정

mc_batch는 조건에 무관한 seed/j/MC-set으로 초기 상태와 난수를 만든다. common.batch_args와 Stage8A 짝대응 근거를 읽고 사용했으며 전 속도 u_threshold 일치 대조 {val['cross_speed_u_threshold_equal_comparisons']}회 PASS. 이것이 전체 과거 실행 trace를 새로 증명하는 것은 아니다. 속도별 MC covariance는 trajectory 차이의 분산에 자동 포함된다. posterior draw 간 공분산을 독립 실험의 반복 수로 바꾸지 않았다.

초기상태 841개에서 복원추출하는 조건부 MC에 대한 SE다. 원 trial의 세션·trigger 군집이나 초기상태 경험분포 추정 오차까지 포괄하지 않는다. 서로 다른 j가 seed를 달리 사용해도 같은 posterior 원천과 bootstrap 회피분포를 공유하므로 posterior 평균 SE를 무조건 sqrt(sum SE²)/200로 단정하지 않았다.

## 규칙 및 민감성

E=p(12)-p(.5), 인접 7구간 전수. 기존 G 구간은 분위수 구간과 평균±z*posterior SD의 외피다. E alpha=.01, 인접 alpha=.05/35, delta=.02. 이 구간은 표준오차 기반 빈도주의 신뢰구간이 아니다. descriptive 95% equal-tail 구간은 별도 열이다.

전체 stable 48개 효과에서 d_j를 각각 ±2 MC SE_j 이동시키는 결정론적 진단을 추가했다. 규칙을 바꾸지 않았으며 이것은 새 posterior나 MC 신뢰구간이 아니다. 이 이동에 분류가 달라지는 효과 {len(mcchanged)}개:

{md(mcchanged[['model','n_exp','effect','classification','minus2se_class','plus2se_class']])}

0 discordance인 짝차이 SE=0은 관측 표본의 결과다. 모집단 차이가 정확히 0이거나 미래 표본에서도 불변임을 보장하지 않는다. 작은 효과의 |d|≤2SE 비율은 민감성 지표이며 새로운 유의성 검정이 아니다. 기존 N 선택 목표 delta/4=.005와 실제 E·구간 MC SE는 h2_effects_rule_G.csv에서 직접 비교해야 하며, 이번 단계에서 N을 변경하지 않았다.

시간 차이 SE는 both-finite 집합에 조건부인 평균 차이의 SE만 제공한다. 검열이 있는 전체 시간 분포/전체 평균을 뜻하지 않는다. inf-inf에서 inf-inf를 계산하거나 비검출을 0초로 치환하지 않았다.
''')
issues=[('scientific_H2','DEFERRED','기계적 G 재계산과 최종 과학적 판정은 구분; 원고 수정 없음'),('full_path_attribution','UNKNOWN','legacy에는 문턱·trigger 원인·매 frame 경로가 없어 전체 38.4M의 위치/속도 차이와 원인분해 불가'),('disk_1mps_replication','NOT REVALIDATED','48조건은 손 resampling; 848/841 trial 원판 검증과 다름'),('H_Z_and_lag','UNRESOLVED','stable 문턱 추출만 수정; 적합-시뮬레이터 누적/정규화/지연 경계 차이는 보존'),('high_speed_extrapolation','UNVALIDATED','약 1m/s 데이터/CFD 밖 공간·시간·속도 배율 가정'),('posterior_weights','LIMITED','기존 PSIS 비복원 200개와 순서 보존; 정확한 전체 posterior 재현 및 ESS UNKNOWN'),('manuscript_latest','USER_NAMED_FILE_FOUND','정확한 사용자 지정 파일 발견; 최신본이라는 외부 버전 계보는 독립 확인 불가'),('prior_Stage8A_exit_code','UNKNOWN','생산 OS exit-code 직접 기록 UNKNOWN은 기존 감사에서 승계')]
pd.DataFrame(issues,columns=['issue','status','detail']).to_csv(O/'unresolved_issues.csv',index=False)
put('verification_and_open_issues.md','# 검증 결과 및 미해결 문제\n\n'+json.dumps(val,ensure_ascii=False,indent=2)+'\n\n'+md(pd.DataFrame(issues,columns=['issue','status','detail']))+'\n\nStage9 독립 과학 검토 GO는 이 한계와 UNKNOWN을 검토 입력으로 제공한다는 뜻이다. H2 최종 판정·출판·추가 실험 GO가 아니다.\n')
put('stage8_paired_analysis_report.md',f'''# Stage 8B — H2 고정 48조건 legacy–stable 전수 짝비교

작성일 2026-11-05. **Stage 9 과학적 독립 검토 GO (검토 입력으로서)**. H2의 최종 과학적 판정·논문 수정·추가 실험은 확정하거나 수행하지 않았다.

## 범위와 근거

original 24 + calibrated 24, v=.5/1/2/3/4/6/8/12 × n=.8/1/1.2, P1/MC-a, 200 posterior draw × N4000 = 38,400,000 trajectory pairs. 생산 시뮬레이터·runner·common·executor를 import/실행하지 않고 NumPy로 저장 배열만 분석했다. `analyze.py`, `finish_reports.py`가 재현 코드이며 `input_manifest.json`이 입력 절대 경로·크기·SHA256을 갖는다. 결과 방향에 따라 조건/구간을 고르지 않았다.

Stage8A 근거: `{B/'mq10_audit_stage8_completion_20261105/stage8_completion_audit.md'}`, checkpoint_audit.json, pairing_inputs.json, stable_schema.json. Stage7 근거: `{S/'conditions.json'}`, input_manifest.json, checkpoints/<condition_id>/jNNN.npz, production/paired_capture_by_draw.json, source/run_original.py, common.py. 보고서 내 모든 CSV는 이 폴더 상대 경로다.

## A. 계산·짝대응 검증

{val['draw_pairs']:,}개 draw pair, {val['trajectory_pairs']:,}개 trajectory pair 전수. legacy의 저장 j=0..199와 두 번째 축 위치를 stable의 명시적 j/trajectory_id=0..3999에 결합했다. legacy float32, stable float64 시간을 유지했다. legacy manifest hash, stable Stage8A hash·commit sidecar·metadata ID·내부 배열 해시, 시간 유효성, 포획=유한 접촉, trigger>i0, stable 조건부 문턱을 대조 PASS. Stage7 기존 draw별 통계와 독립 재산출 결과 전수 일치. 과거 시뮬레이션을 다시 실행해 짝을 증명한 것은 아니며 위치 대응의 과학적 근거는 Stage8A와 저장 공통 난수 코드다.

보호 대상 Stage7/8A 및 legacy 파일 {val['protected_files']:,}개에 대해 전후 경로/크기/mtime_ns/SHA256 대조, 변경 {len(val['protected_changes'])}개. 추가 읽기 근거의 해시는 입력 manifest에 있다. 보호 대상 외 모든 OS 파일 무변경을 증명하는 감사는 아니다.

## B. legacy–stable 수치 변화

{md(agg)}

mean_delta는 조건·draw를 균등 요약한 기술값이지 새 통합 모형/가설 검정이 아니다. 조건별 posterior 차이·분위수·MC SE는 posterior_by_speed_n_model.csv의 quantity=delta, draw별 전환 4종과 SE는 paired_capture_by_condition_draw.csv. 궤적별 차이는 trajectory_differences/48개 NPZ에 (j,trajectory_id)와 함께 저장했다.

시간·검출·접촉 전환 전체 합계:

{md(ts.groupby('metric')[['N','legacy_finite','stable_finite','inf_inf','inf_finite','finite_inf','both_finite','float32_bit_different_finite']].sum().reset_index())}

both-finite 시간 차이의 draw별 평균/중앙값/2.5–97.5%/최대절댓값은 paired_time_by_condition_draw.csv, 조건별 finite_time_delta_summary.csv. finite↔inf는 전체 N 분모에서 따로 기록했다. stable을 float32로 되돌렸을 때 비트가 같은 값은 저장 정밀도와 양립하지만 같은 경로였다는 증명은 아니다. 비트가 다른 값도 문턱 하나의 인과 효과라는 증명은 아니다.

stable_diagnostics_by_draw.csv는 stimulus-only/spontaneous-only/동시 trigger 및 문턱 여유를 기록한다. legacy에는 이 진단과 매 frame X/V가 저장되지 않아 전수 경로 거리·회피 시작 원인·문턱 오류별 포획 기여를 확정할 수 없다. 검출 시각과 포획의 동반 변화까지만 관측 비교다.

## C. 과학·통계적 의미와 남은 한계

n=1의 stable posterior 포획 요약(95% equal-tail; 확률 단위):

{md(stable95)}

모든 n과 legacy/stable의 평균·중앙값·90/95% 구간은 posterior_by_speed_n_model.csv. original-calibrated의 속도별 효과는 model_contrasts.csv에 24조건 모두 제공했다. 두 모형 차이는 기존 k 추출 대 k=8.2 고정과 theta shift=-.87을 함께 포함하므로 한 요인의 효과로 분해하지 않는다. **calibrated stable은 기존 보정값을 유지한 수치 안정화 결과이며 새 재보정이 아니다.**

E와 규칙은 UNKNOWN이 아니라 기존 logs/decisions.md 및 phases/p3b/judge.py에서 확인했다. E=p(12)-p(.5), 99%; 7개 인접 차이는 alpha=.05/35, 최소 효과 .02. draw 분위수와 평균±z SD의 외피 및 기존 G를 재구현했다. 최종 과학적 판정과 분리한 n별 기계적 출력:

{md(mechanical)}

{md(st[st.effect=='E'][['model','n_exp','mean','median','q025','q975','L','U','mcse_median','classification']])}

인접 7구간 × n3 × model2를 모두 h2_effects_rule_G.csv와 h2_effects_by_draw.csv로 제공했다. 기존 legacy 재계산은 previous_effect_comparison.csv와 previous_speed_comparison.csv에서 대조했다. float32로 차이를 계산하던 예전 코드와 현재 float64 평균의 아주 작은 수치 차이는 허용하되 숨기지 않았다.

MC SE와 posterior 분산은 mc_uncertainty_validation_report.md에 분리했다. 200 posterior draw를 독립 실험 반복으로 보지 않는다. ±2 MC SE 민감성에서 분류가 바뀌는 stable 효과는 {len(mcchanged)}개이며 이는 진단이다. 실제 관측된 작은 차이/SE=0을 모집단 동일성으로 단정하지 않는다.

수치적으로 해결된 범위는 조건부 문턱을 안정적으로 추출하고 유한값과 conditioning을 보장한 부분이다. H/Z(누적/조건부 정규화)의 적합-시뮬레이터 구조적 일치, soft/hard 누적·상한·시간집계·지연 시작 경계는 이번 수정 대상이 아니다. 근거 `{B/'mq10_audit_stage3_20261104/audit_report.md'}`의 적합 10ms/시뮬레이터 hard 경계 차이와 Stage6 보고서의 수정 범위를 참고했다. 이전 제한 표본의 survival 차이를 전체 포획효과로 전이하지 않았다. H/Z의 모든 역사적 구현 상태를 이번 자료만으로 완전히 복원했다는 주장은 UNKNOWN이다.

1m/s 손 예측이 원판 검증을 대신하지 않는다. 저장된 원판 재현 실패·보정에 같은 지표 사용·848 대 841 표본 차이는 남는다. 약 1m/s CFD의 고속 배율 및 공간 형태 유지, 종/황혼 범위, 초기상태 복원추출, 지연·회피 지속 가정도 수치 안정화가 검증해 주지 않는다. 안정화 후에도 관측되는 속도별 방향은 모형 예측이며 실증 속도 효과가 아니다.

## 기존 원고 및 수치 대조

지정 파일 `{pdf}`를 찾았고 9쪽 전체 텍스트를 manuscript_extracted_text.txt에 추출했다. 표1은 page3 이미지, 그림1은 page4 이미지다. 원본 내장 이미지를 추출해 함께 제공했다. 사용자가 지정한 파일과 이름은 일치하나 독립적인 최신 버전 계보는 확인하지 못했다.

manuscript_comparison.csv는 본문/그림1 n=1 포획값과 stable 값을, previous_speed_comparison.csv는 전체 n/속도 이전 수치를 대조한다. 표1의 기존 라벨은 새 기계적 G와 대조하되 세 갈래 과학적 종합은 이 단계에서 교체하지 않는다. 원판 1m/s 0.311/0.781/0.178과 관측값은 역사적 값으로만 표시하고 이번 손 조건의 stable로 재검증됐다고 쓰지 않았다. ③ 경계 계산은 입력 근거로만 다루며 재계산하지 않았다.

## D. H2 최종 판정에 필요한 추가 정보

전수 저장 경로/legacy trigger·문턱 진단 부재에 따른 인과 attribution 한계, 원판 1m/s 재현의 독립 검증, H/Z·지연 경계의 통계모형 일관성, 고속 외삽의 경험적 근거, posterior 선택/ESS 및 초기상태 군집 불확실성, 세 갈래의 다른 예측량과 종합 규칙의 타당성에 대한 독립 검토가 필요하다. 이를 위해 추가 시뮬레이션·재적합·재보정을 이 단계에서 수행하지 않았다.

## E. Stage9 GO/NO-GO

**GO — 과학적 독립 검토에 사용할 계산 입력과 전수 비교가 준비됨.** 알려진 UNKNOWN/한계는 unresolved_issues.csv 및 verification_and_open_issues.md에 명시했다. 최종 H2 판정과 논문 수정 승인을 뜻하지 않는다.

## 재현 및 파일 사전

`mq10/.venv/Scripts/python.exe -B <Stage8B>/analyze.py` 후 `finish_reports.py`. 원자료가 바뀌면 해시 assertion 또는 전후 보호 대조가 중단한다. 조건 하나의 legacy 배열/차이 배열과 stable draw 하나만 읽고, 속도 비교는 한 model/n 그룹의 bool 배열 8개만 처리한다. 대규모 전수 시간 배열을 한 번에 적재하지 않았다.

trajectory NPZ: capture_delta=-1/0/+1, det/contact_delta_s는 both-finite만 값이 있고 나머지는 NaN, state=0 inf/inf, 1 finite/inf, 2 inf/finite, 3 finite/finite. 이것은 결측 상태 코드이며 원자료 +inf를 변경한 파일이 아니다. NPZ의 두 축은 저장 j와 trajectory_id이며 condition_id는 파일명이다. CSV 빈 시간 통계는 finite pair 없음 등 정의 불가이며 0이 아니다.

검증 가능한 행수: 포획 draw CSV 9600, 시간 draw CSV 19200, posterior 요약 144, H2 효과 96, H2 draw 효과 19200. 분석 코드 및 산출물 해시는 output_manifest.json.
''')
import runpy
runpy.run_path(str(O/'supplement.py'))
# Final checks and artifact inventory.
assert len(p)==9600 and len(t)==19200 and len(post)==144 and len(e)==96
assert ((p.n01+p.n10+p.n00+p.n11)==4000).all()
assert np.allclose(p.delta,(p.n01-p.n10)/4000)
assert ((t.inf_inf+t.inf_finite+t.finite_inf+t.both_finite)==4000).all()
manifest=[]
for f in sorted(O.rglob('*')):
 if f.is_file() and f.name!='output_manifest.json': manifest.append({'path':str(f),'bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()})
(O/'output_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
print(md(agg));print(md(st[st.effect=='E'][['model','n_exp','median','mcse_median','classification']]));print('MC_SENSITIVE',len(mcchanged));print('REPORT COMPLETE')
