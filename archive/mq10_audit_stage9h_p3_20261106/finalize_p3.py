"""Produce the final scientific report from verified stored results only."""
import p3_core as C
from p3_core import B,read,save,sha,check
from datetime import datetime,timezone,timedelta
import pandas as pd
import numpy as np
KST=timezone(timedelta(hours=9))
def pp(x):return f'{100*float(x):.4f}'
def main():
    verified=read(B/'p3_verification.json');check(verified['status']=='PASS','VERIFICATION_REQUIRED')
    state=read(B/'execution_state.json');gate=read(B/'p3_resource_gate.json')
    df=pd.read_csv(B/'p3_draw_results.csv');cs=pd.read_csv(B/'p3_condition_draw_results.csv');rg=pd.read_csv(B/'p3_rule_g_summary.csv');summary=pd.read_csv(B/'p3_contrast_summary.csv')
    for p,h in read(B/'p3_run_code_manifest.json').items():check(sha(B/p)==h,'RUN_CODE_CHANGED '+p)
    check(len(df)==1200 and len(cs)==2400 and len(rg)==6 and len(summary)==30,'FINAL_TABLE_COUNTS')
    check(df.groupby(['model','n']).size().eq(200).all() and cs.groupby(['model','n','v_mps']).size().eq(200).all(),'ALL_BLOCKS_COMPLETE')
    res=pd.read_csv(B/'resource_usage.csv');post=pd.read_csv(B/'postvalidation_resource_usage.csv');sim=res[res.stage=='simulation']
    check(len(sim)==2364,'RESOURCE_BATCH_COUNT')
    start=datetime.fromisoformat(state['science_started_utc']);end=datetime.fromisoformat(state['science_completed_utc']);now=datetime.now(timezone.utc)
    peak=max(float(res.peak_rss_bytes.max()),float(post.peak_rss_bytes.max()))
    check(peak<1024**3 and now<C.REPORT_DEADLINE and end<C.SCIENCE_DEADLINE,'FINAL_RESOURCE_OR_DEADLINE')
    resource={'status':'PASS','new_L_batches':2364,'reused_L_batches':36,'total_L_batches':2400,'new_trajectories':9456000,'total_L_trajectories_including_reuse':9600000,'new_baseline_batches':0,'science_started_KST':start.astimezone(KST).isoformat(),'science_completed_KST':end.astimezone(KST).isoformat(),'science_elapsed_wall_s':(end-start).total_seconds(),'simulation_wall_s':float(sim.wall_s.sum()),'simulation_cpu_s':float(sim.cpu_s.sum()),'simulation_batch_wall_min_s':float(sim.wall_s.min()),'simulation_batch_wall_max_s':float(sim.wall_s.max()),'observed_phase_wall_sum_s':float(res.wall_s.sum()+post.wall_s.sum()),'peak_RSS_bytes':int(peak),'peak_RSS_MiB':peak/1024**2,'observed_production_phase_read_bytes':int(res.read_bytes.sum()),'observed_production_phase_write_bytes':int(res.write_bytes.sum()),'IO_scope':'measured phases; excludes some orchestration/state/resource-log flush writes, not a whole-process IO total','postvalidation_wall_s':float(post.wall_s.sum()),'conservative_science_budget_s':gate['two_times_estimate_plus_600s_IO_startup_budget_s'],'science_budget_met':(end-start).total_seconds()<gate['two_times_estimate_plus_600s_IO_startup_budget_s'],'science_deadline_KST':'2026-11-06T16:00:00+09:00','report_deadline_KST':'2026-11-06T18:00:00+09:00','disk_free_bytes':__import__('shutil').disk_usage(B).free,'numeric_maxima':verified['numeric_maxima']}
    check(resource['science_budget_met'],'SCIENCE_BUDGET_EXCEEDED')
    save('p3_resource_summary.json',resource)
    retained=rg.decrease_classification_retained.astype(str).str.lower().eq('true')
    allretained=bool(retained.all())
    changed=rg.loc[~retained,['model','n','L_cls']].to_dict('records')
    unstable=rg.loc[~rg.L_MC_perturbation_classification_stable.astype(str).str.lower().eq('true'),['model','n','L_minus2se_cls','L_plus2se_cls']].to_dict('records')
    unstable_text='; '.join(f"{r['model']} n={r['n']:g}: −2SE {r['L_minus2se_cls']}, +2SE {r['L_plus2se_cls']}" for r in unstable) or '없음'
    conclusion=('선언된 L 경계 시나리오에서도 고정 200 draw·MC a의 여섯 8→12 m/s 감소 분류가 모두 유지됐다.' if allretained else '고정 200 draw·MC a에서 하나 이상의 국소 감소 분류가 L 경계 선택에 민감했다. 감소 미유지 블록: '+str(changed)+'.')
    lines=['# Stage 9H P3 탐색적 강건성 분석', '',f'작성: {now.astimezone(KST).isoformat(timespec="seconds")}. **계산·검산 완료. {conclusion}**','',
    '## 범위와 사전 확정 여부','',
    '이번 사용자의 명시적 지시로 P3 및 계획된 국소 Rule G 분석을 실행했다. Stage9G run_04의 P2 열람 전 기록은 claim target을 설명하지만, 당시 과제에는 3 draw 진단으로 충분하고 P3가 필요하지 않다고 명시했다. 따라서 minimal_compute_plan_v2 §6의 “P2 결과 열람 전 P3 필요성 확정” 조건은 **미충족**이다. 이를 사전 확정 검증이라고 주장하지 않으며, 이번 승인을 반영한 탐색적 강건성 분석으로 보고한다. 범위는 결과 방향에 관계없이 처음부터 여섯 model×n 블록, 양 속도, 기존 j0..199 전부로 고정했다.','',
    'original/calibrated × n=(0.8,1,1.2) × v=(8,12) × 기존 200좌표, N=4000, MC a다. run_04 L 36배치를 재사용하고 L 2,364배치·9,456,000궤적만 새로 계산했다. 전체 L 2,400배치·9,600,000궤적을 분석했다. B는 기존 Stage7 저장 2,400배치를 재사용했으며 신규 baseline·Stage7/8/9G 재계산은 0배치다. P3 이후 실험은 실행하지 않았다.','',
    '## 과학 Gate와 모수·난수 연결','',
    '- run_04 CFI0–4, 4개 baseline sentinel의 전체 schema/float64 bit 일치, L 고정배열·독립식 검증을 재사용했다. L executor·baseline·strict sampler·의존 코드와 난수 생성 함수는 검증본과 바이트가 동일하다. 신규 실행부 변경은 경로 깊이, P3 상한·시간 예산·단계 표시와 순회다(`p3_core_changes.diff`).',
    '- 원래 선택 CSV의 j/chain/draw/flat 순서 200개와 필요한 Zarr 변수만 index-first로 읽었다. 실제 접근 metadata·청크는 Stage7 manifest의 현재 SHA256와 연결했다. 물리 청크에 동거하는 다른 좌표가 있을 수 있으나 배열 materialization은 승인된 좌표만 수행했다. 전체 posterior 스캔·새 추정·재표집·재보정은 하지 않았다.',
    '- 기존 `params_from_draw` 변환과 독립 결정적 매핑을 모든 좌표에서 bit 대조했다. sigma=raw sigma+1e-3, a_sp/b_sp shape=(1,), eta_sess=raw sigma_sess×z_sess 및 원 session 순서를 유지했다. raw sigma_sess를 별도 읽고 MC eta=raw sigma_sess×새 MC z를 사용했다. run_04 j0/73/199의 P/raw 값도 정확 일치한다. 저장 full-P 사본과 직접 대조한 것으로 주장하지 않는다.',
    '- seed `[20261105,j,0]` → 초기상태 pick → uniform(N,6) → z_a → z, k seed20261103/j, escape seed20261104/j를 유지했다. 2,400개 저장 B의 u_threshold 및 strict 문턱 재구성을 float64 bit로 확인했다. 재사용 L 36개 전체 입력 fingerprint도 정확 일치했다. 과거 full uniform/pick/z 배열은 저장되어 있지 않으므로 과거 전체 난수의 직접 대조를 주장하지 않는다.',
    '- 생산 중 기존 observer로 매 프레임 유한 구동/상태, strict 꼬리, u 복원 ≤2e-11, 생존 범위·비증가 atol1e-12, 문턱·포획·검열 상태를 검사했다. 허용오차·sigma·seed·N 변경, clipping/fallback/과학 실패 재시도는 없었다.',
    '- 준비 오류 2건은 과학 0배치에서 발생했다: 신규 복사본 줄바꿈을 바이트 보존 복사로 수정; 원래 manifest 경로 구분자를 충돌 거부 조건으로 정규화했다. `preparation_attempt_01/02`에 기록을 보존했다. 원본 모수·데이터·strict 수치 실패는 없었다.','',
    '## 국소 Rule G 결과','',
    'Δ7=p_capture(12)−p_capture(8). 기존 judge.py의 interval/classify/effect_row 함수를 AST 그대로 가져와 사용했다. α=0.05/35, δ=0.02를 유지했다. 구간은 draw 분위수 구간과 평균±z×draw SD의 외피이며, SD/√200을 넣지 않는다. “감소”는 U<0이다. 여섯 블록만 분석해도 α를 0.05/6으로 바꾸지 않았다. 아래 수치 단위는 %p이며 95% 기술구간과 G 구간은 서로 다르다.','',
    '|model|n|B Δ7 중앙값|L Δ7 중앙값|L 95% 경험구간|L G 구간|B→L 분류|L 음수 draw/200|L +2 MC SE 분류|','|---|---:|---:|---:|---|---|---|---:|---|']
    for r in rg.itertuples(index=False):lines.append(f'|{r.model}|{r.n:g}|{pp(r.B_median)}|{pp(r.L_median)}|[{pp(r.L_q025)}, {pp(r.L_q975)}]|[{pp(r.L_L)}, {pp(r.L_U)}]|{r.B_cls}→{r.L_cls}|{r.L_negative_draws}/200|{r.L_plus2se_cls}|')
    lines+=['',conclusion,'',f'L의 ±2 MC SE 결정론적 이동으로 분류가 달라지는 블록: {unstable_text}. 이 진단은 기존 분류를 대체하지 않으며 MC 신뢰구간이나 새 posterior 표본이 아니다. B의 여섯 Δ7 draw 값·G 구간·분류는 기존 Stage8 저장 결과와 검산 일치했다. 전체 H2, E, 나머지 인접 효과, 지수 결합은 재판정하지 않았다.','',
    '## 짝대응 L−B와 속도 효과 차이','',
    '같은 model/n/j/trajectory/MC 키로 네 포획 지표를 연결했다. L−B, B와 L 각각의 12−8, (L12−L8)−(B12−B8)를 궤적 수준에서 먼저 구해 공분산을 보존했다. 각 draw의 MC SE는 짝차이 표본 SD/√4000이다. 속도 효과 차이의 SE 역시 네 지표의 궤적 차이로 직접 계산했다. 아래는 draw 간 중앙값(%p)이다.','',
    '|model|n|L−B v8|L−B v12|속도 효과 차이 L−B|차이의 95% 경험구간|','|---|---:|---:|---:|---:|---|']
    for (m,n),g in summary.groupby(['model','n'],sort=False):
        d=g.set_index('quantity');a=d.loc['L_minus_B_v8'];b=d.loc['L_minus_B_v12'];c=d.loc['speed_delta_L_minus_B']
        lines.append(f'|{m}|{n:g}|{pp(a["median"])}|{pp(b["median"])}|{pp(c["median"])}|[{pp(c.q025)}, {pp(c.q975)}]|')
    lines+=['','각 열은 해당 통계량의 draw 간 중앙값이다. 중앙값은 선형 연산이 아니므로 열별 중앙값의 차이가 속도 효과 차이의 중앙값과 같을 필요는 없다.','',f'전체 draw·속도별 L−B 포획률 범위는 {pp(cs.capture_paired_delta_L_minus_B.min())}~{pp(cs.capture_paired_delta_L_minus_B.max())}%p다. `p3_draw_results.csv`는 1,200행(model×n×j), `p3_condition_draw_results.csv`는 2,400행(model×n×v×j)으로 모든 블록·draw를 제공한다. 각 차이의 MC SE, discordance와 원 포획률 및 draw 간 요약은 CSV에 있다.','',
    '## 조건화·감지·검열 해석','',
    f'B 대비 M0가 달라진 L 궤적은 {int(cs.M0_changed.sum()):,}/{9600000:,}개다. draw별 평균 F0 변화는 {cs.F0_delta_mean.min():.6f}~{cs.F0_delta_mean.max():.6f}, logZ<log(1e−8) 사례는 B {int(cs.B_logZ_sensitive_count.sum()):,}개, L {int(cs.L_logZ_sensitive_count.sum()):,}개다. 표시상 F0 반올림과 유한 logZ를 구분하고 strict 검산을 유지했다.',
    'L은 시작 전 지연 특징을 0으로 연장하는 선언된 경계 시나리오다. 실제 시작 전 자극을 관측하거나 생물학적으로 0이라고 식별한 결과가 아니다. Sz(max(theta,1e−6))를 유지하므로 D 전체/F0를 0 또는 Z를 1로 강제하지 않았다. M0/F0/logZ와 시작생존 조건부 문턱 분포가 바뀌고, 이후 감지·회피 경로도 달라질 수 있다. 공통 난수를 썼더라도 같은 개체 집단의 인과효과로 읽을 수 없다.',
    '조건별 CSV는 포획·감지-before-contact/end·미발동·자발/자극/동시 발동, 접촉 및 행정 검열, M0·F0·logZ 변화를 모두 보존한다. 비감지를 0초로 바꾸거나 무한대 시간을 빼지 않았다. 이 결과로 올바른 생물학적 경계나 원래 H2의 참·거짓을 결정하지 않는다.','',
    '## 불확실성과 주장 한계','',
    '- 고정 MC a·N4000의 유한 MC 오차는 draw 간 분산과 구간에 남는다. MC SE는 초기상태 841개 경험분포에서 조건부 복원추출한 궤적 오차이며, 원 세션/trigger 군집 및 초기상태 경험분포 추정 오차까지 포함하지 않는다.',
    '- 기존 PSIS 확률 비례 비복원 선택 200좌표를 재선택·재가중 없이 동일 가중 경험분포로 요약했다. 이를 전체 재가중 posterior의 정확한 대표표본으로 과장하지 않는다. 극단 분위수/꼬리, 적분오차·ESS·PSIS 근사·보정 및 외삽 불확실성은 해결하지 않았다.',
    '- var(draw 효과)와 mean(MC SE²), 양수 부분 var−mean(SE²)는 구분해 출력했지만 정제된 credible interval이나 정확한 분산분해는 아니다. ±2SE 분류 변화와 posterior 폭을 혼동하지 않는다. 200 draw를 독립 실험 반복으로 취급하지 않는다. discordance가 0일 때 MC SE=0은 관측 표본의 결과이며 모집단 차이가 정확히 0임을 보장하지 않는다.',
    '- 이 탐색적 분석의 국소 G 분류는 다른 구조 대안, 실제 모기 집단, 엄밀한 동시 보장, 전체 H2 최종 판정으로 확장할 수 없다.','',
    '## 자원·마감·산출물','',
    f'실행 전 run_04 최대 simulation {gate["max_simulation_s"]:.6f}s + checkpoint {gate["max_checkpoint_s"]:.6f}s + 짝분석 {gate["max_pair_read_summary_s"]:.6f}s에 2,364배치와 2배 여유를 적용하고 시작/I/O 600초를 더해 과학 예산 {resource["conservative_science_budget_s"]/60:.2f}분을 확보했다. 후검산·보고 90분과 중단정리 30분도 별도 확보했다. 이는 측정 기반 보수적 외삽으로 보장된 ETA가 아니다.',
    f'실제 과학실행: {start.astimezone(KST).isoformat(timespec="seconds")} → {end.astimezone(KST).isoformat(timespec="seconds")}, {resource["science_elapsed_wall_s"]/60:.2f}분. simulation wall {resource["simulation_wall_s"]:.3f}s, CPU {resource["simulation_cpu_s"]:.3f}s, 배치 최대 {resource["simulation_batch_wall_max_s"]:.6f}s, peak RSS {resource["peak_RSS_MiB"]:.2f}MiB. 1GiB/RSS·디스크 여유 5GB 한계를 유지했고 계산 16:00/보고 18:00 KST 전에 완료했다.',
    f'신규+재사용 L strict 수치 최대 u 복원 오차 {verified["numeric_maxima"]["max_uniform_recovery_error"]:.3g}, 기존 tail tolerance 대비 최대 잔차 비율 {verified["numeric_maxima"]["max_tail_residual_fraction_of_existing_tolerance"]:.3g}, 생존 증가 최대 {verified["numeric_maxima"]["max_survival_increase"]:.3g}. 모든 소비 입력·코드·선택 체크포인트 {verified["consumed_input_preservation_files"]:,}개의 제한적 보존 확인이 PASS다. 전수 파일 해시, WRITE_DAC, 반복 환경 감사, 원본·논문 수정은 수행하지 않았다. Python audit guard의 보호는 관측 Python 이벤트 범위이며 native I/O 전수 증명을 주장하지 않는다.','',
    '- `p3_draw_results.csv`: 1,200개 draw의 네 포획률·짝차이·MC SE·discordance.',
    '- `p3_rule_g_summary.csv`: 6개 블록의 기존 B/L 국소 G 구간·분류·±2SE 진단.',
    '- `p3_condition_draw_results.csv`, `p3_contrast_summary.csv`: 조건화/검열과 경험적 posterior·MC 요약.',
    '- `p3_verification.json`, `p3_resource_gate.json`, `p3_resource_summary.json`, `cfi_gate_results.json`: 간결한 Gate·검산·자원 기록.',
    '- `selected_parameter_provenance.json`, `actual_zarr_read_access.json`, `reuse_and_baseline_gate.json`, `checkpoint_output_manifest.json`: 제한 모수 읽기·기존 계산 재사용·검산 계보.','']
    (B/'p3_science_report.md').write_text('\n'.join(lines),encoding='utf-8')
    text=f'''# Stage9H P3 handover

{now.astimezone(KST).isoformat(timespec="seconds")}. 과학 계산·검산·보고 완료. 탐색적 강건성 분석이며 P2 열람 전 P3 필요성 사전 확정 조건은 미충족; 이번 사용자 명시 승인으로 진행했다.

{conclusion}

L 총 2,400배치: run_04 36개 재사용 + 신규 2,364개·9,456,000궤적. N4000/MC a/기존 seed·strict 규약 유지. B Stage7 2,400개 재사용, 기존 Stage7/8/9G 재계산 0. 원래 200좌표만 모수·raw sigma_sess 읽기 및 재구성 검증 PASS. 전체 H2 재판정·P3 이후 실험 없음.

주요 산출물: p3_science_report.md, p3_draw_results.csv(1200행), p3_rule_g_summary.csv(6행). 조건별 상세 p3_condition_draw_results.csv(2400행), p3_contrast_summary.csv(30행). MC ±2SE 민감 블록: {unstable_text}. MC 오차·고정 200 draw 불확실성·시작생존 조건화 집단 차이를 구분해서 읽어야 한다.

Gate: cfi_gate_results.json, reuse_and_baseline_gate.json, p3_verification.json. 자원: p3_resource_gate.json, p3_resource_summary.json, resource_usage.csv 및 postvalidation_resource_usage.csv. 과학 종료 {end.astimezone(KST).isoformat(timespec="seconds")}, peak RSS {resource['peak_RSS_MiB']:.2f}MiB. 계산16:00/보고18:00 마감 충족.

코드: run_p3.py + p3_core.py는 실행 완료, 재실행/재개하지 않는다. analyze_p3.py와 finalize_p3.py는 저장 결과 검산·보고 전용이며 simulator 호출이 없다. 신규 checkpoint는 checkpoints/L/에 j0/73/199를 제외한 2364개만 있다. 이 세 j의 36개 L과 conditioning 파일은 run_04 원경로를 읽는다. 원본 baseline은 Stage7 원경로를 읽는다. 준비 2회는 모두 과학0배치로 중단했고 preparation_attempt_01/02에 보존했다.

원본/논문 수정, 전수 46000파일 해시, WRITE_DAC, 반복 환경 감사 없음. 모든 신규 파일은 현재 Stage9H 폴더에만 작성했다. 후속 추가실험은 실행하지 않았다.
'''
    (B/'handover.md').write_text(text,encoding='utf-8')
    gates=read(B/'cfi_gate_results.json');gates['CFI4']['postvalidation']='PASS: p3_verification.json';save('cfi_gate_results.json',gates)
    save('completion_status.json',{'status':'COMPLETE','completed_KST':now.astimezone(KST).isoformat(),'exploratory':True,'prior_P3_necessity_condition_met':False,'all_six_local_decrease_classifications_retained':allretained,'new_L_batches':2364,'reused_L_batches':36,'new_trajectories':9456000,'new_B_batches':0,'overall_H2_rejudged':False,'additional_experiments_after_P3':0})
    names=['p3_science_report.md','p3_draw_results.csv','p3_rule_g_summary.csv','handover.md','p3_condition_draw_results.csv','p3_contrast_summary.csv','p3_verification.json','p3_resource_gate.json','p3_resource_summary.json','cfi_gate_results.json','completion_status.json','p3_run_code_manifest.json','checkpoint_output_manifest.json']
    save('final_output_manifest.json',{'scope':'named final Stage9H artifacts only; checkpoint hashes in separate manifest','files':{n:{'sha256':sha(B/n),'bytes':(B/n).stat().st_size} for n in names}})
    print(conclusion)
    print('FINAL_COMPLETE',resource['science_elapsed_wall_s'],resource['peak_RSS_MiB'])
if __name__=='__main__':main()
