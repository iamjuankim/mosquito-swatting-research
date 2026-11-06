from pathlib import Path
import json,csv,hashlib,datetime,collections
import numpy as np
B=Path(__file__).resolve().parent
R=B.parent.parent

def load(n):return json.loads((B/n).read_text(encoding='utf-8'))
def write(n,x):(B/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(n):
 with (B/n).open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
state=load('execution_state.json');g=load('cfi_gate_results.json');pairs=rows('paired_screening_results.csv');speed=rows('paired_speed_contrasts.csv');resources=rows('resource_usage.csv')
assert state['completed_batches']==40 and state['completed_trajectories']==160000 and state['P2_complete'] and not state['P3_executed']
assert all(x['status']=='PASS' for x in g.values()) and len(pairs)==36 and len(speed)==18
assert len({(r['condition_id'],r['j']) for r in pairs})==36
for n,h in load('run_code_manifest.json').items():assert sha(B/n)==h,n
checkpoints=list((B/'checkpoints').rglob('j*.json'));assert len(checkpoints)==40
numeric=collections.defaultdict(float)
for p in checkpoints:
 m=json.loads(p.read_text(encoding='utf-8'));assert sha(p.with_suffix('.npz'))==m['sha256']
 assert sha(p.with_name(p.stem+'_conditioning.npz'))==m['conditioning_sha256']
 for key,value in m['numeric'].items():numeric[key]=max(numeric[key],value)
for r in pairs:
 for variant in ['B','L']:
  p=(R/'mq10_audit_stage7_h2_stable_20261105/checkpoints'/r['condition_id']/('j%03d.npz'%int(r['j']))) if variant=='B' else B/'checkpoints/L'/r['condition_id']/('j%03d.npz'%int(r['j']))
  with np.load(p,allow_pickle=False) as z:
   count=int(z['captured'].sum());assert count==int(r[variant+'_captured'])
   assert count/4000==float(r[variant+'_capture_rate'])
 for key in ['B','L']:assert int(r[key+'_captured'])+int(r[key+'_no_capture'])==4000
sim=[r for r in resources if r['stage']=='simulation'];assert len(sim)==40
start=datetime.datetime.fromisoformat((B/'execution_once.lock').read_text());end=datetime.datetime.fromisoformat(state['science_completed_utc'])
summary={'status':'PASS','completed_utc':state['science_completed_utc'],'end_to_end_science_and_targeted_validation_wall_s':(end-start).total_seconds(),'simulation_wall_s':sum(float(x['wall_s']) for x in sim),'simulation_cpu_s':sum(float(x['cpu_s']) for x in sim),'peak_rss_bytes':max(int(x['peak_rss_bytes']) for x in resources),'simulation_batch_wall_range_s':[min(float(x['wall_s']) for x in sim),max(float(x['wall_s']) for x in sim)],'measured_phase_wall_sum_s':sum(float(x['wall_s']) for x in resources),'measured_phase_cpu_sum_s':sum(float(x['cpu_s']) for x in resources),'measured_phase_read_bytes':sum(int(x['read_bytes']) for x in resources),'measured_phase_write_bytes':sum(int(x['write_bytes']) for x in resources),'output_bytes_before_reporting':sum(p.stat().st_size for p in B.rglob('*') if p.is_file()),'scope':'successful run_04 only; prior startup-only attempts retained separately','numeric_maxima':dict(numeric),'final_checks':{'summary_rows':36,'speed_rows':18,'checkpoint_hashes_and_conditioning_hashes':40,'capture_counts_reconciled':72,'run_code_unchanged':True},'P3_executed':False}
write('resource_summary.json',summary)
write('final_artifact_verification.json',summary['final_checks'])
write('paired_screening_results_status.json',{'status':'COMPLETE','data_rows':36,'new_baseline_batches':4,'new_L_batches':36,'N_each':4000,'total_new_trajectories':160000,'P3_executed':False})
write('gate_summary.json',{k:{'status':v['status']} for k,v in g.items()})
lines=['# Stage9G P1/P2 완료 보고','',f"2026-11-06 KST. **완료: 신규 baseline 4 + L 36 = 40배치·160,000궤적. P3 미실행.** 실행 종료 {(end+datetime.timedelta(hours=9)).strftime('%Y-%m-%d %H:%M:%S')} KST.",'','P1 8배치·32,000궤적 뒤 P2 추가 32배치를 연속 실행했다. P1의 L 4배치를 재사용하여 P2 L 36배치를 완결했다. baseline의 나머지 32배치는 Stage7 저장 결과를 읽었으며 재계산하지 않았다.','', 'CFI0–4 모두 PASS. j0/73/199만 index-first로 읽어 기존 params_from_draw 변환·raw sigma_sess·shape·sigma epsilon을 유지했다. 선택 입력/청크 43개와 실제 사용 코드 9개의 기존 manifest 연결을 확인했다. full P 저장 사본 대조를 주장하지 않는다. L은 _lagged만 변경했고 고정배열 및 독립 구동식 검산을 통과했다.','', '4 baseline sentinel은 저장 Stage7 전체 schema·dtype·shape·float64 비트·무한대·이산 상태가 정확 일치했다. 신규 B/L의 전체 입력·난수 bit 비교도 PASS다. 과거 full 난수 배열은 저장되어 있지 않아 과거 전체 난수의 직접 대조를 주장하지 않는다. strict sampler 허용오차, u 복원 2e-11, 생존 범위/비증가 1e-12를 그대로 유지했다.','', '|모형|n|L−B 포획률 변화 범위 (pp, 6사례)|B의 12−8 범위 (pp, 3 draw)|L의 12−8 범위 (pp, 3 draw)|','|---|---:|---:|---:|---:|']
for model in ['original','calibrated']:
 for n in [.8,1.,1.2]:
  rr=[x for x in pairs if x['model']==model and float(x['n'])==n];ss=[x for x in speed if x['model']==model and float(x['n'])==n]
  values=[[100*float(x['capture_paired_delta_L_minus_B']) for x in rr]]+[[100*float(x[v+'_paired_delta_v12_minus_v8']) for x in ss] for v in ['B','L']]
  lines.append('|'+model+'|'+str(n)+'|'+'|'.join(f'{min(v):.3f} ~ {max(v):.3f}' for v in values)+'|')
lines+=['','고정 18개 model×n×draw 사례 모두 L에서도 v12 포획률이 v8보다 낮았다. L−B 포획률 변화는 전체 36사례에서 −1.325~0 pp였다. 이는 선택된 3 draw·고정 MC a의 기술 결과이며 posterior 전체 강건성·같은 개체의 인과효과·생물학적 정답을 뜻하지 않는다. G/H2를 재판정하지 않았다.','',f"M0는 144,000개 L 궤적 모두에서 B와 달랐다. 평균 F0 변화 범위는 −0.762228~−0.164452다. logZ<log(1e−8) 사례는 B 5,655개, L 0개였고 모두 strict 검사를 통과했다. 감지·조건화·검열 상세는 paired_screening_results.csv와 conditioning NPZ에 저장했다.",'',f"성공 실행 시작부터 과학 완료까지 {(end-start).total_seconds():.3f}초; 40배치 simulation wall {summary['simulation_wall_s']:.3f}초, CPU {summary['simulation_cpu_s']:.3f}초, peak RSS {summary['peak_rss_bytes']/1024**2:.2f} MiB. 구간별 자원은 resource_usage.csv, 합계와 수치 최대 오차는 resource_summary.json. 운영 상한 1GiB·디스크 여유 5GB를 유지했다.",'','준비 오류는 과학 계산 없이 수정했다: 원본 manifest의 역슬래시 키를 slash로 정규화하고 충돌/없는 키를 거부; platform.py의 shell `ver` 확인을 sys.getwindowsversion 기반 읽기로 대체; UTF-8 명시. run_02는 버전 확인 subprocess 차단, run_03은 cp949 decode 오류로 각각 0배치에서 종료했다. run_04만 과학 40배치를 실행했다. 기존 실패 기록은 보존했다.','', '현재 환경은 Stage7과 Python/패키지/OS 값이 같고 스레드 환경변수만 1로 명시했다. 차이는 기록하고 baseline 정확 재현으로 검증했다. 전수 감사·WRITE_DAC 검사·권한 변경·외부 원본 쓰기·P3는 수행하지 않았다. 마지막으로 소비 입력 43개·사용 코드 9개·선택 baseline 36개와 sidecar의 보존만 확인했다. Python audit guard는 Python 이벤트 범위의 보호이며 전체 native I/O를 증명하는 전수 검사는 아니다.','', '재사용: run_04의 검증 완료 코드·40 checkpoint·36 conditioning 결과·CSV·Gate·모수 provenance·자원 기록. 이 실행은 완결되어 재개하거나 다시 계산할 필요가 없다. P3는 금지 범위로 남는다.']
(B/'stage9g_p1p2_execution_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
handover='''# Stage9G 완료 handover

2026-11-06 KST. run_04에서 P1→P2 완료. 40배치·160,000궤적(신규 baseline 4, L 36); P1 L 4개를 P2에 재사용. P3 0.

CFI0–4 PASS, 4 baseline 전체 schema/float64 bit 정확, 제한 j0/73/199 모수 공급·L fixture·독립식·strict tail/u 복원·생존 정상성 모두 PASS. 실제 사용 입력/청크와 코드만 검증했고 전수 감사·WRITE_DAC 검사·원본 변경 없음.

고정 18개 model×n×draw에서 L의 v12−v8 포획률은 모두 음수. L−B 포획률 −1.325~0 pp. 3 draw/MC a 사례 결과로 한정하며 posterior 전체 강건성·개체 인과효과·G/H2 재판정 불가.

결과: paired_screening_results.csv(36행), paired_speed_contrasts.csv(18행), checkpoints/B 4·L 36 및 conditioning NPZ. Gate: cfi_gate_results.json, gate_summary.json, sentinel_exact_comparison.json. 모수: selected_parameter_provenance.json, verified_input_manifest.json, actual_zarr_read_access.json. 자원: resource_usage.csv, resource_summary.json. 상세 해석/한계: stage9g_p1p2_execution_report.md.

준비 오류는 원본 manifest 경로 키 정규화(없는 키/충돌 거부), native Windows 버전 읽기, UTF-8 명시로 해결. run_02/run_03은 모두 과학 0배치. 최초 Stage9G 실패 기록은 상위 폴더에 보존. 검증 완료 계산은 run_04만 사용한다. 재계산 불필요. P3 실행 금지 유지.
'''
(B/'handover.md').write_text(handover,encoding='utf-8')
print(json.dumps(summary,indent=2))
