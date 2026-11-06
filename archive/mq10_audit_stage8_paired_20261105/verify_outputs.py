from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
O=Path(__file__).resolve().parent
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''): h.update(b)
 return h.hexdigest()
p=pd.read_csv(O/'paired_capture_by_condition_draw.csv'); t=pd.read_csv(O/'paired_time_by_condition_draw.csv'); q=pd.read_csv(O/'precontact_detection_by_draw.csv')
assert len(p)==9600 and len(t)==19200 and len(q)==9600
assert not p.duplicated(['condition_id','j']).any()
for cid,g in p.groupby('condition_id',sort=False):
 g=g.sort_values('j'); assert np.array_equal(g.j,np.arange(200))
 with np.load(O/'trajectory_differences'/f'{cid}.npz',allow_pickle=False) as z:
  a=z['legacy_capture']; b=z['stable_capture']; d=z['capture_delta']
  assert d.shape==(200,4000) and np.array_equal(d,b.astype(np.int8)-a.astype(np.int8))
  assert np.allclose(d.mean(1),g.delta) and np.allclose(d.std(1,ddof=1)/np.sqrt(4000),g.paired_mcse)
  for metric,state,delta in [('t_det_s','det_state','det_delta_s'),('t_contact_s','contact_state','contact_delta_s')]:
   s=z[state]; dt=z[delta]; h=t[(t.condition_id==cid)&(t.metric==metric)].sort_values('j')
   assert np.array_equal(np.isfinite(dt),s==3)
   for code,key in [(0,'inf_inf'),(1,'finite_inf'),(2,'inf_finite'),(3,'both_finite')]: assert np.array_equal((s==code).sum(1),h[key])
assert ((q.n01+q.n10+q.n00+q.n11)==4000).all()
assert np.allclose(q.delta,(q.n01-q.n10)/4000)
# Verify every original protected file after all postprocessing, including any unexpected additions/deletions.
before=json.loads((O/'protected_inputs_before.json').read_text(encoding='utf-8')); changes=[]
for name,v in before.items():
 f=Path(name)
 if not f.exists() or f.stat().st_size!=v['bytes'] or f.stat().st_mtime_ns!=v['mtime_ns'] or sha(f)!=v['sha256']: changes.append(name)
for folder in ['mq10_audit_stage7_h2_stable_20261105','mq10_audit_stage8_completion_20261105']:
 for f in (O.parent/folder).rglob('*'):
  if f.is_file() and str(f) not in before: changes.append('ADDED '+str(f))
assert not changes
v=json.loads((O/'validation_results.json').read_text(encoding='utf-8'));v['independent_output_array_reconciliation']='PASS';v['protected_after_all_postprocessing']='PASS';v['protected_changes_after_all_postprocessing']=changes
(O/'validation_results.json').write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
(O/'final_verification.md').write_text('# 최종 산출물 검증\n\nverify_outputs.py를 실행해 48개 trajectory NPZ와 포획/시간 draw CSV를 전수 재대조했다. 9600개 포획/접촉 전 감지 행, 19200개 시간 행, 모든 궤적 차이 및 MC SE, 시간 상태 코드별 집계가 일치했다. 모든 후처리 종료 후 원본 Stage7/8A 및 legacy 19422개 파일 크기·mtime_ns·SHA256을 다시 확인했고 추가/삭제/변경은 0개였다.\n\n원고 표1·그림1은 추출된 원본 내장 이미지를 직접 시각 검토했다. PDF 자체는 변경하지 않았다. 최종 과학적 H2 판정은 보류이며 Stage9 과학적 독립 검토 GO다.\n',encoding='utf-8')
manifest=[{'path':str(f),'bytes':f.stat().st_size,'sha256':sha(f)} for f in sorted(O.rglob('*')) if f.is_file() and f.name!='output_manifest.json']
(O/'output_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
assert all(sha(r['path'])==r['sha256'] for r in manifest)
print('PASS: 48 trajectory archives; all CSV reconciliation; 19422 protected files unchanged; '+str(len(manifest))+' artifact hashes verified')
