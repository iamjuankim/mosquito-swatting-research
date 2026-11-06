"""Prepare isolated Stage9H copies; no scientific computation."""
from pathlib import Path
import ast, hashlib, json, difflib
B=Path(__file__).resolve().parent
G=B.parent/'mq10_audit_stage9g_p1p2_20261106/run_04'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
manifest=json.loads((G/'run_code_manifest.json').read_text(encoding='utf-8'))
for name,h in manifest.items():
    assert sha(G/name)==h, name
for name in ['baseline.py','variant_L.py','stable_threshold.py','extracted.py','isolation.py']+[f'source/swat/{n}.py' for n in ['__init__','detection','flow','escape','porous']]:
    p=B/name;p.parent.mkdir(parents=True,exist_ok=True)
    data=(G/name).read_text(encoding='utf-8')
    if name=='isolation.py': data=data.replace("BASE.parent.parent/'mq10/", "BASE.parent/'mq10/")
    if name=='isolation.py':p.write_text(data,encoding='utf-8')
    else:p.write_bytes((G/name).read_bytes())
source=(G/'execute.py').read_text(encoding='utf-8')
core=source[:source.index('def preflight():')]
tree=ast.parse(source)
node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='summary_pair')
core+='\n'+ast.get_source_segment(source,node)+'\n'
core=core.replace('B.parent.parent/', 'B.parent/')
core=core.replace("STATE['attempted_batches']<40", "STATE['attempted_batches']<2364")
core=core.replace("time.perf_counter()-SCI_START<30*60", "time.perf_counter()-SCI_START<SCI_BUDGET")
core=core.replace("'SCIENCE_30MIN_BUDGET'", "'P3_MEASURED_SCIENCE_BUDGET'")
core=core.replace("phase='P1' if not STATE['P1_complete'] else 'P2'", "phase='P3'")
(B/'p3_core.py').write_text(core,encoding='utf-8')
(B/'p3_core_changes.diff').write_text(''.join(difflib.unified_diff(source[:source.index('def preflight():')].splitlines(True),core[:core.index('def summary_pair(')].splitlines(True),fromfile='run_04/execute.py reusable prefix',tofile='p3_core.py reusable prefix')),encoding='utf-8')
judge=B.parent/'mq10/phases/p3b/judge.py'
s=judge.read_text(encoding='utf-8-sig');t=ast.parse(s)
parts=[ast.get_source_segment(s,n) for n in t.body if isinstance(n,ast.FunctionDef) and n.name in ['interval','classify','effect_row']]
assert len(parts)==3
(B/'local_judge.py').write_text('"""Exact AST-selected original judge functions; no production imports."""\nimport numpy as np\nfrom scipy.stats import norm\n\n'+'\n\n'.join(parts)+'\n',encoding='utf-8')
(B/'code_reuse_record.json').write_text(json.dumps({'source':str(G),'source_manifest_verified':manifest,'copied_scientific_functions':'unchanged baseline/L/sampler/dependencies/extracted; compute observer/checkpoint and summary functions reused','core_changes':'root depth, P3 batch cap 2364, measured P3 time budget, phase label; old preflight/main removed','judge_source':str(judge),'judge_sha256':sha(judge),'judge_functions_exact_AST':['interval','classify','effect_row'],'no_H2_combination':True},indent=2),encoding='utf-8')
print('Prepared isolated code copies and unchanged local judge functions.')
