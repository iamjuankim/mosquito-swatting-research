"""Recompute summaries from released draw CSVs; no production driver imports."""
from pathlib import Path
import importlib.util
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
S8 = ROOT / 'archive/mq10_audit_stage8_paired_20261105'
H = ROOT / 'archive/mq10_audit_stage9h_p3_20261106'
spec = importlib.util.spec_from_file_location('historical_judge', H / 'local_judge.py')
judge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(judge)

def near(a, b):
    if not np.isclose(a, b, rtol=1e-11, atol=2e-14):
        raise ValueError(f'Summary mismatch: {a} != {b}')

def main():
    out = ROOT / 'generated'
    out.mkdir(exist_ok=True)
    d = pd.read_csv(H / 'p3_draw_results.csv')
    expected = pd.read_csv(H / 'p3_rule_g_summary.csv')
    rows = []
    for _, e in expected.iterrows():
        block = d[(d.model == e.model) & (d.n == e.n)].sort_values('j')
        assert len(block) == 200 and block.j.tolist() == list(range(200))
        r = dict(model=e.model, n=e.n)
        for variant in ['B', 'L']:
            x = block[f'{variant}_speed_delta'].to_numpy()
            se = block[f'{variant}_speed_delta_mcse'].to_numpy()
            lo, hi, med = judge.interval(x, .05 / 35)
            values = dict(L=lo, U=hi, median=med, mean=float(x.mean()),
                          q025=float(np.quantile(x, .025)), q975=float(np.quantile(x, .975)))
            for key, value in values.items():
                near(value, e[f'{variant}_{key}'])
                r[f'{variant}_{key}'] = value
            cls = judge.classify(lo, hi, med, .02)
            assert cls == e[f'{variant}_cls']
            r[f'{variant}_cls'] = cls
            for label, sign in [('minus2se', -1), ('plus2se', 1)]:
                a, b, c = judge.interval(x + sign * 2 * se, .05 / 35)
                cls = judge.classify(a, b, c, .02)
                assert cls == e[f'{variant}_{label}_cls']
                near(a, e[f'{variant}_{label}_L']); near(b, e[f'{variant}_{label}_U'])
                r[f'{variant}_{label}_cls'] = cls
        rows.append(r)
    pd.DataFrame(rows).to_csv(out / 'stage9h_local_rule_g.csv', index=False)
    d = pd.read_csv(S8 / 'h2_effects_by_draw.csv')
    expected = pd.read_csv(S8 / 'h2_effects_rule_G.csv')
    rows = []
    for _, e in expected.iterrows():
        block = d[(d.model == e.model) & (d.n_exp == e.n_exp) &
                  (d.version == e.version) & (d.effect == e.effect)].sort_values('j')
        assert len(block) == 200 and block.j.tolist() == list(range(200))
        x = block.delta.to_numpy(); se = block.paired_mcse.to_numpy()
        lo, hi, med = judge.interval(x, e.alpha)
        near(lo, e.L); near(hi, e.U); near(med, e['median']); near(x.mean(), e['mean'])
        cls = judge.classify(lo, hi, med, .02)
        assert cls == e.classification
        for key, sign in [('minus2se_class', -1), ('plus2se_class', 1)]:
            assert judge.classify(*judge.interval(x + sign * 2 * se, e.alpha), .02) == e[key]
        rows.append(dict(model=e.model, n_exp=e.n_exp, version=e.version,
                         effect=e.effect, L=lo, U=hi, median=med, classification=cls))
    pd.DataFrame(rows).to_csv(out / 'stage8_effect_rule_g.csv', index=False)
    result = dict(scope='Published CSV summaries only; no simulations or historical Gates rerun',
                  stage9h_blocks=6, stage8_effect_rows=len(rows), comparison='matched',
                  rtol=1e-11, atol=2e-14)
    (out / 'table_check.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result))

if __name__ == '__main__':
    main()
