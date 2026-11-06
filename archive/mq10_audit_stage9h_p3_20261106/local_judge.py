"""Exact AST-selected original judge functions; no production imports."""
import numpy as np
from scipy.stats import norm

def interval(d, alpha):
    """draw 별 차이 d → (L, U, 중앙값). 백분위 구간과 정규 근사 중 넓은 쪽."""
    d = np.asarray(d, float)
    lp, up = np.quantile(d, [alpha / 2, 1 - alpha / 2])
    z = norm.ppf(1 - alpha / 2)
    mu, sd = d.mean(), d.std(ddof=1)
    return float(min(lp, mu - z * sd)), float(max(up, mu + z * sd)), float(np.median(d))

def classify(L, U, m, delta):
    if L > 0 and m >= delta:
        return "증가"
    if U < 0:
        return "감소"
    if U < delta:
        return "배제"
    return "불확실"

def effect_row(name, d, se, alpha, delta, **kw):
    L, U_, m = interval(d, alpha)
    return dict(effect=name, **kw, alpha=alpha, L=L, U=U_, median=m, mean=float(np.mean(d)), cls=classify(L, U_, m, delta),
                mcse_med=float(np.median(se)), mcse_q90=float(np.quantile(se, 0.9)))
