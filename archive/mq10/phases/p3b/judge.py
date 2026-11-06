"""P3b ①② 기계 판정 (P3a 규칙 G, decisions.md :91–104) + H2 세 갈래 종합 (7-12절 6번).

  uv run python phases/p3b/judge.py      # → results/p3b_judge/

입력: 궤적 단위 결과(data/interim/p3b/units/{P1,P0}/, data/interim/p3b_cal/units/P1/), ③ results/p3b_bound/bound_summary.json,
재현 검증 results/p3b/validate.json, ② 보정 results/p3b_cal/calibrate.json. 해석 없이 규칙만 적용한다.

규칙 G 구현 (decisions.md :91–96):
  - 판정량 = draw 별 짝차이(같은 draw·같은 난수, 궤적 인덱스가 짝). 구간 = draw 간 백분위 구간과 평균 ± z·SD 중 넓은 쪽.
  - 주 효과 E 는 99%, 조건 효과 Δ_k 는 α = 0.05/(5K) (양쪽).
  - 분류(기대 방향 양수): '증가' = L > 0 이고 m ≥ δ; '감소' = U < 0; '배제' = U < δ; 나머지 '불확실'.
  - G: 지지 = E 증가 & 감소 Δ_k 없음 / 부분 기각 = E 증가 & 감소 Δ_k 있음, 또는 E 증가 아님 & 증가 Δ_k 있음 /
       기각 = E 배제(감소 포함) & 증가 Δ_k 없음 / 판정 불가 = E 불확실 & 증가 Δ_k 없음.
  - MC SE(판정마다 병기) = draw 별 궤적 짝차이 SD/sqrt(N) 의 draw 간 중앙값·90 백분위 (calib_n 과 같은 정의).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from swat import provenance as pv  # noqa: E402
from swat import simulate as S  # noqa: E402

_spec = importlib.util.spec_from_file_location("p3b_run", ROOT / "phases/p3b/run.py")
R = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R)

OUT = ROOT / "results/p3b_judge"
DELTA_P = 0.02          # δ = 2%p
DELTA_T_S = 0.005       # δ_t = 5 ms
ALPHA_E = 0.01
N_FAMILY = 5
SPEEDS = R.SPEEDS
SEED_BOOT = 20261108    # H4 감지 시점 중앙값 차의 MC SE (궤적 bootstrap)
B_BOOT = 50
OBS_1MPS = dict(resp_opaque=370 / 417, resp_transparent=378 / 431, cap_opaque=72 / 417, cap_transparent=110 / 431)


# ------------------------------------------------------------------ 규칙
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


def rule_g(e_cls, dk_cls):
    inc, dec = "증가" in dk_cls, "감소" in dk_cls
    if e_cls == "증가":
        return "부분 기각" if dec else "지지"
    if inc:
        return "부분 기각"
    if e_cls in ("배제", "감소"):
        return "기각"
    return "판정 불가"


def combine_h2(verdicts):
    """H2 최종(:101): 세 지수 모두 지지일 때만 지지, 모두 같으면 그 판정, 섞이면 부분 기각."""
    v = set(verdicts)
    return v.pop() if len(v) == 1 else "부분 기각"


# ------------------------------------------------------------------ 자료
class Units:
    def __init__(self, prior, root=None, conds=None):
        self.prior, self.root, self._c = prior, root, {}
        self.by_name = {c.name: c for c in (conds if conds is not None else R.conditions(prior))}
        self.n_draws = R.N_DRAWS

    def get(self, name):
        if name not in self._c:
            z = np.load(R.unit_path(self.prior, self.by_name[name], 4000, self.n_draws, self.root))
            res = dict(captured=z["captured"], t_contact_s=z["t_contact_s"].astype(float), t_det_s=z["t_det_s"].astype(float),
                       t_end_obs_s=np.inf)
            cap, det_ok, tdet = S.outcomes(res)
            self._c[name] = dict(cap=cap, det=det_ok, tdet=tdet.astype(np.float32), j=z["j"])
        return self._c[name]

    def drop_except(self, keep):
        for k in [k for k in self._c if k not in keep]:
            del self._c[k]

    def has(self, name):
        return name in self.by_name and R.unit_path(self.prior, self.by_name[name], 4000, self.n_draws, self.root).exists()


def pdiff(U, a, b, key):
    """draw 별 짝차이 p(a) − p(b) 와 그 MC SE (draw 별 궤적 짝차이 SD/√N)."""
    A, B = U.get(a), U.get(b)
    assert np.array_equal(A["j"], B["j"])
    x = A[key].astype(np.float32) - B[key].astype(np.float32)
    return x.mean(1), x.std(1, ddof=1) / np.sqrt(x.shape[1])


def tdiff(U, a, b, rng):
    """draw 별 감지 시점 중앙값 차 med t(a) − med t(b) 와 MC SE (궤적 짝 bootstrap B_BOOT)."""
    A, B = U.get(a)["tdet"], U.get(b)["tdet"]
    d = np.median(A, 1) - np.median(B, 1)
    N = A.shape[1]
    se = np.empty(len(d))
    for i in range(len(d)):
        idx = rng.integers(0, N, (B_BOOT, N))
        se[i] = (np.median(A[i][idx], 1) - np.median(B[i][idx], 1)).std(ddof=1)
    return d, se


def effect_row(name, d, se, alpha, delta, **kw):
    L, U_, m = interval(d, alpha)
    return dict(effect=name, **kw, alpha=alpha, L=L, U=U_, median=m, mean=float(np.mean(d)), cls=classify(L, U_, m, delta),
                mcse_med=float(np.median(se)), mcse_q90=float(np.quantile(se, 0.9)))


# ------------------------------------------------------------------ 가설별
def h2(U, name_of, n_exps, branch, prior):
    """H2: E = p(v_K) − p(v_1) (99%), Δ_k = p(v_k+1) − p(v_k) (α = 0.05/(5(K−1)))."""
    K = len(SPEEDS)
    a_k = 0.05 / (N_FAMILY * (K - 1))
    rows, speed_rows, verdicts = [], [], {}
    for n in n_exps:
        names = [name_of(v, n) for v in SPEEDS]
        if not all(U.has(x) for x in names):
            continue
        for v, x in zip(SPEEDS, names):
            c = U.get(x)["cap"]
            U.drop_except(names)
            p = c.mean(1)
            speed_rows.append(dict(branch=branch, prior=prior, n_exp=n, v_mps=v, p_cap_median=float(np.median(p)),
                                   p_cap_q05=float(np.quantile(p, 0.05)), p_cap_q95=float(np.quantile(p, 0.95)),
                                   mcse_med=float(np.median(np.sqrt(p * (1 - p) / c.shape[1]))), n_draws=len(p), n_traj=c.shape[1]))
        d, se = pdiff(U, names[-1], names[0], "cap")
        e = effect_row("E", d, se, ALPHA_E, DELTA_P, hyp="H2", branch=branch, prior=prior, n_exp=n, v_from=SPEEDS[0], v_to=SPEEDS[-1])
        dk = []
        for k in range(K - 1):
            d, se = pdiff(U, names[k + 1], names[k], "cap")
            dk.append(effect_row(f"Δ_{k + 1}", d, se, a_k, DELTA_P, hyp="H2", branch=branch, prior=prior, n_exp=n,
                                 v_from=SPEEDS[k], v_to=SPEEDS[k + 1]))
        verdicts[n] = rule_g(e["cls"], [r["cls"] for r in dk])
        for r in [e] + dk:
            r["verdict_n"] = verdicts[n]
        rows += [e] + dk
    final = combine_h2(list(verdicts.values())) if len(verdicts) == len(n_exps) else None
    return rows, speed_rows, verdicts, final


def h3(U, prior):
    K = len(SPEEDS)
    a_k = 0.05 / (N_FAMILY * K)
    ds, rows = [], []
    for v in SPEEDS:
        d, se = pdiff(U, f"v{v:g}_n1_clap_a", f"v{v:g}_n1_a", "cap")
        ds.append((d, se))
        rows.append(effect_row(f"Δ_v{v:g}", d, se, a_k, DELTA_P, hyp="H3", prior=prior, v_mps=v))
    dE = np.mean([d for d, _ in ds], 0)
    U1 = U.get(f"v{SPEEDS[0]:g}_n1_a")
    x = np.mean([U.get(f"v{v:g}_n1_clap_a")["cap"].astype(np.float32) - U.get(f"v{v:g}_n1_a")["cap"] for v in SPEEDS], 0)
    seE = x.std(1, ddof=1) / np.sqrt(U1["cap"].shape[1])
    e = effect_row("E", dE, seE, ALPHA_E, DELTA_P, hyp="H3", prior=prior, v_mps=np.nan)
    return [e] + rows, rule_g(e["cls"], [r["cls"] for r in rows])


def h4(U, prior, rng):
    cbs = [cb for cb in (0.25, 0.5, 0.75) if U.has(f"v1_n1_cb{cb:g}_a")]
    K = len(cbs) * len(SPEEDS)
    a_k = 0.05 / (N_FAMILY * K)
    d, se = tdiff(U, "v1_n1_cb0.5_a", "v1_n1_a", rng)
    e = effect_row("E", d, se, ALPHA_E, DELTA_T_S, hyp="H4", prior=prior, c_b=0.5, v_mps=1.0, unit="s")
    rows, rep = [], []
    for cb in cbs:
        for v in SPEEDS:
            a, b = f"v{v:g}_n1_cb{cb:g}_a", f"v{v:g}_n1_a"
            d, se = tdiff(U, a, b, rng)
            rows.append(effect_row(f"Δ_cb{cb:g}_v{v:g}", d, se, a_k, DELTA_T_S, hyp="H4", prior=prior, c_b=cb, v_mps=v, unit="s"))
            dp, sp = pdiff(U, a, b, "cap")                              # 포획 확률 차: 보고만
            rep.append(dict(hyp="H4", prior=prior, c_b=cb, v_mps=v, quantity="p_cap 밝은 − 어두운 (보고만)",
                            median=float(np.median(dp)), q05=float(np.quantile(dp, 0.05)), q95=float(np.quantile(dp, 0.95)),
                            mcse_med=float(np.median(sp))))
    return [e] + rows, rule_g(e["cls"], [r["cls"] for r in rows]), rep, K


def h5_d(U, prior, m=1.0, model="ustar"):
    """H5 감지 부분 D: E = p_det(φ0, 1 m/s) − p_det(φ0.4, 1 m/s), Δ_k = p_det(0, v) − p_det(φ, v)."""
    nm = lambda v, phi: f"v{v:g}_n1_phi{phi:g}_m{m:g}_{model}_a"  # noqa: E731
    phis = [p for p in R.PHI_H5 if all(U.has(nm(v, p)) for v in SPEEDS)]
    if not phis or 0.4 not in phis:
        return [], None, 0
    K = len(phis) * len(SPEEDS)
    a_k = 0.05 / (N_FAMILY * K)
    tag = dict(hyp="H5-D", prior=prior, att_mult=m, att_model=model)
    d, se = pdiff(U, "v1_n1_a", nm(1.0, 0.4), "det")
    e = effect_row("E", d, se, ALPHA_E, DELTA_P, phi=0.4, v_mps=1.0, **tag)
    rows = []
    for phi in phis:
        for v in SPEEDS:
            d, se = pdiff(U, f"v{v:g}_n1_a", nm(v, phi), "det")
            rows.append(effect_row(f"Δ_phi{phi:g}_v{v:g}", d, se, a_k, DELTA_P, phi=phi, v_mps=v, **tag))
    return [e] + rows, rule_g(e["cls"], [r["cls"] for r in rows]), K


def h6(U):
    """H6: 집합 a 에서 φ*_s = argmax p_cap, 집합 b 에서 G_a = p(φ*) − p(0), G_b = p(φ*) − p(0.9), 각 α = 0.01/2."""
    nm = lambda phi, st: f"v1_n1_a" if phi == 0 and st == "a" else (f"v1_n1_{st}" if phi == 0 else f"v1_n1_phi{phi:g}_m1_ustar_{st}")  # noqa: E731
    phis = list(R.PHI_H6)
    pa = np.stack([U.get(nm(p, "a"))["cap"].mean(1) for p in phis], 1)
    capb = [U.get(nm(p, "b"))["cap"] for p in phis]
    star = pa.argmax(1)
    D, N = capb[0].shape
    xa = np.stack([capb[star[i]][i].astype(np.float32) - capb[0][i] for i in range(D)])
    xb = np.stack([capb[star[i]][i].astype(np.float32) - capb[-1][i] for i in range(D)])
    ga = effect_row("G_a", xa.mean(1), xa.std(1, ddof=1) / np.sqrt(N), 0.005, DELTA_P, hyp="H6", prior="P1")
    gb = effect_row("G_b", xb.mean(1), xb.std(1, ddof=1) / np.sqrt(N), 0.005, DELTA_P, hyp="H6", prior="P1")
    phi_star_med = float(np.median(np.asarray(phis)[star]))
    if ga["cls"] == "증가" and gb["cls"] == "증가" and 0 < phi_star_med < 0.9:
        v = "지지"
    elif ga["cls"] in ("배제", "감소"):
        v = "기각"
    elif (ga["cls"] == "증가") != (gb["cls"] == "증가"):
        v = "부분 기각"
    else:
        v = "판정 불가"
    for r in (ga, gb):
        r["phi_star_median"] = phi_star_med
    return [ga, gb], v


# ------------------------------------------------------------------ 종합
BRANCH_ASSUMPTIONS = {
    "①": "감지 모델 P1 사후 그대로(1 m/s 재현 2/4 실패), 수동 계수 k draw 별 U(0, 8.2), P2e 회피 분포",
    "②": "감지 문턱 위치 Δ̂ −0.87(1 m/s 반응률에 보정), k 8.2 고정(보정 미달), 나머지 ①과 같음",
    "③": "감지 = 관측 반응 시점 기류 하위 5% 임계(시각 경로 없음), τ_a 하위 5%, 회피 A_95·반지름 바깥 방향·40 ms, k 8.2",
}
DIR_OF = {"지지": "지지", "지지 방향": "지지", "기각": "반박", "반박 방향": "반박"}


def synthesize(v1, v2, v3):
    """7-12절 6번: ①·② 판정과 ③ 방향이 모두 같으면 강건, ①·② 판정 불가 & ③ 방향 있음 → 경계 수준, 그 밖 판정 불가 + 갈리는 가정."""
    d1, d2, d3 = DIR_OF.get(v1), DIR_OF.get(v2), DIR_OF.get(v3)
    if d1 is not None and d1 == d2 == d3:
        return f"[모델 예측] 수준에서 가정에 대해 강건하게 {d1}", "①·② 판정과 ③ 방향이 모두 같음", []
    if v1 == "판정 불가" and v2 == "판정 불가" and d3 is not None:
        return "경계 계산 수준의 결론만(판정 불가)", "①·② 판정 불가, ③만 방향 있음", []
    split = [dict(branch=b, verdict=v, h2_direction=DIR_OF.get(v, "방향 없음(" + v + ")"), assumption=BRANCH_ASSUMPTIONS[b])
             for b, v in (("①", v1), ("②", v2), ("③", v3))]
    return "판정 불가", f"갈래끼리 엇갈림 (① {v1} · ② {v2} · ③ {v3})", split


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    prov = pv.start_meta()
    rng = np.random.default_rng(SEED_BOOT)
    OUT.mkdir(parents=True, exist_ok=True)
    P1, P0 = Units("P1"), Units("P0")
    cal = json.loads((R.OUT_CAL / "calibrate.json").read_text(encoding="utf-8"))
    C2 = Units("P1", R.INTERIM_CAL, R.conditions_cal(cal))
    sfx = f"_k{cal['k_per_s']:g}_a_d{cal['shift']:+g}_cal"

    h2_rows, sp_rows, summ = [], [], {}
    for branch, U, prior, nm, ns in (
        ("①", P1, "P1", lambda v, n: f"v{v:g}_n{n:g}_a", (0.8, 1.0, 1.2)),
        ("②", C2, "P1", lambda v, n: f"v{v:g}_n{n:g}{sfx}", (0.8, 1.0, 1.2)),
        ("① P0 민감도", P0, "P0", lambda v, n: f"v{v:g}_n{n:g}_a", (1.0,)),
        ("① k=0 민감도(보고만)", P1, "P1", lambda v, n: f"v{v:g}_n{n:g}_k0_a", (1.0,)),
        ("① k=8.2 민감도(보고만)", P1, "P1", lambda v, n: f"v{v:g}_n{n:g}_k8.2_a", (1.0,)),
    ):
        r, s, vn, final = h2(U, nm, ns, branch, prior)
        h2_rows += r
        sp_rows += s
        U._c.clear()
        summ[branch] = dict(by_n={str(k): v for k, v in vn.items()}, final=final)
    pd.DataFrame(h2_rows).to_csv(OUT / "h2_rule_g.csv", index=False)
    sp = pd.DataFrame(sp_rows)
    sp.to_csv(OUT / "h2_speed_table.csv", index=False)
    # P1 − P0 (n = 1) 포획 확률 중앙값 차: 보고만
    a = sp[(sp.branch == "①") & (sp.n_exp == 1.0)].set_index("v_mps").p_cap_median
    b = sp[sp.branch == "① P0 민감도"].set_index("v_mps").p_cap_median
    p1p0 = (a - b).rename("p_cap_median_P1_minus_P0").reset_index()

    # 부록: H3, H4, H5-D, H6 (P1 판정, P0 민감도)
    app, app_v, rep = [], [], []
    for prior, U in (("P1", P1), ("P0", P0)):
        r, v = h3(U, prior)
        app += r
        app_v.append(dict(hyp="H3", prior=prior, verdict=v, K=len(SPEEDS), note="[모델 예측] 박수 결합 가정"))
        U._c.clear()
        r, v, rp, K = h4(U, prior, rng)
        app += r
        rep += rp
        U._c.clear()
        app_v.append(dict(hyp="H4", prior=prior, verdict=v, K=K, note="감지 시점 차(초), δ_t 5 ms" + ("" if prior == "P1" else ", P0 은 c_b 0.5 만")))
        for m, model in ((1.0, "ustar"), (0.5, "ustar"), (2.0, "ustar"), (1.0, "drag")):
            r, v, K = h5_d(U, prior, m, model)
            if not r:
                continue
            app += r
            U._c.clear()
            app_v.append(dict(hyp="H5-D", prior=prior, verdict=v, K=K, att_mult=m, att_model=model,
                              note=("판정 조건" if (m, model) == (1.0, "ustar") else "보고만") + "; H5 최종은 bow wave 부분 B(P5a) 필요 — B 미산출이라 최종 판정 없음"
                              + ("" if prior == "P1" else ", P0 은 φ 0.4 만")))
    r, v = h6(P1)
    app += r
    app_v.append(dict(hyp="H6", prior="P1", verdict=v, K=2, note="v = 1 m/s, φ > 0.4 [모델 예측], 타격 면적 = P3b 통과 확률 φ 가정"))
    app_v.append(dict(hyp="H1", prior="-", verdict="P3a 에 정의 없음", note="P3a 규칙은 H2–H6 만 정의(decisions.md :91–104)"))
    app_v.append(dict(hyp="기류 우세", prior="P1", verdict="보류(확정)", note="P2d-R G2 결정 2번(decisions.md :244): SD 비 0.506 ≥ 0.5"))
    pd.DataFrame(app).to_csv(OUT / "appendix_rule_g.csv", index=False)
    pd.DataFrame(app_v).to_csv(OUT / "appendix_verdicts.csv", index=False)
    pd.DataFrame(rep).to_csv(OUT / "appendix_h4_pcap_report.csv", index=False)

    # 종합
    bnd = json.loads((ROOT / "results/p3b_bound/bound_summary.json").read_text(encoding="utf-8"))
    val = json.loads((R.OUT / "validate.json").read_text(encoding="utf-8"))
    br = pd.read_csv(ROOT / "results/p3b_bound/bound_rates.csv")
    v1, v2, v3 = summ["①"]["final"], summ["②"]["final"], bnd["verdict"]
    sentence, basis, split = synthesize(v1, v2, v3)
    chk = lambda checks, disk, met: next(c for c in checks if c["disk"] == disk and c["metric"] == met)  # noqa: E731
    one = []
    for disk in ("opaque", "transparent"):
        for met in ("respond", "capture"):
            key = f"{'resp' if met == 'respond' else 'cap'}_{disk}"
            c1, c2 = chk(val["checks"], disk, met), chk(cal["checks"], disk, met)
            one.append(dict(quantity=f"1 m/s 원판 {disk} {'반응(회피)률' if met == 'respond' else '포획률'}", observed=OBS_1MPS[key],
                            obs_ci95=c1["obs_ci95"], branch1=c1["sim_median"], branch1_pi90=c1["sim_pi90"], branch1_overlap=c1["overlap"],
                            branch2=c2["sim_median"], branch2_pi90=c2["sim_pi90"], branch2_overlap=c2["overlap"], branch3=None))
    m3 = br[(br.main) & (np.isclose(br.v_mps, 1.0))]
    one.append(dict(quantity="1 m/s 경계: 회피 가능 비율(③, 불투명 기준 없음, 원판 공통 초기 상태)", observed=1 - OBS_1MPS["cap_opaque"],
                    obs_note="1 − 불투명 포획률", branch3={str(r.n_exp): r.avoid for r in m3.itertuples()}))
    for branch in ("①", "②"):
        s1 = sp[(sp.branch == branch) & (sp.n_exp == 1.0) & (sp.v_mps == 1.0)].iloc[0]
        one.append(dict(quantity=f"H2 조건 1 m/s 손(불투명, n = 1) 포획 확률 — 갈래 {branch}", observed=None,
                        **{("branch1" if branch == "①" else "branch2"): s1.p_cap_median}, mcse=s1.mcse_med))
    out = dict(
        rule="handover 7-12절 6번 / decisions.md 'H2 세 갈래 판정' 6번 (결과 전 고정)",
        branches={"①": dict(verdict=v1, by_n=summ["①"]["by_n"], tag=R.VALIDATION_TAG,
                            tag_mismatch="불투명 포획률 0.311 대 관측 0.173(과대), 투명 반응률 0.781 대 0.877(과소)"),
                  "②": dict(verdict=v2, by_n=summ["②"]["by_n"], tag=R.CAL_TAG, tag_note=R.CAL_NOTE,
                            calibration=dict(shift=cal["shift"], k_per_s=cal["k_per_s"], k_note=cal["k_note"], n_overlap=cal["n_overlap"])),
                  "③": dict(verdict=v3, direction_main=bnd["direction_main"], v_star_main=bnd["v_star_main"], tag=bnd["label"])},
        sensitivity={k: v for k, v in summ.items() if k not in ("①", "②")},
        synthesis=dict(sentence=sentence, basis=basis, split_table=split),
        one_mps_table=one, p1_minus_p0=p1p0.to_dict("records"), appendix_verdicts=app_v,
        delta_p=DELTA_P, delta_t_s=DELTA_T_S, alpha_E=ALPHA_E, alpha_dk_h2=0.05 / (N_FAMILY * (len(SPEEDS) - 1)),
        n_traj=4000, n_draws=R.N_DRAWS, seed_boot=SEED_BOOT, b_boot=B_BOOT,
        mcse_def="판정량 MC SE = draw 별 궤적 짝차이 SD/√N 의 draw 간 중앙값(mcse_med)·90 백분위(mcse_q90); 속도별 표는 draw 별 이항 SE 중앙값",
        summarize_commit=pv.git_head(), **prov)
    R.save_json(OUT / "synthesis.json", out)
    print(f"[judge] ① {v1} {summ['①']['by_n']} | ② {v2} {summ['②']['by_n']} | ③ {v3} → {sentence} ({basis})")
    print(f"[judge] 민감도 {out['sensitivity']}")
    for r in app_v:
        print(f"[judge] 부록 {r['hyp']} {r['prior']} {r.get('att_mult', '')}{r.get('att_model', '')}: {r['verdict']}")


if __name__ == "__main__":
    main()
