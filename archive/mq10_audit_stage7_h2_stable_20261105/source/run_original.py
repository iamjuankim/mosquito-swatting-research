"""P3b 시뮬레이터 실행 — 준비·재현 검증·궤적 수 보정·판정 실험(조건 × draw)·요약.

설계: logs/decisions.md 'P3b 시뮬레이터 설계' (결과 전 고정), P3a 규칙('사전 고정 기준' 절), P2d-R G2 결정 6번.

  uv run python phases/p3b/run.py prepare            # 초기 상태 표·기류 격자·사후 draw 추출 (캐시)
  uv run python phases/p3b/run.py validate           # 재현 검증 (1 m/s, Exp2 848 trial × draw)
  uv run python phases/p3b/run.py calib-n            # 궤적 수 N 보정 (H2 n = 1, 20 draw)
  uv run python phases/p3b/run.py run --n N          # 판정 실험: 조건 × draw (조건 단위 저장, 이어서 실행)
  uv run python phases/p3b/run.py run --order priority --n N --max-hours 7
      # ① P1 판정 → ② P0 판정 → ③ P1 보고만 → ④ P0 보고만 을 한 명령에서, 경과 7 h 를 넘으면 새 조건을 시작하지 않고
      # 저장 후 종료 (decisions.md 'P3b 판정 실험 실행 결정'). 같은 명령 재실행 = 완료 조건 건너뛰고 이어서.
  uv run python phases/p3b/run.py summarize          # 조건 × draw 예측량 표 → results/p3b/
  uv run python phases/p3b/run.py run --fix-rho-av 0 --speeds 1,6,8,12 --n 4000   # D2 진단 (decisions.md 'D2 진단'): results/p3b_d2/rho<값>/
  uv run python phases/p3b/run.py summarize-d2 --n 4000                           # D2 표·판단 문구 기계 적용 → results/p3b_d2/
옵션: --draws K (앞 K 개 draw 만), --smoke (축소 격자). 환경변수 MQ10_P3B_DIR 를 주면 중간 결과와 결과를
그 폴더(<DIR>, <DIR>/results)에 쓴다(소규모 시험용, results/ 를 건드리지 않음).
단일 프로세스·순차. 시작할 때 단위 시간·전체 예상 시간을 로그(logs/runs/p3b_<날짜>.log)에 쓴다.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from swat import detection as det  # noqa: E402
from swat import escape as E  # noqa: E402
from swat import flow  # noqa: E402
from swat import provenance as pv  # noqa: E402
from swat import simulate as S  # noqa: E402

SMOKE_DIR = os.environ.get("MQ10_P3B_DIR")
INTERIM = (ROOT / SMOKE_DIR) if SMOKE_DIR else ROOT / "data/interim/p3b"
OUT = (INTERIM / "results") if SMOKE_DIR else ROOT / "results/p3b"
LOG = ROOT / f"logs/runs/p3b{'_smoke' if SMOKE_DIR else ''}_{datetime.now():%Y%m%d}.log"
P0_POST = ROOT / "results/p2d_r/mcmc_P0_d8000/mcmc_exp2.json"
PREP_EXP2 = ROOT / "data/interim/p2d_r/prep_exp2_500.pkl"
P2E_BOOT = ROOT / "results/p2e/p2e_bootstrap.csv"

SEED_DRAW = {"P1": 20261101, "P0": 20261102}
SEED_K, SEED_ESC, SEED_MC, SEED_VAL = 20261103, 20261104, 20261105, 20261106
N_DRAWS = 200
SPEEDS = (0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0)
PHI_H5 = (0.1, 0.2, 0.3, 0.4)
PHI_H6 = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
MC_SET_ID = {"a": 0, "b": 1}
# 재현 검증 실패 표시 (decisions.md 'P3b G3 결정' 2번): 판정 규칙 G 계산은 바꾸지 않는 표시 규칙
VALIDATION = "failed (2/4, 03db936)"
VALIDATION_TAG = "[모델 예측, 1 m/s 원판 재현 검증 실패(2/4)]"
OBS = {"opaque": dict(n=417, collided=72, responded=370), "transparent": dict(n=431, collided=110, responded=378)}
# ② 1 m/s 보정 모형 (decisions.md 'H2 세 갈래 판정' 4번과 9-14 격자 기록): 결과 폴더를 원 모형과 분리
INTERIM_CAL = (INTERIM / "cal") if SMOKE_DIR else ROOT / "data/interim/p3b_cal"
OUT_CAL = (OUT / "cal") if SMOKE_DIR else ROOT / "results/p3b_cal"
CAL_TAG = "[모델 예측, 1 m/s 보정 모형 — 재현 지표 3개를 보정에 사용(계획은 2개였으나 실제로 3개)]"
CAL_NOTE = "실제로 보정에 쓴 지표는 3개(두 원판 반응률 + 불투명 포획률), 준독립은 투명 포획률 1개 (decisions.md 9-14 불일치 공개)"
CAL_SHIFTS = np.round(np.arange(-2.0, 1.0 + 1e-9, 0.01), 2)     # Δ 격자 (문턱 로그 위치)
CAL_KS = np.round(np.arange(0.0, S.K_MAX_PER_S + 1e-9, 0.2), 1)   # k 격자 0–8.2 s⁻¹
CAL_N_EXP = (0.8, 1.0, 1.2)
# D2 진단 (decisions.md 'D2 진단 — ρ_av 고정 민감도', 핸드오버 7-13절 5번): 궤적 단위는 interim, 요약만 results
INTERIM_D2 = (INTERIM / "d2") if SMOKE_DIR else ROOT / "data/interim/p3b_d2"
OUT_D2 = (OUT / "d2") if SMOKE_DIR else ROOT / "results/p3b_d2"
D2_FIXED = (0.0, 0.15, 1.48)
D2_SPEEDS = (1.0, 6.0, 8.0, 12.0)
D2_SEGMENTS = ((6.0, 8.0), (8.0, 12.0))
D2_TAG = "탐색적 진단"
D2_LIMIT = "ρ_av 만 고정하면 사후의 결합 구조가 깨진다(다른 파라미터는 원 ρ_av 와 함께 추정된 값). 조건부 민감도이지 대안 모형이 아니다."


def log(msg):
    line = f"{datetime.now():%H:%M:%S} {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def save_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    os.replace(tmp, path)


def kin_profile():
    k = pd.read_csv(ROOT / "data/interim/2024_bowwave/cfd/Data S3 - CFD/_CFD/full/piston_kinematic_corrected.csv")
    return k.time_s.to_numpy(float), np.maximum.accumulate(np.maximum(k.y_pos.to_numpy(float), 0.0))


# ------------------------------------------------------------------ 준비
def init_states():
    """Exp2 848 trial 관찰 시작 시점의 모기 상태 (원판 A 시작 중심 기준) + 관찰 창·세션·원판·관측 결과."""
    f = INTERIM / "init_states.pkl"
    if f.exists():
        return pd.read_pickle(f)
    import pyarrow.parquet as pq

    with open(PREP_EXP2, "rb") as fh:
        prep = pickle.load(fh)
    trials, ons = prep["trials"], prep["onsets"]
    ids = [t.trial_id for t in trials]
    tab = pq.read_table(ROOT / "data/processed/p1b_trials.parquet",
                        columns=["trial_id", "t_s", "x_m", "y_m", "z_m", "swatter_y_m", "swatter_x_m", "swatter_z_m", "fps"],
                        filters=[("dataset", "==", "2024_exp2")]).to_pandas().set_index("trial_id").loc[ids]
    tk, sk = kin_profile()
    s_max, t_peak = float(sk.max()), float(tk[np.argmax(sk)])
    sessions = sorted({t.session for t in trials})
    rows = []
    for tr in trials:
        r = tab.loc[tr.trial_id]
        t = np.asarray(r.t_s, float)
        X = np.c_[np.asarray(r.x_m, float), np.asarray(r.y_m, float), np.asarray(r.z_m, float)]
        k = int(round(float(r.fps) / 500.0))
        n = len(t) // k
        with np.errstate(invalid="ignore"):
            X = np.nanmean(X[: n * k].reshape(n, k, 3), axis=1)
            t = t[: n * k].reshape(n, k).mean(axis=1)
        y0 = float(np.nanmin(np.asarray(r.swatter_y_m, float))) + s_max
        c0 = np.array([float(r.swatter_x_m), y0 + det.HALF_THICK_M, float(r.swatter_z_m)])
        ts, te = float(tr.t[tr.i_start]), float(tr.t[tr.i_end])
        ok = np.isfinite(X).all(axis=1)
        win = ok & (t >= ts - 0.020 - 1e-9) & (t <= ts + 1e-9)
        for lo, hi in ((-0.020, 0.0), (0.0, 0.020), (-0.020, 0.020), (0.0, 0.040)):
            if win.sum() >= 3:
                break
            win = ok & (t >= ts + lo - 1e-9) & (t <= ts + hi + 1e-9)
        if win.sum() < 3 or not np.isfinite(c0).all():          # 상태 추정 불가 → 제외 (개수 보고)
            continue
        tt, XX = t[win], X[win]
        A = np.c_[np.ones(len(tt)), tt - ts]
        coef, *_ = np.linalg.lstsq(A, XX, rcond=None)
        rows.append(dict(trial_id=tr.trial_id, session=tr.session, session_idx=sessions.index(tr.session),
                         transparent=tr.visual == "transparent", rel0_m=coef[0] - c0, v0_mps=coef[1],
                         s_start_m=float(np.interp(ts + t_peak, tk, sk)), t_start_kin_s=ts + t_peak,
                         t_end_kin_s=te + t_peak, collided=bool(tr.collided), responded=bool(ons[tr.trial_id].p_resp >= 0.5),
                         n_fit=int(win.sum())))
    df = pd.DataFrame(rows)
    df.attrs["n_excluded"] = len(trials) - len(df)
    INTERIM.mkdir(parents=True, exist_ok=True)
    df.to_pickle(f)
    return df


def flow_grid():
    f = INTERIM.parent / "p3b_flow_grid.npz" if SMOKE_DIR else INTERIM / "flow_grid.npz"
    if not f.exists():
        f2 = ROOT / "data/interim/p3b/flow_grid.npz"
        if f2.exists():
            return S.FlowGrid.load(f2)
        t0 = time.time()
        g = S.FlowGrid.build(flow.load_model())
        f.parent.mkdir(parents=True, exist_ok=True)
        g.save(f)
        log(f"[prepare] 기류 격자 {g.u_xi.shape} {time.time() - t0:.0f}s → {f}")
        return g
    return S.FlowGrid.load(f)


def grid_error(g, n=20000, seed=0):
    """격자 보간 대 대리모델 직접 예측 (무작위 점, 원판 앞쪽 0.1 < ξ ≤ 4, ρ* ≤ 2) — 상대 RMSE."""
    fm = flow.load_model()
    rng = np.random.default_rng(seed)
    xi, rho, s = rng.uniform(0.1, 4.0, n), rng.uniform(0, 2.0, n), rng.uniform(0, fm.s_star_max, n)
    ux, ur, _ = fm.predict(xi, rho, s)
    gx, gr = g._fx(np.c_[xi, rho, s]), g._fr(np.c_[xi, rho, s])
    return float(np.sqrt(np.mean((gx - ux) ** 2 + (gr - ur) ** 2) / np.mean(ux**2 + ur**2)))


def posterior():
    import xarray as xr

    m = json.loads(P0_POST.read_text(encoding="utf-8"))
    return xr.open_datatree(ROOT / m["posterior_file"], engine="zarr")["posterior"].to_dataset(), m


def draws(prior):
    """P1: PSIS 가중치 비례 비복원 200, P0: 균등 비복원 200 (P2d-R G2 결정 6번). results/p3b/draws_<P>.csv."""
    f = OUT / f"draws_{prior}.csv"
    if f.exists():
        return pd.read_csv(f)
    post, m = posterior()
    C, D = post.sizes["chain"], post.sizes["draw"]
    rng = np.random.default_rng(SEED_DRAW[prior])
    k_hat = None
    if prior == "P1":
        lw, k_hat = det.prior_reweight(np.log(np.asarray(post["rho_vs"]).ravel()), np.log(np.asarray(post["rho_av"]).ravel()),
                                       "P0", "P1")
        w = np.exp(lw - lw.max())
        idx = rng.choice(C * D, size=N_DRAWS, replace=False, p=w / w.sum())
    else:
        idx = rng.choice(C * D, size=N_DRAWS, replace=False)
    df = pd.DataFrame(dict(j=np.arange(N_DRAWS), flat=idx, chain=idx // D, draw=idx % D))
    df.attrs["k_hat"] = k_hat
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(f, index=False)
    save_json(OUT / f"draws_{prior}_meta.json", dict(prior=prior, seed=SEED_DRAW[prior], n=N_DRAWS, k_hat=k_hat,
              source=str(P0_POST.relative_to(ROOT)), posterior_file=m["posterior_file"], mcmc_seed=m["mcmc_seed"],
              rule="decisions.md 'P2d-R G2 결정' 6번", **pv.start_meta()))
    return df


def draw_params(post, row, n_sessions):
    return det.params_from_draw(post, int(row.chain), int(row.draw), 1, n_sessions)


def draw_k(j, cond):
    return cond.k_fixed if cond.k_fixed is not None else float(np.random.default_rng([SEED_K, j]).uniform(0, S.K_MAX_PER_S))


def draw_escape(boot, j):
    b = int(np.random.default_rng([SEED_ESC, j]).integers(0, boot.b.nunique()))
    out = {}
    for disk in ("opaque", "transparent"):
        r = boot[(boot.b == b) & (boot.disk == disk)].iloc[0]
        out[disk] = E.EscapeParams(k_per_s=float(r.k), w=float(r.w), kappa=float(r.kappa), log_a_mu=float(r.log_a_mu),
                                   log_a_sd=float(r.log_a_sd))
    return out, b


class Ctx:
    def __init__(self):
        self.init = init_states()
        self.grid = flow_grid()
        self.kin = kin_profile()
        fm = flow.load_model()
        self.U = fm.U_ref_cfd_mps
        self.A_ref = det.reference_airflow(fm)
        self.post, _ = posterior()
        self.n_sess = int(self.init.session_idx.max()) + 1
        self.boot = pd.read_csv(P2E_BOOT)


def mc_batch(ctx, j, n, mc_set, sigma_sess):
    """통념 실험 궤적 n 개: 초기 상태 복원추출 + 공통 난수 (조건과 무관; seed, draw, 집합으로만 결정)."""
    rng = np.random.default_rng([SEED_MC, j, MC_SET_ID[mc_set]])
    pick = rng.integers(0, len(ctx.init), n)
    u, z_a = S.common_randoms(rng, n)
    z = rng.normal(size=n)
    I = ctx.init.iloc[pick]
    return S.Batch(rel0_m=np.stack(I.rel0_m.to_numpy()), v0_mps=np.stack(I.v0_mps.to_numpy()),
                   s_start_m=I.s_start_m.to_numpy(float), transparent=np.zeros(n, bool), eta=sigma_sess * z, u=u, z_a=z_a)


# ------------------------------------------------------------------ 조건
def label_of(c: S.Condition, U: float) -> str:
    """판정 줄 꼬리표: 재현 검증 실패 꼬리표는 모든 줄, 1 m/s 외 등 [모델 예측] 은 조건에 따라 앞에."""
    return ("[모델 예측] " if c.model_prediction(U) else "") + VALIDATION_TAG


def cname(c: S.Condition):
    s = f"v{c.v_mps:g}_n{c.n_exp:g}"
    s += "_clap" if c.clap else ""
    s += f"_cb{c.c_b:g}" if c.c_b != 1 else ""
    s += f"_phi{c.phi:g}_m{c.att_mult:g}_{c.att_model}" if c.phi > 0 else ""
    s += f"_k{c.k_fixed:g}" if c.k_fixed is not None else ""
    return s + f"_{c.mc_set}"


def conditions(prior="P1", smoke=False):
    sp = (1.0, 12.0) if smoke else SPEEDS
    C = []
    add = lambda **kw: C.append(S.Condition(name="", **kw))  # noqa: E731
    for v in sp:
        add(v_mps=v)                                                   # 한 손 기준 (H2 n = 1 등 공통)
        add(v_mps=v, clap=True)                                        # H3
        add(v_mps=v, c_b=0.5)                                          # H4 주
        add(v_mps=v, phi=0.4)                                          # H5 주
        if prior == "P1":
            for n in (0.8, 1.2):
                add(v_mps=v, n_exp=n)                                  # H2 지수
            for cb in ((0.25, 0.75) if not smoke else ()):
                add(v_mps=v, c_b=cb)                                   # H4 c_b 격자
            for phi in (PHI_H5 if not smoke else ()):
                for m in (0.5, 1.0, 2.0):
                    add(v_mps=v, phi=phi, att_mult=m)                  # H5 (m = 1 판정, 0.5·2 보고)
                add(v_mps=v, phi=phi, att_model="drag")               # H5 항력 비 시나리오 (보고)
            for kf in (0.0, S.K_MAX_PER_S):
                add(v_mps=v, k_fixed=kf)                               # k 민감도 (보고)
    for st in ("a", "b") if prior == "P1" else ("a",):
        for phi in ((0.0, 0.4, 0.9) if smoke else PHI_H6):
            add(v_mps=1.0, phi=phi, mc_set=st)                         # H6 (집합 a 선택, b 대비)
    out, seen = [], set()
    for c in C:
        nm = cname(c)
        if nm not in seen:
            seen.add(nm)
            out.append(S.Condition(**{**c.__dict__, "name": nm}))
    return out


def role_of(c: S.Condition) -> str:
    """P3a 규칙(decisions.md :91–104)의 역할 표시: 'judge'(판정 조건) | 'report'(보고만).

    보고만 = k 고정(0·8.2) 민감도, H5 감쇠 배수 m ≠ 1 과 항력 비 시나리오('H5 감지 부분에서 보고만'), H6 φ > 0.4 의
    v ≠ 1 m/s('다른 속도 격자 결과는 보고만', 현재 격자에 없음). 나머지(한 손 n ∈ {0.8, 1, 1.2}, 박수, c_b 격자,
    H5 φ ≤ 0.4 m = 1, H6 φ 격자 집합 a·b)는 판정 조건. P0 은 판정 없이 P1 과 차이만 보고하지만 순서 구분은 같은
    역할 표시를 쓴다(decisions.md 'P3b 판정 실험 실행 결정' 2번).
    """
    if c.k_fixed is not None:
        return "report"
    if c.phi > 0 and (c.att_mult != 1.0 or c.att_model != "ustar"):
        return "report"
    if c.phi > 0.4 and abs(c.v_mps - 1.0) > 1e-9:
        return "report"
    return "judge"


ORDER_GROUPS = (("1", "P1", "judge"), ("2", "P0", "judge"), ("3", "P1", "report"), ("4", "P0", "report"))


def plan_units(order="prior", prior="P1", smoke=False):
    """실행 순서 [(묶음, prior, 조건)]. 'priority' = ① P1 판정 → ② P0 판정 → ③ P1 보고만 → ④ P0 보고만."""
    if order != "priority":
        return [("-", prior, c) for c in conditions(prior, smoke)]
    return [(g, p, c) for g, p, r in ORDER_GROUPS for c in conditions(p, smoke) if role_of(c) == r]


def run_plan(plan, is_done, run_one, max_s=None, clock=time.time):
    """plan 을 차례로 실행. 완료 단위는 건너뛰고, 경과가 max_s 를 넘으면 새 단위를 시작하지 않고 멈춘다.

    반환: (이번에 실행한 단위 수, 건너뛴 단위 수, 남은 단위 목록)."""
    t0, ran, skipped = clock(), 0, 0
    for i, u in enumerate(plan):
        if is_done(u):
            skipped += 1
            continue
        if max_s is not None and clock() - t0 > max_s:
            return ran, skipped, [v for v in plan[i:] if not is_done(v)]
        run_one(i, u)
        ran += 1
    return ran, skipped, []


def steps_of(ctx, c):
    tk, sk = ctx.kin
    f = c.v_mps / ctx.U
    dt = min(S.DT_MAX_S, S.DX_MAX_M / c.v_mps)
    return (tk[np.argmax(sk)] - ctx.init.t_start_kin_s.min()) / f / dt


# ------------------------------------------------------------------ 재현 검증
def cmd_validate(n_draws):
    from scipy.stats import beta

    ctx = Ctx()
    dr = draws("P1").iloc[:n_draws]
    I = ctx.init
    cond = S.Condition(name="validate_v1.008", v_mps=ctx.U)
    t0 = time.time()
    rec = []
    for _, row in dr.iterrows():
        P = draw_params(ctx.post, row, ctx.n_sess)
        esc, b = draw_escape(ctx.boot, int(row.j))
        k = draw_k(int(row.j), cond)
        rng = np.random.default_rng([SEED_VAL, int(row.j)])
        u, z_a = S.common_randoms(rng, len(I))
        batch = S.Batch(rel0_m=np.stack(I.rel0_m.to_numpy()), v0_mps=np.stack(I.v0_mps.to_numpy()),
                        s_start_m=I.s_start_m.to_numpy(float), transparent=I.transparent.to_numpy(bool),
                        eta=P["eta_sess"][I.session_idx.to_numpy()], u=u, z_a=z_a, t_end_kin_s=I.t_end_kin_s.to_numpy(float))
        res = S.simulate(batch, cond, P, esc, k, ctx.kin, ctx.grid, ctx.U, ctx.A_ref)
        cap, det_ok, _ = S.outcomes(res, window=True)
        for disk, m in (("opaque", ~I.transparent.to_numpy(bool)), ("transparent", I.transparent.to_numpy(bool))):
            rec.append(dict(j=int(row.j), disk=disk, capture=float(cap[m].mean()), respond=float(det_ok[m].mean()), k=k, boot=b))
        if len(rec) == 2:
            log(f"[validate] draw 1개 {time.time() - t0:.1f}s → {len(dr)} draw 예상 {(time.time() - t0) * len(dr) / 60:.1f} 분")
    df = pd.DataFrame(rec)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "validate_draws.csv", index=False)
    checks, allok = [], True
    for disk, o in OBS.items():
        d = df[df.disk == disk]
        for key, obs_k, col in (("capture", o["collided"], "capture"), ("respond", o["responded"], "respond")):
            lo = beta.ppf(0.025, obs_k, o["n"] - obs_k + 1) if obs_k > 0 else 0.0
            hi = beta.ppf(0.975, obs_k + 1, o["n"] - obs_k) if obs_k < o["n"] else 1.0
            s5, s50, s95 = np.quantile(d[col], [0.05, 0.5, 0.95])
            ok = bool(max(lo, s5) <= min(hi, s95))
            allok &= ok
            checks.append(dict(disk=disk, metric=key, obs=obs_k / o["n"], obs_ci95=[lo, hi], sim_median=s50,
                               sim_pi90=[s5, s95], overlap=ok))
    out = dict(verdict="통과" if allok else "실패", rule="decisions.md 'P3b 시뮬레이터 설계' 재현 검증 기준 (4개 모두 겹침)",
               n_draws=len(dr), checks=checks, seconds=round(time.time() - t0, 1), seed_val=SEED_VAL, seed_k=SEED_K,
               seed_esc=SEED_ESC, prior="P1", run_commit=pv.git_head(), **pv.start_meta())
    save_json(OUT / "validate.json", out)
    for c in checks:
        log(f"[validate] {c['disk']:11s} {c['metric']:8s} 관측 {c['obs']:.3f} [{c['obs_ci95'][0]:.3f}, {c['obs_ci95'][1]:.3f}] "
            f"| 시뮬 {c['sim_median']:.3f} [{c['sim_pi90'][0]:.3f}, {c['sim_pi90'][1]:.3f}] → {'겹침' if c['overlap'] else '안 겹침'}")
    log(f"[validate] 판정: {out['verdict']} ({out['seconds']}s)")
    return out


# ------------------------------------------------------------------ 조건 실행
def unit_path(prior, c, n, n_draws, root=None):
    return (root or INTERIM) / "units" / prior / f"{c.name}_N{n}_D{n_draws}.npz"


def run_unit(ctx, prior, dr, c, n, root=None, fix_rho_av=None):
    f = unit_path(prior, c, n, len(dr), root)
    if f.exists():
        return f, 0.0
    t0 = time.time()
    cap = np.zeros((len(dr), n), bool)
    tc = np.zeros((len(dr), n), np.float32)
    td = np.zeros((len(dr), n), np.float32)
    for a, (_, row) in enumerate(dr.iterrows()):
        P = draw_params(ctx.post, row, ctx.n_sess)
        if fix_rho_av is not None:
            P["rho_av"] = float(fix_rho_av)
        esc, _ = draw_escape(ctx.boot, int(row.j))
        batch = mc_batch(ctx, int(row.j), n, c.mc_set, float(ctx.post["sigma_sess"].values[int(row.chain), int(row.draw)]))
        res = S.simulate(batch, c, P, esc, draw_k(int(row.j), c), ctx.kin, ctx.grid, ctx.U, ctx.A_ref)
        cap[a], tc[a], td[a] = res["captured"], res["t_contact_s"], res["t_det_s"]
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_name(f.stem + "_tmp.npz")
    np.savez_compressed(tmp, captured=cap, t_contact_s=tc, t_det_s=td, j=dr.j.to_numpy())
    os.replace(tmp, f)
    return f, time.time() - t0


def unit_stats(f):
    z = np.load(f)
    res = dict(captured=z["captured"], t_contact_s=z["t_contact_s"].astype(float), t_det_s=z["t_det_s"].astype(float),
               t_end_obs_s=np.inf)
    cap, det_ok, tdet = S.outcomes(res)
    return z["j"], cap.mean(1), det_ok.mean(1), np.median(tdet, axis=1)


def next_cmd(order, prior, n, n_draws, smoke, max_hours):
    return ("uv run python phases/p3b/run.py run --order " + order + ("" if order == "priority" else f" --prior {prior}")
            + f" --n {n}" + (f" --max-hours {max_hours:g}" if max_hours is not None else "")
            + (f" --draws {n_draws}" if n_draws != N_DRAWS else "") + (" --smoke" if smoke else ""))


def cmd_run(prior, n, n_draws, smoke, order="prior", max_hours=None):
    ctx = Ctx()
    plan = plan_units(order, prior, smoke)
    dr = {p: draws(p).iloc[:n_draws] for p in dict.fromkeys(p for _, p, _ in plan)}
    prov = pv.start_meta()
    w = np.array([steps_of(ctx, c) for _, _, c in plan])

    def done(u):
        return unit_path(u[1], u[2], n, len(dr[u[1]])).exists()

    def by_group(us):
        return {g: sum(1 for x in us if x[0] == g) for g in dict.fromkeys(x[0] for x in us)}

    log(f"[run {order}] 단위 {len(plan)} (묶음별 {by_group(plan)}) × draw {n_draws} × 궤적 {n}, 이미 완료 {sum(map(done, plan))}, "
        f"상한 {'없음' if max_hours is None else f'{max_hours:g} h'}, 시작 HEAD {prov['run_start_commit'][:7]} dirty {prov['run_start_dirty']}")
    t_all, st, ran_units = time.time(), dict(w=0.0, sec=0.0), []

    def one(i, u):
        g, p, c = u
        _, sec = run_unit(ctx, p, dr[p], c, n)
        ran_units.append(u)
        st["w"] += w[i]
        st["sec"] += sec
        rest = st["sec"] / max(st["w"], 1e-12) * sum(w[k] for k in range(i + 1, len(plan)) if not done(plan[k]))
        log(f"[run {g}:{p}:{role_of(c)}] {i + 1}/{len(plan)} {c.name}: {sec:.1f}s (단계당 {sec / w[i] / len(dr[p]) * 1e3:.2f} ms/draw) "
            f"→ 남은 예상 {rest / 60:.1f} 분, 경과 {(time.time() - t_all) / 3600:.2f} h")

    ran, skipped, left = run_plan(plan, done, one, None if max_hours is None else max_hours * 3600)
    if left:
        log(f"[run] 경과 {(time.time() - t_all) / 3600:.4f} h > 상한 {max_hours:g} h → 새 조건 시작 안 함, 저장 후 종료. "
            f"이번 실행 {ran}, 건너뜀 {skipped}, 남은 조건 {len(left)} (묶음별 {by_group(left)})")
        log(f"[run] 다음 실행 명령(이어서, 로그는 >>): {next_cmd(order, prior, n, n_draws, smoke, max_hours)}")
    else:
        log(f"[run] 모든 단위 완료 (이번 실행 {ran}, 건너뜀 {skipped})")
    for p in dr:
        units = [x for x in plan if x[1] == p]
        f = OUT / f"run_{p}_meta.json"
        old = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
        sess = old.get("sessions", []) + [dict(order=order, max_hours=max_hours, ran=[x[2].name for x in ran_units if x[1] == p],
                                              seconds=round(time.time() - t_all, 1), run_commit=pv.git_head(), **prov)]
        save_json(f, dict(prior=p, n_traj=n, n_draws=len(dr[p]), n_conditions=len(units), order=order,
                  order_rule="decisions.md 'P3b 판정 실험 실행 결정' 2번" if order == "priority" else None,
                  n_completed=sum(map(done, units)), n_remaining=sum(1 for x in left if x[1] == p),
                  conditions=[dict(c.__dict__, group=g, role=role_of(c)) for g, _, c in units], seed_mc=SEED_MC, seed_k=SEED_K,
                  seed_esc=SEED_ESC, seed_draw=SEED_DRAW[p], seconds=round(time.time() - t_all, 1), run_commit=pv.git_head(),
                  smoke=smoke, validation=VALIDATION, validation_tag=VALIDATION_TAG, sessions=sess, **prov))
        cmd_summarize(p, n, n_draws, smoke, ctx=ctx)
    return left


def cmd_summarize(prior, n, n_draws, smoke, ctx=None):
    dr = draws(prior).iloc[:n_draws]
    U = ctx.U if ctx else flow.load_model().U_ref_cfd_mps
    rows = []
    for c in conditions(prior, smoke):
        f = unit_path(prior, c, n, len(dr))
        if not f.exists():
            continue
        j, pc, pdt, tdm = unit_stats(f)
        nt = np.load(f)["captured"].shape[1]
        for a in range(len(j)):
            rows.append(dict(condition=c.name, v_mps=c.v_mps, n_exp=c.n_exp, clap=c.clap, c_b=c.c_b, phi=c.phi,
                             att_mult=c.att_mult, att_model=c.att_model, k_fixed=c.k_fixed, mc_set=c.mc_set, j=int(j[a]),
                             role=role_of(c), n_traj=nt, p_cap=pc[a], p_det=pdt[a], t_det_med_s=tdm[a],
                             p_cap_mcse=np.sqrt(pc[a] * (1 - pc[a]) / nt), p_det_mcse=np.sqrt(pdt[a] * (1 - pdt[a]) / nt),
                             label=label_of(c, U), validation=VALIDATION))
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / f"conditions_{prior}.csv", index=False)
    if len(df):
        s = df.groupby("condition", sort=False).agg(role=("role", "first"), p_cap=("p_cap", "median"),
                                                    p_cap_mcse=("p_cap_mcse", "median"), p_det=("p_det", "median"),
                                                    p_det_mcse=("p_det_mcse", "median"), n=("j", "size"))
        s.to_csv(OUT / f"conditions_{prior}_summary.csv")
        log(f"[summarize {prior}] 조건 {len(s)}, 행 {len(df)} → {OUT / f'conditions_{prior}.csv'} (판정은 P3b 2/2) {VALIDATION_TAG}")
    save_json(OUT / f"conditions_{prior}_meta.json", dict(prior=prior, n_traj=n, n_draws=len(dr), n_rows=len(df), smoke=smoke,
              validation=VALIDATION, validation_tag=VALIDATION_TAG, label_rule="decisions.md 'P3b G3 결정' 2번",
              mcse_note="p_*_mcse = draw 별 이항 MC 표준오차 sqrt(p(1-p)/N). 판정량(짝차이)의 MC SE 는 판정 단계에서 "
                        "units npz 궤적 짝차이 SD/sqrt(N) 으로 판정마다 병기",
              n_rule="decisions.md :266 (4000 으로도 안 되면 4000, 목표 미충족 보고) — 'P3b 판정 실험 실행 결정' 1번",
              summarize_commit=pv.git_head(), **pv.start_meta()))


# ------------------------------------------------------------------ ② 1 m/s 보정 모형
_W = {}


def _w_init():
    _W["ctx"] = Ctx()


def _val_batch(ctx, j, P):
    """재현 검증(cmd_validate)과 같은 trial 단위 batch·난수 (seed SEED_VAL, draw j)."""
    I = ctx.init
    rng = np.random.default_rng([SEED_VAL, j])
    u, z_a = S.common_randoms(rng, len(I))
    return S.Batch(rel0_m=np.stack(I.rel0_m.to_numpy()), v0_mps=np.stack(I.v0_mps.to_numpy()),
                   s_start_m=I.s_start_m.to_numpy(float), transparent=I.transparent.to_numpy(bool),
                   eta=P["eta_sess"][I.session_idx.to_numpy()], u=u, z_a=z_a, t_end_kin_s=I.t_end_kin_s.to_numpy(float))


def _subset(b: S.Batch, m):
    return S.Batch(**{k: (v[m] if isinstance(v, np.ndarray) else v) for k, v in b.__dict__.items()})


def _cal_task(kind, rows, arg):
    """작업 묶음 하나 (rows = draws 표의 행 dict 목록).

    kind 'shift': 감지 끈 기록 → 모든 Δ 의 원판별 반응 수 (draw 별 k ~ U(0, 8.2), 원 모형).
    kind 'k'    : arg = (Δ, k) → 불투명 trial 포획 수.
    kind 'val'  : arg = (Δ, k 또는 None) → 원판별 (포획, 반응) 비율 (재현 검증 규칙).
    kind 'check': arg = Δ 목록 → 전체 시뮬레이션 반응 수 대 재계산 반응 수 (일치 점검).
    """
    ctx = _W.get("ctx") or _W.setdefault("ctx", Ctx())
    tr = ctx.init.transparent.to_numpy(bool)
    out = []
    for row in rows:
        j = int(row["j"])
        P = det.params_from_draw(ctx.post, int(row["chain"]), int(row["draw"]), 1, ctx.n_sess)
        esc, _ = draw_escape(ctx.boot, j)
        b = _val_batch(ctx, j, P)
        if kind in ("shift", "check"):
            c0 = S.Condition(name="cal", v_mps=ctx.U)
            k = draw_k(j, c0)
            rec = S.simulate(b, c0, P, esc, k, ctx.kin, ctx.grid, ctx.U, ctx.A_ref, record=True)
            lim = np.minimum(np.minimum(rec["t_contact_s"], 0.0), rec["t_end_obs_s"])
            shifts = CAL_SHIFTS if kind == "shift" else arg
            resp = np.array([[(S.detect_given_shift(rec, b, P, d) < lim)[m].sum() for m in (~tr, tr)] for d in shifts])
            if kind == "check":
                full = []
                for d in shifts:
                    r = S.simulate(b, S.Condition(name="cal", v_mps=ctx.U, theta_shift=float(d)), P, esc, k, ctx.kin,
                                   ctx.grid, ctx.U, ctx.A_ref)
                    _, ok, _ = S.outcomes(r, window=True)
                    full.append([ok[~tr].sum(), ok[tr].sum()])
                out.append(dict(j=j, recomputed=resp.tolist(), full=full))
            else:
                out.append(dict(j=j, resp=resp))
        elif kind == "k":
            d, k = arg
            bo = _subset(b, ~tr)
            r = S.simulate(bo, S.Condition(name="cal", v_mps=ctx.U, theta_shift=d), P, esc, k, ctx.kin, ctx.grid, ctx.U,
                           ctx.A_ref)
            out.append(dict(j=j, cap=int(r["captured"].sum())))
        else:
            d, k = arg
            c = S.Condition(name="cal", v_mps=ctx.U, theta_shift=d)
            r = S.simulate(b, c, P, esc, draw_k(j, c) if k is None else k, ctx.kin, ctx.grid, ctx.U, ctx.A_ref)
            cap, ok, _ = S.outcomes(r, window=True)
            out.append(dict(j=j, opaque=(cap[~tr].mean(), ok[~tr].mean()), transparent=(cap[tr].mean(), ok[tr].mean())))
    return out


def _pmap(tasks, workers):
    """tasks = [(kind, rows, arg)] → 결과 목록 (순서 유지). workers ≤ 1 이면 이 프로세스에서."""
    if workers <= 1:
        return [_cal_task(*t) for t in tasks]
    from concurrent.futures import ProcessPoolExecutor

    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = "1"
    with ProcessPoolExecutor(max_workers=workers, initializer=_w_init) as ex:
        return list(ex.map(_cal_task, *zip(*tasks)))


def _chunks(dr, n):
    rows = dr.to_dict("records")
    return [rows[i::n] for i in range(n)]


def cal_checks(val):
    """원판별 (포획, 반응) draw 표 → 재현 지표 4개 (cmd_validate 와 같은 규칙)."""
    from scipy.stats import beta

    checks = []
    used = {("opaque", "capture"): "(b) k", ("opaque", "respond"): "(a) Δ", ("transparent", "respond"): "(a) Δ",
            ("transparent", "capture"): "준독립"}
    for disk, o in OBS.items():
        arr = np.array([v[disk] for v in val])
        for key, obs_k, col in (("capture", o["collided"], 0), ("respond", o["responded"], 1)):
            lo = beta.ppf(0.025, obs_k, o["n"] - obs_k + 1) if obs_k > 0 else 0.0
            hi = beta.ppf(0.975, obs_k + 1, o["n"] - obs_k) if obs_k < o["n"] else 1.0
            s5, s50, s95 = np.quantile(arr[:, col], [0.05, 0.5, 0.95])
            checks.append(dict(disk=disk, metric=key, obs=obs_k / o["n"], obs_ci95=[lo, hi], sim_mean=float(arr[:, col].mean()),
                               sim_median=s50, sim_pi90=[s5, s95], overlap=bool(max(lo, s5) <= min(hi, s95)),
                               used_in=used[(disk, key)]))
    return checks


def cmd_calibrate(n_draws, workers):
    """② 보정: (a) 문턱 이동 Δ (두 원판 반응률 제곱오차 합 최소) → (b) k (불투명 포획률) → 재현 지표 4개."""
    t0 = time.time()
    ctx = Ctx()
    dr = draws("P1").iloc[:n_draws]
    tr = ctx.init.transparent.to_numpy(bool)
    n_o, n_t = int((~tr).sum()), int(tr.sum())
    nw = max(1, min(workers, len(dr)))
    log(f"[calibrate] draw {len(dr)}, trial {len(tr)} (불투명 {n_o}/투명 {n_t}), Δ 격자 {len(CAL_SHIFTS)}점, k 격자 {len(CAL_KS)}점, "
        f"작업자 {nw}")
    # 일치 점검: 앞 draw 2개에서 전체 시뮬레이션 대 재계산 (Δ = 0, −0.5)
    chk = _pmap([("check", [r], [0.0, -0.5]) for r in dr.iloc[:2].to_dict("records")], min(nw, 2))
    chk = [x for c in chk for x in c]
    match = all(np.array_equal(np.array(c["recomputed"]), np.array(c["full"])) for c in chk)
    log(f"[calibrate] 재계산 일치 점검 (draw 2 × Δ 2): {'일치' if match else '불일치'} {[(c['recomputed'], c['full']) for c in chk]} "
        f"({time.time() - t0:.0f}s)")
    # (a)
    ta = time.time()
    res = [x for c in _pmap([("shift", ch, None) for ch in _chunks(dr, nw)], nw) for x in c]
    resp = np.sum([x["resp"] for x in res], axis=0)                      # (n_shift, 2)
    rate = resp / (len(dr) * np.array([n_o, n_t]))
    obs = np.array([OBS["opaque"]["responded"] / OBS["opaque"]["n"], OBS["transparent"]["responded"] / OBS["transparent"]["n"]])
    sse = ((rate - obs) ** 2).sum(axis=1)
    best = np.flatnonzero(np.isclose(sse, sse.min(), rtol=0, atol=1e-15))
    i_d = int(best[np.argmin(np.abs(CAL_SHIFTS[best]))])
    d_hat = float(CAL_SHIFTS[i_d])
    log(f"[calibrate] (a) Δ̂ = {d_hat:+.2f} (반응률 불투명 {rate[i_d, 0]:.4f}, 투명 {rate[i_d, 1]:.4f}; 관측 {obs[0]:.4f}, {obs[1]:.4f}; "
        f"Δ=0 에서 {rate[CAL_SHIFTS == 0][0, 0]:.4f}, {rate[CAL_SHIFTS == 0][0, 1]:.4f}) {time.time() - ta:.0f}s"
        + (" — 격자 경계" if i_d in (0, len(CAL_SHIFTS) - 1) else ""))
    # (b)
    tb = time.time()
    rows = dr.to_dict("records")
    rk = _pmap([("k", rows, (d_hat, float(k))) for k in CAL_KS], nw)
    cap = np.array([sum(x["cap"] for x in r) for r in rk]) / (len(dr) * n_o)
    obs_c = OBS["opaque"]["collided"] / OBS["opaque"]["n"]
    diff = np.abs(cap - obs_c)
    i_k = int(np.flatnonzero(np.isclose(diff, diff.min(), rtol=0, atol=1e-15))[0])
    k_hat = float(CAL_KS[i_k])
    short = not (cap.min() <= obs_c <= cap.max())
    log(f"[calibrate] (b) k̂ = {k_hat:.1f} s⁻¹ (불투명 포획률 {cap[i_k]:.4f}, 관측 {obs_c:.4f}; 격자 범위 [{cap.min():.4f}, {cap.max():.4f}])"
        f"{' — 보정 미달' if short else ''} {time.time() - tb:.0f}s")
    # 보정 뒤 재현 지표 4개 (+ Δ̂ 만 적용·k 원 모형)
    tv = time.time()
    val = [x for c in _pmap([("val", ch, (d_hat, k_hat)) for ch in _chunks(dr, nw)], nw) for x in c]
    checks = cal_checks(val)
    for c in checks:
        log(f"[calibrate] 보정 뒤 {c['disk']:11s} {c['metric']:8s} 관측 {c['obs']:.3f} [{c['obs_ci95'][0]:.3f}, {c['obs_ci95'][1]:.3f}] "
            f"| 시뮬 {c['sim_median']:.3f} [{c['sim_pi90'][0]:.3f}, {c['sim_pi90'][1]:.3f}] → {'겹침' if c['overlap'] else '안 겹침'} ({c['used_in']})")
    OUT_CAL.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(dict(shift=CAL_SHIFTS, resp_opaque=rate[:, 0], resp_transparent=rate[:, 1], sse=sse)).to_csv(
        OUT_CAL / "calibrate_shift.csv", index=False)
    pd.DataFrame(dict(k_per_s=CAL_KS, cap_opaque=cap)).to_csv(OUT_CAL / "calibrate_k.csv", index=False)
    out = dict(shift=d_hat, k_per_s=k_hat, shift_param="문턱 로그 위치 η_sess + Δ (= log w_v − Δ), c_t 아님",
               shift_at_grid_edge=bool(i_d in (0, len(CAL_SHIFTS) - 1)), k_short=short, k_note="보정 미달" if short else "",
               resp_at_shift=rate[i_d].tolist(), resp_at_zero=rate[CAL_SHIFTS == 0][0].tolist(), obs_resp=obs.tolist(),
               cap_at_k=float(cap[i_k]), obs_cap_opaque=obs_c, cap_range=[float(cap.min()), float(cap.max())],
               checks=checks, n_overlap=int(sum(c["overlap"] for c in checks)), recompute_match=bool(match),
               grid_shift=[float(CAL_SHIFTS[0]), float(CAL_SHIFTS[-1]), 0.01], grid_k=[0.0, S.K_MAX_PER_S, 0.2],
               n_draws=len(dr), prior="P1", seed_val=SEED_VAL, seed_k=SEED_K, seed_esc=SEED_ESC, seed_draw=SEED_DRAW["P1"],
               rule="decisions.md 'H2 세 갈래 판정' 4번 + 9-14 ② 격자 기록", tag=CAL_TAG, tag_note=CAL_NOTE,
               note="보정은 원 모형 판정 결과를 쓰지 않는다 (재현 검증 trial 표·관측 비율만)", smoke=bool(SMOKE_DIR),
               seconds=round(time.time() - t0, 1), workers=nw, run_commit=pv.git_head(), **pv.start_meta())
    save_json(OUT_CAL / "calibrate.json", out)
    log(f"[calibrate] 재현 지표 겹침 {out['n_overlap']}/4 ({time.time() - tv:.0f}s), 전체 {out['seconds']}s → {OUT_CAL / 'calibrate.json'}")
    return out


def conditions_cal(cal, smoke=False):
    """② 속도 조건: 한 손 v × n ∈ {0.8, 1, 1.2} (H2 판정 조건만), Δ̂·k̂ 고정."""
    sp = (1.0, 12.0) if smoke else SPEEDS
    out = []
    for v in sp:
        for n in CAL_N_EXP:
            c = S.Condition(name="", v_mps=v, n_exp=n, k_fixed=float(cal["k_per_s"]), theta_shift=float(cal["shift"]))
            out.append(S.Condition(**{**c.__dict__, "name": cname(c) + f"_d{c.theta_shift:+g}_cal"}))
    return out


def cmd_run_cal(n, n_draws, smoke, max_hours=None):
    cal = json.loads((OUT_CAL / "calibrate.json").read_text(encoding="utf-8"))
    ctx = Ctx()
    conds = conditions_cal(cal, smoke)
    dr = draws("P1").iloc[:n_draws]
    prov = pv.start_meta()
    w = np.array([steps_of(ctx, c) for c in conds])
    est = None
    ref = OUT / "run_P1_meta.json"
    if ref.exists() and not smoke:                       # 원 판정 실험의 실행 시간·조건 목록만 사용 (결과 값 미사용)
        m1 = json.loads(ref.read_text(encoding="utf-8"))
        m0 = json.loads((OUT / "run_P0_meta.json").read_text(encoding="utf-8"))
        cs = [S.Condition(**{k: v for k, v in c.items() if k in S.Condition.__dataclass_fields__}) for c in m1["conditions"] + m0["conditions"]]
        w_ref = sum(steps_of(ctx, c) for c in cs) * m1["n_draws"] * m1["n_traj"]
        est = m1["seconds"] / w_ref * w.sum() * len(dr) * n
    done = lambda c: unit_path("P1", c, n, len(dr), INTERIM_CAL).exists()  # noqa: E731
    log(f"[run calibrated] Δ̂ {cal['shift']:+.2f}, k̂ {cal['k_per_s']:.1f}, 조건 {len(conds)} × draw {len(dr)} × 궤적 {n}, 이미 완료 "
        f"{sum(map(done, conds))}, 예상 {'?' if est is None else f'{est / 3600:.2f} h (원 판정 실험 단계당 시간 기준)'}, "
        f"상한 {'없음' if max_hours is None else f'{max_hours:g} h'}, 시작 HEAD {prov['run_start_commit'][:7]} dirty {prov['run_start_dirty']}")
    t_all, st, ran_units = time.time(), dict(w=0.0, sec=0.0), []

    def one(i, c):
        _, sec = run_unit(ctx, "P1", dr, c, n, INTERIM_CAL)
        ran_units.append(c)
        st["w"] += w[i]
        st["sec"] += sec
        rest = st["sec"] / max(st["w"], 1e-12) * sum(w[k] for k in range(i + 1, len(conds)) if not done(conds[k]))
        log(f"[run calibrated] {i + 1}/{len(conds)} {c.name}: {sec:.1f}s → 남은 예상 {rest / 60:.1f} 분, 경과 {(time.time() - t_all) / 3600:.2f} h")

    ran, skipped, left = run_plan(conds, done, one, None if max_hours is None else max_hours * 3600)
    if left:
        log(f"[run calibrated] 상한 도달 → 남은 조건 {len(left)}. 같은 명령으로 이어서 (로그는 >>)")
    else:
        log(f"[run calibrated] 모든 조건 완료 (이번 실행 {ran}, 건너뜀 {skipped})")
    f = OUT_CAL / "run_P1_meta.json"
    old = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
    sess = old.get("sessions", []) + [dict(max_hours=max_hours, ran=[c.name for c in ran_units], seconds=round(time.time() - t_all, 1),
                                          est_seconds=est, run_commit=pv.git_head(), **prov)]
    save_json(f, dict(model="calibrated", prior="P1", n_traj=n, n_draws=len(dr), n_conditions=len(conds),
                      n_completed=sum(map(done, conds)), n_remaining=len(left), conditions=[c.__dict__ for c in conds],
                      calibration=dict(shift=cal["shift"], k_per_s=cal["k_per_s"], k_note=cal["k_note"],
                                       source=str((OUT_CAL / "calibrate.json").relative_to(ROOT))),
                      seed_mc=SEED_MC, seed_esc=SEED_ESC, seed_draw=SEED_DRAW["P1"], smoke=smoke, tag=CAL_TAG, tag_note=CAL_NOTE,
                      sessions=sess, run_commit=pv.git_head(), **prov))
    summarize_cal(n, n_draws, smoke, ctx)
    return left


def summarize_cal(n, n_draws, smoke, ctx=None):
    cal = json.loads((OUT_CAL / "calibrate.json").read_text(encoding="utf-8"))
    dr = draws("P1").iloc[:n_draws]
    U = ctx.U if ctx else flow.load_model().U_ref_cfd_mps
    rows = []
    for c in conditions_cal(cal, smoke):
        f = unit_path("P1", c, n, len(dr), INTERIM_CAL)
        if not f.exists():
            continue
        j, pc, pdt, tdm = unit_stats(f)
        nt = np.load(f)["captured"].shape[1]
        lab = ("[모델 예측] " if c.model_prediction(U) else "") + CAL_TAG
        for a in range(len(j)):
            rows.append(dict(condition=c.name, v_mps=c.v_mps, n_exp=c.n_exp, k_fixed=c.k_fixed, theta_shift=c.theta_shift,
                             j=int(j[a]), role="judge", n_traj=nt, p_cap=pc[a], p_det=pdt[a], t_det_med_s=tdm[a],
                             p_cap_mcse=np.sqrt(pc[a] * (1 - pc[a]) / nt), p_det_mcse=np.sqrt(pdt[a] * (1 - pdt[a]) / nt),
                             label=lab))
    df = pd.DataFrame(rows)
    OUT_CAL.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CAL / "conditions_P1.csv", index=False)
    save_json(OUT_CAL / "conditions_P1_meta.json", dict(model="calibrated", prior="P1", n_traj=n, n_draws=len(dr), n_rows=len(df),
              smoke=smoke, tag=CAL_TAG, tag_note=CAL_NOTE, summarize_commit=pv.git_head(), **pv.start_meta()))
    log(f"[summarize calibrated] 행 {len(df)} → {OUT_CAL / 'conditions_P1.csv'} (판정은 9-15) {CAL_TAG}")


# ------------------------------------------------------------------ 궤적 수 보정
# ------------------------------------------------------------------ D2 진단 (ρ_av 고정 민감도, 탐색적)
def d2_conditions(speeds):
    return [S.Condition(name=cname(S.Condition(name="", v_mps=v)), v_mps=v) for v in speeds]


def d2_name(fix):
    return f"rho{fix:g}"


def cmd_run_d2(fix, speeds, n, n_draws):
    """fix = None 은 소규모 시험의 비교 기준(같은 축소 N 의 원 결과)을 만들 때만. 실제 비교 기준은 재계산하지 않는다."""
    if fix is None and not SMOKE_DIR:
        raise SystemExit("--speeds 단독 실행은 MQ10_P3B_DIR 소규모 시험에서만 (① 원 결과는 재계산하지 않는다)")
    ctx = Ctx()
    prov = pv.start_meta()
    dr = draws("P1").iloc[:n_draws]
    root = INTERIM if fix is None else INTERIM_D2 / d2_name(fix)
    conds = d2_conditions(speeds)
    w = np.array([steps_of(ctx, c) for c in conds])
    log(f"[d2] ρ_av {'원 사후값' if fix is None else fix} × 속도 {list(speeds)} × draw {len(dr)} × 궤적 {n}, "
        f"시작 HEAD {prov['run_start_commit'][:7]} dirty {prov['run_start_dirty']} [{D2_TAG}]")
    t_all, unit_s, rows = time.time(), {}, []
    for i, c in enumerate(conds):
        f, sec = run_unit(ctx, "P1", dr, c, n, root, fix)
        unit_s[c.name] = round(sec, 1)
        rest = sum(unit_s.values()) / max(w[:i + 1].sum(), 1e-12) * w[i + 1:].sum()
        log(f"[d2] {i + 1}/{len(conds)} {c.name}: {sec:.1f}s → 남은 예상 {rest / 60:.1f} 분")
        j, pc, pdt, tdm = unit_stats(f)
        rows += [dict(condition=c.name, v_mps=c.v_mps, n_exp=c.n_exp, j=int(j[a]), n_traj=n, p_cap=pc[a], p_det=pdt[a], t_det_med_s=tdm[a],
                      label=label_of(c, ctx.U)) for a in range(len(j))]
    sec_all = sum(unit_s.values())
    if sec_all > 0:
        log(f"[d2] 이번 실행 {sec_all:.1f}s → N 4000·200 draw·고정값 {len(D2_FIXED)}개 환산 예상 "
            f"{sec_all / (n * len(dr)) * 4000 * 200 * len(D2_FIXED) / 60:.1f} 분 (궤적 수·draw 수 비례 가정)")
    if fix is None:
        return
    out = OUT_D2 / d2_name(fix)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "conditions_P1.csv", index=False)
    rho_av = ctx.post["rho_av"].values[dr.chain.to_numpy(int), dr.draw.to_numpy(int)]
    save_json(out / "run_meta.json", dict(tag=D2_TAG, rule="decisions.md 'D2 진단 — ρ_av 고정 민감도'", limit=D2_LIMIT, fix_rho_av=fix, prior="P1",
              speeds_mps=list(speeds), n_exp=1.0, n_traj=n, n_draws=len(dr), conditions=[c.name for c in conds], unit_seconds=unit_s,
              rho_av_original_draws_q025_q50_q975=[float(x) for x in np.quantile(rho_av, [0.025, 0.5, 0.975])],
              seed_mc=SEED_MC, seed_k=SEED_K, seed_esc=SEED_ESC, seed_draw=SEED_DRAW["P1"], validation=VALIDATION,
              validation_tag=VALIDATION_TAG, seconds=round(time.time() - t_all, 1), run_commit=pv.git_head(), **prov))


def d2_load_cap(f):
    z = np.load(f)
    res = dict(captured=z["captured"], t_contact_s=z["t_contact_s"].astype(float), t_det_s=z["t_det_s"].astype(float), t_end_obs_s=np.inf)
    return S.outcomes(res)[0], z["j"]


def d2_paired(a, b):
    """draw × 궤적 포획 a, b → draw 별 짝차이 평균과 MC SE (궤적 짝차이 SD/√N, judge.py pdiff 와 같은 정의)."""
    x = a.astype(np.float32) - b.astype(np.float32)
    return x.mean(1), x.std(1, ddof=1) / np.sqrt(x.shape[1])


def d2_is_drop(effect_median, mcse):
    """'MC SE 의 2배를 넘는 감소' (decisions.md 'D2 진단' (가))."""
    return bool(effect_median < -2 * mcse)


def d2_phrase(drops):
    """구간 하나의 {고정값: 감소 여부} → 7-13절 5번 판단 문구 (기계 적용, 해석 아님)."""
    yes = [k for k, v in drops.items() if v]
    no = [k for k, v in drops.items() if not v]
    if not yes:
        return "하락은 ρ_av 사후의 큰 값에 의존"
    if not no:
        return "ρ_av 외 요인"
    return f"갈림: 감소 있음 ρ_av = {yes}, 감소 없음 ρ_av = {no}"


def cmd_summarize_d2(n, n_draws):
    prov = pv.start_meta()
    conds = {c.v_mps: c for c in d2_conditions(D2_SPEEDS)}
    dr_n = min(n_draws, len(draws("P1")))
    U = flow.load_model().U_ref_cfd_mps
    roots = {"①": INTERIM, **{f"{v:g}": INTERIM_D2 / d2_name(v) for v in D2_FIXED}}
    paths = {m: {v: unit_path("P1", c, n, dr_n, r) for v, c in conds.items()} for m, r in roots.items()}
    missing = [str(f) for fs in paths.values() for f in fs.values() if not f.exists()]
    if missing:
        raise SystemExit(f"D2 단위 없음 {len(missing)}개, 표를 만들지 않는다: {missing[:4]}")
    cap, js = {}, None
    for m, fs in paths.items():
        cap[m] = {}
        for v, f in fs.items():
            cap[m][v], j = d2_load_cap(f)
            assert js is None or np.array_equal(js, j), "draw 짝 불일치"
            js = j
    obs = OBS["opaque"]["collided"] / OBS["opaque"]["n"]
    speed_rows, eff_rows = [], []
    for m in roots:
        for v in D2_SPEEDS:
            p = cap[m][v].mean(1)
            row = dict(model=m, v_mps=v, p_cap_median=float(np.median(p)), p_cap_q025=float(np.quantile(p, 0.025)),
                       p_cap_q975=float(np.quantile(p, 0.975)), obs_cap_opaque=obs if v == 1.0 else np.nan, label=label_of(conds[v], U))
            if m != "①":
                d, se = d2_paired(cap[m][v], cap["①"][v])
                row.update(diff_vs_orig_median=float(np.median(d)), diff_vs_orig_mcse_med=float(np.median(se)))
            speed_rows.append(row)
        for lo, hi in D2_SEGMENTS:
            d, se = d2_paired(cap[m][hi], cap[m][lo])
            med, se_med, se_q90 = float(np.median(d)), float(np.median(se)), float(np.quantile(se, 0.9))
            row = dict(model=m, v_from=lo, v_to=hi, effect_median=med, effect_q025=float(np.quantile(d, 0.025)),
                       effect_q975=float(np.quantile(d, 0.975)), mcse_med=se_med, mcse_q90=se_q90, drop=d2_is_drop(med, se_med),
                       drop_at_q90=d2_is_drop(med, se_q90))
            if m != "①":
                x = (cap[m][hi].astype(np.float32) - cap[m][lo].astype(np.float32)) - (cap["①"][hi].astype(np.float32) - cap["①"][lo].astype(np.float32))
                row.update(did_vs_orig_median=float(np.median(x.mean(1))), did_vs_orig_mcse_med=float(np.median(x.std(1, ddof=1) / np.sqrt(x.shape[1]))))
            eff_rows.append(row)
    sp, ef = pd.DataFrame(speed_rows), pd.DataFrame(eff_rows)
    fixed = ef[ef.model != "①"]
    phrases = {}
    for lo, hi in D2_SEGMENTS:
        g = fixed[(fixed.v_from == lo) & (fixed.v_to == hi)]
        phrases[f"{lo:g}→{hi:g}"] = dict(by_mcse_med=d2_phrase(dict(zip(g.model, g["drop"]))), by_mcse_q90=d2_phrase(dict(zip(g.model, g.drop_at_q90))))
    OUT_D2.mkdir(parents=True, exist_ok=True)
    sp.to_csv(OUT_D2 / "summary_speed.csv", index=False)
    ef.to_csv(OUT_D2 / "summary_effects.csv", index=False)
    run_commits = {}
    for v in D2_FIXED:
        mf = OUT_D2 / d2_name(v) / "run_meta.json"
        run_commits[d2_name(v)] = json.loads(mf.read_text(encoding="utf-8")).get("run_start_commit") if mf.exists() else None
    save_json(OUT_D2 / "summary_d2.json", dict(tag=D2_TAG, rule="decisions.md 'D2 진단 — ρ_av 고정 민감도'", phrases=phrases, limit=D2_LIMIT,
              verdict_note="판정 문구(7-13절 1번)는 이 결과와 상관없이 바꾸지 않는다", se_def="draw 별 궤적 짝차이 SD/√N 의 draw 간 중앙값(판단), 90 백분위(병기)",
              obs_cap_opaque=obs, obs_note="1 m/s 표 값은 통념 실험 궤적(초기 상태 복원추출·불투명)으로, 재현 검증(848 trial, G2 ① 원판 0.311)과 표본이 다르다",
              n_traj=n, n_draws=dr_n, seed_mc=SEED_MC, seed_k=SEED_K, seed_esc=SEED_ESC, seed_draw=SEED_DRAW["P1"],
              validation=VALIDATION, validation_tag=VALIDATION_TAG, run_start_commits=run_commits, summarize_commit=pv.git_head(), **prov))
    log(f"[summarize-d2] {D2_TAG} {VALIDATION_TAG} N {n} draw {dr_n} → {OUT_D2}")
    log("속도별\n" + sp.drop(columns="label").to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    log("구간 효과\n" + ef.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    log(f"판단 문구 {json.dumps(phrases, ensure_ascii=False)}")


def cmd_calib(n_draws=20, n_pilot=1000):
    """H2 (n = 1) E·Δ_k 짝차이의 draw 별 MC 표준오차 → N 선택 (decisions.md 'P3b 시뮬레이터 설계')."""
    ctx = Ctx()
    dr = draws("P1").iloc[:n_draws]
    t0 = time.time()
    caps = {}
    for v in SPEEDS:
        c = S.Condition(name=f"calib_v{v:g}", v_mps=v)
        m = np.zeros((len(dr), n_pilot), bool)
        for a, (_, row) in enumerate(dr.iterrows()):
            P = draw_params(ctx.post, row, ctx.n_sess)
            esc, _ = draw_escape(ctx.boot, int(row.j))
            batch = mc_batch(ctx, int(row.j), n_pilot, "a", float(ctx.post["sigma_sess"].values[int(row.chain), int(row.draw)]))
            m[a] = S.simulate(batch, c, P, esc, draw_k(int(row.j), c), ctx.kin, ctx.grid, ctx.U, ctx.A_ref)["captured"]
        caps[v] = m
        log(f"[calib] v {v:g}: p_cap 중앙값 {np.median(m.mean(1)):.3f} ({time.time() - t0:.0f}s)")
    pairs = [(SPEEDS[-1], SPEEDS[0])] + [(SPEEDS[i + 1], SPEEDS[i]) for i in range(len(SPEEDS) - 1)]
    se = np.array([[np.std(caps[a][d].astype(float) - caps[b][d], ddof=1) / np.sqrt(n_pilot) for d in range(len(dr))]
                   for a, b in pairs])
    q90 = float(np.quantile(se, 0.9))
    target = 0.02 / 4
    chosen = next((N for N in (500, 1000, 2000, 4000) if q90 * np.sqrt(n_pilot / N) <= target), 4000)
    out = dict(n_pilot=n_pilot, n_draws=len(dr), se_q90_at_pilot=q90, target=target, chosen_N=chosen,
               se_q90_at_chosen=q90 * np.sqrt(n_pilot / chosen), met=bool(q90 * np.sqrt(n_pilot / chosen) <= target),
               seconds=round(time.time() - t0, 1), **pv.start_meta())
    save_json(OUT / "calib_n.json", out)
    log(f"[calib] MC SE 90 백분위 {q90 * 100:.3f}%p (N {n_pilot}) → N = {chosen} (SE {out['se_q90_at_chosen'] * 100:.3f}%p, "
        f"목표 0.5%p, {'충족' if out['met'] else '미충족'})")
    return out


def cmd_prepare():
    t0 = time.time()
    I = init_states()
    g = flow_grid()
    err = grid_error(g)
    for p in ("P1", "P0"):
        draws(p)
    km = json.loads((OUT / "draws_P1_meta.json").read_text(encoding="utf-8"))["k_hat"]
    save_json(OUT / "prepare.json", dict(n_init=len(I), n_excluded=848 - len(I), n_fit_min=int(I.n_fit.min()), s_start_median_m=float(I.s_start_m.median()),
              speed_median_mps=float(np.median(np.linalg.norm(np.stack(I.v0_mps.to_numpy()), axis=1))),
              grid_shape=list(g.u_xi.shape), grid_front_rel_rmse_vs_hgb=err,
              grid_note="격자(Δ0.1R) 대 HGB 직접 예측 차이. HGB 자체가 0.02R 이동에 상대 10.9% 변하는 계단형이라 대부분 HGB 불연속(9-11 점검; Δ0.05R 격자도 15.1%)", p1_k_hat=km, seconds=round(time.time() - t0, 1),
              **pv.start_meta()))
    log(f"[prepare] 초기 상태 {len(I)} (적합 점 최소 {I.n_fit.min()}), 격자 앞쪽 상대 RMSE {err:.4f}, P1 k̂ {km:.3f}")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prepare", "validate", "calib-n", "run", "summarize", "calibrate", "summarize-d2"])
    ap.add_argument("--model", default="original", choices=["original", "calibrated"])
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--prior", default="P1", choices=["P1", "P0"])
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--draws", type=int, default=N_DRAWS)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--pilot", type=int, default=1000)
    ap.add_argument("--order", default="prior", choices=["prior", "priority"])
    ap.add_argument("--max-hours", type=float, default=None)
    ap.add_argument("--fix-rho-av", type=float, default=None, help="D2 진단: 모든 사후 draw 에서 ρ_av 를 이 값으로 대체")
    ap.add_argument("--speeds", type=lambda s: tuple(float(x) for x in s.split(",")), default=None, help="D2 진단 속도 목록 m/s, 예: 1,6,8,12")
    a = ap.parse_args()
    d2 = a.fix_rho_av is not None or a.speeds is not None
    if d2 and (a.cmd != "run" or a.model != "original" or a.prior != "P1" or a.smoke or a.order != "prior" or a.max_hours is not None):
        ap.error("--fix-rho-av·--speeds 는 원 모형 P1 run 전용")
    if a.fix_rho_av is not None and not (np.isfinite(a.fix_rho_av) and a.fix_rho_av >= 0):
        ap.error("--fix-rho-av 는 0 이상의 유한값")
    if a.cmd == "calibrate" or a.model == "calibrated":   # ② 로그는 원 판정 실험 로그와 분리
        global LOG
        LOG = LOG.with_name(LOG.name.replace("p3b", "p3b_cal", 1))
    elif d2 or a.cmd == "summarize-d2":                    # 표준 출력을 받는 p3b_d2_<날짜>.log 와 겹치지 않게
        LOG = LOG.with_name(LOG.name.replace("p3b", "p3b_d2_detail", 1))
    log(f"P3b {a.cmd} | 인자 {vars(a)} | HEAD {(pv.git_head() or '')[:7]} | INTERIM {INTERIM} | OUT {OUT}")
    if a.cmd == "prepare":
        cmd_prepare()
    elif a.cmd == "validate":
        cmd_validate(a.draws)
    elif a.cmd == "calib-n":
        cmd_calib(min(a.draws, 20), a.pilot)
    elif a.cmd == "calibrate":
        cmd_calibrate(a.draws, a.workers)
    elif a.cmd == "summarize-d2":
        cmd_summarize_d2(a.n, a.draws)
    elif d2:
        cmd_run_d2(a.fix_rho_av, a.speeds or D2_SPEEDS, a.n, a.draws)
    elif a.cmd == "run" and a.model == "calibrated":
        cmd_run_cal(a.n, a.draws, a.smoke, a.max_hours)
    elif a.cmd == "run":
        cmd_run(a.prior, a.n, a.draws, a.smoke, a.order, a.max_hours)
    elif a.model == "calibrated":
        summarize_cal(a.n, a.draws, a.smoke)
    else:
        cmd_summarize(a.prior, a.n, a.draws, a.smoke)
    log(f"P3b {a.cmd} 끝")


if __name__ == "__main__":
    main()
