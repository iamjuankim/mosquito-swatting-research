"""회피 시작 이후 모기 운동 모델 (P2e).

a(t) = a_act(t) + k·u_air(x, t)   (logs/decisions.md 'P2e 회피 운동')
 - 수동: 스와터 기류 u_air(swat.flow)에 비례하는 선형 항력 가속, 계수 k [1/s]. 반응 전 구간에서 추정.
 - 능동: 나머지. 방향 d ~ vMF(μ, κ), μ = normalize(w·d_away + (1 − w)·d_bow), 크기 A ~ LogNormal.
   d_away = 원판 최근접점 → 모기(위협 반대), d_bow = 반응 시점 기류 방향.
좌표: world frame(z 위, m). 회피 축 ê = det.away_from_disk(원 논문 escape 축 정의).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import savgol_filter

from swat import detection as det
from swat import flow

G_MPS2 = 9.81
FPS_ESC = 500.0
SG_WINDOW_S = 0.03
SG_ORDER = 3
GUARD_S = 0.010            # 반응 시점 이전 이 시간 안은 k 추정에서 뺀다
WIN_S = 0.040              # 회피 창 길이
MIN_WIN_S = 0.010
MIN_AIR_MPS = 1e-3
MIN_PRE_FRAMES = 10
Z_HAT = np.array([0.0, 0.0, 1.0])


# ------------------------------------------------------------------ 자료
@dataclass
class EscapeTrial:
    trial_id: str
    session: str
    visual: str             # 'opaque' | 'transparent'
    dt: float
    t: np.ndarray           # [s]
    x_m: np.ndarray         # (T, 3) 평활 위치
    v_mps: np.ndarray       # (T, 3)
    a_mps2: np.ndarray      # (T, 3) 운동학 가속
    u_air_mps: np.ndarray   # (T, 3) 스와터 기류 (실험실 기준)
    e_away: np.ndarray      # (T, 3) 회피 축
    i_start: int
    i_end: int
    i_on: int               # 반응 시점 프레임 (없으면 −1)
    p_resp: float
    meta: dict = field(default_factory=dict)


def _interp_gaps(X):
    """내부 결측만 선형 보간 (앞뒤 결측은 그대로)."""
    X = np.asarray(X, float).copy()
    ok = np.isfinite(X).all(axis=1)
    if ok.sum() < 2:
        return X
    i = np.arange(len(X))
    lo, hi = np.nonzero(ok)[0][[0, -1]]
    for j in range(3):
        X[lo:hi + 1, j] = np.interp(i[lo:hi + 1], i[ok], X[ok, j])
    return X


def _sg(X, fps, deriv):
    w = max(SG_ORDER + 2, int(round(SG_WINDOW_S * fps)) | 1)
    return savgol_filter(X, w, SG_ORDER, deriv=deriv, delta=1.0 / fps, axis=0, mode="interp")


def prepare_escape(row, kin, flow_model=None, fps=FPS_ESC, p_resp_min=0.5) -> EscapeTrial | None:
    """공통 스키마 한 행(Exp2) → EscapeTrial. 관찰 창·반응 시점은 det.prepare_trial 과 같은 것을 쓴다."""
    m = flow_model or flow.potential_model()
    tr = det.prepare_trial(row, flow_model=flow.potential_model(), kin=kin, target_fps=fps)
    if tr is None:
        return None
    on = det.changepoint_onset(tr.v_esc, tr.t, tr.i_start, tr.i_end)
    # det.prepare_trial 과 같은 블록 평균 → 같은 시각 격자에서 잘라 맞춘다
    t = np.asarray(row.t_s, float)
    X = np.c_[np.asarray(row.x_m, float), np.asarray(row.y_m, float), np.asarray(row.z_m, float)]
    k = int(round(float(row.fps) / fps)) if float(row.fps) > fps else 1
    n = len(t) // k
    with np.errstate(invalid="ignore"):
        X = np.nanmean(X[: n * k].reshape(n, k, 3), axis=1)
        t = t[: n * k].reshape(n, k).mean(axis=1)
    f = float(row.fps) / k
    lo = int(np.argmin(np.abs(t - tr.t[0])))
    if not np.allclose(t[lo:lo + len(tr.t)], tr.t, atol=1e-9):
        raise ValueError(f"시각 격자 불일치: {row.trial_id}")
    X = _interp_gaps(X[lo:lo + len(tr.t)])
    ok = np.isfinite(X).all(axis=1)
    if ok.sum() < 3 * int(round(SG_WINDOW_S * f)):
        return None
    a0, a1 = np.nonzero(ok)[0][[0, -1]]
    Xs = np.full_like(X, np.nan)
    V = np.full_like(X, np.nan)
    A = np.full_like(X, np.nan)
    Xs[a0:a1 + 1] = _sg(X[a0:a1 + 1], f, 0)
    V[a0:a1 + 1] = _sg(X[a0:a1 + 1], f, 1)
    A[a0:a1 + 1] = _sg(X[a0:a1 + 1], f, 2)
    # 원판 위치 (det.prepare_trial 과 같은 식)
    s_max = float(np.max(kin[1]))
    t_peak = float(kin[0][np.argmax(kin[1])])
    s = np.interp(tr.t + t_peak, kin[0], kin[1], left=0.0, right=float(kin[1][-1]))
    yc = np.nanmin(np.asarray(row.swatter_y_m, float)) + s_max - s + det.HALF_THICK_M
    rel = np.c_[Xs[:, 0] - row.swatter_x_m, Xs[:, 1] - yc, Xs[:, 2] - row.swatter_z_m]
    good = np.isfinite(rel).all(axis=1)
    U = np.zeros_like(rel)
    E = np.full_like(rel, np.nan)
    if good.any():
        U[good], _ = flow.air_velocity(rel[good], s[good] / det.R_DISK_M, m.U_ref_cfd_mps, det.R_DISK_M,
                                       det.NORMAL, model=m)
        E[good] = det.away_from_disk(rel[good])
    i_on = on.k_map if (on.p_resp >= p_resp_min and on.k_map >= 0) else -1
    return EscapeTrial(trial_id=str(row.trial_id), session=str(row.session), visual=tr.visual, dt=1.0 / f,
                       t=tr.t, x_m=Xs, v_mps=V, a_mps2=A, u_air_mps=U, e_away=E, i_start=tr.i_start,
                       i_end=tr.i_end, i_on=i_on, p_resp=on.p_resp, meta=dict(outcome=str(row.outcome)))


# ------------------------------------------------------------------ 수동 계수 k
def pre_onset_frames(tr: EscapeTrial, guard_s=GUARD_S):
    """k 추정용 반응 전 프레임 인덱스 (관찰 창 안, 값이 모두 유한)."""
    hi = tr.i_end if tr.i_on < 0 else tr.i_on - int(round(guard_s / tr.dt))
    idx = np.arange(tr.i_start, hi + 1)
    if len(idx) == 0:
        return idx
    ok = np.isfinite(tr.a_mps2[idx]).all(axis=1) & np.isfinite(tr.u_air_mps[idx]).all(axis=1)
    return idx[ok]


def passive_moments(tr: EscapeTrial, guard_s=GUARD_S):
    """trial 고정효과(성분별 절편)를 뺀 Σ u·a, Σ u·u. k = ΣSxy / ΣSxx."""
    idx = pre_onset_frames(tr, guard_s)
    if len(idx) < MIN_PRE_FRAMES:
        return 0.0, 0.0
    a = tr.a_mps2[idx] - tr.a_mps2[idx].mean(axis=0)
    u = tr.u_air_mps[idx] - tr.u_air_mps[idx].mean(axis=0)
    return float(np.sum(u * a)), float(np.sum(u * u))


def fit_k(trials, guard_s=GUARD_S):
    m = np.array([passive_moments(tr, guard_s) for tr in trials])
    return float(m[:, 0].sum() / m[:, 1].sum()) if m[:, 1].sum() > 0 else np.nan


# ------------------------------------------------------------------ 회피 창
def escape_window(tr: EscapeTrial, win_s=WIN_S):
    """[i_on, i_on + 창] (관찰 끝에서 자름). 짧거나 반응이 없으면 None."""
    if tr.i_on < 0:
        return None
    j = min(tr.i_on + int(round(win_s / tr.dt)), tr.i_end)
    idx = np.arange(tr.i_on, j + 1)
    if (len(idx) - 1) * tr.dt < MIN_WIN_S:
        return None
    if not (np.isfinite(tr.a_mps2[idx]).all() and np.isfinite(tr.e_away[idx]).all()):
        return None
    return idx


def force_split(tr: EscapeTrial, k, win_s=WIN_S):
    """회피 창의 회피 축 성분 적분 (총 공기역학, 수동) [m/s]. 능동 = 총 − 수동."""
    idx = escape_window(tr, win_s)
    if idx is None:
        return None
    e = tr.e_away[idx]
    a_aero = tr.a_mps2[idx] + G_MPS2 * Z_HAT
    tot = float(np.sum(np.einsum("ij,ij->i", a_aero, e)) * tr.dt)
    pas = float(np.sum(np.einsum("ij,ij->i", k * tr.u_air_mps[idx], e)) * tr.dt)
    return tot, pas


def active_vector(tr: EscapeTrial, k, win_s=WIN_S):
    """능동 속도 변화 Δv_act = Δv − k∫u_air dt, 창 길이 [s], 반응 시점의 d_away·d_bow."""
    idx = escape_window(tr, win_s)
    if idx is None:
        return None
    i0, i1 = idx[0], idx[-1]
    dur = (i1 - i0) * tr.dt
    dv = tr.v_mps[i1] - tr.v_mps[i0]
    dv_act = dv - k * np.sum(tr.u_air_mps[idx[:-1]], axis=0) * tr.dt
    u0 = tr.u_air_mps[i0]
    nu = np.linalg.norm(u0)
    return dict(dv_act=dv_act, dv=dv, dur_s=dur, d_away=tr.e_away[i0],
                d_bow=u0 / nu if nu >= MIN_AIR_MPS else None, e_axis=tr.e_away[i0])


# ------------------------------------------------------------------ von Mises–Fisher (S²)
def _unit(X):
    X = np.asarray(X, float)
    return X / np.linalg.norm(X, axis=-1, keepdims=True)


def vmf_logpdf(x, mu, kappa):
    """S² 위 vMF 로그 밀도: log κ − log 2π − κ − log(1 − e^{−2κ}) + κ μ·x."""
    c = np.einsum("...i,...i->...", np.asarray(x, float), np.asarray(mu, float))
    return np.log(kappa) - np.log(2 * np.pi) - kappa - np.log1p(-np.exp(-2 * kappa)) + kappa * c


def vmf_cos_cdf(c, kappa):
    """cosθ = μ·x 의 누적분포: (e^{κc} − e^{−κ}) / (e^{κ} − e^{−κ})."""
    c = np.asarray(c, float)
    return np.expm1(kappa * (c + 1)) / np.expm1(2 * kappa)


def vmf_cos_quantile(p, kappa):
    return np.log1p(np.asarray(p, float) * np.expm1(2 * kappa)) / kappa - 1


def sample_vmf(mu, kappa, rng, n=None):
    """vMF 표본 (S², 역CDF). mu (3,) 또는 (N, 3)."""
    mu = _unit(np.atleast_2d(mu))
    N = len(mu) if n is None else n
    mu = np.broadcast_to(mu, (N, 3))
    c = vmf_cos_quantile(rng.uniform(size=N), kappa)
    phi = rng.uniform(0, 2 * np.pi, size=N)
    ref = np.where(np.abs(mu[:, :1]) < 0.9, np.array([[1.0, 0, 0]]), np.array([[0, 1.0, 0]]))
    b1 = _unit(np.cross(mu, ref))
    b2 = np.cross(mu, b1)
    s = np.sqrt(np.clip(1 - c**2, 0, 1))
    return c[:, None] * mu + s[:, None] * (np.cos(phi)[:, None] * b1 + np.sin(phi)[:, None] * b2)


def mix_mean(w, d_away, d_bow):
    """μ = normalize(w·d_away + (1 − w)·d_bow). 합이 0 에 가까우면 d_away."""
    m = w * np.asarray(d_away, float) + (1 - w) * np.asarray(d_bow, float)
    nm = np.linalg.norm(m, axis=-1, keepdims=True)
    return np.where(nm > 1e-9, m / np.maximum(nm, 1e-12), d_away)


def fit_vmf_mix(d, d_away, d_bow, w_fixed=None):
    """(w, κ) 최대우도. w_fixed 를 주면 κ 만 (비교 모형). 반환: dict(w, kappa, loglik)."""
    from scipy.optimize import minimize

    d, Da, Db = _unit(d), _unit(d_away), _unit(d_bow)

    def nll(p):
        w = 1 / (1 + np.exp(-p[0])) if w_fixed is None else w_fixed
        return -np.sum(vmf_logpdf(d, mix_mean(w, Da, Db), np.exp(p[-1])))

    best = None
    starts = [[0.0]] if w_fixed is not None else [[-2.0], [0.0], [2.0]]
    for s0 in starts:
        x0 = ([] if w_fixed is not None else s0) + [np.log(2.0)]
        r = minimize(nll, np.array(x0, float), method="Nelder-Mead", options=dict(xatol=1e-8, fatol=1e-10, maxiter=4000))
        if best is None or r.fun < best.fun:
            best = r
    w = float(1 / (1 + np.exp(-best.x[0]))) if w_fixed is None else float(w_fixed)
    return dict(w=w, kappa=float(np.exp(best.x[-1])), loglik=float(-best.fun))


# ------------------------------------------------------------------ 시뮬레이션 (P3b 용)
@dataclass
class EscapeParams:
    k_per_s: float
    w: float
    kappa: float
    log_a_mu: float         # ln(능동 가속 크기 [m/s²]) 평균
    log_a_sd: float


def draw_active(p: EscapeParams, d_away, d_bow, rng, n=None):
    """능동 가속 벡터 표본 [m/s²] = A·d, d ~ vMF(μ, κ), A ~ LogNormal."""
    d_bow = d_away if d_bow is None else d_bow
    mu = mix_mean(p.w, d_away, d_bow)
    d = sample_vmf(mu, p.kappa, rng, n)
    A = np.exp(rng.normal(p.log_a_mu, p.log_a_sd, size=len(d)))
    return A[:, None] * d


def simulate_escape(x0_m, v0_mps, a_act_mps2, u_air_fn, t0_s, dur_s, dt_s, k_per_s):
    """회피 시작부터 dur_s 동안 전진 오일러: v̇ = a_act + k·u_air(x, t). u_air_fn(x (3,), t) → (3,) m/s."""
    n = int(round(dur_s / dt_s))
    X = np.empty((n + 1, 3))
    V = np.empty((n + 1, 3))
    X[0], V[0] = x0_m, v0_mps
    for i in range(n):
        t = t0_s + i * dt_s
        acc = np.asarray(a_act_mps2, float) + k_per_s * np.asarray(u_air_fn(X[i], t), float)
        V[i + 1] = V[i] + acc * dt_s
        X[i + 1] = X[i] + V[i] * dt_s
    return X, V
