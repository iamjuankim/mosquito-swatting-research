"""스와터(손) 타격 → 모기 감지·회피·포획 몬테카를로 시뮬레이터 (P3b).

설계·근거: logs/decisions.md 'P3b 시뮬레이터 설계' (P3a 규칙 H2–H6, P2d-R G2 결정 6번, H5·P2e 사전 고정).

한 궤적의 흐름 (실시간 t, 스트로크 끝 = 최대 신장에서 끝)
    스와터: Exp2 CFD 운동학 s_kin(τ) 를 시간 척도 f = v / U_cfd 로 압축, τ = f·t.
    기류  : P1d 대리모델 무차원 장(FlowGrid) × U_cfd × f^n × r(β)  (박수는 두 원판 장의 합)
    감지  : detection.drive(sum) 와 같은 식을 시간 진행으로, 실행 최댓값 M 이 문턱 Θ 를 넘거나 자발 반응.
    운동  : 감지 전 등속 + k·u_air, 감지 후 40 ms 동안 능동 가속 A·d (P2e) + k·u_air.
    포획  : 원판 부피 진입 또는 원판면 통과(스윕) = 접촉, 다공 원판은 확률 β 로 통과.
좌표: 실험실(z 위, m), 원점 = 원판 A 시작 중심, 원판 A 는 −y 로 이동.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import ndtr, ndtri

from swat import detection as det
from swat import escape as E
from swat import porous

R_M = det.R_DISK_M
H_M = det.HALF_THICK_M
N_A = np.array([0.0, -1.0, 0.0])
N_B = np.array([0.0, 1.0, 0.0])
K_MAX_PER_S = 8.2                       # P2e 수동 k 민감도 상한 (decisions.md :226)
LOG10L_TWILIGHT = float(np.log10(det.LUMINANCE_CDM2["twilight"]))
DT_MAX_S = 0.002
DX_MAX_M = 0.005
ESC_WIN_S = E.WIN_S                     # 능동 가속 지속 (P2e 창 40 ms)


# ------------------------------------------------------------------ 기류 격자
class FlowGrid:
    """대리모델 무차원 장 (u_ξ, u_ρ)(ξ, ρ*, s*) 를 격자에 미리 계산해 선형 보간. 격자 밖은 0 [가정]."""

    def __init__(self, xi, rho, s, u_xi, u_rho):
        from scipy.interpolate import RegularGridInterpolator

        self.xi, self.rho, self.s = (np.asarray(a, float) for a in (xi, rho, s))
        self.u_xi, self.u_rho = np.asarray(u_xi, np.float32), np.asarray(u_rho, np.float32)
        kw = dict(bounds_error=False, fill_value=0.0)
        self._fx = RegularGridInterpolator((self.xi, self.rho, self.s), self.u_xi, **kw)
        self._fr = RegularGridInterpolator((self.xi, self.rho, self.s), self.u_rho, **kw)

    @classmethod
    def build(cls, fm, d_xi=0.1, d_rho=0.1, n_s=106, xi_lim=(-8.0, 10.0), rho_max=6.0):
        xi = np.round(np.arange(xi_lim[0], xi_lim[1] + 1e-9, d_xi), 6)
        rho = np.round(np.arange(0.0, rho_max + 1e-9, d_rho), 6)
        s = np.linspace(0.0, fm.s_star_max, n_s)
        X, Rr, S = np.meshgrid(xi, rho, s, indexing="ij")
        ux, ur, _ = fm.predict(X.ravel(), Rr.ravel(), S.ravel())
        return cls(xi, rho, s, np.reshape(ux, X.shape), np.reshape(ur, X.shape))

    def save(self, path):
        np.savez_compressed(path, xi=self.xi, rho=self.rho, s=self.s, u_xi=self.u_xi, u_rho=self.u_rho)

    @classmethod
    def load(cls, path):
        z = np.load(path)
        return cls(z["xi"], z["rho"], z["s"], z["u_xi"], z["u_rho"])

    def velocity(self, rel_m, s_star, normal):
        """원판 중심 기준 상대 위치 (N, 3) → 무차원 공기 속도 벡터 (N, 3) (× U 를 곱하면 m/s)."""
        xi, rho_s, e_rho, n = _disk_frame(rel_m, normal)
        pts = np.c_[xi, rho_s, np.full(len(xi), np.clip(s_star, 0.0, self.s[-1]))]
        return self._fx(pts)[:, None] * n[None, :] + self._fr(pts)[:, None] * e_rho


def _disk_frame(rel, n):
    along = rel @ n
    radial = rel - along[:, None] * n[None, :]
    rho = np.linalg.norm(radial, axis=1)
    e_rho = np.where(rho[:, None] > 1e-12, radial / np.maximum(rho, 1e-12)[:, None], 0.0)
    return along / R_M, rho / R_M, e_rho, n


def _theta_deg(rel, n):
    """det.disk_view 와 같은 각지름 (deg)."""
    d = np.linalg.norm(rel, axis=1)
    cos_a = np.where(d > 0, np.abs(rel @ n) / np.maximum(d, 1e-12), 1.0)
    return np.degrees(2 * np.arctan2(R_M * np.sqrt(cos_a), d))


# ------------------------------------------------------------------ 조건·입력
@dataclass(frozen=True)
class Condition:
    name: str
    v_mps: float                 # 최고 손(원판) 속도
    n_exp: float = 1.0           # bow wave 속도 지수 (H2)
    clap: bool = False           # 박수 (H3)
    c_b: float = 1.0             # 밝은 손 시각 이득 (H4)
    phi: float = 0.0             # 공극률 (H5·H6)
    att_mult: float = 1.0        # 감쇠 계수 배수 m (H5 민감도)
    att_model: str = "ustar"     # 'ustar' (주) | 'drag' (항력 비 시나리오)
    k_fixed: float | None = None  # None 이면 draw 별 k ~ U(0, 8.2)
    mc_set: str = "a"            # 공통 난수 집합 (H6 독립 집합 'b')
    theta_shift: float = 0.0     # 문턱 로그 위치 이동 Δ (② 보정 모형, decisions.md 'H2 세 갈래 판정')

    def residual(self):
        if self.phi == 0:
            return 1.0
        if self.att_model == "drag":
            return float(porous.cd_momentum(porous.u_star_from_beta(self.phi)) / porous.CD_SOLID)
        return float(porous.residual_ratio(self.phi, self.att_mult))

    def model_prediction(self, U_cfd):
        """[모델 예측] 표기 대상 (P3a 표기 규칙: 1 m/s 외 속도, 공극률 > 0.4, 박수, 밝은 손)."""
        return bool(abs(self.v_mps - 1.0) > 1e-9 and abs(self.v_mps - U_cfd) > 1e-9) or self.phi > 0.4 or self.clap or self.c_b != 1.0


@dataclass
class Batch:
    """궤적 N 개의 초기 상태와 공통 난수."""
    rel0_m: np.ndarray           # (N, 3) 모기 위치 − 원판 A 시작 중심
    v0_mps: np.ndarray           # (N, 3)
    s_start_m: np.ndarray        # (N,) 시작 시점 스와터 이동 거리 (위상)
    transparent: np.ndarray      # (N,) bool
    eta: np.ndarray              # (N,) 세션 효과 (로그 문턱 위치)
    u: np.ndarray                # (N, 6) 균등 난수: 문턱, 자발, vMF cos, vMF φ, 다공 통과, (예비)
    z_a: np.ndarray              # (N,) 능동 크기 정규 난수
    t_end_kin_s: np.ndarray | None = None   # (N,) 관찰 창 끝 (kin 시간, 재현 검증용)

    @property
    def n(self):
        return len(self.rel0_m)


def common_randoms(rng, n):
    return rng.uniform(size=(n, 6)), rng.normal(size=n)


# ------------------------------------------------------------------ 시뮬레이션
def _vmf_from_uniform(mu, kappa, u_c, u_phi):
    c = E.vmf_cos_quantile(np.clip(u_c, 1e-12, 1 - 1e-12), kappa)
    phi = 2 * np.pi * u_phi
    ref = np.where(np.abs(mu[:, :1]) < 0.9, np.array([[1.0, 0, 0]]), np.array([[0, 1.0, 0]]))
    b1 = E._unit(np.cross(mu, ref))
    b2 = np.cross(mu, b1)
    s = np.sqrt(np.clip(1 - c**2, 0, 1))
    return c[:, None] * mu + s[:, None] * (np.cos(phi)[:, None] * b1 + np.sin(phi)[:, None] * b2)


def _lagged(hist, i, i0, f):
    """hist (N, T) 의 i − f 프레임 값 (선형 보간, 행 시작 i0 이전은 시작 값)."""
    j0 = int(np.floor(f))
    w = f - j0
    rows = np.arange(hist.shape[0])
    a = hist[rows, np.maximum(i - j0, i0)]
    b = hist[rows, np.maximum(i - j0 - 1, i0)]
    return (1 - w) * a + w * b


def simulate(batch: Batch, cond: Condition, P: dict, esc: dict, k_per_s: float, kin, grid: FlowGrid,
             U_cfd: float, A_ref: float, log10L: float = LOG10L_TWILIGHT, dt_max_s=DT_MAX_S, dx_max_m=DX_MAX_M,
             record=False):
    """궤적 N 개 → dict(captured, t_contact_s, t_det_s, t_end_s) — 시각은 스트로크 끝(최대 신장) 기준 [s], 없으면 +inf.

    P  : 감지 파라미터 (det.params_from_draw 형식; a_sp·b_sp 는 종 0 = Anopheles)
    esc: {'opaque': EscapeParams, 'transparent': EscapeParams}
    kin: (τ [s], s [m]) Exp2 CFD 운동학 (단조)
    record=True: 감지를 끄고(문턱·자발 반응 없음) 실행 최댓값 M (N, T)·시작 프레임 i0·시각 t_s (T,) 를 함께 돌려준다.
        감지 전 운동은 문턱과 무관하므로 detect_given_shift 로 모든 Δ 의 감지를 정확히 다시 계산할 수 있다.
    """
    tau_k, s_k = kin
    f = cond.v_mps / U_cfd
    tau_peak = float(tau_k[np.argmax(s_k)])
    s_max = float(s_k.max())
    tau0 = tau_k[np.minimum(np.searchsorted(s_k, np.maximum(batch.s_start_m, 1e-4), side="right"), np.argmax(s_k))]
    dt = min(dt_max_s, dx_max_m / cond.v_mps)
    t0, t_end = float(tau0.min()) / f, tau_peak / f
    n_steps = int(np.ceil((t_end - t0) / dt))
    N = batch.n
    i0 = np.minimum(np.ceil((tau0 / f - t0) / dt - 1e-9).astype(int), n_steps - 1)
    t_of = lambda i: t0 + i * dt  # noqa: E731
    s_of = lambda t: float(np.interp(min(f * t, tau_peak), tau_k, s_k))  # noqa: E731
    U_air = U_cfd * f ** cond.n_exp * cond.residual()
    y_c = -s_max - H_M                                   # 박수: 두 앞면이 만나는 면

    def centers(t):
        s = s_of(t)
        cA = np.array([0.0, -s, 0.0])
        return s, cA, np.array([0.0, 2 * y_c + s, 0.0])   # 원판 B = y_c 면 대칭

    def air(X, t):
        s, cA, cB = centers(t)
        u = grid.velocity(X - cA, s / R_M, N_A)
        if cond.clap:
            u = u + grid.velocity(X - cB, s / R_M, N_B)
        return u * U_air

    # 감지 파라미터
    g = 1.0 / (1.0 + np.exp(-(P["a_sp"][0] + P["b_sp"][0] * log10L)))
    c_vis = np.where(batch.transparent, P["c_t"], 1.0) * cond.c_b * g
    air_scale = det.V_REF / A_ref / U_cfd                # |u| [m/s] → 구동 단위 (prepare_trial: air = |u|/U_cfd)
    f_v, f_a = P["tau_v"] / dt, P["tau_a"] / dt
    u_rand, z_a = batch.u, batch.z_a
    t_sp = -np.log(np.clip(u_rand[:, 1], 1e-300, 1.0)) / P["lam0"]
    k = k_per_s

    X = batch.rel0_m.astype(float).copy()
    V = batch.v0_mps.astype(float).copy()
    T = n_steps + 1
    hist_thA = np.zeros((N, T))
    hist_thB = np.zeros((N, T))
    hist_V = np.zeros((N, T))
    hist_air = np.zeros((N, T))
    M = np.zeros(N)
    Theta = np.full(N, np.inf)
    detected = np.zeros(N, bool)
    t_det = np.full(N, np.inf)
    a_act = np.zeros((N, 3))
    contact = np.zeros(N, bool)
    t_contact = np.full(N, np.inf)
    captured = np.zeros(N, bool)
    started = np.zeros(N, bool)
    phi = cond.phi
    rows = np.arange(N)
    eta = batch.eta + cond.theta_shift
    hist_M = np.zeros((N, T)) if record else None

    for i in range(n_steps):
        t = t_of(i)
        s, cA, cB = centers(t)
        act = (i >= i0) & ~captured
        relA = X - cA
        thA = _theta_deg(relA, N_A)
        hist_thA[:, i] = thA
        thd = np.where(i > i0, (thA - hist_thA[rows, np.maximum(i - 1, 0)]) / dt, 0.0)
        Vf = det.C1_MV_PER_DEG_S * np.maximum(thd, 0.0)
        if cond.clap:
            thB = _theta_deg(X - cB, N_B)
            hist_thB[:, i] = thB
            thdB = np.where(i > i0, (thB - hist_thB[rows, np.maximum(i - 1, 0)]) / dt, 0.0)
            Vf = Vf + det.C1_MV_PER_DEG_S * np.maximum(thdB, 0.0)
        hist_V[:, i] = Vf
        u_air = air(X, t)
        hist_air[:, i] = np.linalg.norm(u_air, axis=1) * air_scale
        # 구동 (지연)
        th_l = _lagged(hist_thA, i, i0, f_v)
        Sz = det.C2_MV * np.exp(-((np.log(np.maximum(th_l, 1e-6)) - np.log(P["theta_pk"])) ** 2) / (2 * P["sigma_theta"] ** 2))
        if cond.clap:
            thB_l = _lagged(hist_thB, i, i0, f_v)
            Sz = Sz + det.C2_MV * np.exp(-((np.log(np.maximum(thB_l, 1e-6)) - np.log(P["theta_pk"])) ** 2)
                                         / (2 * P["sigma_theta"] ** 2))
        D = c_vis * P["w_v"] * (_lagged(hist_V, i, i0, f_v) + Sz / P["rho_vs"]) \
            + P["w_v"] * P["rho_av"] * _lagged(hist_air, i, i0, f_a)
        new = act & ~started
        if new.any():                                   # 시작 시점: 문턱을 M(시작) 에 대해 왼쪽 절단
            M[new] = D[new]
            u0 = ndtr((np.log(M[new] + 1e-12) - eta[new]) / P["sigma"])
            q = np.clip(u0 + (1 - u0) * u_rand[new, 0], 0.0, 1 - 1e-12)
            Theta[new] = np.exp(eta[new] + P["sigma"] * ndtri(q))
            started |= new
        M = np.where(act, np.maximum(M, D), M)
        if record:
            hist_M[:, i] = M
        trig = act & ~detected & (i > i0) & ((M >= Theta) | (t - (t0 + i0 * dt) >= t_sp)) & (not record)
        if trig.any():
            detected |= trig
            t_det[trig] = t
            d_away = det.away_from_disk(relA[trig], N_A)
            if cond.clap:                               # 더 가까운 원판 기준
                relB = X[trig] - cB
                nearB = np.linalg.norm(relB, axis=1) < np.linalg.norm(relA[trig], axis=1)
                d_away = np.where(nearB[:, None], det.away_from_disk(relB, N_B), d_away)
            ua = u_air[trig]
            nu = np.linalg.norm(ua, axis=1, keepdims=True)
            d_bow = np.where(nu > E.MIN_AIR_MPS, ua / np.maximum(nu, 1e-12), d_away)
            tr = batch.transparent[trig]
            w = np.where(tr, esc["transparent"].w, esc["opaque"].w)
            kap = np.where(tr, esc["transparent"].kappa, esc["opaque"].kappa)
            mu = E.mix_mean(w[:, None], d_away, d_bow)
            d = np.empty_like(mu)
            for kv in np.unique(kap):
                m_ = kap == kv
                d[m_] = _vmf_from_uniform(mu[m_], kv, u_rand[trig][m_, 2], u_rand[trig][m_, 3])
            la_mu = np.where(tr, esc["transparent"].log_a_mu, esc["opaque"].log_a_mu)
            la_sd = np.where(tr, esc["transparent"].log_a_sd, esc["opaque"].log_a_sd)
            a_act[trig] = np.exp(la_mu + la_sd * z_a[trig])[:, None] * d
        # 운동
        acc = k * u_air + np.where((detected & (t - t_det < ESC_WIN_S))[:, None], a_act, 0.0)
        Xn = X + V * dt
        Vn = V + acc * dt
        t1 = t + dt
        s1, cA1, cB1 = centers(t1)
        hit = _swept_contact(relA, Xn - cA1, N_A)
        if cond.clap:
            hit |= _swept_contact(X - cB, Xn - cB1, N_B)
        newc = act & ~contact & hit
        if newc.any():
            contact |= newc
            t_contact[newc] = t1
            cap = newc & (u_rand[:, 4] >= phi)
            captured |= cap
        X = np.where(act[:, None], Xn, X)
        V = np.where(act[:, None], Vn, V)
    t_end_obs = np.full(N, np.inf) if batch.t_end_kin_s is None else batch.t_end_kin_s / f - t_end
    out = dict(captured=captured, t_contact_s=t_contact - t_end, t_det_s=t_det - t_end, t_end_obs_s=t_end_obs,
               dt_s=dt, n_steps=n_steps)
    if record:
        out.update(M=hist_M[:, :n_steps], i0=i0, t_s=t0 + np.arange(n_steps) * dt - t_end, t_sp_s=t_sp)
    return out


def detect_given_shift(rec, batch: Batch, P: dict, shift: float):
    """record=True 결과 + 문턱 이동 Δ → 감지 시각 t_det_s (스트로크 끝 기준, 없으면 +inf).

    simulate 의 감지 규칙과 같다: 시작 프레임 i0 에서 M(i0) 에 대해 왼쪽 절단한 Θ = exp(η + Δ + σ·z),
    i > i0 에서 처음으로 M ≥ Θ 또는 (t − t(i0)) ≥ t_sp. 감지 전 궤적은 Δ 와 무관하다(감지 전 운동 = 등속 + k·u_air).
    """
    M, i0, t = rec["M"], rec["i0"], rec["t_s"]
    N, T = M.shape
    rows = np.arange(N)
    eta = batch.eta + shift
    m0 = M[rows, i0]
    u0 = ndtr((np.log(m0 + 1e-12) - eta) / P["sigma"])
    q = np.clip(u0 + (1 - u0) * batch.u[:, 0], 0.0, 1 - 1e-12)
    theta = np.exp(eta + P["sigma"] * ndtri(q))
    idx = np.arange(T)[None, :]
    after = idx > i0[:, None]
    trig = after & ((M >= theta[:, None]) | ((t[None, :] - t[i0][:, None]) >= rec["t_sp_s"][:, None]))
    first = np.where(trig.any(axis=1), trig.argmax(axis=1), -1)
    return np.where(first >= 0, t[np.maximum(first, 0)], np.inf)


def _swept_contact(rel_old, rel_new, n):
    """원판 부피 진입 또는 한 Δt 동안 원판면(두께 2h) 통과 (통과 지점 ρ ≤ R)."""
    a0, a1 = rel_old @ n, rel_new @ n
    rad0 = rel_old - a0[:, None] * n[None, :]
    rad1 = rel_new - a1[:, None] * n[None, :]
    inside = (np.abs(a1) <= H_M) & (np.linalg.norm(rad1, axis=1) <= R_M)
    crossed = ((a0 > H_M) & (a1 < -H_M)) | ((a0 < -H_M) & (a1 > H_M))
    q = np.clip(a0 / np.where(a0 != a1, a0 - a1, 1.0), 0.0, 1.0)
    rho_x = np.linalg.norm((1 - q)[:, None] * rad0 + q[:, None] * rad1, axis=1)
    return inside | (crossed & (rho_x <= R_M))


def outcomes(res, window=False):
    """궤적 결과 → (포획, 접촉 전 감지, t_det 절단) 배열. window=True 면 관찰 창 끝 이후 감지는 무시(재현 검증)."""
    t_det = res["t_det_s"]
    lim = np.minimum(res["t_contact_s"], 0.0)
    if window:
        lim = np.minimum(lim, res["t_end_obs_s"])
    det_ok = t_det < lim
    return res["captured"], det_ok, np.minimum(t_det, lim)
