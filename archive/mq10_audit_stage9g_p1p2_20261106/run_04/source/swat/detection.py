"""모기 감지 모델 (시각 looming + 스와터 기류) — P2b.

설계·근거: logs/decisions.md (P2b), logs/phase_reports/p2b.md.

구조
    V(t)  = C1 · θ̇⁺(t − τ_v)                                  각속도 선형 (Ache 2019 LC4 성분)
    Sz(t) = C2 · exp(−[ln θ(t − τ_v) − ln θ_pk]² / (2σ_θ²))     각크기 (ln θ) 가우시안 (Ache 2019 LPLC2 성분)
    Air(t)= (A(t − τ_a) / A_ref) · V_ref                       기류 (A = |u_air|/U_ref, swat.flow)
    통합 규칙 sum / max / rel → 구동 D(t) ≥ 0
    M(t) = 단조 매끄러운 실행 최댓값, Θ_i ~ LogNormal(η_session, σ) → P(반응 ≤ t) = Φ((ln M − η)/σ)
    자발 반응 경쟁 위험 λ0, 관찰 시작에서 왼쪽 절단
    우도: 반응 trial Σ_k P_k q_k (q = 반응 시점 사후분포), 미반응 S(t_end), 혼합 p·(…) + (1 − p)·S(t_end)

커넥톰 사전정보 P0/P1/P2 는 ρ_vs = w_v/w_s 와 ρ_av(기준 자극에서 기류 구동/LC4 구동) 의 사전분포만 다르다.
MCMC 는 nutpie 만 쓴다 (sample_nutpie).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import numpy as np
from scipy.special import erfc as _np_erfc
from scipy.special import ndtr, ndtri
from scipy.signal import savgol_filter

from swat import flow

# ------------------------------------------------------------------ 상수 (근거: decisions.md P2b)
C1_MV_PER_DEG_S = 2.567e-4      # Ache 2019 e3: V_LC4 = C1·θ̇
C2_MV = 1.7                     # Ache 2019 e4: V_LPLC2 진폭
THETA_PK_FLY_DEG = 42.0         # Ache 2019 e4: C3
SIGMA_THETA_FLY = 0.52          # Ache 2019 e4: C4 (ln θ 척도)
R_DISK_M = 0.05
HALF_THICK_M = 0.005
NORMAL = (0.0, -1.0, 0.0)       # 스와터 이동 방향 (실험실 −y)
LUMINANCE_CDM2 = {"dark": 0.0201, "twilight": 0.347, "sunrise": 28.6, "overcast": 781.0}
REF_DIST_M = 0.10               # 기준 자극: 원판 축 위, 중심에서 2R 앞
KAPPA = 100.0                   # 매끄러운 최댓값 날카로움 (구동 단위 ~1 에서 오차 ≤ ln(k)/κ)
D_CLIP = 6.0                    # e^{κD} 넘침 방지 (Φ(ln 6 / σ) ≈ 1 이라 영향 없음)
RULES = ("sum", "max", "rel")
SCHEMA_COLUMNS = ["dataset", "species", "light", "disk_type", "with_disk", "swatter_on", "condition_disk", "session",
                  "trial_id", "trigger_index", "obj_id", "outcome", "collision_manual", "in_trigger_set", "in_track_set",
                  "fps", "age_days", "trigger_frame", "kin_frame_delay", "frame_zero", "pos_source", "swatter_virtual",
                  "swatter_x_m", "swatter_z_m", "frame", "t_s", "x_m", "y_m", "z_m", "swatter_y_m", "outcome_label",
                  "frame_body_touch", "frame_leg_touch", "grade", "sex", "n_obj_in_trigger", "same_file_as_2024_exp1",
                  "subset_2022_paper", "subset_2024_exp1_paper", "subset_2024_exp2_paper"]   # P1b 공통 스키마 (p1b.md 2절)
PRIOR_LEVELS = ("P0", "P1", "P2")

# (평균, SD) — 로그 척도. P 단계는 이 두 파라미터만 다르다.
PRIORS = {
    "P0": {"rho_vs": (0.0, 1.5), "rho_av": (0.0, 1.5)},
    "P1": {"rho_vs": (np.log(1.9), 0.5), "rho_av": (np.log(0.15), 1.5)},
    "P2": {"rho_vs": (np.log(2.6), 0.15), "rho_av": (np.log(0.15), 0.25)},
}
COMMON_PRIORS = {
    "w_v": (0.0, 1.0),                         # LogNormal(로그 평균, SD)
    "theta_pk": (np.log(THETA_PK_FLY_DEG), 0.35),
    "sigma_theta": (np.log(SIGMA_THETA_FLY), 0.3),
    "tau_v": (np.log(0.030), 0.5),
    "tau_a": (np.log(0.030), 0.5),
    "lam0": (np.log(0.05), 1.5),
    "sigma_hn": 1.0, "c_t_beta": (2.0, 2.0), "k_h_beta": (2.0, 2.0),
    "a_sp": (0.0, 1.5), "b_sp_hn": 1.0, "sigma_sess_hn": 0.5,
}

V_REF = C1_MV_PER_DEG_S * np.degrees(2 * R_DISK_M * 1.0 / (REF_DIST_M**2 + R_DISK_M**2))  # 1 m/s 기준


# ------------------------------------------------------------------ 기하
def disk_view(rel_pos_m, normal=NORMAL, R_m=R_DISK_M):
    """원판 중심 기준 상대 위치 → (θ [deg], 거리 d [m], ξ [m], ρ [m], e_ρ, n).

    θ = 2·atan(R·√|cos α| / d): 기울어진 원판의 투영 면적을 같은 면적의 원으로 본 각지름 근사.
    """
    rel = np.atleast_2d(np.asarray(rel_pos_m, float))
    n = np.asarray(normal, float)
    n = n / np.linalg.norm(n)
    along = rel @ n
    radial = rel - along[:, None] * n[None, :]
    rho = np.linalg.norm(radial, axis=1)
    d = np.linalg.norm(rel, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos_a = np.where(d > 0, np.abs(along) / d, 1.0)
        e_rho = np.where(rho[:, None] > 1e-12, radial / rho[:, None], 0.0)
        theta = np.degrees(2 * np.arctan2(R_m * np.sqrt(cos_a), d))
    return theta, d, along, rho, e_rho, n


def angular_size_deg(rel_pos_m, normal=NORMAL, R_m=R_DISK_M):
    return disk_view(rel_pos_m, normal, R_m)[0]


def away_from_disk(rel_pos_m, normal=NORMAL, R_m=R_DISK_M, half_m=HALF_THICK_M):
    """원판(반지름 R, 두께 2h)의 최근접점 → 모기 방향 단위벡터. 원판 안이면 법선 방향."""
    _, _, along, rho, e_rho, n = disk_view(rel_pos_m, normal, R_m)
    xa = along - np.clip(along, -half_m, half_m)
    xr = rho - np.minimum(rho, R_m)
    vec = xa[:, None] * n[None, :] + xr[:, None] * e_rho
    norm = np.linalg.norm(vec, axis=1, keepdims=True)
    fallback = np.where(along[:, None] >= 0, n[None, :], -n[None, :])
    return np.where(norm > 1e-12, vec / np.maximum(norm, 1e-12), fallback)


def smooth_positions(X, fps, window_s=0.05):
    """내부 결측을 선형 보간한 뒤 Savitzky–Golay(2차) 평활. 앞뒤 결측은 그대로 둔다."""
    X = np.asarray(X, float).copy()
    ok = np.isfinite(X).all(axis=1)
    if ok.sum() < 3:
        return X
    idx = np.arange(len(X))
    lo, hi = np.nonzero(ok)[0][[0, -1]]
    for j in range(3):
        X[lo:hi + 1, j] = np.interp(idx[lo:hi + 1], idx[ok], X[ok, j])
    w = max(5, int(round(window_s * fps)) | 1)
    seg = X[lo:hi + 1]
    if len(seg) > w:
        X[lo:hi + 1] = savgol_filter(seg, w, 2, axis=0, mode="interp")
    return X


# ------------------------------------------------------------------ trial 자극
@dataclass
class TrialData:
    trial_id: str
    session: str
    species: str
    light: str
    visual: str            # 'opaque' | 'transparent'
    hollow: bool
    swatter_on: bool
    dt: float
    t: np.ndarray          # [s], 0 = 최대 신장 (스키마 t_s 기준)
    theta: np.ndarray      # [deg]
    theta_dot: np.ndarray  # [deg/s] (부호 포함)
    air: np.ndarray        # |u_air| / U_ref
    extrap: np.ndarray     # 기류 외삽 표시
    v_esc: np.ndarray      # 회피 속도 [m/s] (원판 최근접점 → 모기 방향 성분)
    valid: np.ndarray      # 위치가 있는 프레임
    i_start: int           # 관찰 시작 (스와터 운동 시작 또는 기록 시작)
    i_end: int             # 관찰 끝 (최대 신장·충돌·트랙 끝 중 이른 것)
    collided: bool
    meta: dict = field(default_factory=dict)


def visual_category(disk_type: str, condition_disk: str = "") -> str | None:
    s = f"{disk_type} {condition_disk}".lower()
    if "no disk" in s or "nodisk" in s:
        return None
    if "black" in s or "opaque" in s:
        return "opaque"
    if "transparent" in s:
        return "transparent"
    return None


def prepare_trial(row, *, flow_model=None, shift_frames: float = 0.0, kin=None, target_fps: float | None = None,
                  pre_s: float = 0.15, smooth_s: float = 0.05, U_ref_mps: float | None = None,
                  collision_t_s: float | None = None) -> TrialData | None:
    """P1b 공통 스키마 한 행 → TrialData.

    dataset '2022_looming'·'synthetic': swatter_y_m(앞면)을 프레임마다 쓰고 shift_frames 만큼 이동(음수 = 앞당김).
    dataset '2024_exp2': kin=(time_s, 이동 거리 m) 운동학을 frame_zero(최대 신장, t = 0)에 맞춰 쓰고,
                         target_fps 로 위치를 블록 평균해 낮춘다.
    스와터 off(대조군): 원판은 시작 위치에 정지, 기류 0, 관찰 창은 가상 스와터 운동 구간.
    """
    m = flow_model or flow.potential_model()
    U = U_ref_mps or m.U_ref_cfd_mps
    vis = visual_category(str(row.disk_type), str(getattr(row, "condition_disk", "")))
    if vis is None:
        return None
    t = np.asarray(row.t_s, float)
    X = np.c_[np.asarray(row.x_m, float), np.asarray(row.y_m, float), np.asarray(row.z_m, float)]
    fps = float(row.fps)
    ys = np.asarray(row.swatter_y_m, float)
    if row.dataset == "2024_exp2":
        if kin is None:
            raise ValueError("Exp2 는 운동학 kin=(time_s, s_m) 이 필요하다")
        if target_fps and fps > target_fps:
            k = int(round(fps / target_fps))
            n = len(t) // k
            with np.errstate(invalid="ignore"):
                X = np.nanmean(X[: n * k].reshape(n, k, 3), axis=1)
                t = t[: n * k].reshape(n, k).mean(axis=1)
            fps = fps / k
        s_max = float(np.max(kin[1]))
        t_peak = float(kin[0][np.argmax(kin[1])])
        s = np.interp(t + t_peak, kin[0], kin[1], left=0.0, right=float(kin[1][-1]))
        y0 = np.nanmin(ys) + s_max
        yf = y0 - s
    else:
        frames = np.arange(len(t), dtype=float)
        if shift_frames:
            ys = np.interp(frames - shift_frames, frames, ys)
        y0 = np.nanmax(ys)
        yf = ys
        s = y0 - yf
    dt = 1.0 / fps
    moving = s > 1e-4
    if not moving.any():
        return None
    i_mov = int(np.argmax(moving))
    i_peak = int(np.argmax(s))
    ok = np.isfinite(X).all(axis=1)
    if not ok.any():
        return None
    last_ok = int(np.nonzero(ok)[0][-1])
    first_ok = int(np.nonzero(ok)[0][0])
    i_start = max(i_mov, first_ok)
    i_end = min(i_peak, last_ok)
    if i_end <= i_start + 2:
        return None
    lo = max(0, i_start - int(round(pre_s * fps)))
    sl = slice(lo, i_end + 1)
    t, X, s, yf, ok = t[sl], X[sl], s[sl], yf[sl], ok[sl]
    i_start -= lo
    i_end -= lo
    on = bool(row.swatter_on)
    yc = (yf if on else np.full_like(yf, y0)) + HALF_THICK_M
    Xs = smooth_positions(X, fps, smooth_s)
    rel = np.c_[Xs[:, 0] - row.swatter_x_m, Xs[:, 1] - yc, Xs[:, 2] - row.swatter_z_m]
    good = np.isfinite(rel).all(axis=1)
    theta = np.zeros(len(t))
    theta[good] = angular_size_deg(rel[good])
    theta_dot = np.zeros(len(t))
    if good.sum() > 2:
        theta_dot[good] = np.gradient(theta[good], t[good])
    air = np.zeros(len(t))
    extrap = np.zeros(len(t), bool)
    hollow = "hollow" in str(row.disk_type).lower()
    if on and good.any():
        v, e = flow.air_velocity(rel[good], s[good] / R_DISK_M, U, R_DISK_M, NORMAL,
                                 disk_type="hollow" if hollow else "full", model=m)
        air[good] = np.linalg.norm(v, axis=1) / U
        extrap[good] = e
    v_esc = np.full(len(t), np.nan)
    if good.sum() > 2:
        vel = np.gradient(Xs[good], t[good], axis=0)
        u = away_from_disk(rel[good])
        v_esc[good] = np.einsum("ij,ij->i", vel, u)
    collided = str(getattr(row, "outcome", "")) in ("collision", "Collision")
    if collision_t_s is not None and np.isfinite(collision_t_s):
        k = int(np.searchsorted(t, collision_t_s))
        i_end = max(i_start + 1, min(i_end, k))
    elif collided and good.any():
        _, _, along, rho, _, _ = disk_view(rel[good])
        inside = (np.abs(along) <= HALF_THICK_M + 0.002) & (rho <= R_DISK_M + 0.002)
        if inside.any():
            k = int(np.nonzero(good)[0][np.argmax(inside)])
            i_end = max(i_start + 1, min(i_end, k))
    return TrialData(trial_id=str(row.trial_id), session=str(row.session), species=str(row.species),
                     light=str(row.light), visual=vis, hollow=hollow, swatter_on=on, dt=dt, t=t, theta=theta,
                     theta_dot=theta_dot, air=air, extrap=extrap, v_esc=v_esc, valid=good, i_start=i_start,
                     i_end=i_end, collided=collided, meta=dict(dataset=str(row.dataset), shift_frames=shift_frames))


# ------------------------------------------------------------------ 반응 시점 (변화점)
@dataclass
class Onset:
    q: np.ndarray       # 프레임별 반응 시점 확률 (관찰 창 밖은 0, 합 1 또는 0)
    p_resp: float       # 변화점(반응)이 있을 확률
    k_map: int          # 최빈 프레임 (없으면 −1)
    ci95: tuple         # 95% 구간 (프레임 인덱스)


def changepoint_onset(v, t, i_start, i_end, min_seg=3, prior_resp=0.5) -> Onset:
    """회피 속도에 조각선형 변화점 v = a + b·t + c·(t − t_c)₊ (c > 0) 을 맞춘다.

    t_c 는 관찰 창의 프레임 격자, 균등 사전분포. 증거는 BIC 근사(σ 포함 모수 수: 변화 5, 없음 3).
    c ≤ 0 인 후보는 '변화 없음' 과 같게 둔다(회피 = 멀어지는 쪽 가속).
    """
    T = len(v)
    q = np.zeros(T)
    idx = np.arange(i_start, i_end + 1)
    idx = idx[np.isfinite(v[idx])]
    n = len(idx)
    if n < 2 * min_seg + 3:
        return Onset(q, 0.0, -1, (-1, -1))
    y, tt = v[idx], t[idx]
    X0 = np.c_[np.ones(n), tt - tt[0]]
    b0, *_ = np.linalg.lstsq(X0, y, rcond=None)
    rss0 = float(np.sum((y - X0 @ b0) ** 2))
    cand = np.arange(min_seg, n - min_seg)
    logw = np.empty(len(cand))
    for j, c in enumerate(cand):
        X1 = np.c_[X0, np.maximum(tt - tt[c], 0.0)]
        b1, *_ = np.linalg.lstsq(X1, y, rcond=None)
        rss = float(np.sum((y - X1 @ b1) ** 2)) if b1[2] > 0 else rss0
        logw[j] = -0.5 * (n * np.log(max(rss, 1e-300) / n) + 5 * np.log(n))
    bic0 = n * np.log(max(rss0, 1e-300) / n) + 3 * np.log(n)
    mx = logw.max()
    w = np.exp(logw - mx)
    log_ev1 = mx + np.log(w.mean())
    log_ev0 = -0.5 * bic0
    logit = log_ev1 - log_ev0 + np.log(prior_resp / (1 - prior_resp))
    p = float(1 / (1 + np.exp(-np.clip(logit, -700, 700))))
    w = w / w.sum()
    q[idx[cand]] = w
    cdf = np.cumsum(w)
    lo = idx[cand][min(np.searchsorted(cdf, 0.025), len(cand) - 1)]
    hi = idx[cand][min(np.searchsorted(cdf, 0.975), len(cand) - 1)]
    return Onset(q, p, int(idx[cand][np.argmax(w)]), (int(lo), int(hi)))


def hmm_escape_onset(v, i_start, i_end, hm: dict) -> int:
    """교차 확인용: 원저자 HMM(P1b, mhmm.mat: prior, A, mu, var, mixmat, esc_state)의 첫 escape 상태 프레임 (없으면 −1)."""
    idx = np.arange(i_start, i_end + 1)
    idx = idx[np.isfinite(v[idx])]
    if len(idx) == 0:
        return -1
    x = v[idx]
    mix = hm.get("mixmat", np.ones_like(hm["mu"]))
    lb = (np.log(mix)[None, :] - 0.5 * np.log(2 * np.pi * hm["var"])[None, :]
          - (x[:, None] - hm["mu"][None, :]) ** 2 / (2 * hm["var"][None, :]))
    with np.errstate(divide="ignore"):
        lA, lp = np.log(hm["A"]), np.log(hm["prior"])
    n, nq = lb.shape
    delta = lp + lb[0]
    psi = np.zeros((n, nq), int)
    for k in range(1, n):
        s = delta[:, None] + lA
        psi[k] = np.argmax(s, axis=0)
        delta = s[psi[k], np.arange(nq)] + lb[k]
    path = np.zeros(n, int)
    path[-1] = int(np.argmax(delta))
    for k in range(n - 1, 0, -1):
        path[k - 1] = psi[k, path[k]]
    hit = np.nonzero(path == hm["esc_state"])[0]
    return int(idx[hit[0]]) if len(hit) else -1


# ------------------------------------------------------------------ 자료 묶음 (같은 dt 끼리)
@dataclass
class Group:
    dt: float
    theta: np.ndarray       # (N, T)
    theta_dot_pos: np.ndarray
    air: np.ndarray
    valid: np.ndarray       # (N, T) float 0/1
    q: np.ndarray           # (N, T)
    p_resp: np.ndarray      # (N,)
    i_start: np.ndarray     # (N,) int
    i_end: np.ndarray
    transparent: np.ndarray  # (N,) float 0/1
    hollow: np.ndarray       # (N,) float 0/1
    log10L: np.ndarray
    species_idx: np.ndarray
    session_idx: np.ndarray
    trial_ids: list
    sessions: list
    weight: np.ndarray | None = None   # 채움 trial 은 0 (P2c 고정 크기용)


GROUP_ARRAYS = ["theta", "theta_dot_pos", "air", "valid", "q", "p_resp", "i_start", "i_end", "transparent", "hollow",
                "log10L", "species_idx", "session_idx", "weight"]


def pad_group(G: Group, N: int, T: int) -> Group:
    """(N, T) 고정 크기로 채운다(채움 trial: 자극 0, q 0, weight 0). nutpie with_data 로 컴파일을 재사용하기 위함."""
    n, t = G.theta.shape
    if n > N or t > T:
        raise ValueError(f"group {n}x{t} > pad {N}x{T}")

    def p2(x):
        out = np.zeros((N, T))
        out[:n, :t] = x
        return out

    def p1(x, fill=0):
        out = np.full(N, fill, dtype=np.asarray(x).dtype)
        out[:n] = x
        return out

    w = np.ones(n) if G.weight is None else G.weight
    return Group(dt=G.dt, theta=p2(G.theta), theta_dot_pos=p2(G.theta_dot_pos), air=p2(G.air), valid=p2(G.valid),
                 q=p2(G.q), p_resp=p1(G.p_resp), i_start=p1(G.i_start), i_end=p1(G.i_end, T - 1),
                 transparent=p1(G.transparent), hollow=p1(G.hollow), log10L=p1(G.log10L),
                 species_idx=p1(G.species_idx), session_idx=p1(G.session_idx),
                 trial_ids=list(G.trial_ids) + [""] * (N - n), sessions=list(G.sessions) + [""] * (N - n),
                 weight=p1(w.astype(float)))


def make_groups(trials: list[TrialData], onsets: dict, species_levels: list, session_levels: list,
                onset_shift_s: float = 0.0) -> list[Group]:
    """trial 을 dt 별로 묶어 (N, T) 배열로. onsets[trial_id] = Onset (없으면 미반응: p = 0).

    onset_shift_s: 반응 시점 추정의 계통 편향 보정(양수 = q 를 늦춤). 보정값은 Exp2 고해상도 비교나
    합성 자료로 따로 정한다 (decisions.md P2b).
    """
    out = []
    for dt in sorted({round(tr.dt, 9) for tr in trials}):
        ts = [tr for tr in trials if round(tr.dt, 9) == dt]
        N, T = len(ts), max(len(tr.t) for tr in ts)
        Z = lambda: np.zeros((N, T))  # noqa: E731
        th, td, ai, va, qq = Z(), Z(), Z(), Z(), Z()
        p = np.zeros(N)
        i0 = np.zeros(N, int)
        i1 = np.zeros(N, int)
        for i, tr in enumerate(ts):
            L = len(tr.t)
            th[i, :L] = np.where(tr.valid, tr.theta, 0.0)
            td[i, :L] = np.where(tr.valid, np.maximum(tr.theta_dot, 0.0), 0.0)
            ai[i, :L] = np.where(tr.valid, tr.air, 0.0)
            va[i, :L] = tr.valid
            i0[i], i1[i] = tr.i_start, tr.i_end
            o = onsets.get(tr.trial_id)
            if o is not None and o.q.sum() > 0:
                qv = o.q[:L].copy()
                if onset_shift_s:
                    qv = np.interp(np.arange(L) - onset_shift_s / tr.dt, np.arange(L), qv, left=0.0, right=0.0)
                qv[: tr.i_start + 1] = 0.0
                qv[tr.i_end + 1:] = 0.0
                if qv.sum() > 0:
                    qq[i, :L] = qv / qv.sum()
                    p[i] = o.p_resp
        out.append(Group(dt=float(dt), theta=th, theta_dot_pos=td, air=ai, valid=va, q=qq, p_resp=p, i_start=i0,
                         i_end=i1, transparent=np.array([tr.visual == "transparent" for tr in ts], float),
                         hollow=np.array([tr.hollow for tr in ts], float),
                         log10L=np.array([np.log10(LUMINANCE_CDM2[tr.light]) for tr in ts]),
                         species_idx=np.array([species_levels.index(tr.species) for tr in ts]),
                         session_idx=np.array([session_levels.index(tr.session) for tr in ts]),
                         trial_ids=[tr.trial_id for tr in ts], sessions=[tr.session for tr in ts]))
    return out


def reference_airflow(flow_model=None) -> float:
    """A_ref: 최고 속도 시점 s* 에서 원판 축 위 중심 2R 앞의 |u|/U_ref."""
    m = flow_model or flow.potential_model()
    s_pk = float(m.kin_s_star[int(np.argmax(m.kin_u_rel))])
    v, _ = flow.air_velocity(np.array([[0.0, -REF_DIST_M, 0.0]]), s_pk, m.U_ref_cfd_mps, R_DISK_M, NORMAL, model=m)
    return float(np.linalg.norm(v) / m.U_ref_cfd_mps)


# ------------------------------------------------------------------ 계산 백엔드 (numpy / PyTensor 공용)
class _NumpyOps:
    exp, log, expm1, log1p, sqrt = np.exp, np.log, np.expm1, np.log1p, np.sqrt
    maximum, minimum, where = np.maximum, np.minimum, np.where

    @staticmethod
    def erfc(x):
        return _np_erfc(x)

    @staticmethod
    def cumsum(x):
        return np.cumsum(x, axis=1)

    @staticmethod
    def floor_int(x):
        return int(np.floor(x))

    @staticmethod
    def arange(n):
        return np.arange(n)

    @staticmethod
    def clip(x, lo, hi):
        return np.clip(x, lo, hi)

    @staticmethod
    def concat_ones_left(x):
        return np.concatenate([np.ones((x.shape[0], 1)), x[:, :-1]], axis=1)

    @staticmethod
    def rowpick(x, idx):
        return x[np.arange(x.shape[0]), idx]

    @staticmethod
    def const(x):
        return x


class _PytensorOps:
    def __init__(self):
        import pytensor.tensor as pt

        self.pt = pt
        self.exp, self.log, self.expm1, self.log1p, self.sqrt = pt.exp, pt.log, pt.expm1, pt.log1p, pt.sqrt
        self.maximum, self.minimum, self.where, self.erfc = pt.maximum, pt.minimum, pt.switch, pt.erfc

    def cumsum(self, x):
        return self.pt.cumsum(x, axis=1)

    def floor_int(self, x):
        return self.pt.floor(x).astype("int64")

    def arange(self, n):
        return self.pt.arange(n)

    def clip(self, x, lo, hi):
        return self.pt.clip(x, lo, hi)

    def concat_ones_left(self, x):
        return self.pt.concatenate([self.pt.ones((x.shape[0], 1)), x[:, :-1]], axis=1)

    def rowpick(self, x, idx):
        return x[self.pt.arange(x.shape[0]), idx]

    def const(self, x):
        from pytensor.graph.basic import Variable

        return x if isinstance(x, Variable) else self.pt.as_tensor_variable(np.asarray(x))


NP = _NumpyOps()


LAG_ARRAYS = ["theta_lag", "V_lag", "air_lag"]


def crop_group(G: Group, T: int | None = None) -> Group:
    """시간축을 반응 가능 창 [0, max(i_end)] 으로 자른다 (P2c 최적화). i_end 뒤 프레임은 S(i_end)·q 에 들어가지 않으므로
    로그 우도가 바뀌지 않는다. 채움 trial(weight 0)의 i_end 는 T − 1 로 줄인다."""
    real = (G.weight > 0) if G.weight is not None else np.ones(len(G.i_end), bool)
    T_need = int(G.i_end[real].max()) + 1
    T = T or T_need
    if T < T_need:
        raise ValueError(f"crop T={T} < max(i_end)+1={T_need}")
    kw = {f: getattr(G, f)[:, :T] for f in ["theta", "theta_dot_pos", "air", "valid", "q"]}
    return dataclasses.replace(G, i_end=np.minimum(G.i_end, T - 1), i_start=np.minimum(G.i_start, T - 1), **kw)


def convolve_onsets(G: Group, offsets_frames, weights) -> Group:
    """반응 시점 보정 (P2c-R ③): q 를 커널(오프셋 프레임 → 가중치, + = 늦춤)로 합성곱하고 관찰 창
    (i_start < k ≤ i_end) 밖을 0 으로 한 뒤 다시 정규화. p_resp 는 그대로. 창 안 질량이 0 이 되면 원래 q 를 둔다."""
    w = np.asarray(weights, float)
    w = w / w.sum()
    N, T = G.q.shape
    q = np.zeros_like(G.q, dtype=float)
    for o, wo in zip(np.asarray(offsets_frames, int), w):
        if o >= 0:
            q[:, o:] += wo * G.q[:, :T - o]
        else:
            q[:, :o] += wo * G.q[:, -o:]
    k = np.arange(T)[None, :]
    q *= (k > G.i_start[:, None]) & (k <= G.i_end[:, None])
    s = q.sum(axis=1, keepdims=True)
    had = G.q.sum(axis=1, keepdims=True) > 0
    q = np.where((s > 0) & had, q / np.where(s > 0, s, 1.0), G.q)
    return dataclasses.replace(G, q=q.astype(G.q.dtype))


def onset_kernel(G: Group, offsets_frames, weights, use_kernel=None) -> Group:
    """시점 커널 우도 (P2d-R, decisions.md 'P2d-R 설계 변경 (10/29)' 2번). j = 추정 시점 − 참 시점 (프레임), w_j 는 합 1 로 정규화.

    L = Σ_j w_j · f(t_est − j). trial_loglik 의 반응 항은 프레임 이산(Σ_k Pk·q_k)이므로 q 를
    q_K[m] = Σ_j w_j · q[m + j] 로 바꾸면 정확히 이 우도가 된다(t_est 는 기존처럼 변화점 사후 q 로 적분).
    참 시점이 관찰 창(i_start < m ≤ i_end) 밖이면 관찰된 반응과 맞지 않으므로 0 이고, 다시 정규화하지 않는다
    (③ convolve_onsets 와 다른 점). p_resp 는 그대로. 델타 커널(j = 0 만 1)이면 q 가 바뀌지 않는다.
    use_kernel: (N,) bool. False 인 trial(≥1 kHz 시점)은 커널 없이 원래 q (기존 점 우도). 기본은 전부 True."""
    w = np.asarray(weights, float)
    w = w / w.sum()
    N, T = G.q.shape
    q = np.zeros_like(G.q, dtype=float)
    for j, wj in zip(np.asarray(offsets_frames, int), w):
        if j >= T or -j >= T:
            continue
        if j >= 0:
            q[:, : T - j] += wj * G.q[:, j:]
        else:
            q[:, -j:] += wj * G.q[:, : T + j]
    k = np.arange(T)[None, :]
    q *= (k > G.i_start[:, None]) & (k <= G.i_end[:, None])
    use = np.ones(N, bool) if use_kernel is None else np.asarray(use_kernel, bool)
    q = np.where(use[:, None], q, G.q)
    return dataclasses.replace(G, q=q.astype(G.q.dtype))


def kernel_from_errors(err_frames, max_frames: int = 10):
    """오차 표본(추정 − 참, 프레임 정수) → (offsets, weights). ±max_frames 에서 자른다(P2c-R 커널과 같은 처리)."""
    e = np.clip(np.asarray(err_frames, int), -max_frames, max_frames)
    vals, cnt = np.unique(e, return_counts=True)
    return vals, cnt / cnt.sum()


def aggregate_group(G: Group, target_dt: float = 0.010) -> Group:
    """프레임을 target_dt(기본 10 ms) 블록으로 묶는다: 자극은 블록 평균, q 는 블록 합, 인덱스는 블록 번호.
    dt ≥ target_dt 이면(2022 자료 90 fps) 그대로 둔다. 고속 자료(Exp2 12,500 fps)용 근사이며 정확 변환이 아니다."""
    k = int(round(target_dt / G.dt))
    if k < 2:
        return G
    N, T = G.theta.shape
    n = T // k

    def blk(x, how):
        x = x[:, : n * k].reshape(N, n, k)
        return x.mean(axis=2) if how == "mean" else (x.sum(axis=2) if how == "sum" else x.min(axis=2))

    return dataclasses.replace(G, dt=G.dt * k, theta=blk(G.theta, "mean"), theta_dot_pos=blk(G.theta_dot_pos, "mean"),
                               air=blk(G.air, "mean"), valid=blk(G.valid, "min"), q=blk(G.q, "sum"),
                               i_start=np.minimum(G.i_start // k, n - 1), i_end=np.minimum(G.i_end // k, n - 1))


def lag_stack(X, K):
    """(K + 2, N, T): k 번째 = X 를 k 프레임 늦춘 것 (앞은 첫 값). _shift 의 고정 격자 사전 계산."""
    N, T = X.shape
    out = np.empty((K + 2, N, T))
    for k in range(K + 2):
        idx = np.clip(np.arange(T) - k, 0, T - 1)
        out[k] = X[:, idx]
    return out


def add_lags(G: Group, A_ref: float, tau_max_s: float = 0.30) -> Group:
    """지연 고정 격자(0 … K+1 프레임, (K+1)·dt ≥ tau_max_s)와 자극 특징(V = C1·θ̇⁺, Air 척도)을 미리 계산해 붙인다.
    지연이 (K+1)·dt 를 넘으면 (K+1) 프레임에서 멈춘다 (τ 사전분포에서 P(τ > 0.3 s) ≈ 2e-6)."""
    K = int(np.ceil(tau_max_s / G.dt)) - 1
    G2 = dataclasses.replace(G)
    G2.theta_lag = lag_stack(G.theta, K)
    G2.V_lag = lag_stack(C1_MV_PER_DEG_S * G.theta_dot_pos, K)
    G2.air_lag = lag_stack(G.air * (V_REF / A_ref), K)
    G2.lag_K = K
    return G2


def _shift_lag(B, Xs, f, K):
    """사전 계산된 지연 격자에서 f 프레임 지연을 선형 보간 (f ∈ [0, K+1] 에서 _shift 와 같다)."""
    Xs = B.const(Xs)
    j0 = B.clip(B.floor_int(f), 0, K)
    w = B.clip(f - j0, 0.0, 1.0)
    return (1 - w) * Xs[j0] + w * Xs[j0 + 1]


def _shift(B, X, f, T):
    """X(t − f·dt): 프레임 축 선형 보간 지연 (앞은 첫 값으로 채움)."""
    X = B.const(X)
    j0 = B.floor_int(f)
    w = f - j0
    i0 = B.clip(B.arange(T) - j0, 0, T - 1)
    i1 = B.clip(i0 - 1, 0, T - 1)
    return (1 - w) * X[:, i0] + w * X[:, i1]


def soft_max_nonneg(B, xs, kappa=KAPPA):
    """비음수 성분들의 매끄러운 최댓값: (1/κ)·ln(1 + Σ(e^{κx} − 1)). 0 이면 0, ≥ max, ≤ max + ln(k)/κ."""
    acc = 0.0
    for x in xs:
        acc = acc + B.expm1(kappa * B.minimum(x, D_CLIP))
    return B.log1p(acc) / kappa


def soft_running_max(B, D, kappa=KAPPA):
    """단조 매끄러운 실행 최댓값 M(t) = (1/κ)·ln(1 + Σ_{u≤t}(e^{κD(u)} − 1)), D ≥ 0."""
    return B.log1p(B.cumsum(B.expm1(kappa * B.minimum(D, D_CLIP)))) / kappa


def drive(B, P, G: Group, rule: str, A_ref: float):
    """구동 D (N, T). P: 파라미터(dict, numpy 값 또는 PyTensor 변수)."""
    T = G.theta.shape[1]
    if getattr(G, "theta_lag", None) is not None:      # 사전 계산 경로 (add_lags)
        K = G.lag_K
        th = _shift_lag(B, G.theta_lag, P["tau_v"] / G.dt, K)
        V = _shift_lag(B, G.V_lag, P["tau_v"] / G.dt, K)
        air_s = _shift_lag(B, G.air_lag, P["tau_a"] / G.dt, K)
    else:
        th = _shift(B, G.theta, P["tau_v"] / G.dt, T)
        V = C1_MV_PER_DEG_S * _shift(B, G.theta_dot_pos, P["tau_v"] / G.dt, T)
        air_s = _shift(B, G.air, P["tau_a"] / G.dt, T) * (V_REF / A_ref)
    lth = B.log(B.maximum(th, 1e-6))
    Sz = C2_MV * B.exp(-((lth - B.log(P["theta_pk"])) ** 2) / (2 * P["sigma_theta"] ** 2))
    kh = 1.0 - G.hollow + G.hollow * P["k_h"]
    Air = air_s * kh[:, None]
    lin = P["a_sp"][G.species_idx] + P["b_sp"][G.species_idx] * G.log10L
    g = 1.0 / (1.0 + B.exp(-lin))
    c = 1.0 - G.transparent + G.transparent * P["c_t"]
    vis_v = P["w_v"] * V
    vis_s = P["w_v"] * Sz / P["rho_vs"]
    air_t = P["w_v"] * P["rho_av"] * Air
    cg = (c * g)[:, None]
    if rule == "sum":
        D = cg * (vis_v + vis_s) + air_t
    elif rule == "max":
        D = soft_max_nonneg(B, [cg * vis_v, cg * vis_s, air_t])
    elif rule == "rel":
        D = cg * (vis_v + vis_s) + (1.0 - g)[:, None] * air_t
    else:
        raise ValueError(rule)
    return D * G.valid


def survival(B, P, G: Group, rule: str, A_ref: float):
    """왼쪽 절단된 생존함수 S (N, T)."""
    D = drive(B, P, G, rule, A_ref)
    M = soft_running_max(B, D)
    eta = P["eta_sess"][G.session_idx]
    z = (B.log(M + 1e-12) - eta[:, None]) / P["sigma"]
    F = 0.5 * B.erfc(-z / np.sqrt(2.0))
    T = G.theta.shape[1]
    tt = (B.arange(T)[None, :] - G.i_start[:, None]) * G.dt
    S_raw = B.exp(-P["lam0"] * B.maximum(tt, 0.0)) * (1.0 - F)
    S0 = B.rowpick(S_raw, G.i_start)
    return S_raw / (S0[:, None] + 1e-300)


def trial_loglik(B, P, G: Group, rule: str, A_ref: float, mode="mixture", cut: float = 0.5):
    """trial 별 로그 우도 (N,). mode: 'mixture' 또는 'cut'(p ≥ cut 이면 반응, 아니면 미반응).

    P 에 'eps' 가 있으면 허위 검출 혼합 (decisions.md 'P2d-R G3 분기' 6번): 확률 ε 로 관측 시점이 참 반응과 무관하게
    관찰 창(i_start < k ≤ i_end) 안 균등 → 반응 항 (1 − ε)·ev + ε/n_win, 미반응 항 (1 − ε)·S_end. ε = 0 이면 기존 우도."""
    S = survival(B, P, G, rule, A_ref)
    Pk = B.maximum(B.concat_ones_left(S) - S, 0.0)
    ev = (Pk * G.q).sum(axis=1)
    S_end = B.rowpick(S, G.i_end)
    # make_groups 가 q 가 없는 trial 의 p 를 0 으로 둔다
    p = G.p_resp if mode == "mixture" else (G.p_resp >= cut) * 1.0
    eps = P.get("eps")
    if eps is not None:
        n_win = B.maximum((G.i_end - G.i_start) * 1.0, 1.0)
        ev = (1.0 - eps) * ev + eps / n_win
        S_end = (1.0 - eps) * S_end
    return B.log(p * ev + (1.0 - p) * S_end + 1e-300)


def loglik_numpy(P, groups, rule, A_ref, mode="mixture", cut=0.5):
    """numpy 참조 구현: 묶음별 trial 로그 우도를 이어 붙인다."""
    return np.concatenate([trial_loglik(NP, P, g, rule, A_ref, mode, cut) for g in groups])


# ------------------------------------------------------------------ PyMC 모델 (nutpie 전용)
def build_model(groups, rule: str, prior_level: str, n_species: int, n_sessions: int, A_ref: float,
                mode="mixture", cut=0.5, data_containers: bool = False, eps_beta: tuple | None = None):
    """9종 중 하나: rule ∈ RULES, prior_level ∈ PRIOR_LEVELS.

    data_containers=True: 묶음 배열을 pm.Data('<필드>_g<번호>') 로 넣어 nutpie with_data 로 자료만 바꿔 쓸 수 있게 한다
    (배열 크기는 고정 — pad_group). weight 가 있으면 trial 로그 우도에 곱한다.
    eps_beta=(a, b): 허위 검출 혼합 ε ~ Beta(a, b) 를 추가 (trial_loglik 참조). None 이면 기존 모델.
    """
    import pymc as pm

    if rule not in RULES or prior_level not in PRIOR_LEVELS:
        raise ValueError((rule, prior_level))
    Bp = _PytensorOps()
    pr = PRIORS[prior_level]
    cp = COMMON_PRIORS
    trial_ll = []
    with pm.Model() as model:
        lw = pm.Normal("log_w_v", *cp["w_v"])
        lvs = pm.Normal("log_rho_vs", *pr["rho_vs"])
        lav = pm.Normal("log_rho_av", *pr["rho_av"])
        lpk = pm.Normal("log_theta_pk", *cp["theta_pk"])
        lst = pm.Normal("log_sigma_theta", *cp["sigma_theta"])
        ltv = pm.Normal("log_tau_v", *cp["tau_v"])
        lta = pm.Normal("log_tau_a", *cp["tau_a"])
        llam = pm.Normal("log_lam0", *cp["lam0"])
        sigma = pm.HalfNormal("sigma", cp["sigma_hn"])
        c_t = pm.Beta("c_t", *cp["c_t_beta"])
        k_h = pm.Beta("k_h", *cp["k_h_beta"])
        a_sp = pm.Normal("a_sp", *cp["a_sp"], shape=n_species)
        b_sp = pm.HalfNormal("b_sp", cp["b_sp_hn"], shape=n_species)
        s_sess = pm.HalfNormal("sigma_sess", cp["sigma_sess_hn"])
        z_sess = pm.Normal("z_sess", 0.0, 1.0, shape=n_sessions)
        P = dict(w_v=pm.math.exp(lw), rho_vs=pm.math.exp(lvs), rho_av=pm.math.exp(lav),
                 theta_pk=pm.math.exp(lpk), sigma_theta=pm.math.exp(lst), tau_v=pm.math.exp(ltv),
                 tau_a=pm.math.exp(lta), lam0=pm.math.exp(llam), sigma=sigma + 1e-3, c_t=c_t, k_h=k_h,
                 a_sp=a_sp, b_sp=b_sp, eta_sess=s_sess * z_sess)
        if eps_beta is not None:
            P["eps"] = pm.Beta("eps", *eps_beta)
        for name in ["w_v", "rho_vs", "rho_av", "theta_pk", "sigma_theta", "tau_v", "tau_a", "lam0"]:
            pm.Deterministic(name, P[name])
        for gi, g in enumerate(groups):
            if data_containers:
                from types import SimpleNamespace

                arr = {k[: -len(f"_g{gi}")]: pm.Data(k, v) for k, v in group_data([g], start=gi).items()}
                gg = SimpleNamespace(dt=g.dt, lag_K=getattr(g, "lag_K", None), **arr)
            else:
                gg = g
            ll = trial_loglik(Bp, P, gg, rule, A_ref, mode, cut)
            w = gg.weight if getattr(gg, "weight", None) is not None else 1.0
            pm.Potential(f"loglik_g{gi}", (ll * w).sum())
            trial_ll.append(ll)
    model.trial_ll = trial_ll          # trial 별 로그 우도 (MAP + Laplace 의 LOO 용; 표본에 저장하지 않음)
    return model


def group_data(groups, start: int = 0) -> dict:
    """nutpie with_data 용 {이름: 배열} (build_model(data_containers=True) 와 같은 이름). add_lags 배열이 있으면 함께."""
    out = {}
    for gi, g in enumerate(groups, start):
        fields = GROUP_ARRAYS + (LAG_ARRAYS if getattr(g, "theta_lag", None) is not None else [])
        for f in fields:
            v = getattr(g, f)
            v = np.ones(g.theta.shape[0]) if (f == "weight" and v is None) else v
            is_int = f in ("i_start", "i_end", "species_idx", "session_idx")
            out[f"{f}_g{gi}"] = np.asarray(v, "int64" if is_int else "float64")
    return out


def sample_nutpie(model, draws=1000, tune=1000, chains=4, seed=20261028, initial_points=None, **kw):
    """nutpie (numba 백엔드) 샘플링 → ArviZ 객체. initial_points={자유 RV 이름: 값(제약 척도)} 이면 모든 체인을
    그 점에서 흔들림 없이 시작한다 (MAP 시작값)."""
    import nutpie

    ck = {} if initial_points is None else dict(initial_points=initial_points, jitter_rvs=set())
    compiled = nutpie.compile_pymc_model(model, backend="numba", **ck)
    return nutpie.sample(compiled, draws=draws, tune=tune, chains=chains, seed=seed, progress_bar=False, **kw)


PARAM_NAMES = ["w_v", "rho_vs", "rho_av", "theta_pk", "sigma_theta", "tau_v", "tau_a", "lam0", "sigma", "c_t", "k_h"]


def params_from_draw(post, chain, draw, n_species, n_sessions) -> dict:
    """사후 표본 하나 → numpy 파라미터 dict (pointwise 로그 우도·LOO 용)."""
    g = lambda k: np.asarray(post[k].values[chain, draw])  # noqa: E731
    P = {k: float(g(k)) for k in PARAM_NAMES if k != "sigma"}
    P["sigma"] = float(g("sigma")) + 1e-3
    P["a_sp"], P["b_sp"] = g("a_sp").reshape(n_species), g("b_sp").reshape(n_species)
    P["eta_sess"] = float(g("sigma_sess")) * g("z_sess").reshape(n_sessions)
    return P


def session_loglik(ll_trials: np.ndarray, sessions: list) -> tuple[np.ndarray, list]:
    """세션 단위 PSIS(leave-one-session-out 근사)용: trial 로그 우도를 세션별로 합친다 (..., N) → (..., S)."""
    levels = sorted(set(sessions))
    idx = np.array([levels.index(s) for s in sessions])
    out = np.zeros(ll_trials.shape[:-1] + (len(levels),))
    for j in range(len(levels)):
        out[..., j] = ll_trials[..., idx == j].sum(axis=-1)
    return out, levels


# ------------------------------------------------------------------ MAP + Laplace, 사전분포 재가중 (P2c)
_UNCONSTRAIN = {"log_w_v": ("w_v", np.exp), "log_rho_vs": ("rho_vs", np.exp), "log_rho_av": ("rho_av", np.exp),
                "log_theta_pk": ("theta_pk", np.exp), "log_sigma_theta": ("sigma_theta", np.exp),
                "log_tau_v": ("tau_v", np.exp), "log_tau_a": ("tau_a", np.exp), "log_lam0": ("lam0", np.exp),
                "sigma_log__": ("sigma", np.exp), "c_t_logodds__": ("c_t", lambda x: 1 / (1 + np.exp(-x))),
                "k_h_logodds__": ("k_h", lambda x: 1 / (1 + np.exp(-x))), "a_sp": ("a_sp", lambda x: x),
                "b_sp_log__": ("b_sp", np.exp), "sigma_sess_log__": ("sigma_sess", np.exp), "z_sess": ("z_sess", lambda x: x),
                "eps_logodds__": ("eps", lambda x: 1 / (1 + np.exp(-x)))}


class FlatModel:
    """모델 logp(제약 없는 척도, Jacobian 포함)·기울기·trial 로그 우도를 평평한 벡터 x 로 계산 (한 번 컴파일).
    data_containers=True 모델이면 set_data 로 자료만 바꾼다."""

    def __init__(self, model, mode="NUMBA"):
        import pytensor

        self.model = model
        self.vv = model.value_vars
        ip = model.initial_point()
        self.names = [v.name for v in self.vv]
        self.shapes = [np.shape(ip[n]) for n in self.names]
        self.sizes = [int(np.prod(s)) for s in self.shapes]
        self.x0 = np.concatenate([np.ravel(ip[n]) for n in self.names]).astype(float)
        lp = model.logp()
        grads = pytensor.grad(lp, self.vv)
        ll = model.replace_rvs_by_values(model.trial_ll)
        llv = ll[0] if len(ll) == 1 else pytensor.tensor.concatenate(ll)
        self._fg = pytensor.function(self.vv, [lp] + grads, mode=mode)
        self._fl = pytensor.function(self.vv, [lp, llv], mode=mode, on_unused_input="ignore")

    def split(self, x):
        out, i = [], 0
        for s, n in zip(self.shapes, self.sizes):
            out.append(np.asarray(x[i:i + n]).reshape(s))
            i += n
        return out

    def logp_grad(self, x):
        r = self._fg(*self.split(x))
        return float(r[0]), np.concatenate([np.ravel(g) for g in r[1:]])

    def logp_ll(self, x):
        lp, ll = self._fl(*self.split(x))
        return float(lp), np.asarray(ll)

    def set_data(self, data: dict):
        import pymc as pm

        with self.model:
            pm.set_data(data)

    def constrained(self, X):
        """(S, d) 제약 없는 표본 → {파라미터: (S, ...)} (params_from_draw 와 같은 이름)."""
        X = np.atleast_2d(X)
        out, i = {}, 0
        for nm, s, n in zip(self.names, self.shapes, self.sizes):
            name, f = _UNCONSTRAIN[nm]
            out[name] = f(X[:, i:i + n]).reshape((X.shape[0],) + s)
            i += n
        return out


def map_laplace(fm: FlatModel, starts, n_draws=1000, seed=0, h=1e-4, maxiter=3000, eig_floor=0.25):
    """여러 시작점 L-BFGS 로 MAP → 기울기 중심 차분 Hessian → 정규 근사 표본.
    반환: dict(x_map, logp_map, cov, draws (S, d), log_q (S,), hess_pd, n_eval, converged, all_logp)."""
    from scipy.optimize import minimize

    best, runs = None, []
    n_eval = 0
    def neg(x):
        lp, g = fm.logp_grad(x)
        if not (np.isfinite(lp) and np.isfinite(g).all()):   # NaN 영역(예: max 규칙)은 +inf → 선 탐색이 물러난다
            return np.inf, np.zeros_like(x)
        return -lp, -g

    for x0 in starts:
        r = minimize(neg, np.asarray(x0, float), jac=True, method="L-BFGS-B",
                     options=dict(maxiter=maxiter, maxfun=maxiter * 2, ftol=1e-13, gtol=1e-7))
        n_eval += r.nfev
        runs.append(-float(r.fun))
        if np.isfinite(r.fun) and (best is None or r.fun < best.fun):   # NaN·inf 결과는 최선 후보에서 뺀다
            best = r
    if best is None:
        raise FloatingPointError("map_laplace: 모든 시작점에서 logp 가 유한하지 않다")
    x = best.x
    d = len(x)
    H = np.empty((d, d))
    for j in range(d):
        e = np.zeros(d)
        e[j] = h
        H[j] = -(fm.logp_grad(x + e)[1] - fm.logp_grad(x - e)[1]) / (2 * h)
    n_eval += 2 * d
    H = 0.5 * (H + H.T)
    w, U = np.linalg.eigh(H)
    hess_pd = bool(w.min() > 0)
    bad_dirs = [(float(w[i]), int(np.argmax(np.abs(U[:, i])))) for i in np.nonzero(w < eig_floor)[0]]
    w = np.maximum(np.abs(w), eig_floor)  # 음의 고윳값은 절댓값(SoftAbs 식; 꺾인 점의 차분 곡률), 작은 값은 eig_floor(SD 2)로 (hess_pd·bad_dirs 로 보고)
    cov = (U / w) @ U.T
    L = U / np.sqrt(w)
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n_draws, d))
    draws = x + Z @ L.T
    log_q = -0.5 * (Z**2).sum(axis=1) + 0.5 * np.log(w).sum() - 0.5 * d * np.log(2 * np.pi)
    return dict(x_map=x, logp_map=-float(best.fun), cov=cov, draws=draws, log_q=log_q, hess_pd=hess_pd, bad_dirs=bad_dirs,
                n_eval=int(n_eval), converged=bool(best.success), all_logp=runs, message=str(best.message))


def prior_reweight(log_rho_vs, log_rho_av, level_from="P0", level_to="P1"):
    """P0 표본을 다른 P 단계로 PSIS 재가중: log w = log p_to(ρ) − log p_from(ρ) (ρ_vs, ρ_av 의 로그 정규).
    반환: (정규화된 로그 가중치, k̂)."""
    from arviz_stats.base.array import array_stats

    def lnorm(x, mu_sd):
        mu, sd = mu_sd
        return -0.5 * ((x - mu) / sd) ** 2 - np.log(sd)

    a, b = PRIORS[level_from], PRIORS[level_to]
    lr = (lnorm(log_rho_vs, b["rho_vs"]) - lnorm(log_rho_vs, a["rho_vs"])
          + lnorm(log_rho_av, b["rho_av"]) - lnorm(log_rho_av, a["rho_av"]))
    lw, k = array_stats.psislw(-np.asarray(lr, float).ravel())   # psislw 는 −(로그 비)를 받는다 (LOO 관례)
    return np.asarray(lw), float(k)


def weighted_quantile(x, q, log_w=None):
    """가중 분위수 (log_w 가 None 이면 보통 분위수)."""
    x = np.asarray(x, float).ravel()
    if log_w is None:
        return np.quantile(x, q)
    w = np.exp(np.asarray(log_w) - np.max(log_w))
    o = np.argsort(x)
    c = np.cumsum(w[o])
    c = (c - 0.5 * w[o]) / c[-1]
    return np.interp(q, c, x[o])


def weighted_sd(x, log_w=None):
    x = np.asarray(x, float).ravel()
    if log_w is None:
        return float(np.std(x))
    w = np.exp(np.asarray(log_w) - np.max(log_w))
    w /= w.sum()
    m = (w * x).sum()
    return float(np.sqrt((w * (x - m) ** 2).sum()))


# ------------------------------------------------------------------ 합성 자료 (테스트·P2c 전용; 실데이터 수치 아님)
SYNTH_TRUTH_DEFAULT = dict(w_v=2.0, rho_vs=1.9, rho_av=0.3, theta_pk=42.0, sigma_theta=0.52, tau_v=0.03, tau_a=0.02,
                           lam0=0.05, sigma=0.4, c_t=0.4, k_h=0.5)


def _min_jerk(t, T, S):
    x = np.clip(t / T, 0.0, 1.0)
    return S * (10 * x**3 - 15 * x**4 + 6 * x**5)


def simulate_dataset(n_trials: int, rule: str = "sum", truth: dict | None = None, seed: int = 0, *, flow_model=None,
                     fps: float = 90.0, n_sessions: int = 12, conditions=None, control_frac: float = 0.15,
                     a_sp=(0.5, 0.0), b_sp=(0.5, 0.5), sigma_sess: float = 0.2, pos_noise_m: float = 5e-4,
                     escape_acc_mps2: float = 5.0, target_sd_m: float = 0.04):
    """P1b 공통 스키마 형식의 합성 trial 표와 참값을 만든다 (dataset = 'synthetic').

    스와터: 최소 저크 운동(0.26 m, 0.53 s, 시작 앞면 y = 0.243 m) — 합성용 해석 함수(실측 운동학 아님).
    모기: 원판 경로 끝 근처를 지나는 직선 비행(0.15–0.35 m/s). 감지 모델(rule, truth)로 반응 시점을 뽑고,
    반응하면 원판에서 멀어지는 방향으로 escape_acc_mps2 가속(0.2 s). 원판 부피에 들어가면 충돌.
    반환: (trials DataFrame, truth dict, per-trial truth DataFrame)
    """
    import pandas as pd

    rng = np.random.default_rng(seed)
    P = dict(SYNTH_TRUTH_DEFAULT, **(truth or {}))
    m = flow_model or flow.potential_model()
    A_ref = reference_airflow(m)
    conditions = conditions or [("Anopheles coluzzii", "dark", "full / black"), ("Anopheles coluzzii", "twilight", "full / transparent"),
                                ("Aedes aegypti", "sunrise", "full / black"), ("Aedes aegypti", "overcast", "full / transparent")]
    species_levels = sorted({c[0] for c in conditions})
    sess = []
    for k in range(n_sessions):
        spc, light, disk = conditions[k % len(conditions)]
        sess.append(dict(session=f"syn{k:03d}", species=spc, light=light, disk_type=disk,
                         swatter_on=bool(rng.random() >= control_frac), eta=rng.normal(0.0, sigma_sess)))
    eta_sess = np.array([s["eta"] for s in sess])
    T_move, S_move, y0 = 0.53, 0.26, 0.243
    t_rel = np.arange(-0.15 - T_move, 0.12, 1.0 / fps)          # 0 = 최대 신장
    s_t = _min_jerk(t_rel + T_move, T_move, S_move)
    yf = y0 - s_t
    rows, truth_rows = [], []
    for i in range(n_trials):
        ss = sess[i % n_sessions]
        target = rng.normal(0, target_sd_m, 3) + np.array([0.0, -0.01, 0.0])
        d = rng.normal(size=3)
        d /= np.linalg.norm(d)
        speed = rng.uniform(0.15, 0.35)
        X = target[None, :] + (t_rel[:, None] - rng.uniform(-0.15, 0.05)) * speed * d[None, :]
        row = dict(dataset="synthetic", species=ss["species"], light=ss["light"], disk_type=ss["disk_type"],
                   with_disk=1, swatter_on=ss["swatter_on"],
                   condition_disk=(ss["disk_type"].replace(" / ", "_") if ss["swatter_on"] else "nopiston"),
                   session=ss["session"], trial_id=f"{ss['session']}_t{i:05d}", trigger_index=i, obj_id=i,
                   fps=fps, age_days=7, trigger_frame=np.nan, kin_frame_delay=np.nan, frame_zero=np.nan,
                   pos_source="synthetic", swatter_virtual=not ss["swatter_on"], swatter_x_m=0.0, swatter_z_m=0.0,
                   frame=np.arange(len(t_rel)), t_s=t_rel.copy(), x_m=X[:, 0], y_m=X[:, 1], z_m=X[:, 2],
                   swatter_y_m=yf.copy(), outcome="no_collision", collision_manual=False, in_trigger_set=True,
                   in_track_set=True, outcome_label=None, frame_body_touch=np.nan, frame_leg_touch=np.nan, grade="",
                   sex="", n_obj_in_trigger=1, same_file_as_2024_exp1=False, subset_2022_paper=False,
                   subset_2024_exp1_paper=False, subset_2024_exp2_paper=False)
        tr = prepare_trial(pd.Series(row), flow_model=m)
        onset_k, coll_k = -1, -1
        if tr is not None:
            G = make_groups([tr], {}, species_levels, [s["session"] for s in sess])[0]
            Pn = dict(P, a_sp=np.array(a_sp[: len(species_levels)], float), b_sp=np.array(b_sp[: len(species_levels)], float),
                      eta_sess=eta_sess)
            D = drive(NP, Pn, G, rule, A_ref)[0]
            M = soft_running_max(NP, D[None, :])[0]
            # 관찰 시작 전에 넘지 않았다는 조건(왼쪽 절단)에 맞춰 임계값을 절단분포에서 뽑는다
            eta_i = eta_sess[G.session_idx[0]]
            u0 = float(ndtr((np.log(M[tr.i_start] + 1e-12) - eta_i) / P["sigma"]))
            theta_i = np.exp(eta_i + P["sigma"] * ndtri(min(u0 + (1 - u0) * rng.random(), 1 - 1e-12)))
            cross = np.nonzero((M >= theta_i) & (np.arange(len(M)) > tr.i_start))[0]
            k_stim = int(cross[0]) if len(cross) else 10**9
            t_sp = rng.exponential(1.0 / P["lam0"])
            k_sp = tr.i_start + int(np.ceil(t_sp * fps))
            k_on = min(k_stim, k_sp)
            off = int(np.nonzero(np.isclose(t_rel, tr.t[0]))[0][0])
            if k_on <= tr.i_end:
                onset_k = off + k_on
                yc_on = (yf[onset_k] if ss["swatter_on"] else y0) + HALF_THICK_M
                u = away_from_disk(np.array([[X[onset_k, 0], X[onset_k, 1] - yc_on, X[onset_k, 2]]]))[0]
                tau = np.clip(t_rel - t_rel[onset_k], 0.0, None)
                acc_t = np.minimum(tau, 0.2)
                disp = 0.5 * escape_acc_mps2 * acc_t**2 + escape_acc_mps2 * 0.2 * np.maximum(tau - 0.2, 0.0)
                X = X + disp[:, None] * u[None, :]
            if ss["swatter_on"]:
                _, _, along, rho, _, _ = disk_view(np.c_[X[:, 0], X[:, 1] - (yf + HALF_THICK_M), X[:, 2]])
                inside = (np.abs(along) <= HALF_THICK_M) & (rho <= R_DISK_M) & (t_rel <= 0)
                if inside.any():
                    coll_k = int(np.argmax(inside))
        Xn = X + rng.normal(0.0, pos_noise_m, X.shape)
        row.update(x_m=Xn[:, 0], y_m=Xn[:, 1], z_m=Xn[:, 2])
        if coll_k >= 0:
            row.update(outcome="collision", collision_manual=True)
        responded = onset_k >= 0 and (coll_k < 0 or onset_k < coll_k)
        rows.append(row)
        truth_rows.append(dict(trial_id=row["trial_id"], responded=bool(responded),
                               onset_t_s=float(t_rel[onset_k]) if responded else np.nan,
                               collision_t_s=float(t_rel[coll_k]) if coll_k >= 0 else np.nan))
    truth_all = dict(P, rule=rule, a_sp=list(a_sp[: len(species_levels)]), b_sp=list(b_sp[: len(species_levels)]),
                     sigma_sess=sigma_sess, eta_sess=eta_sess.tolist(), species_levels=species_levels,
                     sessions=[s["session"] for s in sess], A_ref=A_ref, seed=seed, n_trials=n_trials, fps=fps,
                     note="합성 자료: 해석적 운동학·직선 비행·감지 모델 참값으로 생성. 실데이터 수치 아님.")
    return pd.DataFrame(rows)[SCHEMA_COLUMNS], truth_all, pd.DataFrame(truth_rows)
