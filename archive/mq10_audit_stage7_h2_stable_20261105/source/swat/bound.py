"""③ 모기에게 가장 유리한 경계 계산 (감지 모델 적합도와 무관).

설계: logs/decisions.md 'H2 세 갈래 판정' 5번과 9-14 ③ 수치 설정 (결과 전 기록).
    감지  : 시작 이후 처음으로 모기 위치 기류 크기 |u_air| ≥ u_th (u_th = Exp2 관측 반응 시점 기류 크기의 하위 분위수)
    회피  : 감지 + τ_a 부터 40 ms 동안 능동 가속 A (원판 이동축에서 바깥 반지름 방향), 그 뒤 관성.
    수동  : 전 구간 k·u_air (k = 8.2 s⁻¹).
    접촉  : P3b 와 같은 스윕 판정 (simulate._swept_contact), 스트로크 끝까지. '회피 가능' = 접촉 없음.
스와터 운동·기류 척도·Δt·초기 상태는 simulate.simulate 와 같다 (f = v / U_cfd, u = U_cfd·f^n·u*).
"""

from __future__ import annotations

import dataclasses

import numpy as np

from swat import detection as det
from swat import escape as E
from swat import flow
from swat import simulate as S

# CFD 원자료 격자 → GridPredictor (phases/p1_repro/flow_fit.py:57–60, 242–251 과 같은 값·식)
Y0_M = 0.241
THICK_M = 0.01
T_MAX_MS = 527


def cfd_grid_model(fm_ref: flow.FlowModel, d) -> flow.FlowModel:
    """CFD 원자료(cfd_full.npz) 1 ms 단면 선형 보간 예측기를 쓰는 FlowModel (범위·운동학은 fm_ref 그대로)."""
    kt, ky = d["kin_t_s"], d["kin_y_m"]
    s = np.maximum.accumulate(np.maximum(ky, 0.0))
    U_ref = fm_ref.U_ref_cfd_mps
    rho_u, y_u = np.unique(d["rho_m"]), np.unique(d["y_m"])
    tt = np.arange(0, T_MAX_MS + 1)
    uxi = np.empty((len(tt), len(rho_u), len(y_u)), np.float32)
    urh = np.empty_like(uxi)
    ir, iy = np.searchsorted(rho_u, d["rho_m"]), np.searchsorted(y_u, d["y_m"])
    mean = np.asarray(d["mean"][: T_MAX_MS + 1])          # npz 는 접근마다 압축을 푸므로 한 번만 읽는다
    for j, t in enumerate(tt):
        uxi[j][ir, iy] = -mean[t, :, 2] / U_ref
        urh[j][ir, iy] = mean[t, :, 0] / U_ref
    keep = np.r_[True, np.diff(s[tt]) > 1e-9]
    pred = flow.GridPredictor(rho_u, y_u, s[tt][keep], (Y0_M - s[tt] + THICK_M / 2)[keep], uxi[keep], urh[keep])
    return dataclasses.replace(fm_ref, name="A_grid_interp_cfd", predictor=pred)


def air_mag_at(rel_m, s_m, grid: S.FlowGrid, U_cfd):
    """1 m/s(실측 운동학) 기류 크기 [m/s] — 원판 A 중심 기준 상대 위치, 이동 거리 s."""
    out = np.empty(len(rel_m))
    for i, (r, s) in enumerate(zip(rel_m, s_m)):
        out[i] = np.linalg.norm(grid.velocity(r[None, :], s / S.R_M, S.N_A)[0]) * U_cfd
    return out


def simulate_bound(rel0_m, v0_mps, s_start_m, v_mps, n_exp, u_th_mps, tau_a_s, A_mps2, k_per_s, kin,
                   grid: S.FlowGrid, U_cfd, win_s=E.WIN_S, dt_max_s=S.DT_MAX_S, dx_max_m=S.DX_MAX_M):
    """궤적 N 개 → dict(avoided, detected, t_det_s, t_contact_s) (시각은 스트로크 끝 기준, 없으면 +inf)."""
    tau_k, s_k = kin
    f = v_mps / U_cfd
    tau_peak = float(tau_k[np.argmax(s_k)])
    tau0 = tau_k[np.minimum(np.searchsorted(s_k, np.maximum(s_start_m, 1e-4), side="right"), np.argmax(s_k))]
    dt = min(dt_max_s, dx_max_m / v_mps)
    t0, t_end = float(tau0.min()) / f, tau_peak / f
    n_steps = int(np.ceil((t_end - t0) / dt))
    N = len(rel0_m)
    i0 = np.minimum(np.ceil((tau0 / f - t0) / dt - 1e-9).astype(int), n_steps - 1)
    U_air = U_cfd * f ** n_exp
    X = np.asarray(rel0_m, float).copy()
    V = np.asarray(v0_mps, float).copy()
    detected = np.zeros(N, bool)
    t_det = np.full(N, np.inf)
    esc_dir = np.zeros((N, 3))
    esc_set = np.zeros(N, bool)
    contact = np.zeros(N, bool)
    t_contact = np.full(N, np.inf)
    for i in range(n_steps):
        t = t0 + i * dt
        s = float(np.interp(min(f * t, tau_peak), tau_k, s_k))
        cA = np.array([0.0, -s, 0.0])
        act = (i >= i0) & ~contact
        rel = X - cA
        u_air = grid.velocity(rel, s / S.R_M, S.N_A) * U_air
        trig = act & ~detected & (np.linalg.norm(u_air, axis=1) >= u_th_mps)
        detected |= trig
        t_det[trig] = t
        go = act & detected & ~esc_set & (t >= t_det + tau_a_s - 1e-12)
        if go.any():                                   # 회피 시작 시점의 바깥 반지름 방향 (원판 이동축 기준)
            r = rel[go] - (rel[go] @ S.N_A)[:, None] * S.N_A[None, :]
            nr = np.linalg.norm(r, axis=1, keepdims=True)
            esc_dir[go] = np.where(nr > 1e-9, r / np.maximum(nr, 1e-12), np.array([[1.0, 0.0, 0.0]]))
            esc_set |= go
        on = esc_set & (t < t_det + tau_a_s + win_s)
        acc = k_per_s * u_air + np.where(on[:, None], A_mps2 * esc_dir, 0.0)
        Xn = X + V * dt
        Vn = V + acc * dt
        s1 = float(np.interp(min(f * (t + dt), tau_peak), tau_k, s_k))
        hit = S._swept_contact(rel, Xn - np.array([0.0, -s1, 0.0]), S.N_A)
        newc = act & hit
        contact |= newc
        t_contact[newc] = t + dt
        X = np.where(act[:, None], Xn, X)
        V = np.where(act[:, None], Vn, V)
    return dict(avoided=~contact, detected=detected & (t_det < t_contact), t_det_s=t_det - t_end,
                t_contact_s=t_contact - t_end, dt_s=dt, n_steps=n_steps)


def direction(speeds, a, n):
    """9-14 ③ 방향 조작적 정의: '감소' | '증가' | '비단조' (인접 차·끝점 차를 이항 SE 2배와 비교)."""
    a = np.asarray(a, float)
    se = lambda x, y: np.sqrt(x * (1 - x) / n + y * (1 - y) / n)  # noqa: E731
    d = np.diff(a)
    s = se(a[1:], a[:-1])
    end, s_end = a[-1] - a[0], se(a[-1], a[0])
    if np.all(d <= 2 * s) and end < -2 * s_end:
        return "감소"
    if np.all(d >= -2 * s) and end > 2 * s_end:
        return "증가"
    return "비단조"


def v_star(speeds, a, level=0.5):
    """회피 가능 비율이 처음 level 아래가 되는 속도 (log v 선형 보간). 범위 밖이면 문자열."""
    v, a = np.asarray(speeds, float), np.asarray(a, float)
    if a[0] < level:
        return f"< {v[0]:g} m/s"
    below = np.flatnonzero(a < level)
    if len(below) == 0:
        return f"> {v[-1]:g} m/s"
    i = int(below[0])
    w = (a[i - 1] - level) / (a[i - 1] - a[i])
    return float(np.exp(np.log(v[i - 1]) + w * (np.log(v[i]) - np.log(v[i - 1]))))
