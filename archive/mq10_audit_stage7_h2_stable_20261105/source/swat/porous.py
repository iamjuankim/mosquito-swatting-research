"""다공성(천공) 스와터의 bow wave 감쇠 계수 (P5a).

근거: Steiros & Hultmark 2018 (JFM 853 R3). references/notes/porous.md S&H 블록에서 원문 쪽 이미지로
확인된 식만 쓴다 (β = A_p/A 개구율 = 공극률, u* = u/U∞).
 - 식 (2.12b)  C_D = (4/3)(1 − u*)(2 + u*)/(2 − u*)                     [원문 PDF p6 / R3-6]
 - 식 (2.15)   C_D = u*²(1/β² − 1) − (4/3)(1 − u*)³/(2 − u*)²           [원문 PDF p7 / R3-7]
 - β → u*: 원문(R3-7)은 두 식의 C_D 를 같게 놓는다고만 쓴다. 여기서는 그 방정식을 수치 근으로 푼다.
   porous.md 의 명시 결합식 1/β² = 1 + (4/3)(1 − u*)(5 − 2u*)/(u*²(2 − u*)²) 는 **[유도, 원문 식 아님]**
   이며 이 모듈에서는 검산 함수(inv_beta2_derived)로만 둔다.
 - u 는 판 총면적 A 기준 평균 통과 속도 (원문 p4 본문 'mass flux through the plate is ρuA').

감쇠 계수 — **[모델 가정, 원문 식 아님]** (logs/decisions.md P5a, references/notes/porous.md 'P5a 유도'):
 원판이 정지 공기 속을 V 로 움직이면 원판면의 실험실 기준 평균 공기 속도는 V − u*V = (1 − u*)V 이다.
 원판 앞쪽 흐름을 면 경계조건에 선형인 흐름으로 보면 bow wave 는 고체 원판 장의 (1 − u*) 배다.
   잔존 비 r(β) = 1 − u*(β),   감쇠 계수 a(β) = u*(β) = 1 − r(β)  (bow wave 가 줄어드는 비율)
 민감도: a 에 배수 m 을 곱한다 (a_m = clip(m·a, 0, 1)). β = 0 에서는 m 과 무관하게 고체 원판 장 그대로.

적용 한계 (원문 R3-9~10, porous.md): 2차원 평판 이론, β ≥ 0.23 에서만 모델 유효(그 아래는 와류 방출),
실험 개구율 0–0.6, 정상 유동, Re = 6000 (순폭 (1 − β)w 기준). 원판·비정상(가속) 운동·다른 Re 는 모두 외삽이다.
β > 0 인 결과는 항상 extrapolated=True (CFD 는 고체 원판뿐).
"""

from __future__ import annotations

import numpy as np

from swat import flow

SH_BETA_MIN = 0.23        # 모델 유효 하한 (원문 R3-9~10)
SH_BETA_EXP_MAX = 0.6     # 실험 개구율 상한 (판 6장, 0–60 %)
SH_RE_NET = 6000.0        # 검증 Re, 순폭 (1 − β)w 기준
SH_RE_C = 1000.0          # 고체판 C_D 가 평탄해지는 Re (Hoerner 1965, 원문 p8 재인용)
CD_SOLID = 4.0 / 3.0      # 식 (2.12b) 의 u* = 0 값


def cd_momentum(u_star):
    """식 (2.12b): 운동량 이론 항력계수."""
    u = np.asarray(u_star, float)
    return (4.0 / 3.0) * (1 - u) * (2 + u) / (2 - u)


def cd_screen(u_star, beta):
    """식 (2.15): Taylor & Davies 형 판 통과 손실로 쓴 항력계수."""
    u = np.asarray(u_star, float)
    b = np.asarray(beta, float)
    return u**2 * (1 / b**2 - 1) - (4.0 / 3.0) * (1 - u) ** 3 / (2 - u) ** 2


def inv_beta2_derived(u_star):
    """[유도, 원문 식 아님] (2.12b) = (2.15) 를 1/β² 로 푼 명시식 (porous.md). 검산용으로만 쓴다."""
    u = np.asarray(u_star, float)
    return 1 + (4.0 / 3.0) * (1 - u) * (5 - 2 * u) / (u**2 * (2 - u) ** 2)


def u_star_from_beta(beta):
    """β → u*: (2.12b) − (2.15) = 0 의 (0, 1) 안 근. β = 0 → 0, β = 1 → 1.

    차이 함수는 u* = 0 에서 5/3 > 0, u* = 1 에서 −(1/β² − 1) < 0 이고 (0, 1) 에서 근이 하나뿐이다
    (같은 방정식의 1/β² 형태 우변이 u* 에 대해 단조 감소).
    """
    from scipy.optimize import brentq

    b = np.atleast_1d(np.asarray(beta, float))
    if np.any((b < 0) | (b > 1)) or not np.all(np.isfinite(b)):
        raise ValueError("beta 는 0 이상 1 이하")
    out = np.empty_like(b)
    for i, bi in enumerate(b):
        if bi == 0.0:
            out[i] = 0.0
        elif bi == 1.0:
            out[i] = 1.0
        else:
            out[i] = brentq(lambda u: cd_momentum(u) - cd_screen(u, bi), 1e-12, 1 - 1e-12, xtol=1e-14, rtol=1e-14)
    return out if np.ndim(beta) else out[0]


def attenuation(beta, k_mult=1.0):
    """감쇠 계수 a = clip(k_mult·u*(β), 0, 1) [모델 가정]. bow wave 가 고체 원판 대비 줄어드는 비율."""
    return np.clip(k_mult * u_star_from_beta(beta), 0.0, 1.0)


def residual_ratio(beta, k_mult=1.0):
    """잔존 비 r = 1 − a: 다공 원판 bow wave / 고체 원판 bow wave."""
    return 1.0 - attenuation(beta, k_mult)


def reynolds_net(beta, U_ref_mps, R_m, nu_m2ps=flow.NU_AIR_M2PS):
    """S&H 기준과 같은 순폭 Re = (1 − β)·2R·U/ν."""
    return (1 - np.asarray(beta, float)) * flow.reynolds(U_ref_mps, R_m, nu_m2ps)


def validity(beta, U_ref_mps, R_m):
    """S&H 적용 범위 점검. 모두 참이어도 원판·비정상 운동이라는 외삽은 남는다."""
    b = np.asarray(beta, float)
    re = reynolds_net(b, U_ref_mps, R_m)
    return {
        "beta_in_model_range": (b == 0) | (b >= SH_BETA_MIN),
        "beta_in_exp_range": b <= SH_BETA_EXP_MAX,
        "re_net": re,
        "re_net_over_sh": re / SH_RE_NET,
        "re_net_ge_re_c": re >= SH_RE_C,
    }


def air_velocity_porous(rel_pos_m, s_star, U_ref_mps, R_m, beta, *, k_mult=1.0, normal=(0.0, -1.0, 0.0),
                        motion_dir=None, model: flow.FlowModel | None = None):
    """공극률 β 원판 주변 공기 속도 (실험실 기준, m/s) = r(β) × 고체 원판 장 [모델 가정].

    인자는 flow.air_velocity 와 같고 beta(공극률)·k_mult(감쇠 계수 배수, 민감도용)가 더해진다.
    반환: (vel_mps (N, 3), extrapolated (N,) bool). β > 0 이면 extrapolated 는 모두 True.
    """
    vel, extrap = flow.air_velocity(rel_pos_m, s_star, U_ref_mps, R_m, normal, motion_dir=motion_dir,
                                    disk_type="full", model=model)
    if beta == 0:
        return vel, extrap
    return residual_ratio(beta, k_mult) * vel, np.ones_like(extrap, dtype=bool)
