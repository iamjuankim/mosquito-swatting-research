"""스와터(원판) 주변 기류 속도장 대리모델.

CFD(Cribellier et al. 2024 Data S3, 고체 원판 지름 0.1 m) 를 원판 기준 무차원 좌표로 옮겨 학습한 모델을
불러 쓴다. 학습·평가: phases/p1_repro/flow_fit.py, 근거·가정: logs/decisions.md (P1d), 오차: results/p1d/.

무차원화 (logs/decisions.md 'P1d')
- 원점 = 원판 중심(두께 중앙), n = 이동 방향 단위벡터 (= 원판 법선, 기울기 없을 때)
- ξ = (x − x_c)·n / R  (양수 = 원판 앞, bow wave 쪽), ρ* = 축에서의 거리 / R
- s* = 원판 이동 거리 / R  (흐름 상태 변수; 학습 범위 0 ≤ s* ≤ 5.23, 최대 신장까지)
- 속도 = 실험실 기준 공기 속도 / U_ref (U_ref = 최고 스와터 속도)

CFD 범위 밖(다른 U_ref·R, 기울기, hollow, 영역 밖, s* 범위 밖)은 계산은 하되 extrapolated=True 로 돌려준다.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

CFD_R_M = 0.05
CFD_THICKNESS_RATIO = 0.1  # 두께 / 지름 (0.01 m / 0.1 m)
NU_AIR_M2PS = 1.5e-5
SCALE_TOL = 0.01  # U_ref·R 이 CFD 와 이만큼 넘게 다르면 외삽
TILT_TOL_RAD = np.deg2rad(1.0)

DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[2] / "data" / "interim" / "p1d" / "flow_model.joblib"


# ---------------------------------------------------------------- 무차원 좌표 변환
def to_disk_frame(rel_pos_m, normal, R_m):
    """원판 중심 기준 상대 위치(m) → (ξ, ρ*, e_ρ). normal 은 이동 방향(정규화함)."""
    rel = np.atleast_2d(np.asarray(rel_pos_m, float))
    n = np.asarray(normal, float)
    n = n / np.linalg.norm(n)
    along = rel @ n
    radial = rel - along[:, None] * n[None, :]
    rho = np.linalg.norm(radial, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        e_rho = np.where(rho[:, None] > 1e-12, radial / rho[:, None], 0.0)
    return along / R_m, rho / R_m, e_rho, n


def from_disk_frame(u_xi, u_rho, n, e_rho, U_ref_mps):
    """무차원 (u_ξ, u_ρ) → 실험실 기준 속도 벡터 (m/s)."""
    return U_ref_mps * (np.asarray(u_xi)[:, None] * n[None, :] + np.asarray(u_rho)[:, None] * e_rho)


def reynolds(U_ref_mps, R_m, nu_m2ps=NU_AIR_M2PS):
    return U_ref_mps * 2 * R_m / nu_m2ps


# ---------------------------------------------------------------- 물리 기준선 (C)
def potential_flow_disk(xi, rho_star, u_inst_rel):
    """얇은 원판(반지름 1)이 +ξ 로 속도 u_inst_rel 로 움직일 때 정지 유체의 퍼텐셜 흐름 (무차원).

    편구면 좌표 ρ = sqrt(1+ζ²)·sqrt(1−μ²), ξ = ζ·μ, φ = (2u/π)·μ·(ζ·arccot ζ − 1).
    원판면(ζ=0)에서 ∂φ/∂ξ = u 를 만족한다. 속도는 φ 의 수치 미분.
    """
    xi = np.asarray(xi, float)
    rho_star = np.asarray(rho_star, float)

    def phi(x, r):
        q = r**2 + x**2 - 1.0
        zeta = np.sqrt(np.maximum((q + np.sqrt(q**2 + 4 * x**2)) / 2, 0.0))
        with np.errstate(invalid="ignore", divide="ignore"):
            mu = np.where(zeta > 1e-9, x / np.maximum(zeta, 1e-300), np.sign(x) * np.sqrt(np.clip(1 - r**2, 0, 1)))
            acot = np.where(zeta > 0, np.arctan2(1.0, zeta), np.pi / 2)
        return (2.0 / np.pi) * mu * (zeta * acot - 1.0)

    h = 1e-4
    # φ 는 원판면(ξ=0, ρ*<1)을 사이에 두고 불연속 → 그 근처에서는 간격을 줄여 같은 쪽에서만 미분
    near = (rho_star < 1.0) & (np.abs(xi) < 2 * h)
    h_xi = np.where(near, np.maximum(np.abs(xi) / 2, 1e-9), h)
    u_xi = (phi(xi + h_xi, rho_star) - phi(xi - h_xi, rho_star)) / (2 * h_xi)
    u_rho = (phi(xi, rho_star + h) - phi(xi, np.abs(rho_star - h))) / (2 * h)
    u_rho = np.where(rho_star < h, 0.0, u_rho)
    u = np.asarray(u_inst_rel, float)
    return u * u_xi, u * u_rho


# ---------------------------------------------------------------- 예측기 (joblib 저장 대상)
class GridPredictor:
    """(A) CFD 실험실 격자를 시간 단면별로 선형 보간하고, s* 사이는 인접 두 단면을 선형 보간."""

    def __init__(self, rho_grid_m, y_grid_m, s_m, y_center_m, u_xi, u_rho, R_m=CFD_R_M, axis_sign=-1.0):
        self.rho_grid_m = np.asarray(rho_grid_m, float)   # (n_rho,)
        self.y_grid_m = np.asarray(y_grid_m, float)       # (n_y,) 오름차순
        self.s_m = np.asarray(s_m, float)                 # (n_t,) 단조 증가
        self.y_center_m = np.asarray(y_center_m, float)   # (n_t,)
        self.u_xi = np.asarray(u_xi, np.float32)          # (n_t, n_rho, n_y)
        self.u_rho = np.asarray(u_rho, np.float32)
        self.R_m = R_m
        self.axis_sign = axis_sign  # 이동 방향 = axis_sign · (+y)

    def _slice(self, k, rho_m, y_m):
        from scipy.interpolate import RegularGridInterpolator

        pts = np.c_[rho_m, y_m]
        out = []
        for f in (self.u_xi[k], self.u_rho[k]):
            g = RegularGridInterpolator((self.rho_grid_m, self.y_grid_m), f, bounds_error=False, fill_value=np.nan)
            out.append(g(pts))
        return out

    def __call__(self, xi, rho_star, s_star):
        s_m = np.asarray(s_star, float) * self.R_m
        k = np.clip(np.searchsorted(self.s_m, s_m) - 1, 0, len(self.s_m) - 2)
        a_xi = np.full(len(xi), np.nan)
        a_rho = np.full(len(xi), np.nan)
        for kk in np.unique(k):
            m = k == kk
            w = np.clip((s_m[m] - self.s_m[kk]) / max(self.s_m[kk + 1] - self.s_m[kk], 1e-12), 0, 1)
            res = []
            for j in (kk, kk + 1):
                y_m = self.y_center_m[j] + self.axis_sign * xi[m] * self.R_m
                res.append(self._slice(j, rho_star[m] * self.R_m, y_m))
            a_xi[m] = (1 - w) * res[0][0] + w * res[1][0]
            a_rho[m] = (1 - w) * res[0][1] + w * res[1][1]
        return a_xi, a_rho


class SklearnPredictor:
    """(B) 특징 (ξ, ρ*, s*) → (u_ξ, u_ρ) 회귀기 두 개."""

    def __init__(self, reg_xi, reg_rho):
        self.reg_xi, self.reg_rho = reg_xi, reg_rho

    def __call__(self, xi, rho_star, s_star):
        X = np.c_[xi, rho_star, s_star]
        return self.reg_xi.predict(X), self.reg_rho.predict(X)


class MultiOutputPredictor:
    """(B) 출력 2개를 한 번에 내는 회귀기(예: MLP) 감싸개."""

    def __init__(self, reg):
        self.reg = reg

    def __call__(self, xi, rho_star, s_star):
        Y = self.reg.predict(np.c_[xi, rho_star, s_star])
        return Y[:, 0], Y[:, 1]


# ---------------------------------------------------------------- 모델 묶음
@dataclass
class FlowModel:
    """학습된 대리모델 + CFD 범위 정보.

    predictor(xi, rho_star, s_star) → (u_xi, u_rho) 무차원. None 이면 퍼텐셜 흐름만 쓴다.
    """

    name: str
    predictor: object = None
    U_ref_cfd_mps: float = 1.008
    R_cfd_m: float = CFD_R_M
    s_star_max: float = 5.23
    rho_star_max: float = 4.96
    xi_bounds_s: np.ndarray = field(default_factory=lambda: np.array([[0.0, -1.8, 9.6], [5.23, -6.6, 4.4]]))
    kin_s_star: np.ndarray = field(default_factory=lambda: np.array([0.0, 5.23]))
    kin_u_rel: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0]))
    meta: dict = field(default_factory=dict)

    def u_inst_rel(self, s_star):
        """CFD 운동학에서 s* 일 때의 순간 원판 속도 / U_ref."""
        return np.interp(s_star, self.kin_s_star, self.kin_u_rel)

    def xi_range(self, s_star):
        b = np.asarray(self.xi_bounds_s, float)
        return np.interp(s_star, b[:, 0], b[:, 1]), np.interp(s_star, b[:, 0], b[:, 2])

    def in_domain(self, xi, rho_star, s_star):
        lo, hi = self.xi_range(s_star)
        return (rho_star <= self.rho_star_max) & (xi >= lo) & (xi <= hi) & (s_star >= 0) & (s_star <= self.s_star_max)

    def predict(self, xi, rho_star, s_star):
        xi, rho_star = np.asarray(xi, float), np.asarray(rho_star, float)
        s = np.broadcast_to(np.asarray(s_star, float), xi.shape)
        s_c = np.clip(s, 0.0, self.s_star_max)
        inside = self.in_domain(xi, rho_star, s_c)
        u_xi, u_rho = potential_flow_disk(xi, rho_star, self.u_inst_rel(s_c))
        if self.predictor is not None and inside.any():
            a, b = self.predictor(xi[inside], rho_star[inside], s_c[inside])
            ok = np.isfinite(a) & np.isfinite(b)
            idx = np.nonzero(inside)[0][ok]
            u_xi[idx], u_rho[idx] = np.asarray(a)[ok], np.asarray(b)[ok]
            inside = inside.copy()
            inside[np.nonzero(inside)[0][~ok]] = False
        in_solid = (np.abs(xi) <= CFD_THICKNESS_RATIO) & (rho_star <= 1.0)
        u_solid = self.u_inst_rel(s_c)
        u_xi = np.where(in_solid, u_solid, u_xi)
        u_rho = np.where(in_solid, 0.0, u_rho)
        out_of_range = ~inside | (s != s_c)
        return u_xi, u_rho, out_of_range


_CACHE: dict = {}


def load_model(path: str | os.PathLike | None = None) -> FlowModel:
    """학습된 모델 파일(joblib)을 읽는다. 기본 경로: data/interim/p1d/flow_model.joblib (환경변수 MQ10_FLOW_MODEL)."""
    import joblib

    p = Path(path or os.environ.get("MQ10_FLOW_MODEL", DEFAULT_MODEL_PATH))
    key = str(p)
    if key not in _CACHE:
        if not p.exists():
            raise FileNotFoundError(f"기류 모델 파일이 없습니다: {p} (phases/p1_repro/flow_fit.py fit 로 만든다)")
        _CACHE[key] = joblib.load(p)
    return _CACHE[key]


def potential_model() -> FlowModel:
    """학습 모델 없이 퍼텐셜 흐름(C)만 쓰는 모델 (항상 '물리 기준선')."""
    return FlowModel(name="potential", predictor=None, xi_bounds_s=np.array([[0.0, -np.inf, np.inf], [5.23, -np.inf, np.inf]]),
                     rho_star_max=np.inf)


# ---------------------------------------------------------------- 공개 함수
def air_velocity(rel_pos_m, s_star, U_ref_mps, R_m, normal=(0.0, -1.0, 0.0), *, motion_dir=None,
                 disk_type="full", model: FlowModel | None = None):
    """원판 주변 공기 속도 (실험실 기준, m/s).

    rel_pos_m : (N, 3) 원판 중심 기준 상대 위치 [m]
    s_star    : 원판 이동 거리 / R (스칼라 또는 (N,))
    U_ref_mps : 최고 스와터 속도 [m/s]
    R_m       : 원판 반지름 [m]
    normal    : 원판 법선 = 이동 방향 (기본: CFD 와 같은 −y)
    motion_dir: 실제 이동 방향 (주면 법선과의 각도로 기울기 판정)
    disk_type : 'full' 만 CFD 범위. 'hollow' 등은 같은 장을 쓰고 extrapolated.

    반환: (vel_mps (N, 3), extrapolated (N,) bool)
    """
    m = model or load_model()
    xi, rho_s, e_rho, n = to_disk_frame(rel_pos_m, normal, R_m)
    u_xi, u_rho, out = m.predict(xi, rho_s, s_star)
    scale_off = (abs(U_ref_mps * R_m / (m.U_ref_cfd_mps * m.R_cfd_m) - 1) > SCALE_TOL
                 or abs(R_m / m.R_cfd_m - 1) > SCALE_TOL or abs(U_ref_mps / m.U_ref_cfd_mps - 1) > SCALE_TOL)
    tilt = False
    if motion_dir is not None:
        d = np.asarray(motion_dir, float)
        cosang = abs(d @ n) / np.linalg.norm(d)
        tilt = np.arccos(np.clip(cosang, -1, 1)) > TILT_TOL_RAD
    extrap = out | scale_off | tilt | (disk_type != "full")
    return from_disk_frame(u_xi, u_rho, n, e_rho, U_ref_mps), extrap
