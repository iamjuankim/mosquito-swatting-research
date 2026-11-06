"""Isolated numerical design; not integrated into any mq10 simulator.

Preserves the existing cutoff epsilon M0 + 1e-12 and original uniform orientation.
Returns log thresholds, never silently clips a quantile or forces it with nextafter.
Unresolvable float64 interior quantiles are flagged, and strict=True stops.
"""
from dataclasses import dataclass
import numpy as np
from scipy.special import log_ndtr, ndtri_exp

class UnresolvedQuantile(FloatingPointError):
    pass

@dataclass
class ThresholdResult:
    log_theta: np.ndarray
    z: np.ndarray
    a: np.ndarray
    log_cutoff: np.ndarray
    log_survival0: np.ndarray
    log_survival_target: np.ndarray
    hazard_budget: np.ndarray
    unresolved: np.ndarray
    log_tail_residual: np.ndarray

def sample_log_threshold(M0, eta, sigma, u, *, strict=True):
    M0,eta,sigma,u=np.broadcast_arrays(*[np.asarray(v,dtype=np.float64) for v in (M0,eta,sigma,u)])
    if not all(np.isfinite(x).all() for x in (M0,eta,sigma,u)):
        raise ValueError('Inputs must be finite')
    if np.any(M0<0) or np.any(sigma<=0) or np.any(u<0) or np.any(u>=1):
        raise ValueError('Require M0 >= 0, sigma > 0, and 0 <= u < 1')
    # Keep the original epsilon and expression, not a new conditioning model.
    cutoff=np.log(M0+1e-12)
    with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
        a=(cutoff-eta)/sigma
        log_s0=log_ndtr(-a)
        h=-np.log1p(-u)
        log_st=log_s0-h
        z=-ndtri_exp(log_st)
    # Generalized inverse at u=0 is exactly the lower endpoint, including a<<0.
    z=np.where(u==0,a,z)
    with np.errstate(over='ignore',invalid='ignore'):
        log_theta=cutoff+sigma*(z-a)
    log_theta=np.where(u==0,cutoff,log_theta)
    residual=log_ndtr(-z)-log_st
    tol=64*np.finfo(np.float64).eps*np.maximum(1.,np.abs(log_st))
    unresolved=(~np.isfinite(a))|(~np.isfinite(log_s0))|(~np.isfinite(log_st))|(~np.isfinite(z))|(~np.isfinite(log_theta))
    unresolved |= (u>0)&((log_st>=log_s0)|(z<=a)|(log_theta<=cutoff))
    unresolved |= (~np.isfinite(residual))|(np.abs(residual)>tol)
    # The scaled log-tail tolerance alone is too weak for enormous a (e.g. 1e6).
    with np.errstate(over='ignore',invalid='ignore'):
        recovered_u=-np.expm1(log_ndtr(-z)-log_s0)
    unresolved |= (~np.isfinite(recovered_u))|(np.abs(recovered_u-u)>2e-11)
    if strict and np.any(unresolved):
        raise UnresolvedQuantile(f'Float64 quantile not resolved for {int(np.sum(unresolved))} entries; no clipping or repair applied')
    return ThresholdResult(log_theta,z,a,cutoff,log_s0,log_st,h,unresolved,residual)

def threshold_crossed(M, result):
    """Proposed comparison using a log threshold; no simulator integration here."""
    M=np.asarray(M,dtype=np.float64)
    if not np.isfinite(M).all() or np.any(M<0): raise ValueError('Finite nonnegative drive required')
    if np.any(result.unresolved): raise UnresolvedQuantile('Cannot compare unresolved threshold')
    with np.errstate(divide='ignore'):
        return np.log(M)>=result.log_theta

