from __future__ import annotations
def draw_k(j, cond):
    return cond.k_fixed if cond.k_fixed is not None else float(np.random.default_rng([SEED_K, j]).uniform(0, S.K_MAX_PER_S))

def draw_escape(boot, j):
    b = int(np.random.default_rng([SEED_ESC, j]).integers(0, boot.b.nunique()))
    out = {}
    for disk in ('opaque', 'transparent'):
        r = boot[(boot.b == b) & (boot.disk == disk)].iloc[0]
        out[disk] = E.EscapeParams(k_per_s=float(r.k), w=float(r.w), kappa=float(r.kappa), log_a_mu=float(r.log_a_mu), log_a_sd=float(r.log_a_sd))
    return (out, b)

def mc_batch(ctx, j, n, mc_set, sigma_sess):
    """통념 실험 궤적 n 개: 초기 상태 복원추출 + 공통 난수 (조건과 무관; seed, draw, 집합으로만 결정)."""
    rng = np.random.default_rng([SEED_MC, j, MC_SET_ID[mc_set]])
    pick = rng.integers(0, len(ctx.init), n)
    u, z_a = S.common_randoms(rng, n)
    z = rng.normal(size=n)
    I = ctx.init.iloc[pick]
    return S.Batch(rel0_m=np.stack(I.rel0_m.to_numpy()), v0_mps=np.stack(I.v0_mps.to_numpy()), s_start_m=I.s_start_m.to_numpy(float), transparent=np.zeros(n, bool), eta=sigma_sess * z, u=u, z_a=z_a)

def validate_arrays(z):
    for key in ['captured', 't_det_s', 't_contact_s', 'M0', 'theta', 'i0', 'trigger_i', 'stimulus', 'spontaneous', 'trajectory_id']:
        if z[key].shape != (4000,):
            raise RuntimeError('Bad shape ' + key)
    if z['captured'].dtype != np.bool_ or not np.array_equal(z['trajectory_id'], np.arange(4000)):
        raise RuntimeError('Bad IDs/dtype')
    for key in ['t_det_s', 't_contact_s']:
        if not np.all(np.isfinite(z[key]) | np.isposinf(z[key])):
            raise RuntimeError('Bad time state')
    if not np.array_equal(z['captured'], np.isfinite(z['t_contact_s'])):
        raise RuntimeError('Solid capture/contact mismatch')
    finite = np.isfinite(z['t_det_s'])
    if not np.array_equal(finite, z['trigger_i'] >= 0) or not np.all(z['trigger_i'][finite] > z['i0'][finite]):
        raise RuntimeError('Illegal trigger')
    if not np.all((z['stimulus'] | z['spontaneous'])[finite]):
        raise RuntimeError('Trigger condition missing')
    if not np.isfinite(z['theta']).all() or not np.isfinite(z['M0']).all():
        raise RuntimeError('Bad thresholds')
