"""Plant models and excitation signals for the numerical examples.

The continuous-time dynamics here are the ground truth the neural predictors are
trained against; `multisine` generates the persistently-exciting inputs used to
collect that training data.
"""

import numpy as np

# Ratio of peak to RMS amplitude; lower is better for excitation signals.
crest_factor = lambda uk: np.max(np.abs(uk)) / np.sqrt(np.mean(uk**2))

duplicate = lambda uk, n: np.concatenate([uk] * n)


def multisine(
    N_points_per_period,
    N_periods=1,
    pmin=1,
    pmax=21,
    prule=lambda p: p % 2 == 1 and p % 6 != 1,
    par=None,
    n_crest_factor_optim=1,
    seed=None,
):
    """Multisine with odd-only frequencies and random phases, unit variance.

    With `n_crest_factor_optim > 1`, draws that many phase realisations
    (seeds `seed`, `seed+1`, ...) and keeps the one with the lowest crest factor.
    """
    if isinstance(seed, int) or seed is None:
        rng = np.random.RandomState(seed)
    else:
        rng = seed
    assert isinstance(rng, np.random.mtrand.RandomState)

    assert pmax < N_points_per_period // 2

    if n_crest_factor_optim > 1:
        ybest = None
        crest_best = float("inf")
        for i in range(n_crest_factor_optim):
            seedi = None if seed is None else seed + i
            uk = multisine(
                N_points_per_period,
                N_periods=1,
                pmax=pmax,
                pmin=pmin,
                prule=prule,
                n_crest_factor_optim=1,
                seed=seedi,
            )
            crest = crest_factor(uk)
            if crest < crest_best:
                ybest = uk
                crest_best = crest
        return duplicate(ybest, N_periods)

    N = N_points_per_period
    uf = np.zeros((N,), dtype=complex)
    for p in range(pmin, pmax) if par is None else par:
        if par is None and not prule(p):
            continue
        uf[p] = np.exp(1j * rng.uniform(0, np.pi * 2))
        uf[N - p] = np.conjugate(uf[p])  # keep the spectrum Hermitian -> real signal

    uk = np.real(np.fft.ifft(uf / 2) * N)
    uk /= np.std(uk)

    return duplicate(uk, N_periods)


def van_der_pol(t, y, mu, u):
    """Van der Pol oscillator, SISO. State [x, dx], input u. For `solve_ivp`."""
    x, dx = y
    ddx = mu * (1 - x**2) * dx - x + u
    return [dx, ddx]


def four_tank(t, x, u1, u2, params):
    """Quadruple-tank process, MIMO. State = four tank levels, inputs = two pumps.

    `params` supplies outlet areas `a1..a4`, cross-section `Sc`, valve splits
    `gamma_a`/`gamma_b` and gravity `g`. Levels are clamped at zero so the
    square roots stay real when a tank empties.
    """
    x1, x2, x3, x4 = x
    a1, a2, a3, a4 = params["a1"], params["a2"], params["a3"], params["a4"]
    Sc, gamma_a, gamma_b = params["Sc"], params["gamma_a"], params["gamma_b"]
    g = params["g"]
    dx1 = (
        (-a1 / Sc) * np.sqrt(max(0, 2 * g * x1))
        + (a3 / Sc) * np.sqrt(max(0, 2 * g * x3))
        + gamma_a / (3600 * Sc) * u1
    )
    dx2 = (
        (-a2 / Sc) * np.sqrt(max(0, 2 * g * x2))
        + (a4 / Sc) * np.sqrt(max(0, 2 * g * x4))
        + gamma_b / (3600 * Sc) * u2
    )
    dx3 = (-a3 / Sc) * np.sqrt(max(0, 2 * g * x3)) + (1 - gamma_b) / (3600 * Sc) * u2
    dx4 = (-a4 / Sc) * np.sqrt(max(0, 2 * g * x4)) + (1 - gamma_a) / (3600 * Sc) * u1
    return [dx1, dx2, dx3, dx4]
