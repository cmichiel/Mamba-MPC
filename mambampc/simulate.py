"""Closed-loop simulation of an MPC controller against a plant.

Every closed-loop experiment in the paper is this same loop: solve the NLP for
the current measurement, apply the first input, step the plant, repeat. The
variations between experiments -- how many initial conditions, which reference,
whether the measurement is corrupted by noise -- are arguments.

The solver is warm-started from the previous solution, including the dual
variables, which is what keeps the per-step solve time low enough to report.
"""

import time

import casadi as cs
import numpy as np

__all__ = ["run_closed_loop"]


def run_closed_loop(
    solver,
    opt_x_num,
    opt_p_num,
    plant,
    x0,
    ref,
    N,
    steps,
    n_u=1,
    n_x=1,
    noise=None,
    lbx=None,
    ubx=None,
    progress=False,
):
    """Run one closed-loop MPC trajectory.

    Args:
        solver, opt_x_num, opt_p_num: as returned by `mambampc.mpc.setup_mpc`.
        plant: `f(state, u) -> next_state`, the true dynamics.
        x0: initial plant state, shape `(n_states,)`.
        ref: reference signal, at least `steps + N` samples. Shape
            `(T,)` for a single tracked output or `(T, n_x)` for several.
        N: prediction horizon.
        steps: number of closed-loop steps to simulate.
        n_u: number of inputs. n_x: number of tracked states in the NLP.
        noise: optional `(steps, n_states)` array added to the *measurement*
            handed to the controller. The plant itself is stepped from the
            clean state, so this is sensor noise, not process noise.
        lbx, ubx: optional bound structures from `setup_mpc`.
        progress: print a progress indicator.

    Returns:
        dict with `states` `(steps + 1, n_states)`, `inputs` `(steps + 1,)`
        or `(steps + 1, n_u)`, `solve_time` `(steps,)` and `cost` `(steps,)`.
    """
    x0 = np.atleast_1d(np.asarray(x0, dtype=float))
    n_states = x0.size
    ref = np.asarray(ref, dtype=float)
    if ref.ndim == 1:
        ref = ref[:, None]
    if ref.shape[0] < steps + N:
        raise ValueError(
            "reference has %d samples, need at least steps+N = %d"
            % (ref.shape[0], steps + N)
        )
    if noise is not None:
        noise = np.asarray(noise, dtype=float)
        if noise.shape[0] < steps:
            raise ValueError("noise has %d rows, need at least %d" % (noise.shape[0], steps))

    states = np.zeros((steps + 1, n_states))
    states[0] = x0
    inputs = np.zeros((steps + 1, n_u))
    solve_time = np.zeros(steps)
    cost = np.zeros(steps)

    n_var = (n_x + n_u) * N
    guess_x = np.zeros((n_var, 1))
    guess_lam_x = np.zeros((n_var, 1))
    guess_lam_g = 0

    solve_kwargs = {"lbg": 0, "ubg": 0}
    if lbx is not None:
        solve_kwargs["lbx"], solve_kwargs["ubx"] = lbx, ubx

    for k in range(steps):
        measured = states[k] + (noise[k] if noise is not None else 0.0)

        opt_p_num["ref"] = ref[k : k + N]
        opt_p_num["u_prev"] = inputs[k]
        for i in range(n_states):
            opt_p_num["x%d" % (i + 1)] = measured[i]

        tic = time.time()
        r = solver(
            x0=guess_x,
            lam_x0=guess_lam_x,
            lam_g0=guess_lam_g,
            p=opt_p_num,
            **solve_kwargs,
        )
        solve_time[k] = time.time() - tic

        opt_x_num.master = r["x"]
        cost[k] = float(r["f"])
        guess_x = np.array(r["x"]).reshape(n_var, 1)
        guess_lam_x = r["lam_x"]
        guess_lam_g = r["lam_g"]

        # "u_N" is a single (N, n_u) matrix entry, so index the row rather
        # than the struct -- ["u_N", 0] would pick one element, not the first input.
        u = np.array(opt_x_num["u_N"])[0, :].reshape(n_u)
        inputs[k + 1] = u
        states[k + 1] = plant(states[k], u)

        if progress:
            print(
                "\rclosed loop: %5.1f%%" % (100.0 * (k + 1) / steps),
                end="",
                flush=True,
            )
    if progress:
        print()

    return {
        "states": states,
        "inputs": inputs.squeeze() if n_u == 1 else inputs,
        "solve_time": solve_time,
        "cost": cost,
    }


def step_reference(amplitudes, samples_per_level, pad=0):
    """Piecewise-constant reference: each amplitude held for `samples_per_level`.

    `pad` repeats the final level, so the horizon can look past the last step.
    """
    ref = np.repeat(np.asarray(amplitudes, dtype=float), samples_per_level)
    if pad:
        ref = np.concatenate([ref, np.full(pad, ref[-1])])
    return ref
