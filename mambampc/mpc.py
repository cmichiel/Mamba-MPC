"""The model predictive control problem, built once for every experiment.

`setup_mpc` assembles the CasADi nonlinear program that every experiment in this
repository solves. The learned sequence model enters as an equality constraint
linking the predicted state trajectory to the model's output, so the solver
optimises over inputs while the network supplies the dynamics.

Decision variables
    x    (N, n_x)   predicted state trajectory
    u_N  (N, n_u)   input sequence

Parameters
    ref     (N, n_x)  reference trajectory
    u_prev  (1, n_u)  last applied input, for the input-rate penalty
    x1..xk  scalars   current measured state, held constant across the horizon

Objective
    sum_k  e_k Q e_k' + u_k R u_k' + du_k Rd du_k'   ( + terminal S )

The three experiments use genuinely different formulations -- the Van der Pol
study tracks a single output with scalar weights and replaces the last stage cost
with a terminal cost, Four Tank tracks all four states with matrix weights and no
terminal cost, Aero2 adds a terminal cost on top of a full-horizon sum. Rather
than hide that behind defaults, each choice is an explicit argument.
"""

import casadi as cs
import numpy as np
from casadi.tools import entry, struct_symMX

from .casadi_models import CasadiMamba

__all__ = ["setup_mpc"]


def _quad(e, W):
    """Quadratic form e W e'. `W` may be a scalar weight or a matrix."""
    if np.isscalar(W) or (hasattr(W, "shape") and np.prod(np.shape(W)) == 1):
        return W * cs.sumsqr(e)
    return cs.mtimes([e, W, e.T])


def setup_mpc(
    model,
    N,
    n_u,
    n_x,
    Q,
    R=0.0,
    Rd=0.0,
    S=0.0,
    n_state_params=None,
    n_model_states=None,
    terminal="add",
    u_bounds=None,
    x_bounds=None,
    solver="ipopt",
    solver_opts=None,
):
    """Build the MPC nonlinear program around a trained sequence model.

    Args:
        model: trained `MambaMPC` whose CasADi twin supplies the dynamics.
        N: prediction horizon.
        n_u: number of inputs.
        n_x: number of predicted states (the constraint dimension).
        Q, R, Rd: stage weights on tracking error, input, and input rate.
            Scalars or matrices; a zero disables that term.
        S: terminal weight. Zero disables the terminal cost.
        n_state_params: how many scalar state parameters to declare.
            Defaults to `n_x`. Aero2 declares more than it feeds to the model.
        n_model_states: how many of those parameters are fed to the model.
            Defaults to `n_state_params`.
        terminal: "add" appends a terminal cost to a full-horizon sum;
            "replace" sums to N-1 and applies the terminal cost at the last
            step instead; None omits it.
        u_bounds, x_bounds: `(lower, upper)` pairs, or None to leave unbounded.
            Each bound may be a scalar or a per-channel sequence.
        solver: CasADi `nlpsol` plugin name.
        solver_opts: options dict; defaults to silencing IPOPT.

    Returns:
        `(solver, opt_x_num, opt_p_num, lbx, ubx)` -- the callable solver plus
        zero-initialised variable/parameter structures for warm-starting, and
        the bound structures.
    """
    n_state_params = n_x if n_state_params is None else n_state_params
    n_model_states = n_state_params if n_model_states is None else n_model_states

    opt_x = struct_symMX(
        [entry("x", shape=(N, n_x)), entry("u_N", shape=(N, n_u))]
    )
    opt_p = struct_symMX(
        [entry("ref", shape=(N, n_x)), entry("u_prev", shape=(1, n_u))]
        + [entry(f"x{i + 1}", shape=(1), repeat=1) for i in range(n_state_params)]
    )

    opt_x_num = opt_x(0)
    opt_p_num = opt_p(0)

    # "replace" leaves the final stage to the terminal cost alone.
    n_stages = N - 1 if terminal == "replace" else N

    obj = 0
    for k in range(n_stages):
        obj += _quad(opt_x["x", k, :] - cs.vertcat(opt_p["ref", k, :]), Q)
        obj += _quad(opt_x["u_N", k, :], R)
        prev = opt_p["u_prev"] if k == 0 else opt_x["u_N", k - 1, :]
        obj += _quad(opt_x["u_N", k, :] - prev, Rd)

    if terminal is not None:
        obj += _quad(opt_x["x", N - 1, :] - cs.vertcat(opt_p["ref", N - 1, :]), S)

    # Model input: the input sequence alongside the current state, held constant
    # across the horizon -- the seq2seq predictor's input-output formulation.
    columns = [opt_x["u_N", :, j] for j in range(n_u)]
    columns += [cs.repmat(*opt_p[f"x{i + 1}"], N, 1) for i in range(n_model_states)]
    inputs = cs.horzcat(*columns)

    x_model = CasadiMamba(model, N)(inputs)
    constraint = cs.reshape(opt_x["x", :, :] - x_model[:, :n_x], N * n_x, 1)

    lbx, ubx = opt_x(-np.inf), opt_x(np.inf)
    for name, bounds in (("u_N", u_bounds), ("x", x_bounds)):
        if bounds is None:
            continue
        lo, hi = bounds
        if np.isscalar(lo):
            lbx[name], ubx[name] = lo, hi
        else:
            for j, (l, h) in enumerate(zip(lo, hi)):
                lbx[name, :, j], ubx[name, :, j] = l, h

    if solver_opts is None:
        solver_opts = {"ipopt.print_level": 0, "print_time": 0}
    nlp = {"x": opt_x, "f": obj, "g": constraint, "p": opt_p}

    return cs.nlpsol("S", solver, nlp, solver_opts), opt_x_num, opt_p_num, lbx, ubx
