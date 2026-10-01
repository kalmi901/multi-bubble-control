from collections.abc import Callable
from typing import Any

import numpy as np

Array = np.ndarray
RHS = Callable[..., Array]


def _expand_env_vector(values: Array, ndim: int) -> Array:
    """Expand an environment vector for broadcasting over state dimensions."""
    return values.reshape((values.shape[0],) + (1,) * (ndim - 1))


def _evaluate_rhs(
    rhs: RHS,
    t: Array,
    y: Array,
    args: tuple[Any, ...],
) -> Array:
    """Evaluate and validate the right-hand side."""
    dydt = np.asarray(
        rhs(t, y, *args),
        dtype=np.float64,
    )

    if dydt.shape != y.shape:
        raise ValueError(
            "RHS output must have the same shape as the state: "
            f"expected {y.shape}, got {dydt.shape}."
        )

    if not np.all(np.isfinite(dydt)):
        raise FloatingPointError("RHS returned non-finite values.")

    return dydt


def rkck45_step(
    rhs: RHS,
    t: Array,
    y: Array,
    h: Array,
    *,
    args: tuple[Any, ...] = (),
) -> tuple[Array, Array]:
    """Evaluate one batched Runge-Kutta-Cash-Karp 4(5) step.

    Parameters
    ----------
    rhs
        Vectorized right-hand-side function with signature
        ``rhs(t, y, *args)``.

    t
        Current time for each environment, shape ``(n_envs,)``.

    y
        Current state, shape ``(n_envs, *state_shape)``.

    h
        Trial step size for each environment, shape ``(n_envs,)``.

    args
        Additional positional arguments forwarded unchanged to ``rhs``.

    Returns
    -------
    y5
        Fifth-order solution estimate, with the same shape as ``y``.

    error
        Embedded local error estimate ``y5 - y4``, with the same
        shape as ``y``.
    """
    y = np.asarray(y, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64)

    if y.ndim < 1:
        raise ValueError("State must contain an environment dimension.")

    n_envs = y.shape[0]

    if t.shape != (n_envs,):
        raise ValueError(f"t must have shape ({n_envs},), got {t.shape}.")

    if h.shape != (n_envs,):
        raise ValueError(f"h must have shape ({n_envs},), got {h.shape}.")

    hb = _expand_env_vector(h, y.ndim)

    k1 = _evaluate_rhs(
        rhs,
        t,
        y,
        args,
    )

    k2 = _evaluate_rhs(
        rhs,
        t + h * (1.0 / 5.0),
        y + hb * ((1.0 / 5.0) * k1),
        args,
    )

    k3 = _evaluate_rhs(
        rhs,
        t + h * (3.0 / 10.0),
        y + hb * ((3.0 / 40.0) * k1 + (9.0 / 40.0) * k2),
        args,
    )

    k4 = _evaluate_rhs(
        rhs,
        t + h * (3.0 / 5.0),
        y + hb * ((3.0 / 10.0) * k1 - (9.0 / 10.0) * k2 + (6.0 / 5.0) * k3),
        args,
    )

    k5 = _evaluate_rhs(
        rhs,
        t + h,
        y + hb * ((-11.0 / 54.0) * k1 + (5.0 / 2.0) * k2 - (70.0 / 27.0) * k3 + (35.0 / 27.0) * k4),
        args,
    )

    k6 = _evaluate_rhs(
        rhs,
        t + h * (7.0 / 8.0),
        y
        + hb
        * (
            (1631.0 / 55296.0) * k1
            + (175.0 / 512.0) * k2
            + (575.0 / 13824.0) * k3
            + (44275.0 / 110592.0) * k4
            + (253.0 / 4096.0) * k5
        ),
        args,
    )

    y5 = y + hb * (
        (37.0 / 378.0) * k1 + (250.0 / 621.0) * k3 + (125.0 / 594.0) * k4 + (512.0 / 1771.0) * k6
    )

    y4 = y + hb * (
        (2825.0 / 27648.0) * k1
        + (18575.0 / 48384.0) * k3
        + (13525.0 / 55296.0) * k4
        + (277.0 / 14336.0) * k5
        + (1.0 / 4.0) * k6
    )

    error = y5 - y4

    return y5, error
