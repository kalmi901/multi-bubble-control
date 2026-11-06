from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
RHS = Callable[..., FloatArray]
Event = Callable[..., FloatArray]


def _expand_env_vector(values: FloatArray, ndim: int) -> FloatArray:
    """Expand an environment vector for broadcasting over state dimensions."""
    return values.reshape((values.shape[0],) + (1,) * (ndim - 1))


def _evaluate_rhs(
    rhs: RHS,
    t: FloatArray,
    y: FloatArray,
    args: tuple[Any, ...],
) -> FloatArray:
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


def _evaluate_event(
    event: Event,
    t: FloatArray,
    y: FloatArray,
    args: tuple[Any, ...],
) -> FloatArray:
    """Evaluate and validate one terminal event function."""
    values = np.asarray(
        event(t, y, *args),
        dtype=np.float64,
    )

    expected_shape = (y.shape[0],)

    if values.shape != expected_shape:
        raise ValueError(
            "Event output must contain one scalar per environment: "
            f"expected {expected_shape}, got {values.shape}."
        )

    if not np.all(np.isfinite(values)):
        raise FloatingPointError("Event returned non-finite values.")

    return values


def rkck45_step(
    rhs: RHS,
    t: FloatArray,
    y: FloatArray,
    h: FloatArray,
    *,
    args: tuple[Any, ...] = (),
) -> tuple[FloatArray, FloatArray]:
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


def compute_scaled_error(
    error: FloatArray,
    y: FloatArray,
    y_trial: FloatArray,
    *,
    rtol: float,
    atol: float | FloatArray,
) -> FloatArray:
    """Scale a local truncation error using relative and absolute tolerances."""
    error = np.asarray(error, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    y_trial = np.asarray(y_trial, dtype=np.float64)
    atol = np.asarray(atol, dtype=np.float64)

    if error.shape != y.shape or y_trial.shape != y.shape:
        raise ValueError("error, y and y_trial must have identical shapes.")

    if rtol < 0.0:
        raise ValueError("rtol must be non-negative.")

    if np.any(atol < 0.0):
        raise ValueError("atol must be non-negative.")

    try:
        scale = atol + rtol * np.maximum(np.abs(y), np.abs(y_trial))

    except ValueError as exc:
        raise ValueError(
            f"atol with shape {atol.shape} is not broadcastable to state shape {y.shape}."
        ) from exc

    if np.any(scale <= 0.0):
        raise ValueError("Error scale must be positive for every state component.")

    return np.abs(error) / scale


def max_error_reducer(
    scaled_error: FloatArray,
) -> FloatArray:
    """Return the worst scaled component error for each environment."""
    scaled_error = np.asarray(
        scaled_error,
        dtype=np.float64,
    )

    if scaled_error.ndim < 2:
        raise ValueError("Scaled error must have shape (n_envs, *state_shape).")

    state_axes = tuple(range(1, scaled_error.ndim))

    return np.max(
        scaled_error,
        axis=state_axes,
    )


def update_step_size(
    error_norm: FloatArray,
    h: FloatArray,
    *,
    safety: float = 0.9,
    min_factor: float = 0.2,
    max_factor: float = 5.0,
) -> tuple[BoolArray, FloatArray]:
    """Accept/reject trial steps and propose new step sizes."""
    error_norm = np.asarray(error_norm, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64)

    if error_norm.shape != h.shape:
        raise ValueError("error_norm ahd  h must have identical shapes.")

    if error_norm.ndim != 1:
        raise ValueError("error_norm and h must have shape (num_envs, ).")

    if not np.all(np.isfinite(error_norm)):
        raise FloatingPointError("error_norm contains non-finite values.")

    accepted = error_norm <= 1.0

    factor = np.empty_like(error_norm)

    zero_error = error_norm == 0.0
    factor[zero_error] = max_factor

    nonzero = ~zero_error
    factor[nonzero] = safety * error_norm[nonzero] ** (-1.0 / 5.0)

    factor = np.clip(factor, min_factor, max_factor)

    h_new = h * factor

    return accepted, h_new


def adaptive_rkck45_step(
    rhs,
    t,
    y,
    h,
    *,
    rtol,
    atol,
    args=(),
    error_reducer=max_error_reducer,
):
    """Evaluate one adaptive batched RKCK45 trial step.

    Returns
    -------
    y_trial
        Fifth-order trial solution for every environment.

    h_new
        Suggested next step size for every environment.

    accepted
        Boolean acceptance mask with shape ``(n_envs,)``.

    error_norm
        Reduced scaled error for every environment.
    """
    y_trial, error = rkck45_step(
        rhs,
        t,
        y,
        h,
        args=args,
    )

    scaled_error = compute_scaled_error(
        error,
        y,
        y_trial,
        rtol=rtol,
        atol=atol,
    )

    error_norm = error_reducer(scaled_error)

    accepted, h_new = update_step_size(
        error_norm,
        h,
    )

    return y_trial, h_new, accepted, error_norm


def _localize_terminal_event(
    rhs: RHS,
    event: Event,
    t: FloatArray,
    y: FloatArray,
    h: FloatArray,
    event_mask: BoolArray,
    event_value: FloatArray,
    *,
    args: tuple[Any, ...],
    event_time_tolerance: float,
    max_event_iterations: int,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Locate a terminal event inside accepted RKCK45 steps.

    The event defines a valid domain through ``event > 0``. A detected
    positive-to-nonpositive crossing is bracketed and refined by bisection.
    The returned state is always taken from the positive (safe) side.
    """
    left_h = np.zeros_like(h)
    right_h = np.where(
        event_mask,
        h,
        0.0,
    )

    left_y = y.copy()
    left_value = event_value.copy()

    for _ in range(max_event_iterations):
        unresolved = event_mask & ((right_h - left_h) > event_time_tolerance)

        if not np.any(unresolved):
            break

        mid_h = np.zeros_like(h)
        mid_h[unresolved] = 0.5 * (left_h[unresolved] + right_h[unresolved])

        y_mid, _ = rkck45_step(
            rhs,
            t,
            y,
            mid_h,
            args=args,
        )

        t_mid = t + mid_h

        mid_value = _evaluate_event(
            event,
            t_mid,
            y_mid,
            args,
        )

        safe = unresolved & (mid_value > 0.0)
        crossed = unresolved & ~safe

        left_h[safe] = mid_h[safe]
        left_y[safe] = y_mid[safe]
        left_value[safe] = mid_value[safe]

        right_h[crossed] = mid_h[crossed]

    unresolved = event_mask & ((right_h - left_h) > event_time_tolerance)

    if np.any(unresolved):
        indices = np.flatnonzero(unresolved)
        raise RuntimeError(
            f"Terminal event localization did not converge for environment(s) {indices.tolist()}."
        )

    event_t = t + left_h

    return event_t, left_y, left_value, left_h


def solve_rkck45(
    rhs,
    y0,
    t_span,
    *,
    initial_step,
    rtol,
    atol,
    args=(),
    error_reducer=max_error_reducer,
    max_steps=100_000,
    store_trajectory=False,
    event=None,
    event_time_tolerance=1e-10,
    max_event_iterations=64,
):
    """Integrate a batch of ODE systems using adaptive RKCK45.

    When ``event`` is provided, it must return one scalar per environment.
    Positive values denote the valid integration domain. A
    positive-to-nonpositive crossing terminates that environment after
    localizing the event from the safe side.
    """
    y = np.asarray(y0, dtype=np.float64).copy()

    if y.ndim < 1:
        raise ValueError("y0 must contain an environment dimension.")

    if not np.all(np.isfinite(y)):
        raise ValueError("y0 contains non-finite values.")

    t_start, t_end = map(float, t_span)

    if not np.isfinite(t_start) or not np.isfinite(t_end):
        raise ValueError("t_span must contain finite values.")

    if t_end <= t_start:
        raise ValueError("t_span must satisfy t_end > t_start.")

    if not np.isfinite(event_time_tolerance):
        raise ValueError("event_time_tolerance must be finite.")

    if event_time_tolerance <= 0.0:
        raise ValueError("event_time_tolerance must be positive.")

    if max_event_iterations < 1:
        raise ValueError("max_event_iterations must be positive.")

    n_envs = y.shape[0]

    n_accepted = np.zeros(n_envs, dtype=np.int64)
    n_rejected = np.zeros(n_envs, dtype=np.int64)

    t = np.full(
        n_envs,
        t_start,
        dtype=np.float64,
    )

    h = np.asarray(
        initial_step,
        dtype=np.float64,
    )

    if h.ndim == 0:
        h = np.full(
            n_envs,
            float(h),
            dtype=np.float64,
        )
    elif h.shape != (n_envs,):
        raise ValueError(f"initial_step must be a scalar or have shape ({n_envs},).")
    else:
        h = h.copy()

    if not np.all(np.isfinite(h)):
        raise ValueError("initial_step contains non-finite values.")

    if np.any(h <= 0.0):
        raise ValueError("initial_step must be positive.")

    terminated = np.zeros(
        n_envs,
        dtype=bool,
    )

    event_triggered = np.zeros(
        n_envs,
        dtype=bool,
    )

    event_time = np.full(
        n_envs,
        np.nan,
        dtype=np.float64,
    )

    if event is not None:
        event_value = _evaluate_event(
            event,
            t,
            y,
            args,
        )

        initial_event = event_value <= 0.0

        terminated[initial_event] = True
        event_triggered[initial_event] = True
        event_time[initial_event] = t_start

    else:
        event_value = None

    if store_trajectory:
        trajectory_t = [[t_start] for _ in range(n_envs)]
        trajectory_y = [[y[env].copy()] for env in range(n_envs)]
    else:
        trajectory_t = None
        trajectory_y = None

    for iteration in range(max_steps):
        remaining = t_end - t

        active = (remaining > 0.0) & ~terminated

        if not np.any(active):
            if store_trajectory and trajectory_t is not None and trajectory_y is not None:
                trajectory = {
                    "t": [np.asarray(values, dtype=np.float64) for values in trajectory_t],
                    "y": [np.stack(values) for values in trajectory_y],
                }
            else:
                trajectory = None

            info = {
                "n_accepted": n_accepted,
                "n_rejected": n_rejected,
                "n_iterations": iteration,
                "trajectory": trajectory,
                "event_triggered": event_triggered,
                "event_time": event_time,
            }

            return t, y, info

        # Do not step beyond the requested endpoint.
        h_trial = np.minimum(
            h,
            remaining,
        )

        # Finished or terminated environments remain in the batch because
        # this keeps the RHS batch shape and environment-specific args
        # unchanged.
        h_trial = np.where(
            active,
            h_trial,
            0.0,
        )

        # Detect inability to advance floating-point time.
        underflow = active & ((t + h_trial) == t)

        if np.any(underflow):
            indices = np.flatnonzero(underflow)
            raise RuntimeError(f"RKCK45 step-size underflow for environment(s) {indices.tolist()}.")

        y_trial, h_new, accepted, _ = adaptive_rkck45_step(
            rhs,
            t,
            y,
            h_trial,
            rtol=rtol,
            atol=atol,
            args=args,
            error_reducer=error_reducer,
        )

        # Inactive environments must never be committed.
        accepted &= active

        rejected = active & ~accepted
        n_rejected[rejected] += 1

        event_hit = np.zeros(
            n_envs,
            dtype=bool,
        )

        event_trial_value = None
        event_t = None
        event_y = None
        event_safe_value = None
        event_step = None

        if event is not None and event_value is not None and np.any(accepted):
            event_eval_t = np.where(
                accepted,
                t + h_trial,
                t,
            )

            event_eval_y = y.copy()
            event_eval_y[accepted] = y_trial[accepted]

            event_trial_value = _evaluate_event(
                event,
                event_eval_t,
                event_eval_y,
                args,
            )

            event_hit = accepted & (event_value > 0.0) & (event_trial_value <= 0.0)

            if np.any(event_hit):
                (
                    event_t,
                    event_y,
                    event_safe_value,
                    event_step,
                ) = _localize_terminal_event(
                    rhs,
                    event,
                    t,
                    y,
                    h_trial,
                    event_hit,
                    event_value,
                    args=args,
                    event_time_tolerance=event_time_tolerance,
                    max_event_iterations=max_event_iterations,
                )

        normal_accepted = accepted & ~event_hit

        n_accepted[normal_accepted] += 1

        # Remember which normal accepted steps were endpoint-limited.
        endpoint_step = normal_accepted & (h_trial >= remaining)

        # Commit normal accepted systems.
        y[normal_accepted] = y_trial[normal_accepted]
        t[normal_accepted] += h_trial[normal_accepted]

        # Force exact endpoint equality instead of relying on
        # floating-point addition to reproduce t_end exactly.
        t[endpoint_step] = t_end

        if event is not None and event_value is not None and event_trial_value is not None:
            event_value[normal_accepted] = event_trial_value[normal_accepted]

        # Commit terminal event states from the safe side of the bracket.
        if np.any(event_hit):
            assert event_t is not None
            assert event_y is not None
            assert event_safe_value is not None
            assert event_step is not None
            assert event_value is not None

            event_advanced = event_hit & (event_step > 0.0)
            n_accepted[event_advanced] += 1

            y[event_hit] = event_y[event_hit]
            t[event_hit] = event_t[event_hit]
            event_value[event_hit] = event_safe_value[event_hit]

            terminated[event_hit] = True
            event_triggered[event_hit] = True
            event_time[event_hit] = t[event_hit]

        if store_trajectory and trajectory_t is not None and trajectory_y is not None:
            committed = normal_accepted | event_hit

            for env in np.flatnonzero(committed):
                # An event may lie within event_time_tolerance of the
                # current state. Avoid storing duplicate time points.
                if trajectory_t[env][-1] < t[env]:
                    trajectory_t[env].append(t[env])
                    trajectory_y[env].append(y[env].copy())

        # Accepted and rejected active systems both receive their
        # individually suggested next step size.
        h[active] = h_new[active]

    raise RuntimeError(
        f"RKCK45 exceeded max_steps={max_steps} before all environments reached the endpoint."
    )
