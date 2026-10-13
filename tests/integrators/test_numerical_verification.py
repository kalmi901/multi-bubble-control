import numpy as np
import pytest
from numpy.testing import assert_allclose

from multibubble.integrators import rkck45_step, solve_rkck45


def test_exponential_decay_matches_analytical_solution():
    decay_rate = 2.0

    y0 = np.array([[1.0]])

    def rhs(t, y, rate):
        return -rate * y

    t, y, _ = solve_rkck45(
        rhs,
        y0,
        t_span=(0.0, 2.0),
        initial_step=0.2,
        rtol=1e-9,
        atol=1e-11,
        args=(decay_rate,),
    )

    expected = np.array([[np.exp(-decay_rate * 2.0)]])

    assert_allclose(
        t,
        np.array([2.0]),
        rtol=0.0,
        atol=0.0,
    )

    assert_allclose(
        y,
        expected,
        rtol=1e-8,
        atol=1e-11,
    )


def test_harmonic_oscillator_matches_analytical_solution():
    omega = 1.0
    t_end = 4.0 * np.pi

    # state = [position, velocity]
    y0 = np.array([[1.0, 0.0]])

    def rhs(t, y, omega):
        dydt = np.empty_like(y)

        dydt[:, 0] = y[:, 1]
        dydt[:, 1] = -(omega**2) * y[:, 0]

        return dydt

    t, y, _ = solve_rkck45(
        rhs,
        y0,
        t_span=(0.0, t_end),
        initial_step=0.2,
        rtol=1e-9,
        atol=np.array([1e-11, 1e-11]),
        args=(omega,),
    )

    expected = np.array(
        [
            [
                np.cos(omega * t_end),
                -omega * np.sin(omega * t_end),
            ]
        ]
    )

    assert_allclose(
        t,
        np.array([t_end]),
        rtol=0.0,
        atol=0.0,
    )

    assert_allclose(
        y,
        expected,
        rtol=1e-7,
        atol=1e-9,
    )


def test_nonautonomous_system_matches_analytical_solution():
    y0 = np.array([[0.0]])
    t_end = 3.0

    def rhs(t, y):
        return np.broadcast_to(t[:, None], y.shape)

    t, y, _ = solve_rkck45(
        rhs,
        y0,
        t_span=(0.0, t_end),
        initial_step=0.4,
        rtol=1e-9,
        atol=1e-11,
    )

    expected = np.array([[0.5 * t_end**2]])

    assert_allclose(
        t,
        np.array([t_end]),
        rtol=0.0,
        atol=0.0,
    )

    assert_allclose(
        y,
        expected,
        rtol=1e-10,
        atol=1e-12,
    )


def test_tighter_tolerances_reduce_global_error():
    decay_rate = 2.0
    t_end = 2.0
    y0 = np.array([[1.0]])

    def rhs(t, y, rate):
        return -rate * y

    expected = np.exp(-decay_rate * t_end)

    tolerances = [
        (1e-4, 1e-6),
        (1e-6, 1e-8),
        (1e-8, 1e-10),
    ]

    errors = []

    for rtol, atol in tolerances:
        _, y, _ = solve_rkck45(
            rhs,
            y0,
            t_span=(0.0, t_end),
            initial_step=0.5,
            rtol=rtol,
            atol=atol,
            args=(decay_rate,),
        )

        errors.append(abs(y[0, 0] - expected))

    assert errors[1] < errors[0]
    assert errors[2] < errors[1]


def test_batched_solution_matches_independent_solves():
    rates = np.array([0.5, 1.0, 2.0])
    y0 = np.ones((3, 1))
    t_end = 2.0

    def batched_rhs(t, y, rate):
        return -rate[:, None] * y

    _, y_batch, _ = solve_rkck45(
        batched_rhs,
        y0,
        t_span=(0.0, t_end),
        initial_step=np.array([0.2, 0.15, 0.1]),
        rtol=1e-9,
        atol=1e-11,
        args=(rates,),
    )

    y_independent = []

    for env, rate in enumerate(rates):

        def rhs(t, y, rate=rate):
            return -rate * y

        _, y_single, _ = solve_rkck45(
            rhs,
            y0[env : env + 1],
            t_span=(0.0, t_end),
            initial_step=0.1,
            rtol=1e-9,
            atol=1e-11,
        )

        y_independent.append(y_single[0])

    y_independent = np.stack(y_independent)

    assert_allclose(
        y_batch,
        y_independent,
        rtol=1e-9,
        atol=1e-11,
    )


def test_cash_karp_embedded_pair_has_expected_local_orders():
    y0 = np.array([[1.0]])
    t = np.array([0.0])

    def rhs(t, y):
        return y

    def errors_for_step(h_value):
        h = np.array([h_value])

        y5, error_estimate = rkck45_step(
            rhs,
            t,
            y0,
            h,
        )

        y4 = y5 - error_estimate
        exact = np.exp(h_value)

        error_y4 = abs(y4[0, 0] - exact)
        error_y5 = abs(y5[0, 0] - exact)
        error_embedded = abs(error_estimate[0, 0])

        return error_y4, error_y5, error_embedded

    error4_h, error5_h, embedded_h = errors_for_step(0.1)
    error4_h2, error5_h2, embedded_h2 = errors_for_step(0.05)

    order4_local = np.log2(error4_h / error4_h2)
    order5_local = np.log2(error5_h / error5_h2)
    order_embedded = np.log2(embedded_h / embedded_h2)

    # Fourth-order RK solution:
    # local truncation error ~ O(h^5).
    assert 4.7 < order4_local < 5.2

    # Fifth-order RK solution:
    # local truncation error ~ O(h^6).
    assert 5.7 < order5_local < 6.3

    # Embedded difference y5 - y4 is dominated by the
    # fourth-order local error and therefore scales as O(h^5).
    assert 4.7 < order_embedded < 5.2


def test_highly_dynamic_subsystem_controls_coupled_step_size():
    t_end = 2.0 * np.pi

    # state shape:
    # (n_envs, n_subsystems, [position, velocity])
    y0 = np.zeros((2, 2, 2))
    y0[..., 0] = 1.0

    # Environment 0:
    #   both subsystems are slow.
    #
    # Environment 1:
    #   subsystem 0 is slow,
    #   subsystem 1 is much faster.
    omega = np.array(
        [
            [1.0, 1.0],
            [1.0, 20.0],
        ]
    )

    def rhs(t, y, omega):
        dydt = np.empty_like(y)

        dydt[..., 0] = y[..., 1]
        dydt[..., 1] = -(omega**2) * y[..., 0]

        return dydt

    t, y, info = solve_rkck45(
        rhs,
        y0,
        t_span=(0.0, t_end),
        initial_step=0.1,
        rtol=1e-8,
        atol=np.array([1e-10, 1e-10]),
        args=(omega,),
    )

    assert_allclose(
        t,
        np.array([t_end, t_end]),
        rtol=0.0,
        atol=0.0,
    )

    # The environment containing the fast subsystem must use
    # more accepted steps, hence a smaller average common step size.
    assert info["n_accepted"][1] > info["n_accepted"][0]

    # Both environments must still produce accurate final states.
    expected = np.zeros_like(y0)

    expected[..., 0] = np.cos(omega * t_end)
    expected[..., 1] = -omega * np.sin(omega * t_end)

    position_error = np.abs(y[..., 0] - expected[..., 0])

    velocity_error = np.abs(y[..., 1] - expected[..., 1])

    # Position amplitude scale is O(1).
    assert np.max(position_error) < 1e-6

    # Velocity amplitude scale is O(omega), so compare
    # dimensionlessly against each oscillator's natural scale.
    scaled_velocity_error = velocity_error / omega

    assert np.max(scaled_velocity_error) < 1e-6


def test_solver_raises_when_max_steps_is_exceeded():
    y0 = np.array([[1.0]])

    def rhs(t, y):
        return y

    with pytest.raises(RuntimeError, match="exceeded max_steps"):
        solve_rkck45(
            rhs,
            y0,
            t_span=(0.0, 10.0),
            initial_step=1e-3,
            rtol=1e-10,
            atol=1e-12,
            max_steps=1,
        )


def test_solver_raises_on_nonfinite_rhs():
    y0 = np.array([[1.0]])

    def rhs(t, y):
        return np.full_like(y, np.nan)

    with pytest.raises(
        FloatingPointError,
        match="RHS returned non-finite values",
    ):
        solve_rkck45(
            rhs,
            y0,
            t_span=(0.0, 1.0),
            initial_step=0.1,
            rtol=1e-6,
            atol=1e-9,
        )


def test_solver_rejects_nonfinite_initial_state():
    y0 = np.array([[np.nan]])

    def rhs(t, y):
        return y

    with pytest.raises(
        ValueError,
        match="y0 contains non-finite values",
    ):
        solve_rkck45(
            rhs,
            y0,
            t_span=(0.0, 1.0),
            initial_step=0.1,
            rtol=1e-6,
            atol=1e-9,
        )


def test_solver_raises_on_step_size_underflow():
    y0 = np.array([[1.0]])

    def rhs(t, y):
        return y

    t_start = 1e16
    t_end = t_start + 10.0

    with pytest.raises(
        RuntimeError,
        match="step-size underflow",
    ):
        solve_rkck45(
            rhs,
            y0,
            t_span=(t_start, t_end),
            initial_step=1e-6,
            rtol=1e-6,
            atol=1e-9,
        )


def test_rejected_environment_does_not_affect_other_environment():
    y0 = np.ones((2, 1))

    def rhs(t, y):
        return y

    _, y_batch, info_batch = solve_rkck45(
        rhs,
        y0,
        t_span=(0.0, 1.0),
        initial_step=np.array([0.01, 1.0]),
        rtol=1e-10,
        atol=1e-12,
    )

    _, y_single, info_single = solve_rkck45(
        rhs,
        y0[0:1],
        t_span=(0.0, 1.0),
        initial_step=0.01,
        rtol=1e-10,
        atol=1e-12,
    )

    # Environment 1 must actually exercise rejection.
    assert info_batch["n_rejected"][1] > 0

    # Environment 0 must behave exactly as if it had been integrated alone.
    assert_allclose(
        y_batch[0],
        y_single[0],
        rtol=1e-12,
        atol=1e-12,
    )

    assert info_batch["n_accepted"][0] == info_single["n_accepted"][0]

    assert info_batch["n_rejected"][0] == info_single["n_rejected"][0]
