import numpy as np
from numpy.testing import assert_allclose

from multibubble.integrators import solve_rkck45


def test_solver_reaches_exact_endpoint():
    y0 = np.ones((2, 1))

    def rhs(t, y):
        return y

    t, y, _ = solve_rkck45(
        rhs,
        y0,
        t_span=(0.0, 1.0),
        initial_step=0.3,
        rtol=1e-8,
        atol=1e-10,
    )

    assert_allclose(t, np.array([1.0, 1.0]), rtol=0.0, atol=0.0)

    expected = np.exp(1.0) * y0
    assert_allclose(y, expected, rtol=1e-7)


def test_solver_retries_rejected_steps():
    y0 = np.ones((1, 1))

    def rhs(t, y):
        return y

    t, y, _ = solve_rkck45(
        rhs,
        y0,
        t_span=(0.0, 1.0),
        initial_step=1.0,  # deliberately aggressive
        rtol=1e-10,
        atol=1e-12,
    )

    assert_allclose(t, np.array([1.0]), rtol=0.0, atol=0.0)
    assert_allclose(y, np.exp(1.0) * y0, rtol=1e-9)


def test_batched_solver_matches_independent_solves():
    rates = np.array([0.5, 1.0, 2.0])
    y0 = np.ones((3, 1))

    def rhs(t, y, rate):
        return rate[:, None] * y

    t_batch, y_batch, _ = solve_rkck45(
        rhs,
        y0,
        t_span=(0.0, 1.0),
        initial_step=np.array([0.1, 0.2, 0.3]),
        rtol=1e-9,
        atol=1e-11,
        args=(rates,),
    )

    assert_allclose(t_batch, 1.0, rtol=0.0, atol=0.0)

    expected = np.exp(rates)[:, None]

    assert_allclose(
        y_batch,
        expected,
        rtol=1e-8,
        atol=1e-10,
    )


def test_solver_reports_step_statistics():
    y0 = np.ones((1, 1))

    def rhs(t, y):
        return y

    t, y, info = solve_rkck45(
        rhs,
        y0,
        t_span=(0.0, 1.0),
        initial_step=1.0,
        rtol=1e-10,
        atol=1e-12,
    )

    assert_allclose(t, np.array([1.0]), rtol=0.0, atol=0.0)
    assert_allclose(y, np.exp(1.0) * y0, rtol=1e-9)

    assert info["n_accepted"].shape == (1,)
    assert info["n_rejected"].shape == (1,)

    assert info["n_accepted"].dtype == np.int64
    assert info["n_rejected"].dtype == np.int64

    assert info["n_accepted"][0] > 0

    # The deliberately aggressive first step should be rejected
    # under the strict tolerance.
    assert info["n_rejected"][0] > 0

    assert info["n_iterations"] >= 1


def test_solver_does_not_store_trajectory_by_default():
    y0 = np.ones((2, 1))

    def rhs(t, y):
        return y

    _, _, info = solve_rkck45(
        rhs,
        y0,
        t_span=(0.0, 1.0),
        initial_step=0.1,
        rtol=1e-8,
        atol=1e-10,
    )

    assert info["trajectory"] is None


def test_solver_stores_accepted_trajectory_when_requested():
    y0 = np.array(
        [
            [1.0],
            [2.0],
        ]
    )

    def rhs(t, y):
        return y

    t_final, y_final, info = solve_rkck45(
        rhs,
        y0,
        t_span=(0.0, 1.0),
        initial_step=np.array([0.1, 0.2]),
        rtol=1e-8,
        atol=1e-10,
        store_trajectory=True,
    )

    trajectory = info["trajectory"]

    assert trajectory is not None

    for env in range(2):
        t_history = trajectory["t"][env]
        y_history = trajectory["y"][env]

        # Initial state is stored.
        assert t_history[0] == 0.0
        assert_allclose(
            y_history[0],
            y0[env],
        )

        # Final accepted state is stored.
        assert t_history[-1] == 1.0
        assert_allclose(
            y_history[-1],
            y_final[env],
        )

        # Stored trajectory must advance monotonically.
        assert np.all(np.diff(t_history) > 0.0)

        assert_allclose(
            t_final[env],
            1.0,
            rtol=0.0,
            atol=0.0,
        )


def test_trajectory_contains_only_accepted_steps():
    y0 = np.ones((1, 1))

    def rhs(t, y):
        return y

    _, _, info = solve_rkck45(
        rhs,
        y0,
        t_span=(0.0, 1.0),
        initial_step=1.0,
        rtol=1e-10,
        atol=1e-12,
        store_trajectory=True,
    )

    trajectory = info["trajectory"]

    assert trajectory is not None

    n_accepted = info["n_accepted"][0]
    n_rejected = info["n_rejected"][0]

    # Make sure this run actually exercised rejection.
    assert n_rejected > 0

    t_history = trajectory["t"][0]
    y_history = trajectory["y"][0]

    # One stored point for the initial condition plus one for
    # every accepted step. Rejected trials must not appear.
    assert len(t_history) == n_accepted + 1
    assert len(y_history) == n_accepted + 1

    assert np.all(np.diff(t_history) > 0.0)
