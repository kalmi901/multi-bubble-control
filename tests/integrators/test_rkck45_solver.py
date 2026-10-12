import numpy as np
from numpy.testing import assert_allclose

from multibubble.integrators import solve_rkck45


def test_solver_reaches_exact_endpoint():
    y0 = np.ones((2, 1))

    def rhs(t, y):
        return y

    t, y = solve_rkck45(
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

    t, y = solve_rkck45(
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

    t_batch, y_batch = solve_rkck45(
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
