import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from multibubble.integrators import rkck45_step


def test_rkck45_step_supports_batched_independent_step_sizes():
    t = np.zeros(3)
    h = np.array([0.05, 0.10, 0.20])

    y0 = np.array(
        [
            [1.0, 2.0],
            [1.0, 2.0],
            [1.0, 2.0],
        ]
    )
    y0_before = y0.copy()

    # dy/dt = y
    def rhs(t, y):
        return y

    y5, error = rkck45_step(rhs, t, y0, h)

    expected = y0 * np.exp(h)[:, None]

    assert_allclose(y5, expected, rtol=1e-7, atol=1e-12)
    assert error.shape == y0.shape

    # An integration step must not modify the input state.
    assert_array_equal(y0, y0_before)


def test_rkck45_step_supports_batched_nonautonomous_rhs():
    t = np.array([0.0, 0.5, 1.0])
    h = np.array([0.10, 0.05, 0.20])
    y0 = np.zeros((3, 1))

    # dy/dt = t
    def rhs(t, y):
        return np.broadcast_to(t[:, None], y.shape)

    y5, _ = rkck45_step(rhs, t, y0, h)

    expected = y0 + t[:, None] * h[:, None] + 0.5 * h[:, None] ** 2

    assert_allclose(y5, expected, rtol=1e-13, atol=1e-14)


def test_rkck45_step_passes_rhs_arguments():
    t = np.zeros(2)
    h = np.array([0.1, 0.2])
    y0 = np.ones((2, 1))

    rate = np.array([1.0, 2.0])

    # dy/dt = rate
    def rhs(t, y, rate):
        return np.broadcast_to(rate[:, None], y.shape)

    y5, _ = rkck45_step(
        rhs,
        t,
        y0,
        h,
        args=(rate,),
    )

    expected = y0 + rate[:, None] * h[:, None]

    assert_allclose(y5, expected, rtol=1e-14, atol=1e-14)


def test_rkck45_step_supports_multidimensional_state():
    t = np.zeros(2)
    h = np.array([0.05, 0.10])

    # (n_envs, n_bubbles, n_state)
    y0 = np.ones((2, 3, 2))

    def rhs(t, y):
        return y

    y5, error = rkck45_step(rhs, t, y0, h)

    expected = y0 * np.exp(h)[:, None, None]

    assert y5.shape == (2, 3, 2)
    assert error.shape == (2, 3, 2)

    assert_allclose(
        y5,
        expected,
        rtol=1e-7,
        atol=1e-12,
    )
