import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from multibubble.integrators import adaptive_rkck45_step


def test_adaptive_step_accepts_and_rejects_independently():
    t = np.zeros(2)
    y0 = np.ones((2, 1))

    # Same ODE, very different trial step sizes.
    h = np.array([0.01, 1.0])

    def rhs(t, y):
        return y

    _, h_new, accepted, error_norm = adaptive_rkck45_step(
        rhs,
        t,
        y0,
        h,
        rtol=1e-6,
        atol=1e-9,
    )

    assert accepted.shape == (2,)
    assert accepted[0]
    assert not accepted[1]

    assert error_norm[0] <= 1.0
    assert error_norm[1] > 1.0

    assert h_new[1] < h[1]


def test_adaptive_step_forwards_rhs_arguments():
    t = np.zeros(2)
    h = np.array([0.1, 0.2])
    y0 = np.ones((2, 1))

    rate = np.array([2.0, 3.0])

    # dy/dt = rate
    # Constant RHS -> Cash-Karp integrates it exactly.
    def rhs(t, y, rate):
        return np.broadcast_to(rate[:, None], y.shape)

    y_trial, h_new, accepted, error_norm = adaptive_rkck45_step(
        rhs,
        t,
        y0,
        h,
        rtol=1e-6,
        atol=1e-9,
        args=(rate,),
    )

    expected = y0 + rate[:, None] * h[:, None]

    assert_allclose(y_trial, expected, rtol=1e-14, atol=1e-14)

    # Constant RHS produces zero embedded error.
    assert_allclose(error_norm, 0.0)
    assert_array_equal(accepted, np.array([True, True]))

    # Zero error should allow the controller to increase the step.
    assert np.all(h_new > h)


def test_adaptive_step_uses_custom_error_reducer():
    t = np.zeros(2)
    h = np.array([0.1, 0.1])
    y0 = np.ones((2, 1))

    def rhs(t, y):
        return y

    def custom_error_reducer(scaled_error):
        assert scaled_error.shape == y0.shape

        # Deliberately override the actual numerical error:
        # environment 0 -> accept
        # environment 1 -> reject
        return np.array([0.5, 2.0])

    _, h_new, accepted, error_norm = adaptive_rkck45_step(
        rhs,
        t,
        y0,
        h,
        rtol=1e-6,
        atol=1e-9,
        error_reducer=custom_error_reducer,
    )

    assert_allclose(
        error_norm,
        np.array([0.5, 2.0]),
    )

    assert_array_equal(
        accepted,
        np.array([True, False]),
    )

    assert h_new[0] > h[0]
    assert h_new[1] < h[1]
