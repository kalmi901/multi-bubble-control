import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from multibubble.integrators.rkck45 import compute_scaled_error, max_error_reducer, update_step_size


def test_scaled_error_uses_relative_and_absolute_tolerances():
    y = np.array(
        [
            [[1.0, 10.0]],
            [[2.0, 20.0]],
        ]
    )

    y_trial = np.array(
        [
            [[1.1, 11.0]],
            [[2.2, 22.0]],
        ]
    )

    error = np.array(
        [
            [[1e-3, 1e-2]],
            [[2e-3, 2e-2]],
        ]
    )

    rtol = 1e-3
    atol = np.array([1e-6, 1e-4])

    scaled = compute_scaled_error(
        error=error,
        y=y,
        y_trial=y_trial,
        rtol=rtol,
        atol=atol,
    )

    scale = atol + rtol * np.maximum(
        np.abs(y),
        np.abs(y_trial),
    )

    expected = np.abs(error) / scale

    assert_allclose(scaled, expected)


def test_max_error_reducer_uses_worst_state_component():
    scaled_error = np.full(
        (2, 32, 2),
        0.1,
    )

    # One problematic equation in environment 0.
    scaled_error[0, 17, 1] = 1.25

    # Environment 1 remains completely acceptable.
    scaled_error[1, 5, 0] = 0.95

    result = max_error_reducer(scaled_error)

    assert_allclose(
        result,
        np.array([1.25, 0.95]),
    )


def test_componentwise_atol_broadcasts_over_environments_and_bubbles():
    y = np.ones((4, 3, 2))
    y_trial = y.copy()
    error = np.ones_like(y)

    atol = np.array([1.0, 2.0])

    scaled = compute_scaled_error(
        error=error,
        y=y,
        y_trial=y_trial,
        rtol=0.0,
        atol=atol,
    )

    assert_allclose(scaled[..., 0], 1.0)
    assert_allclose(scaled[..., 1], 0.5)


def test_custom_coupled_reducer_can_select_worst_subsystem():
    scaled_error = np.array(
        [
            [
                [0.1, 0.2],
                [0.2, 0.1],
                [3.0, 4.0],
            ]
        ]
    )

    def worst_subsystem_reducer(values):
        subsystem_error = np.sqrt(np.mean(values**2, axis=-1))
        return np.max(subsystem_error, axis=-1)

    result = worst_subsystem_reducer(scaled_error)

    expected = np.array([np.sqrt((3.0**2 + 4.0**2) / 2.0)])

    assert_allclose(result, expected)


def test_step_update_accepts_and_rejects_independently():
    error_norm = np.array([0.25, 1.0, 4.0])
    h = np.array([0.1, 0.1, 0.1])

    accepted, h_new = update_step_size(error_norm, h)

    assert_array_equal(accepted, np.array([True, True, False]))

    assert h_new[0] > h[0]
    assert h_new[2] < h[2]


def test_step_update_uses_max_growth_for_zero_error():
    error_norm = np.array([0.0])
    h = np.array([0.1])

    accepted, h_new = update_step_size(error_norm, h, max_factor=5.0)

    assert accepted[0]
    assert_allclose(h_new, np.array([0.5]))


def test_step_update_limits_step_reduction():
    error_norm = np.array([1e10])
    h = np.array([0.1])

    accepted, h_new = update_step_size(
        error_norm,
        h,
        min_factor=0.2,
    )

    assert not accepted[0]
    assert_allclose(h_new, np.array([0.02]))


def test_step_update_matches_controller_formula():
    error_norm = np.array([0.5])
    h = np.array([0.2])

    _, h_new = update_step_size(
        error_norm,
        h,
        safety=0.9,
    )

    expected = 0.2 * 0.9 * 0.5 ** (-1.0 / 5.0)

    assert_allclose(h_new, np.array([expected]))
