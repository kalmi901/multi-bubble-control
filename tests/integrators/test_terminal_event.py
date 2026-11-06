import numpy as np
import pytest
from numpy.testing import assert_allclose

from multibubble.integrators import solve_rkck45


def _constant_decay_rhs(t, y):
    return -np.ones_like(y)


def _state_event(t, y):
    return y[:, 0]


def test_terminal_event_localizes_known_root_from_safe_side():
    y0 = np.array([[1.0]])

    t, y, info = solve_rkck45(
        _constant_decay_rhs,
        y0,
        t_span=(0.0, 2.0),
        initial_step=0.7,
        rtol=1e-9,
        atol=1e-12,
        event=_state_event,
        event_time_tolerance=1e-10,
        store_trajectory=True,
    )

    assert info["event_triggered"][0]
    assert_allclose(
        info["event_time"][0],
        1.0,
        rtol=0.0,
        atol=1e-10,
    )
    assert_allclose(
        t[0],
        1.0,
        rtol=0.0,
        atol=1e-10,
    )

    # The localized state must remain on the valid side.
    assert y[0, 0] >= 0.0
    assert y[0, 0] < 2e-10

    trajectory = info["trajectory"]
    assert trajectory is not None
    assert trajectory["t"][0][-1] == t[0]
    assert_allclose(
        trajectory["y"][0][-1],
        y[0],
    )


def test_terminal_events_are_independent_between_environments():
    y0 = np.array(
        [
            [0.5],
            [1.0],
            [2.0],
        ]
    )

    expected_event_time = y0[:, 0]

    t, y, info = solve_rkck45(
        _constant_decay_rhs,
        y0,
        t_span=(0.0, 3.0),
        initial_step=np.array([0.4, 0.7, 1.3]),
        rtol=1e-9,
        atol=1e-12,
        event=_state_event,
        event_time_tolerance=1e-10,
    )

    assert np.all(info["event_triggered"])

    assert_allclose(
        info["event_time"],
        expected_event_time,
        rtol=0.0,
        atol=1e-10,
    )

    assert_allclose(
        t,
        expected_event_time,
        rtol=0.0,
        atol=1e-10,
    )

    assert np.all(y[:, 0] >= 0.0)
    assert np.all(y[:, 0] < 2e-10)


def test_nontriggering_environment_continues_to_endpoint():
    y0 = np.array(
        [
            [1.0],
            [10.0],
        ]
    )

    t, y, info = solve_rkck45(
        _constant_decay_rhs,
        y0,
        t_span=(0.0, 2.0),
        initial_step=0.7,
        rtol=1e-9,
        atol=1e-12,
        event=_state_event,
        event_time_tolerance=1e-10,
    )

    assert info["event_triggered"].tolist() == [True, False]

    assert_allclose(
        t,
        np.array([1.0, 2.0]),
        rtol=0.0,
        atol=1e-10,
    )

    assert_allclose(
        y[:, 0],
        np.array([0.0, 8.0]),
        rtol=0.0,
        atol=2e-10,
    )

    assert_allclose(
        info["event_time"][0],
        1.0,
        rtol=0.0,
        atol=1e-10,
    )

    assert np.isnan(info["event_time"][1])


def test_initial_nonpositive_event_terminates_immediately():
    y0 = np.array(
        [
            [0.0],
            [-1.0],
            [1.0],
        ]
    )

    t, _, info = solve_rkck45(
        _constant_decay_rhs,
        y0,
        t_span=(0.0, 2.0),
        initial_step=0.2,
        rtol=1e-9,
        atol=1e-12,
        event=_state_event,
    )

    assert info["event_triggered"].tolist() == [True, True, True]

    assert_allclose(
        t[:2],
        0.0,
        rtol=0.0,
        atol=0.0,
    )

    assert_allclose(
        t[2],
        1.0,
        rtol=0.0,
        atol=1e-10,
    )

    assert_allclose(
        info["event_time"][:2],
        0.0,
        rtol=0.0,
        atol=0.0,
    )


def test_event_output_must_have_one_scalar_per_environment():
    y0 = np.ones((2, 1))

    def bad_event(t, y):
        return y

    with pytest.raises(
        ValueError,
        match="one scalar per environment",
    ):
        solve_rkck45(
            _constant_decay_rhs,
            y0,
            t_span=(0.0, 1.0),
            initial_step=0.1,
            rtol=1e-6,
            atol=1e-9,
            event=bad_event,
        )


def test_nonfinite_event_value_raises():
    y0 = np.ones((1, 1))

    def bad_event(t, y):
        return np.array([np.nan])

    with pytest.raises(
        FloatingPointError,
        match="Event returned non-finite values",
    ):
        solve_rkck45(
            _constant_decay_rhs,
            y0,
            t_span=(0.0, 1.0),
            initial_step=0.1,
            rtol=1e-6,
            atol=1e-9,
            event=bad_event,
        )
