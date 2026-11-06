from types import SimpleNamespace

import numpy as np
from numpy.testing import assert_allclose

from multibubble.physics.coupled_bubbles import (
    coupled_bubbles_collision_event,
)
from multibubble.physics.coupled_translating_bubbles import (
    coupled_translating_bubble_collision_event,
)


def test_fixed_position_collision_event_returns_minimum_pairwise_margin():
    position = np.zeros(
        (3, 3, 3),
        dtype=np.float64,
    )

    # Environment 0:
    # pair distances = 4, 7, 3
    position[0, :, 0] = [0.0, 4.0, 7.0]

    # Environment 1:
    # minimum distance = 2.04 -> exactly safety boundary
    position[1, :, 0] = [0.0, 4.0, 6.04]

    # Environment 2:
    # minimum distance = 2.00 -> inside safety boundary
    position[2, :, 0] = [0.0, 4.0, 6.0]

    params = SimpleNamespace(
        position=position,
    )

    y = np.zeros(
        (3, 3, 2),
        dtype=np.float64,
    )

    y[..., 0] = 1.0

    value = coupled_bubbles_collision_event(
        np.zeros(3),
        y,
        params,  # type: ignore[arg-type]
        safety_factor=1.02,
    )

    # R_i + R_j = 2
    # safety contact distance = 1.02 * 2 = 2.04
    assert_allclose(
        value,
        np.array(
            [
                0.96,
                0.0,
                -0.04,
            ]
        ),
        rtol=0.0,
        atol=1e-14,
    )


def test_translating_collision_event_uses_current_state_positions():
    y = np.zeros(
        (3, 3, 8),
        dtype=np.float64,
    )

    y[..., 0] = 1.0

    y[0, :, 1] = [0.0, 4.0, 7.0]
    y[1, :, 1] = [0.0, 4.0, 6.04]
    y[2, :, 1] = [0.0, 4.0, 6.0]

    params = SimpleNamespace()

    value = coupled_translating_bubble_collision_event(
        np.zeros(3),
        y,
        params,  # type: ignore[arg-type]
        safety_factor=1.02,
    )

    assert_allclose(
        value,
        np.array(
            [
                0.96,
                0.0,
                -0.04,
            ]
        ),
        rtol=0.0,
        atol=1e-14,
    )
