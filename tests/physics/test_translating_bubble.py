import numpy as np
from numpy.testing import assert_allclose

from multibubble.physics import MaterialProperties
from multibubble.physics.acoustic_field import AcousticField, PlaneStandingWave, UniformHarmonic
from multibubble.physics.common import ReferenceScales
from multibubble.physics.translating_bubble import (
    TranslatingBubbleParameters,
    prepare_translating_bubble_parameters,
    translating_bubble_rhs,
)


def test_unforced_equilibrium_has_zero_rhs():
    material = MaterialProperties()

    bubble = TranslatingBubbleParameters(
        equilibrium_radius=50e-6,
    )

    field = AcousticField(
        components=(
            UniformHarmonic(
                pressure_amplitude=0.0,
                frequency=25e3,
            ),
        )
    )

    reference = ReferenceScales(
        radius=100e-6,
        frequency=25e3,
    )

    params = prepare_translating_bubble_parameters(
        material,
        bubble,
        field,
        reference,
        n_envs=1,
    )

    y = np.array(
        [[0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
        dtype=np.float64,
    )

    dydt = translating_bubble_rhs(
        np.array([0.0]),
        y,
        params,
    )

    assert dydt.shape == y.shape

    assert_allclose(
        dydt,
        0.0,
        rtol=0.0,
        atol=1e-13,
    )


def test_rhs_preserves_batch_shape_and_coordinate_derivative_equals_velocity():
    material = MaterialProperties()

    radii = np.array(
        [40e-6, 50e-6, 60e-6],
        dtype=np.float64,
    )

    frequencies = np.array(
        [20e3, 25e3, 30e3],
        dtype=np.float64,
    )

    bubble = TranslatingBubbleParameters(
        equilibrium_radius=radii,
    )

    field = AcousticField(
        components=(
            UniformHarmonic(
                pressure_amplitude=0.0,
                frequency=frequencies,
            ),
        )
    )

    reference = ReferenceScales(
        radius=100e-6,
        frequency=frequencies,
    )

    params = prepare_translating_bubble_parameters(
        material,
        bubble,
        field,
        reference,
        n_envs=3,
    )

    y = np.array(
        [
            [0.4, 1.0, 2.0, 3.0, 0.10, 0.01, -0.02, 0.03],
            [0.5, 2.0, 1.0, 4.0, -0.20, -0.04, 0.05, 0.01],
            [0.6, 3.0, 2.0, 1.0, 0.15, 0.02, 0.01, -0.03],
        ],
        dtype=np.float64,
    )

    dydt = translating_bubble_rhs(
        np.zeros(3, dtype=np.float64),
        y,
        params,
    )

    assert dydt.shape == y.shape

    assert_allclose(
        dydt[:, 0:4],
        y[:, 4:8],
        rtol=0.0,
        atol=0.0,
    )


def test_translation_drag_opposes_motion_and_affects_radial_acceleration():
    material = MaterialProperties()

    bubble = TranslatingBubbleParameters(
        equilibrium_radius=50e-6,
    )

    field = AcousticField(
        components=(
            UniformHarmonic(
                pressure_amplitude=0.0,
                frequency=25e3,
            ),
        )
    )

    reference = ReferenceScales(
        radius=50e-6,
        frequency=25e3,
    )

    params = prepare_translating_bubble_parameters(
        material,
        bubble,
        field,
        reference,
        n_envs=1,
    )

    y_stationary = np.array(
        [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
        dtype=np.float64,
    )

    y_moving = y_stationary.copy()
    y_moving[0, 5:8] = np.array(
        [0.10, -0.20, 0.30],
        dtype=np.float64,
    )

    t = np.array([0.0])

    rhs_stationary = translating_bubble_rhs(
        t,
        y_stationary,
        params,
    )

    rhs_moving = translating_bubble_rhs(
        t,
        y_moving,
        params,
    )

    # Drag opposes translational motion.
    assert rhs_moving[0, 5] < 0.0
    assert rhs_moving[0, 6] > 0.0
    assert rhs_moving[0, 7] < 0.0

    # Translational kinetic term adds +|V|^2 / 4
    # to the radial KM numerator.
    assert rhs_moving[0, 4] > rhs_stationary[0, 4]


def test_standing_wave_gradient_drives_translation_along_field_direction():
    material = MaterialProperties()

    pressure_amplitude = 1.0e5
    frequency = 25e3
    reference_radius = 50e-6

    bubble = TranslatingBubbleParameters(
        equilibrium_radius=reference_radius,
    )

    field = AcousticField(
        components=(
            PlaneStandingWave(
                pressure_amplitude=pressure_amplitude,
                frequency=frequency,
                direction=np.array([1.0, 0.0, 0.0]),
            ),
        )
    )

    reference = ReferenceScales(
        radius=reference_radius,
        frequency=frequency,
    )

    params = prepare_translating_bubble_parameters(
        material,
        bubble,
        field,
        reference,
        n_envs=1,
    )

    # Physical x = lambda / 4 -> maximum pressure gradient.
    # Position state uses x* = x / R_ref.
    x_node = material.sound_speed / (4.0 * frequency * reference_radius)

    y = np.array(
        [[1.0, x_node, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
        dtype=np.float64,
    )

    # sin(2*pi*tau) = 1 and cos(2*pi*tau) = 0:
    # nonzero pressure gradient, zero acoustic fluid velocity.
    dydt = translating_bubble_rhs(
        np.array([0.25]),
        y,
        params,
    )

    # grad(p) points in -x, therefore -V grad(p) drives +x.
    assert dydt[0, 5] > 0.0

    assert_allclose(
        dydt[0, 6:8],
        0.0,
        rtol=0.0,
        atol=1e-13,
    )
