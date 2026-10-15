import numpy as np
from numpy.testing import assert_allclose

from multibubble.physics.coupled_bubbles import (
    AcousticForcing,
    CoupledBubblesParameters,
    MaterialProperties,
    coupled_bubbles_rhs,
    prepare_coupled_bubbles_parameters,
)
from multibubble.physics.single_bubble import (
    BubbleParameters,
    prepare_single_bubble_parameters,
    single_bubble_rhs,
)


def test_prepare_supports_environment_and_bubble_specific_parameters():
    material = MaterialProperties()

    equilibrium_radius = np.array(
        [
            [10e-6, 20e-6],
            [20e-6, 30e-6],
            [30e-6, 40e-6],
        ],
        dtype=np.float64,
    )

    position = np.array(
        [
            [
                [0.0],
                [1.0],
            ],
            [
                [0.0],
                [1.5],
            ],
            [
                [0.0],
                [2.0],
            ],
        ],
        dtype=np.float64,
    )  # shape (E, N, D)

    bubbles = CoupledBubblesParameters(
        equilibrium_radius=equilibrium_radius,
        position=position,
    )

    forcing = AcousticForcing(
        pressure_amplitude=np.array(
            [0.1e5, 0.2e5, 0.3e5],
            dtype=np.float64,
        ),
        frequency=np.array(
            [20e3, 25e3, 30e3],
            dtype=np.float64,
        ),
    )

    reference_radius = np.array(
        [50e-6, 75e-6, 100e-6],
        dtype=np.float64,
    )

    params = prepare_coupled_bubbles_parameters(
        material,
        bubbles,
        forcing,
        reference_radius,
    )

    assert params.equilibrium_radius_ratio.shape == (3, 2)
    assert params.gas_pressure_coefficient.shape == (3, 2)

    assert params.static_pressure_coefficient.shape == (3, 1)
    assert params.wall_mach_scale.shape == (3, 1)

    assert params.forcing_amplitude.shape == (3, 1)
    assert params.forcing_angular_frequency.shape == (3, 1)

    assert params.position.shape == (3, 2, 1)
    assert_allclose(
        params.position,
        position,
    )


def test_prepare_broadcasts_shared_bubble_parameters_across_environments():
    material = MaterialProperties()

    equilibrium_radius = np.array(
        [10e-6, 20e-6],
        dtype=np.float64,
    )  # shape (N,)

    position = np.array(
        [
            [0.0],
            [1.0],
        ],
        dtype=np.float64,
    )  # shape (N, D)

    bubbles = CoupledBubblesParameters(
        equilibrium_radius=equilibrium_radius,
        position=position,
    )

    forcing = AcousticForcing(
        pressure_amplitude=np.array(
            [0.1e5, 0.2e5, 0.3e5],
            dtype=np.float64,
        ),
        frequency=np.array(
            [20e3, 25e3, 30e3],
            dtype=np.float64,
        ),
    )

    reference_radius = np.array(
        [50e-6, 75e-6, 100e-6],
        dtype=np.float64,
    )

    params = prepare_coupled_bubbles_parameters(
        material,
        bubbles,
        forcing,
        reference_radius,
    )

    assert params.equilibrium_radius_ratio.shape == (3, 2)

    expected_radius_ratio = equilibrium_radius[None, :] / reference_radius[:, None]

    assert_allclose(
        params.equilibrium_radius_ratio,
        expected_radius_ratio,
    )

    assert params.gas_pressure_coefficient.shape == (3, 2)

    assert params.static_pressure_coefficient.shape == (3, 1)
    assert params.wall_mach_scale.shape == (3, 1)

    assert params.forcing_amplitude.shape == (3, 1)
    assert params.forcing_angular_frequency.shape == (3, 1)

    assert params.position.shape == (3, 2, 1)

    assert_allclose(
        params.position[0],
        position,
    )

    assert_allclose(
        params.position[1],
        position,
    )

    assert_allclose(
        params.position[2],
        position,
    )


def test_rhs_preserves_state_shape_and_radius_derivative_equals_wall_velocity():
    material = MaterialProperties()

    bubbles = CoupledBubblesParameters(
        equilibrium_radius=np.array(
            [30e-6, 40e-6, 50e-6],
            dtype=np.float64,
        ),
        position=np.array(
            [
                [0.0],
                [1.0],
                [2.0],
            ],
            dtype=np.float64,
        ),
    )

    forcing = AcousticForcing(
        pressure_amplitude=np.array(
            [0.2e5, 0.4e5],
            dtype=np.float64,
        ),
        frequency=np.array(
            [20e3, 30e3],
            dtype=np.float64,
        ),
    )

    reference_radius = np.array(
        [50e-6, 70e-6],
        dtype=np.float64,
    )

    params = prepare_coupled_bubbles_parameters(
        material,
        bubbles,
        forcing,
        reference_radius,
    )

    y = np.array(
        [
            [
                [0.8, 0.10],
                [1.0, -0.05],
                [1.2, 0.20],
            ],
            [
                [0.7, -0.15],
                [0.9, 0.08],
                [1.1, -0.12],
            ],
        ],
        dtype=np.float64,
    )

    t = np.array(
        [0.2, 0.6],
        dtype=np.float64,
    )

    dydt = coupled_bubbles_rhs(
        t,
        y,
        params,
    )

    assert dydt.shape == y.shape

    assert_allclose(
        dydt[..., 0],
        y[..., 1],
        rtol=0.0,
        atol=0.0,
    )


def test_unforced_equilibrium_is_stationary_for_multiple_bubbles():
    material = MaterialProperties()

    equilibrium_radius = np.array(
        [
            [30e-6, 40e-6, 50e-6],
            [35e-6, 45e-6, 60e-6],
        ],
        dtype=np.float64,
    )

    bubbles = CoupledBubblesParameters(
        equilibrium_radius=equilibrium_radius,
        position=np.array(
            [
                [0.0],
                [1.0],
                [2.0],
            ],
            dtype=np.float64,
        ),
    )

    forcing = AcousticForcing(
        pressure_amplitude=0.0,
        frequency=np.array(
            [20e3, 30e3],
            dtype=np.float64,
        ),
    )

    reference_radius = np.array(
        [60e-6, 80e-6],
        dtype=np.float64,
    )

    params = prepare_coupled_bubbles_parameters(
        material,
        bubbles,
        forcing,
        reference_radius,
    )

    y = np.stack(
        (
            params.equilibrium_radius_ratio,
            np.zeros_like(params.equilibrium_radius_ratio),
        ),
        axis=-1,
    )

    t = np.array(
        [0.25, 0.75],
        dtype=np.float64,
    )

    dydt = coupled_bubbles_rhs(
        t,
        y,
        params,
    )

    assert dydt.shape == (2, 3, 2)

    assert_allclose(
        dydt,
        0.0,
        rtol=0.0,
        atol=1e-12,
    )


def test_single_bubble_limit_matches_single_bubble_rhs():
    material = MaterialProperties()

    equilibrium_radius = np.array(
        [40e-6, 55e-6, 70e-6],
        dtype=np.float64,
    )

    reference_radius = np.array(
        [50e-6, 60e-6, 80e-6],
        dtype=np.float64,
    )

    forcing = AcousticForcing(
        pressure_amplitude=np.array(
            [0.15e5, 0.35e5, 0.55e5],
            dtype=np.float64,
        ),
        frequency=np.array(
            [20e3, 25e3, 31e3],
            dtype=np.float64,
        ),
        phase=np.array(
            [0.1, 0.3, -0.2],
            dtype=np.float64,
        ),
    )

    single_bubble = BubbleParameters(
        equilibrium_radius=equilibrium_radius,
    )

    coupled_bubbles = CoupledBubblesParameters(
        equilibrium_radius=equilibrium_radius[:, None],
        position=np.array([[0.0]], dtype=np.float64),
    )

    single_params = prepare_single_bubble_parameters(
        material,
        single_bubble,
        forcing,
        reference_radius,
    )

    coupled_params = prepare_coupled_bubbles_parameters(
        material,
        coupled_bubbles,
        forcing,
        reference_radius,
    )

    t = np.array(
        [0.17, 0.33, 0.61],
        dtype=np.float64,
    )

    y_single = np.array(
        [
            [0.90, 0.12],
            [1.05, -0.08],
            [0.77, 0.20],
        ],
        dtype=np.float64,
    )

    y_coupled = y_single[:, None, :]

    dydt_single = single_bubble_rhs(
        t,
        y_single,
        single_params,
    )

    dydt_coupled = coupled_bubbles_rhs(
        t,
        y_coupled,
        coupled_params,
    )

    assert dydt_coupled.shape == (3, 1, 2)

    assert_allclose(
        dydt_coupled[:, 0, :],
        dydt_single,
        rtol=1e-13,
        atol=1e-13,
    )
