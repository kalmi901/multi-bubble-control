import numpy as np
from numpy.testing import assert_allclose

from multibubble.physics.coupled_bubbles import (
    AcousticForcing,
    CoupledBubblesParameters,
    MaterialProperties,
    prepare_coupled_bubbles_parameters,
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

    # Shared 1D geometry: shape (N, D)
    position = np.array(
        [
            [0.0],
            [1.0],
        ],
        dtype=np.float64,
    )

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
