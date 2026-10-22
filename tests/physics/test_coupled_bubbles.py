import numpy as np
from numpy.testing import assert_allclose

from multibubble.physics import MaterialProperties
from multibubble.physics.acoustic_field import (
    AcousticField,
    PlaneStandingWave,
    UniformHarmonic,
    evaluate_acoustic_field,
)
from multibubble.physics.common import ReferenceScales
from multibubble.physics.coupled_bubbles import (
    CoupledBubblesParameters,
    _assemble_radial_coupling_system,
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

    field = AcousticField(
        components=(
            UniformHarmonic(
                pressure_amplitude=np.array([0.1e5, 0.2e5, 0.3e5], dtype=np.float64),
                frequency=np.array([20e3, 25e3, 30e3], dtype=np.float64),
            ),
        )
    )

    reference = ReferenceScales(
        radius=np.array([50e-6, 75e-6, 100e-6]), frequency=field.components[0].frequency
    )

    params = prepare_coupled_bubbles_parameters(material, bubbles, field, reference, n_envs=3)

    assert np.shape(params.equilibrium_radius_ratio) == (3, 2)
    assert np.shape(params.gas_pressure_coefficient) == (3, 2)

    assert np.shape(params.static_pressure_coefficient) == (3, 1)
    assert np.shape(params.wall_mach_scale) == (3, 1)

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
    )  # shared across environments: (N,)

    position = np.array(
        [
            [0.0],
            [1.0],
        ],
        dtype=np.float64,
    )  # shared across environments: (N, D)

    bubbles = CoupledBubblesParameters(
        equilibrium_radius=equilibrium_radius,
        position=position,
    )

    # Environment-specific forcing only.
    field = AcousticField(
        components=(
            UniformHarmonic(
                pressure_amplitude=np.array(
                    [0.1e5, 0.2e5, 0.3e5],
                    dtype=np.float64,
                ),
                frequency=25e3,
            ),
        )
    )

    # Shared reference scales.
    reference = ReferenceScales(
        radius=50e-6,
        frequency=25e3,
    )

    params = prepare_coupled_bubbles_parameters(
        material,
        bubbles,
        field,
        reference,
        n_envs=3,
    )

    # ---------------------------------------------------------
    # Bubble-scoped quantities are canonicalized to (E, N)
    # ---------------------------------------------------------

    assert params.equilibrium_radius_ratio.shape == (3, 2)

    expected_radius_ratio = equilibrium_radius / reference.radius

    assert_allclose(
        params.equilibrium_radius_ratio,
        np.broadcast_to(
            expected_radius_ratio,
            (3, 2),
        ),
    )

    assert params.gas_pressure_coefficient.shape == (3, 2)
    assert params.gas_pressure_velocity_coefficient.shape == (3, 2)

    # ---------------------------------------------------------
    # Shared environment parameters remain scalar Parameters
    # ---------------------------------------------------------

    assert np.shape(params.static_pressure_coefficient) == ()
    assert np.shape(params.surface_tension_coefficient) == ()
    assert np.shape(params.viscous_coefficient) == ()

    assert np.shape(params.acoustic_pressure_scale) == ()
    assert np.shape(params.acoustic_pressure_derivative_scale) == ()
    assert np.shape(params.wall_mach_scale) == ()
    assert np.shape(params.gas_pressure_exponent) == ()

    assert np.shape(params.reference_radius) == ()
    assert np.shape(params.reference_frequency) == ()

    # ---------------------------------------------------------
    # Geometry is canonicalized to (E, N, D)
    # ---------------------------------------------------------

    assert params.position.shape == (3, 2, 1)

    assert_allclose(
        params.position,
        np.broadcast_to(
            position,
            (3, 2, 1),
        ),
    )

    # Coupling matrix is always system-scoped.
    assert params.radial_coupling.shape == (3, 2, 2)


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

    field = AcousticField(
        components=(
            UniformHarmonic(
                pressure_amplitude=np.array([0.2e5, 0.4e5], dtype=np.float64),
                frequency=np.array([20e3, 30e3], dtype=np.float64),
            ),
        )
    )

    reference = ReferenceScales(
        radius=np.array([50e-6, 70e-6], dtype=np.float64), frequency=field.components[0].frequency
    )

    params = prepare_coupled_bubbles_parameters(material, bubbles, field, reference, 2)

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

    field = AcousticField(
        components=(
            UniformHarmonic(
                pressure_amplitude=0, frequency=np.array([20e3, 30e3], dtype=np.float64)
            ),
        )
    )

    reference = ReferenceScales(
        radius=np.array([60e-6, 80e-6], dtype=np.float64), frequency=field.components[0].frequency
    )

    params = prepare_coupled_bubbles_parameters(material, bubbles, field, reference, 2)

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

    field = AcousticField(
        components=(
            UniformHarmonic(
                pressure_amplitude=np.array([0.15e5, 0.35e5, 0.55e5], dtype=np.float64),
                frequency=np.array([20e3, 25e3, 31e3], dtype=np.float64),
                temporal_phase=np.array([0.1, 0.3, -0.2], dtype=np.float64),
            ),
        )
    )

    reference = ReferenceScales(radius=reference_radius, frequency=field.components[0].frequency)

    single_bubble = BubbleParameters(
        equilibrium_radius=equilibrium_radius,
    )

    coupled_bubbles = CoupledBubblesParameters(
        equilibrium_radius=equilibrium_radius[:, None],
        position=np.array([[0.0]], dtype=np.float64),
    )

    single_params = prepare_single_bubble_parameters(material, single_bubble, field, reference, 3)

    coupled_params = prepare_coupled_bubbles_parameters(
        material, coupled_bubbles, field, reference, 3
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


def test_assemble_radial_coupling_system():
    R = np.array(
        [[2.0, 3.0]],
        dtype=np.float64,
    )

    U = np.array(
        [[0.5, -0.25]],
        dtype=np.float64,
    )

    numerator = np.array(
        [[10.0, 20.0]],
        dtype=np.float64,
    )

    denominator = np.array(
        [[2.0, 4.0]],
        dtype=np.float64,
    )

    coupling = np.array(
        [
            [
                [0.0, 0.1],
                [0.2, 0.0],
            ]
        ],
        dtype=np.float64,
    )

    matrix, rhs = _assemble_radial_coupling_system(
        R,
        U,
        numerator,
        denominator,
        coupling,
    )

    expected_matrix = np.array(
        [
            [
                [
                    1.0,
                    (1.0 / 2.0) * 0.1 * 3.0**2,
                ],
                [
                    (1.0 / 4.0) * 0.2 * 2.0**2,
                    1.0,
                ],
            ]
        ],
        dtype=np.float64,
    )

    expected_rhs = np.array(
        [
            [
                (1.0 / 2.0) * (10.0 - 2.0 * 0.1 * 3.0 * (-0.25) ** 2),
                (1.0 / 4.0) * (20.0 - 2.0 * 0.2 * 2.0 * 0.5**2),
            ]
        ],
        dtype=np.float64,
    )

    assert_allclose(
        matrix,
        expected_matrix,
    )

    assert_allclose(
        rhs,
        expected_rhs,
    )


def test_radial_coupling_excludes_self_interaction():
    material = MaterialProperties()

    bubbles = CoupledBubblesParameters(
        equilibrium_radius=np.array(
            [40e-6, 50e-6, 60e-6],
            dtype=np.float64,
        ),
        position=np.array(
            [
                [0.0],
                [0.25],
                [1.0],
            ],
            dtype=np.float64,
        ),
    )

    field = AcousticField(components=(UniformHarmonic(pressure_amplitude=0.3e5, frequency=25e3),))

    reference = ReferenceScales(radius=50e-6, frequency=25e3)

    params = prepare_coupled_bubbles_parameters(material, bubbles, field, reference, 1)

    coupling = params.radial_coupling

    assert coupling.shape == (1, 3, 3)

    # Self-interaction must be excluded exactly.
    diagonal = np.diagonal(
        coupling,
        axis1=1,
        axis2=2,
    )

    assert_allclose(
        diagonal,
        0.0,
        rtol=0.0,
        atol=0.0,
    )

    # x* = x / (c/f), therefore
    # K_ij = (R_ref / (c/f)) / |x_i* - x_j*|.
    reference_length = material.sound_speed / reference.frequency

    expected_k01 = (
        reference.radius / reference_length / abs(bubbles.position[0, 0] - bubbles.position[1, 0])
    )

    assert_allclose(
        coupling[0, 0, 1],
        expected_k01,
        rtol=1e-14,
        atol=0.0,
    )

    assert_allclose(
        coupling[0, 1, 0],
        expected_k01,
        rtol=1e-14,
        atol=0.0,
    )


def test_coupled_bubbles_evaluate_spatial_acoustic_field_at_each_bubble():
    material = MaterialProperties()

    bubbles = CoupledBubblesParameters(
        equilibrium_radius=np.array(
            [50e-6, 50e-6],
            dtype=np.float64,
        ),
        position=np.array(
            [
                [0.0],  # pressure antinode
                [0.25],  # pressure node for f = f_ref
            ],
            dtype=np.float64,
        ),
    )

    pressure_amplitude = 1.0e5
    frequency = 25e3

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
        radius=50e-6,
        frequency=frequency,
    )

    params = prepare_coupled_bubbles_parameters(
        material,
        bubbles,
        field,
        reference,
        n_envs=1,
    )

    # tau = 1/4:
    # sin(2*pi*tau) = 1
    #
    # Bubble 0:
    # x* = 0 -> cos(k*x) = 1 -> p = P_A
    #
    # Bubble 1:
    # x* = 1/4 -> cos(pi/2) = 0 -> p = 0
    field_state = evaluate_acoustic_field(
        t=np.array([0.25]),
        position=params.position,
        field=params.acoustic_field,
    )

    assert field_state.pressure.shape == (1, 2)

    assert_allclose(
        field_state.pressure[0, 0],
        pressure_amplitude,
        rtol=1e-14,
        atol=1e-9,
    )

    assert_allclose(
        field_state.pressure[0, 1],
        0.0,
        atol=1e-9,
    )
