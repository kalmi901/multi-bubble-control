import numpy as np
from numpy.testing import assert_allclose

from multibubble.physics import MaterialProperties
from multibubble.physics.acoustic_field import AcousticField, UniformHarmonic
from multibubble.physics.common import ReferenceScales
from multibubble.physics.coupled_translating_bubbles import (
    CoupledTranslatingBubbleParameters,
    coupled_translating_bubble_rhs,
    prepare_coupled_translating_bubble_parameters,
)


def test_unforced_equilibrium_is_stationary():
    material = MaterialProperties()

    equilibrium_radius = np.array(
        [40e-6, 50e-6],
        dtype=np.float64,
    )

    bubble = CoupledTranslatingBubbleParameters(
        equilibrium_radius=equilibrium_radius,
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

    params = prepare_coupled_translating_bubble_parameters(
        material,
        bubble,
        field,
        reference,
        n_envs=1,
    )

    y = np.zeros((1, 2, 8), dtype=np.float64)

    y[0, :, 0] = equilibrium_radius / reference.radius

    y[0, 0, 1:4] = np.array([0.0, 0.0, 0.0])
    y[0, 1, 1:4] = np.array([10.0, 0.0, 0.0])

    dydt = coupled_translating_bubble_rhs(
        np.array([0.0]),
        y,
        params,
    )

    assert_allclose(
        dydt,
        0.0,
        rtol=0.0,
        atol=1e-12,
    )


def test_rhs_preserves_shape_and_coordinate_derivatives_equal_velocities():
    material = MaterialProperties()

    bubble = CoupledTranslatingBubbleParameters(
        equilibrium_radius=np.array(
            [40e-6, 50e-6],
            dtype=np.float64,
        ),
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

    params = prepare_coupled_translating_bubble_parameters(
        material,
        bubble,
        field,
        reference,
        n_envs=1,
    )

    y = np.array(
        [
            [
                [0.8, 0.0, 0.0, 0.0, 0.10, 0.01, -0.02, 0.03],
                [1.0, 5.0, 2.0, 1.0, -0.05, -0.04, 0.05, 0.01],
            ],
        ],
        dtype=np.float64,
    )

    dydt = coupled_translating_bubble_rhs(
        np.array([0.0]),
        y,
        params,
    )

    assert dydt.shape == y.shape

    assert_allclose(
        dydt[..., 0],
        y[..., 4],
        rtol=0.0,
        atol=0.0,
    )

    assert_allclose(
        dydt[..., 1:4],
        y[..., 5:8],
        rtol=0.0,
        atol=0.0,
    )


from multibubble.physics.translating_bubble import (
    TranslatingBubbleParameters,
    prepare_translating_bubble_parameters,
    translating_bubble_rhs,
)


def test_single_bubble_limit_matches_translating_bubble_rhs():
    material = MaterialProperties()

    equilibrium_radius = 50e-6

    field = AcousticField(
        components=(
            UniformHarmonic(
                pressure_amplitude=0.3e5,
                frequency=25e3,
            ),
        )
    )

    reference = ReferenceScales(
        radius=50e-6,
        frequency=25e3,
    )

    single_params = prepare_translating_bubble_parameters(
        material,
        TranslatingBubbleParameters(
            equilibrium_radius=equilibrium_radius,
        ),
        field,
        reference,
        n_envs=1,
    )

    coupled_params = prepare_coupled_translating_bubble_parameters(
        material,
        CoupledTranslatingBubbleParameters(
            equilibrium_radius=np.array(
                [equilibrium_radius],
                dtype=np.float64,
            ),
        ),
        field,
        reference,
        n_envs=1,
    )

    t = np.array([0.37])

    y_single = np.array(
        [
            [
                1.1,
                2.0,
                -1.0,
                0.5,
                0.12,
                0.03,
                -0.02,
                0.01,
            ]
        ],
        dtype=np.float64,
    )

    y_coupled = y_single[:, np.newaxis, :]

    rhs_single = translating_bubble_rhs(
        t,
        y_single,
        single_params,
    )

    rhs_coupled = coupled_translating_bubble_rhs(
        t,
        y_coupled,
        coupled_params,
    )

    assert_allclose(
        rhs_coupled[:, 0, :],
        rhs_single,
        rtol=1e-12,
        atol=1e-12,
    )


def test_symmetric_pair_produces_equal_and_opposite_translation():
    material = MaterialProperties()

    bubble = CoupledTranslatingBubbleParameters(
        equilibrium_radius=np.array(
            [50e-6, 50e-6],
            dtype=np.float64,
        ),
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

    params = prepare_coupled_translating_bubble_parameters(
        material,
        bubble,
        field,
        reference,
        n_envs=1,
    )

    y = np.zeros((1, 2, 8), dtype=np.float64)

    y[..., 0] = 1.0

    y[0, 0, 1:4] = [-5.0, 0.0, 0.0]
    y[0, 1, 1:4] = [+5.0, 0.0, 0.0]

    # Same radial state for both bubbles.
    y[..., 4] = 0.1

    dydt = coupled_translating_bubble_rhs(
        np.array([0.0]),
        y,
        params,
    )

    assert_allclose(
        dydt[0, 0, 5],
        -dydt[0, 1, 5],
        rtol=1e-12,
        atol=1e-12,
    )

    assert_allclose(
        dydt[..., 6:8],
        0.0,
        rtol=0.0,
        atol=1e-12,
    )

    assert_allclose(
        dydt[0, 0, 4],
        dydt[0, 1, 4],
        rtol=1e-12,
        atol=1e-12,
    )


from types import SimpleNamespace

from multibubble.physics.coupled_translating_bubbles import (
    _assemble_coupling_system,
)


def test_assemble_coupled_system_for_three_bubble_noncollinear_geometry():
    R = np.array(
        [[1.0, 2.0, 3.0]],
        dtype=np.float64,
    )

    U = np.zeros(
        (1, 3),
        dtype=np.float64,
    )

    X = np.array(
        [
            [
                [0.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [0.0, 4.0, 0.0],
            ]
        ],
        dtype=np.float64,
    )

    V = np.zeros(
        (1, 3, 3),
        dtype=np.float64,
    )

    numerator = np.array(
        [[10.0, 20.0, 30.0]],
        dtype=np.float64,
    )

    denominator = np.array(
        [[2.0, 4.0, 5.0]],
        dtype=np.float64,
    )

    translation_acceleration = np.array(
        [
            [
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
                [0.7, 0.8, 0.9],
            ]
        ],
        dtype=np.float64,
    )

    # U = V = 0, therefore all explicit coupling terms vanish.
    params = SimpleNamespace(
        translational_coupling_viscous_scale=0.0,
    )

    matrix, rhs = _assemble_coupling_system(
        R,
        U,
        X,
        V,
        numerator,
        denominator,
        translation_acceleration,
        params,  # type: ignore
    )

    # ---------------------------------------------------------
    # Expected implicit blocks
    #
    # Acceleration ordering:
    #
    # [a_R0, a_R1, a_R2,
    #  a_x0, a_x1, a_x2,
    #  a_y0, a_y1, a_y2,
    #  a_z0, a_z1, a_z2]
    # ---------------------------------------------------------

    expected_rr = np.array(
        [
            [1.0, 2.0 / 3.0, 9.0 / 8.0],
            [1.0 / 12.0, 1.0, 9.0 / 20.0],
            [1.0 / 20.0, 4.0 / 25.0, 1.0],
        ],
        dtype=np.float64,
    )

    expected_rx = np.array(
        [
            [0.0, -2.0 / 9.0, 0.0],
            [1.0 / 72.0, 0.0, 81.0 / 1000.0],
            [0.0, -12.0 / 625.0, 0.0],
        ],
        dtype=np.float64,
    )

    expected_ry = np.array(
        [
            [0.0, 0.0, -27.0 / 64.0],
            [0.0, 0.0, -27.0 / 250.0],
            [1.0 / 160.0, 16.0 / 625.0, 0.0],
        ],
        dtype=np.float64,
    )

    expected_xr = np.array(
        [
            [0.0, 4.0 / 3.0, 0.0],
            [-1.0 / 3.0, 0.0, -81.0 / 125.0],
            [0.0, 36.0 / 125.0, 0.0],
        ],
        dtype=np.float64,
    )

    expected_yr = np.array(
        [
            [0.0, 0.0, 27.0 / 16.0],
            [0.0, 0.0, 108.0 / 125.0],
            [-3.0 / 16.0, -48.0 / 125.0, 0.0],
        ],
        dtype=np.float64,
    )

    expected_translation = np.array(
        [
            [1.0, -8.0 / 9.0, -81.0 / 64.0],
            [-1.0 / 9.0, 1.0, -81.0 / 125.0],
            [-3.0 / 64.0, -24.0 / 125.0, 1.0],
        ],
        dtype=np.float64,
    )

    zero = np.zeros(
        (3, 3),
        dtype=np.float64,
    )

    expected_matrix = np.block(
        [
            [
                expected_rr,
                expected_rx,
                expected_ry,
                zero,
            ],
            [
                expected_xr,
                expected_translation,
                zero,
                zero,
            ],
            [
                expected_yr,
                zero,
                expected_translation,
                zero,
            ],
            [
                zero,
                zero,
                zero,
                expected_translation,
            ],
        ]
    )[np.newaxis, ...]

    expected_rhs = np.array(
        [
            [
                # Radial: numerator / denominator
                5.0,
                5.0,
                6.0,
                # x translation
                0.1,
                0.4,
                0.7,
                # y translation
                0.2,
                0.5,
                0.8,
                # z translation
                0.3,
                0.6,
                0.9,
            ]
        ],
        dtype=np.float64,
    )

    assert matrix.shape == (1, 12, 12)
    assert rhs.shape == (1, 12)

    assert_allclose(
        matrix,
        expected_matrix,
        rtol=1e-13,
        atol=1e-13,
    )

    assert_allclose(
        rhs,
        expected_rhs,
        rtol=1e-13,
        atol=1e-13,
    )
