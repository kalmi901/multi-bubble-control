from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.integrate import solve_ivp

from multibubble.integrators import solve_rkck45
from multibubble.physics import MaterialProperties
from multibubble.physics.acoustic_field import AcousticField, UniformHarmonic
from multibubble.physics.common import ReferenceScales
from multibubble.physics.single_bubble import (
    BubbleParameters,
    prepare_single_bubble_parameters,
    single_bubble_rhs,
)

REFERENCE_PATH = Path(__file__).parents[1] / "fixtures" / "legacy_single_bubble_reference.npz"


LEGACY_CASES = [
    (0, "notebook-r50"),
    (1, "moderate-r50"),
    (2, "sonochem-r41p3"),
]


def _make_driven_case():
    material = MaterialProperties()

    bubble = BubbleParameters(
        equilibrium_radius=50e-6,
    )

    field = AcousticField(
        components=(
            UniformHarmonic(
                pressure_amplitude=0.3e5,
                frequency=25e3,
            ),
        )
    )

    reference = ReferenceScales(radius=bubble.equilibrium_radius, frequency=25e3)

    params = prepare_single_bubble_parameters(material, bubble, field, reference, 1)

    y0 = np.array(
        [[1.0, 0.0]],
        dtype=np.float64,
    )

    return params, y0


def _solve_scipy_at_checkpoints(
    params,
    y0,
    checkpoints,
):
    def scipy_rhs(t, y):
        # SciPy integrates one system with shape (2,).
        # Our RHS expects the batch dimension (n_envs, 2).
        dydt = single_bubble_rhs(
            np.array([t], dtype=np.float64),
            y[None, :],
            params,
        )

        return dydt[0]

    result = solve_ivp(
        scipy_rhs,
        t_span=(0.0, checkpoints[-1]),
        y0=y0[0],
        method="DOP853",
        t_eval=checkpoints,
        rtol=1e-12,
        atol=np.array([1e-13, 1e-13]),
    )

    assert result.success

    return result.y.T


def _solve_rkck45_at_checkpoints(
    params,
    y0,
    checkpoints,
    *,
    rtol=1e-9,
    atol=None,
):
    if atol is None:
        atol = np.array([1e-10, 1e-10])

    states = []

    for t_end in checkpoints:
        _, y_final, _ = solve_rkck45(
            single_bubble_rhs,
            y0,
            t_span=(0.0, t_end),
            initial_step=1e-3,
            rtol=rtol,
            atol=atol,
            args=(params,),
        )

        states.append(y_final[0])

    return np.asarray(states)


def _load_legacy_reference():
    with np.load(REFERENCE_PATH, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def _make_legacy_case(reference, case_index):
    material = MaterialProperties(
        density=float(reference["density_kg_m3"]),
        sound_speed=float(reference["sound_speed_m_s"]),
        surface_tension=float(reference["surface_tension_n_m"]),
        dynamic_viscosity=float(reference["dynamic_viscosity_pa_s"]),
        vapor_pressure=float(reference["vapor_pressure_pa"]),
        ambient_pressure=float(reference["ambient_pressure_pa"]),
        polytropic_exponent=float(reference["polytropic_exponent"]),
    )

    equilibrium_radius = float(reference["equilibrium_radius_m"][case_index])

    reference_radius = float(reference["reference_radius_m"][case_index])

    bubble = BubbleParameters(
        equilibrium_radius=equilibrium_radius,
    )

    field = AcousticField(
        components=(
            UniformHarmonic(
                pressure_amplitude=float(reference["pressure_amplitude_pa"][case_index]),
                frequency=float(reference["frequency_hz"][case_index]),
                temporal_phase=float(reference["forcing_phase_rad"]),
            ),
        )
    )

    reference = ReferenceScales(
        radius=reference_radius, frequency=float(reference["frequency_hz"][case_index])
    )

    params = prepare_single_bubble_parameters(material, bubble, field, reference, 1)

    y0 = np.array(
        [
            [
                equilibrium_radius / reference_radius,
                0.0,
            ]
        ],
        dtype=np.float64,
    )

    return params, y0, equilibrium_radius, reference_radius


def test_unforced_equilibrium_remains_equilibrium_when_integrated():
    material = MaterialProperties()

    bubble = BubbleParameters(
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

    reference = ReferenceScales(radius=bubble.equilibrium_radius, frequency=25e3)

    params = prepare_single_bubble_parameters(material, bubble, field, reference, 1)

    y0 = np.array(
        [[1.0, 0.0]],
        dtype=np.float64,
    )

    t_final, y_final, _ = solve_rkck45(
        single_bubble_rhs,
        y0,
        t_span=(0.0, 10.0),
        initial_step=1e-3,
        rtol=1e-9,
        atol=np.array([1e-10, 1e-10]),
        args=(params,),
    )

    np.testing.assert_allclose(
        t_final,
        10.0,
        rtol=0.0,
        atol=1e-14,
    )

    radius_drift = np.abs(y_final[0, 0] - y0[0, 0])

    velocity_drift = np.abs(y_final[0, 1] - y0[0, 1])

    assert radius_drift < 1e-9
    assert velocity_drift < 1e-8


def test_driven_bubble_matches_scipy_dop853_at_checkpoints():
    params, y0 = _make_driven_case()

    checkpoints = np.array(
        [
            0.25,
            0.50,
            1.00,
            2.00,
            3.00,
            5.00,
        ],
        dtype=np.float64,
    )

    scipy_state = _solve_scipy_at_checkpoints(
        params,
        y0,
        checkpoints,
    )

    rkck_state = _solve_rkck45_at_checkpoints(
        params,
        y0,
        checkpoints,
    )

    assert_allclose(
        rkck_state,
        scipy_state,
        rtol=1e-7,
        atol=1e-9,
    )


def test_batched_integration_matches_independent_simulations():
    material = MaterialProperties()

    radii = np.array(
        [40e-6, 50e-6, 60e-6],
        dtype=np.float64,
    )

    frequencies = np.array(
        [20e3, 25e3, 30e3],
        dtype=np.float64,
    )

    amplitudes = np.array(
        [0.1e5, 0.3e5, 0.5e5],
        dtype=np.float64,
    )

    reference_radius = 100e-6

    bubble = BubbleParameters(
        equilibrium_radius=radii,
    )

    field = AcousticField(
        components=(
            UniformHarmonic(
                pressure_amplitude=amplitudes,
                frequency=frequencies,
            ),
        )
    )

    reference = ReferenceScales(radius=100e-6, frequency=frequencies)

    batch_params = prepare_single_bubble_parameters(material, bubble, field, reference, 3)

    y0_batch = np.column_stack(
        (
            radii / reference_radius,
            np.zeros_like(radii),
        )
    )

    _, y_batch, _ = solve_rkck45(
        single_bubble_rhs,
        y0_batch,
        t_span=(0.0, 5.0),
        initial_step=1e-3,
        rtol=1e-9,
        atol=np.array([1e-10, 1e-10]),
        args=(batch_params,),
    )

    independent_states = []

    for radius, amplitude, frequency in zip(
        radii,
        amplitudes,
        frequencies,
        strict=True,
    ):
        bubble_independent = BubbleParameters(
            equilibrium_radius=float(radius),
        )

        field_independent = AcousticField(
            components=(
                UniformHarmonic(pressure_amplitude=float(amplitude), frequency=float(frequency)),
            )
        )

        reference = ReferenceScales(
            radius=reference_radius, frequency=field_independent.components[0].frequency
        )

        params_independent = prepare_single_bubble_parameters(
            material, bubble_independent, field_independent, reference, 1
        )

        y0 = np.array(
            [[float(radius) / reference_radius, 0.0]],
            dtype=np.float64,
        )

        _, y_final, _ = solve_rkck45(
            single_bubble_rhs,
            y0,
            t_span=(0.0, 5.0),
            initial_step=1e-3,
            rtol=1e-9,
            atol=np.array([1e-10, 1e-10]),
            args=(params_independent,),
        )

        independent_states.append(y_final[0])

    y_independent = np.asarray(independent_states)

    np.testing.assert_allclose(
        y_batch,
        y_independent,
        rtol=1e-11,
        atol=1e-12,
    )


def test_tighter_tolerances_reduce_driven_bubble_error():
    params, y0 = _make_driven_case()

    checkpoints = np.array(
        [
            0.5,
            1.0,
            2.0,
            3.0,
            5.0,
        ],
        dtype=np.float64,
    )

    reference = _solve_scipy_at_checkpoints(
        params,
        y0,
        checkpoints,
    )

    tolerances = [
        (1e-5, 1e-6),
        (1e-7, 1e-8),
        (1e-9, 1e-10),
    ]

    errors = []

    for rtol, atol_value in tolerances:
        solution = _solve_rkck45_at_checkpoints(
            params,
            y0,
            checkpoints,
            rtol=rtol,
            atol=np.array(
                [atol_value, atol_value],
                dtype=np.float64,
            ),
        )

        error = np.max(np.abs(solution - reference))

        errors.append(error)

    assert errors[1] < errors[0]
    assert errors[2] < errors[1]


@pytest.mark.parametrize(
    ("case_index", "_case_name"),
    LEGACY_CASES,
    ids=[case[1] for case in LEGACY_CASES],
)
def test_driven_bubble_matches_legacy_reference(
    case_index,
    _case_name,
):
    reference = _load_legacy_reference()

    params, y0, _, _ = _make_legacy_case(
        reference,
        case_index,
    )

    checkpoints = reference["tau"][case_index]
    expected = reference["state"][case_index]

    # tau = 0 is exactly the initial condition, so avoid asking the
    # solver to integrate over a zero-length interval.
    solved = _solve_rkck45_at_checkpoints(
        params,
        y0,
        checkpoints[1:],
        rtol=float(reference["legacy_rtol"]),
        atol=np.array(
            [
                float(reference["legacy_atol"]),
                float(reference["legacy_atol"]),
            ],
            dtype=np.float64,
        ),
    )

    actual = np.vstack(
        (
            y0[0],
            solved,
        )
    )

    assert actual.shape == expected.shape

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=2e-7,
        atol=2e-9,
    )


@pytest.mark.parametrize(
    ("case_index", "_case_name"),
    LEGACY_CASES,
    ids=[case[1] for case in LEGACY_CASES],
)
def test_legacy_radius_extrema_and_timing(
    case_index,
    _case_name,
):
    reference = _load_legacy_reference()

    (
        params,
        y0,
        equilibrium_radius,
        reference_radius,
    ) = _make_legacy_case(
        reference,
        case_index,
    )

    tau_min = float(reference["tau_r_min"][case_index])

    tau_max = float(reference["tau_r_max"][case_index])

    expected_r_min = float(reference["r_min_over_r0"][case_index])

    expected_r_max = float(reference["r_max_over_r0"][case_index])

    # Small neighborhood around the legacy extremum times.
    # One acoustic period corresponds to Delta tau = 1.
    delta_tau = 1e-2

    checkpoints = np.array(
        [
            tau_min - delta_tau,
            tau_min,
            tau_min + delta_tau,
            tau_max - delta_tau,
            tau_max,
            tau_max + delta_tau,
        ],
        dtype=np.float64,
    )

    states = _solve_rkck45_at_checkpoints(
        params,
        y0,
        checkpoints,
        rtol=float(reference["legacy_rtol"]),
        atol=np.array(
            [
                float(reference["legacy_atol"]),
                float(reference["legacy_atol"]),
            ],
            dtype=np.float64,
        ),
    )

    equilibrium_radius_dimensionless = equilibrium_radius / reference_radius

    radius_over_r0 = states[:, 0] / equilibrium_radius_dimensionless

    r_before_min, r_at_min, r_after_min = radius_over_r0[0:3]

    r_before_max, r_at_max, r_after_max = radius_over_r0[3:6]

    # Radius values at the legacy extremum times agree.
    np.testing.assert_allclose(
        r_at_min,
        expected_r_min,
        rtol=2e-4,
        atol=2e-5,
    )

    np.testing.assert_allclose(
        r_at_max,
        expected_r_max,
        rtol=2e-4,
        atol=2e-5,
    )

    # And the stored legacy time is actually near a local extremum
    # of the new trajectory as well.
    assert r_at_min < r_before_min
    assert r_at_min < r_after_min

    assert r_at_max > r_before_max
    assert r_at_max > r_after_max
