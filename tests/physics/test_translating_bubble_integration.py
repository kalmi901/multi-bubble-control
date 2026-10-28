import numpy as np
from numpy.testing import assert_allclose
from scipy.integrate import solve_ivp

from multibubble.integrators import solve_rkck45
from multibubble.physics import MaterialProperties
from multibubble.physics.acoustic_field import AcousticField, PlaneStandingWave, UniformHarmonic
from multibubble.physics.common import ReferenceScales
from multibubble.physics.translating_bubble import (
    TranslatingBubbleParameters,
    prepare_translating_bubble_parameters,
    translating_bubble_rhs,
)


def test_unforced_equilibrium_remains_equilibrium_when_integrated():
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

    y0 = np.array(
        [[1.0, 2.0, -1.0, 0.5, 0.0, 0.0, 0.0, 0.0]],
        dtype=np.float64,
    )

    t_final, y_final, _ = solve_rkck45(
        translating_bubble_rhs,
        y0,
        t_span=(0.0, 10.0),
        initial_step=1e-3,
        rtol=1e-9,
        atol=np.full(8, 1e-10),
        args=(params,),
    )

    assert_allclose(
        t_final,
        10.0,
        rtol=0.0,
        atol=1e-14,
    )

    radius_drift = abs(y_final[0, 0] - y0[0, 0])
    wall_velocity_drift = abs(y_final[0, 4])

    assert radius_drift < 1e-9
    assert wall_velocity_drift < 1e-8

    assert_allclose(
        y_final[0, 1:4],
        y0[0, 1:4],
        atol=1e-12,
    )

    assert_allclose(
        y_final[0, 5:8],
        0.0,
        atol=1e-12,
    )


def test_driven_translating_bubble_matches_scipy_dop853_at_checkpoints():
    material = MaterialProperties()

    radius = 50e-6
    frequency = 25e3

    bubble = TranslatingBubbleParameters(
        equilibrium_radius=radius,
    )

    field = AcousticField(
        components=(
            PlaneStandingWave(
                pressure_amplitude=0.2e5,
                frequency=frequency,
                direction=np.array([1.0, 0.0, 0.0]),
            ),
        )
    )

    reference = ReferenceScales(
        radius=radius,
        frequency=frequency,
    )

    params = prepare_translating_bubble_parameters(
        material,
        bubble,
        field,
        reference,
        n_envs=1,
    )

    # Start halfway between antinode and node: x = lambda / 8.
    x0 = material.sound_speed / (8.0 * frequency * radius)

    y0 = np.array(
        [[1.0, x0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
        dtype=np.float64,
    )

    checkpoints = np.array(
        [0.25, 0.50, 1.00, 2.00],
        dtype=np.float64,
    )

    def scipy_rhs(t, y):
        return translating_bubble_rhs(
            np.array([t]),
            y[None, :],
            params,
        )[0]

    scipy_result = solve_ivp(
        scipy_rhs,
        t_span=(0.0, checkpoints[-1]),
        y0=y0[0],
        method="DOP853",
        t_eval=checkpoints,
        rtol=1e-12,
        atol=np.full(8, 1e-13),
    )

    assert scipy_result.success

    rkck_states = []

    for t_end in checkpoints:
        _, y_final, _ = solve_rkck45(
            translating_bubble_rhs,
            y0,
            t_span=(0.0, float(t_end)),
            initial_step=1e-3,
            rtol=1e-9,
            atol=np.full(8, 1e-10),
            args=(params,),
        )

        rkck_states.append(y_final[0])

    rkck_states = np.asarray(rkck_states)
    scipy_states = scipy_result.y.T

    assert_allclose(
        rkck_states,
        scipy_states,
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
        [0.10e5, 0.20e5, 0.30e5],
        dtype=np.float64,
    )

    reference_radius = np.array(
        [50e-6, 60e-6, 70e-6],
        dtype=np.float64,
    )

    bubble = TranslatingBubbleParameters(
        equilibrium_radius=radii,
    )

    field = AcousticField(
        components=(
            PlaneStandingWave(
                pressure_amplitude=amplitudes,
                frequency=frequencies,
                direction=np.array([1.0, 0.0, 0.0]),
            ),
        )
    )

    reference = ReferenceScales(
        radius=reference_radius,
        frequency=frequencies,
    )

    params = prepare_translating_bubble_parameters(
        material,
        bubble,
        field,
        reference,
        n_envs=3,
    )

    # x = lambda / 8 in each environment,
    # stored internally as x / R_ref.
    x0 = material.sound_speed / (8.0 * frequencies * reference_radius)

    y0 = np.zeros(
        (3, 8),
        dtype=np.float64,
    )

    y0[:, 0] = radii / reference_radius
    y0[:, 1] = x0

    _, y_batch, _ = solve_rkck45(
        translating_bubble_rhs,
        y0,
        t_span=(0.0, 2.0),
        initial_step=1e-3,
        rtol=1e-9,
        atol=np.full(8, 1e-10),
        args=(params,),
    )

    independent_states = []

    for env in range(3):
        bubble_env = TranslatingBubbleParameters(
            equilibrium_radius=float(radii[env]),
        )

        field_env = AcousticField(
            components=(
                PlaneStandingWave(
                    pressure_amplitude=float(amplitudes[env]),
                    frequency=float(frequencies[env]),
                    direction=np.array([1.0, 0.0, 0.0]),
                ),
            )
        )

        reference_env = ReferenceScales(
            radius=float(reference_radius[env]),
            frequency=float(frequencies[env]),
        )

        params_env = prepare_translating_bubble_parameters(
            material,
            bubble_env,
            field_env,
            reference_env,
            n_envs=1,
        )

        y0_env = y0[env : env + 1]

        _, y_final, _ = solve_rkck45(
            translating_bubble_rhs,
            y0_env,
            t_span=(0.0, 2.0),
            initial_step=1e-3,
            rtol=1e-9,
            atol=np.full(8, 1e-10),
            args=(params_env,),
        )

        independent_states.append(y_final[0])

    y_independent = np.asarray(independent_states)

    assert_allclose(
        y_batch,
        y_independent,
        rtol=1e-9,
        atol=1e-10,
    )
