from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from multibubble.integrators import solve_rkck45
from multibubble.physics import MaterialProperties
from multibubble.physics.acoustic_field import AcousticField, UniformHarmonic
from multibubble.physics.common import ReferenceScales
from multibubble.physics.coupled_translating_bubbles import (
    CoupledTranslatingBubbleParameters,
    coupled_translating_bubble_rhs,
    prepare_coupled_translating_bubble_parameters,
)
from multibubble.physics.translating_bubble import (
    TranslatingBubbleParameters,
    prepare_translating_bubble_parameters,
    translating_bubble_rhs,
)

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "coupled_translating_bubbles"

FIXTURE_CASES = (
    pytest.param("doinikov_fig2_a.npz", id="doinikov_fig2_a-baseline"),
    pytest.param("doinikov_fig2_c.npz", id="doinikov_fig2_c-reversed-radii"),
    pytest.param("doinikov_fig2_f.npz", id="doinikov_fig2_f-strong-close"),
)


FULL_FIXTURE_CASES = (
    pytest.param("doinikov_fig2_a.npz", id="doinikov_fig2_a"),
    pytest.param("doinikov_fig2_b.npz", id="doinikov_fig2_b"),
    pytest.param("doinikov_fig2_c.npz", id="doinikov_fig2_c"),
    pytest.param("doinikov_fig2_e.npz", id="doinikov_fig2_e"),
    pytest.param("doinikov_fig2_f.npz", id="doinikov_fig2_f"),
)


def _load_fixture(name: str) -> dict[str, np.ndarray]:
    with np.load(FIXTURE_DIR / name) as data:
        return {key: data[key] for key in data.files}


@pytest.mark.parametrize(
    "fixture_name",
    FIXTURE_CASES,
)
def test_doinikov_figure2_reference_trajectory(
    fixture_name: str,
):
    fixture = _load_fixture(fixture_name)

    all_checkpoints = fixture["tau"]
    all_expected_states = fixture["state"]

    # CI regression only covers the first 10 acoustic cycles.
    # The full 100-cycle Figure-2 trajectories are retained as
    # validation fixtures but are not exercised in the regular test suite.
    mask = all_checkpoints <= 10.0

    checkpoints = all_checkpoints[mask]
    expected_states = all_expected_states[mask]

    equilibrium_radius = fixture["equilibrium_radius_m"]

    pressure_amplitude = float(fixture["pressure_amplitude_pa"])

    frequency = float(fixture["frequency_hz"])

    reference_radius = float(fixture["reference_radius_m"])

    material = MaterialProperties(
        density=998.0,
        sound_speed=1500.0,
        surface_tension=0.0725,
        dynamic_viscosity=0.001,
        vapor_pressure=0.0,
        ambient_pressure=1.013e5,
        polytropic_exponent=1.4,
    )

    bubbles = CoupledTranslatingBubbleParameters(
        equilibrium_radius=equilibrium_radius,
    )

    field = AcousticField(
        components=(
            UniformHarmonic(
                pressure_amplitude=pressure_amplitude,
                frequency=frequency,
            ),
        )
    )

    reference = ReferenceScales(
        radius=reference_radius,
        frequency=frequency,
    )

    params = prepare_coupled_translating_bubble_parameters(
        material,
        bubbles,
        field,
        reference,
        n_envs=1,
    )

    # Fixture state:
    #
    # (checkpoint, bubble, state)
    #
    # Add the environment dimension required by the model.
    state = fixture["state"][0][
        np.newaxis,
        ...,
    ]

    assert state.shape == (1, 2, 8)

    assert expected_states.shape == (
        checkpoints.size,
        2,
        8,
    )

    for checkpoint_index in range(
        1,
        checkpoints.size,
    ):
        t_start = float(checkpoints[checkpoint_index - 1])

        t_end = float(checkpoints[checkpoint_index])

        t_final, state, _ = solve_rkck45(
            coupled_translating_bubble_rhs,
            state,
            t_span=(t_start, t_end),
            initial_step=1e-3,
            rtol=1e-6,
            atol=np.full(
                8,
                1e-8,
                dtype=np.float64,
            ),
            args=(params,),
            max_steps=int(1e6),
        )

        assert_allclose(
            t_final,
            t_end,
            rtol=0.0,
            atol=1e-13,
        )

        expected = expected_states[checkpoint_index]

        # Radius and x-position are the primary
        # Figure-2 observables.
        assert_allclose(
            state[0, :, 0],
            expected[:, 0],
            rtol=5e-4,
            atol=2e-4,
        )

        assert_allclose(
            state[0, :, 1],
            expected[:, 1],
            rtol=1e-3,
            atol=2e-3,
        )

        # The legacy reference is one-dimensional.
        assert_allclose(
            state[0, :, 2:4],
            0.0,
            rtol=0.0,
            atol=1e-12,
        )

        # Velocity states are more phase-sensitive
        assert_allclose(
            state[0, :, 4],
            expected[:, 4],
            rtol=5e-3,
            atol=2e-3,
        )

        assert_allclose(
            state[0, :, 5],
            expected[:, 5],
            rtol=1e-2,
            atol=2e-5,
        )

        assert_allclose(
            state[0, :, 6:8],
            0.0,
            rtol=0.0,
            atol=1e-12,
        )


def test_batched_environments_match_independent_integrations():
    material = MaterialProperties()

    frequencies = np.array(
        [20e3, 25e3, 30e3],
        dtype=np.float64,
    )

    reference_radii = np.array(
        [50e-6, 60e-6, 70e-6],
        dtype=np.float64,
    )

    equilibrium_radii = np.array(
        [
            [35e-6, 45e-6, 60e-6],
            [40e-6, 55e-6, 65e-6],
            [45e-6, 60e-6, 75e-6],
        ],
        dtype=np.float64,
    )

    amplitudes = np.array(
        [0.15e5, 0.20e5, 0.25e5],
        dtype=np.float64,
    )

    bubbles = CoupledTranslatingBubbleParameters(
        equilibrium_radius=equilibrium_radii,
    )

    field = AcousticField(
        components=(
            UniformHarmonic(
                pressure_amplitude=amplitudes,
                frequency=frequencies,
            ),
        )
    )

    reference = ReferenceScales(
        radius=reference_radii,
        frequency=frequencies,
    )

    params = prepare_coupled_translating_bubble_parameters(
        material,
        bubbles,
        field,
        reference,
        n_envs=3,
    )

    y0 = np.zeros(
        (3, 3, 8),
        dtype=np.float64,
    )

    y0[..., 0] = equilibrium_radii / reference_radii[:, None]

    # Different geometry in every environment.
    y0[0, :, 1] = [0.0, 20.0, 55.0]
    y0[1, :, 1] = [0.0, 25.0, 70.0]
    y0[2, :, 1] = [0.0, 30.0, 80.0]

    _, y_batch, _ = solve_rkck45(
        coupled_translating_bubble_rhs,
        y0,
        t_span=(0.0, 1.0),
        initial_step=1e-3,
        rtol=1e-9,
        atol=np.full(8, 1e-10),
        args=(params,),
    )

    for env in range(3):
        bubbles_env = CoupledTranslatingBubbleParameters(
            equilibrium_radius=equilibrium_radii[env],
        )

        field_env = AcousticField(
            components=(
                UniformHarmonic(
                    pressure_amplitude=float(amplitudes[env]),
                    frequency=float(frequencies[env]),
                ),
            )
        )

        reference_env = ReferenceScales(
            radius=float(reference_radii[env]),
            frequency=float(frequencies[env]),
        )

        params_env = prepare_coupled_translating_bubble_parameters(
            material,
            bubbles_env,
            field_env,
            reference_env,
            n_envs=1,
        )

        _, y_single_env, _ = solve_rkck45(
            coupled_translating_bubble_rhs,
            y0[env : env + 1],
            t_span=(0.0, 1.0),
            initial_step=1e-3,
            rtol=1e-9,
            atol=np.full(8, 1e-10),
            args=(params_env,),
        )

        assert_allclose(
            y_batch[env],
            y_single_env[0],
            rtol=1e-9,
            atol=1e-10,
        )


def test_rhs_is_invariant_under_rigid_spatial_translation():
    material = MaterialProperties()

    bubbles = CoupledTranslatingBubbleParameters(
        equilibrium_radius=np.array(
            [40e-6, 50e-6, 65e-6],
            dtype=np.float64,
        ),
    )

    field = AcousticField(
        components=(
            UniformHarmonic(
                pressure_amplitude=0.2e5,
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
        bubbles,
        field,
        reference,
        n_envs=1,
    )

    y = np.array(
        [
            [
                [0.8, 0.0, 0.0, 0.0, 0.10, 0.03, -0.01, 0.02],
                [1.0, 8.0, 2.0, 0.0, -0.05, -0.02, 0.04, 0.01],
                [1.3, 35.0, 15.0, 4.0, 0.08, 0.01, 0.02, -0.03],
            ],
        ],
        dtype=np.float64,
    )

    shifted = y.copy()

    shifted[..., 1:4] += np.array(
        [123.0, -57.0, 81.0],
        dtype=np.float64,
    )

    t = np.array([0.37])

    rhs = coupled_translating_bubble_rhs(
        t,
        y,
        params,
    )

    rhs_shifted = coupled_translating_bubble_rhs(
        t,
        shifted,
        params,
    )

    assert_allclose(
        rhs_shifted,
        rhs,
        rtol=1e-13,
        atol=1e-13,
    )


def test_rhs_is_permutation_equivariant():
    material = MaterialProperties()

    equilibrium_radius = np.array(
        [40e-6, 55e-6, 70e-6],
        dtype=np.float64,
    )

    field = AcousticField(
        components=(
            UniformHarmonic(
                pressure_amplitude=0.25e5,
                frequency=25e3,
                temporal_phase=0.2,
            ),
        )
    )

    reference = ReferenceScales(
        radius=50e-6,
        frequency=25e3,
    )

    params = prepare_coupled_translating_bubble_parameters(
        material,
        CoupledTranslatingBubbleParameters(
            equilibrium_radius=equilibrium_radius,
        ),
        field,
        reference,
        n_envs=1,
    )

    y = np.array(
        [
            [
                [0.8, 0.0, 0.0, 0.0, 0.10, 0.02, -0.01, 0.00],
                [1.1, 6.0, 1.0, 0.0, -0.05, -0.03, 0.02, 0.01],
                [1.4, 40.0, 12.0, 3.0, 0.08, 0.01, 0.03, -0.02],
            ],
        ],
        dtype=np.float64,
    )

    t = np.array([0.31])

    rhs_original = coupled_translating_bubble_rhs(
        t,
        y,
        params,
    )

    permutation = np.array(
        [2, 0, 1],
        dtype=np.int64,
    )

    permuted_params = prepare_coupled_translating_bubble_parameters(
        material,
        CoupledTranslatingBubbleParameters(
            equilibrium_radius=equilibrium_radius[permutation],
        ),
        field,
        reference,
        n_envs=1,
    )

    rhs_permuted = coupled_translating_bubble_rhs(
        t,
        y[:, permutation],
        permuted_params,
    )

    assert_allclose(
        rhs_permuted[0],
        rhs_original[0, permutation],
        rtol=1e-12,
        atol=1e-12,
    )


def test_far_separated_bubbles_approach_independent_translation():
    material = MaterialProperties()

    frequency = 25e3
    reference_radius = 50e-6

    wavelength = material.sound_speed / frequency

    equilibrium_radius = np.array(
        [40e-6, 50e-6, 60e-6],
        dtype=np.float64,
    )

    bubbles = CoupledTranslatingBubbleParameters(
        equilibrium_radius=equilibrium_radius,
    )

    field = AcousticField(
        components=(
            UniformHarmonic(
                pressure_amplitude=0.1e5,
                frequency=frequency,
            ),
        )
    )

    reference = ReferenceScales(
        radius=reference_radius,
        frequency=frequency,
    )

    params = prepare_coupled_translating_bubble_parameters(
        material,
        bubbles,
        field,
        reference,
        n_envs=1,
    )

    y0 = np.zeros(
        (1, 3, 8),
        dtype=np.float64,
    )

    y0[0, :, 0] = equilibrium_radius / reference_radius

    separation = 10.0 * wavelength / reference_radius

    y0[0, :, 1] = np.array(
        [0.0, separation, 2.0 * separation],
        dtype=np.float64,
    )

    _, y_coupled, _ = solve_rkck45(
        coupled_translating_bubble_rhs,
        y0,
        t_span=(0.0, 1.0),
        initial_step=1e-3,
        rtol=1e-9,
        atol=np.full(8, 1e-10),
        args=(params,),
    )

    for bubble_index, radius in enumerate(equilibrium_radius):
        single_params = prepare_translating_bubble_parameters(
            material,
            TranslatingBubbleParameters(
                equilibrium_radius=float(radius),
            ),
            field,
            reference,
            n_envs=1,
        )

        y0_single = y0[
            :,
            bubble_index,
            :,
        ]

        _, y_single, _ = solve_rkck45(
            translating_bubble_rhs,
            y0_single,
            t_span=(0.0, 1.0),
            initial_step=1e-3,
            rtol=1e-9,
            atol=np.full(8, 1e-10),
            args=(single_params,),
        )

        # Radial state.
        assert_allclose(
            y_coupled[0, bubble_index, [0, 4]],
            y_single[0, [0, 4]],
            rtol=3e-4,
            atol=1e-7,
        )

        # Compare translational displacement rather than
        # the very large absolute x coordinate.
        coupled_displacement = y_coupled[0, bubble_index, 1:4] - y0[0, bubble_index, 1:4]

        single_displacement = y_single[0, 1:4] - y0_single[0, 1:4]

        assert_allclose(
            coupled_displacement,
            single_displacement,
            rtol=0.0,
            atol=1e-5,
        )

        assert_allclose(
            y_coupled[0, bubble_index, 5:8],
            y_single[0, 5:8],
            rtol=0.0,
            atol=1e-5,
        )
