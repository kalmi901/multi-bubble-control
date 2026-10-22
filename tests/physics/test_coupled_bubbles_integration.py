from dataclasses import replace

import numpy as np
from numpy.testing import assert_allclose

from multibubble.integrators import solve_rkck45
from multibubble.physics import MaterialProperties
from multibubble.physics.acoustic_field import AcousticField, UniformHarmonic
from multibubble.physics.common import ReferenceScales
from multibubble.physics.coupled_bubbles import (
    CoupledBubblesParameters,
    coupled_bubbles_rhs,
    prepare_coupled_bubbles_parameters,
)
from multibubble.physics.single_bubble import (
    BubbleParameters,
    prepare_single_bubble_parameters,
    single_bubble_rhs,
)


def _integrate_coupled(
    params,
    y0,
    *,
    t_end=5.0,
    rtol=1e-10,
    atol=1e-12,
    store_trajectory=False,
):
    t, y, info = solve_rkck45(
        coupled_bubbles_rhs,
        y0,
        t_span=(0.0, t_end),
        initial_step=1e-3,
        rtol=rtol,
        atol=np.array([atol, atol]),
        args=(params,),
        store_trajectory=store_trajectory,
    )

    return t, y, info


def _integrate_single(
    params,
    y0,
    *,
    t_end=5.0,
    rtol=1e-10,
    atol=1e-12,
):
    return solve_rkck45(
        single_bubble_rhs,
        y0,
        t_span=(0.0, t_end),
        initial_step=1e-3,
        rtol=rtol,
        atol=np.array([atol, atol]),
        args=(params,),
    )


def test_zero_coupling_matches_independent_single_bubbles():
    material = MaterialProperties()

    radii = np.array(
        [35e-6, 50e-6, 65e-6],
        dtype=np.float64,
    )

    reference_radius = 50e-6

    field = AcousticField(components=(UniformHarmonic(pressure_amplitude=0.3e5, frequency=25e3),))

    reference = ReferenceScales(radius=reference_radius, frequency=field.components[0].frequency)

    bubbles = CoupledBubblesParameters(
        equilibrium_radius=radii,
        position=np.array(
            [
                [0.0],
                [1.0],
                [2.0],
            ],
            dtype=np.float64,
        ),
    )

    params = prepare_coupled_bubbles_parameters(material, bubbles, field, reference, 1)

    params = replace(
        params,
        radial_coupling=np.zeros_like(params.radial_coupling),
    )

    y0 = np.stack(
        (
            params.equilibrium_radius_ratio,
            np.zeros_like(params.equilibrium_radius_ratio),
        ),
        axis=-1,
    )

    _, y_coupled, _ = _integrate_coupled(
        params,
        y0,
    )

    for j, radius in enumerate(radii):
        single_params = prepare_single_bubble_parameters(
            material,
            BubbleParameters(
                equilibrium_radius=radius,
            ),
            field,
            reference,
            1,
        )

        y0_single = np.array(
            [[radius / reference_radius, 0.0]],
            dtype=np.float64,
        )

        _, y_single, _ = _integrate_single(
            single_params,
            y0_single,
        )

        assert_allclose(
            y_coupled[0, j],
            y_single[0],
            rtol=1e-8,
            atol=1e-10,
        )


def test_identical_two_bubbles_remain_symmetric():
    material = MaterialProperties()

    frequency = 25e3
    reference_radius = 50e-6
    reference_length = material.sound_speed / frequency

    bubbles = CoupledBubblesParameters(
        equilibrium_radius=np.array(
            [50e-6, 50e-6],
            dtype=np.float64,
        ),
        position=np.array(
            [
                [-0.5 * reference_length / reference_radius],
                [0.5 * reference_length / reference_radius],
            ],
            dtype=np.float64,
        ),
    )

    field = AcousticField(
        components=(UniformHarmonic(pressure_amplitude=0.4e5, frequency=frequency),)
    )

    reference = ReferenceScales(radius=reference_radius, frequency=field.components[0].frequency)

    params = prepare_coupled_bubbles_parameters(material, bubbles, field, reference, 1)

    y0 = np.stack(
        (
            params.equilibrium_radius_ratio,
            np.zeros_like(params.equilibrium_radius_ratio),
        ),
        axis=-1,
    )

    _, _, info = _integrate_coupled(
        params,
        y0,
        store_trajectory=True,
    )

    y = info["trajectory"]["y"][0]

    assert_allclose(
        y[:, 0, :],
        y[:, 1, :],
        rtol=1e-12,
        atol=1e-12,
    )


def test_permutation_equivariance():
    material = MaterialProperties()

    frequency = 25e3
    reference_radius = 50e-6
    reference_length = material.sound_speed / frequency

    equilibrium_radius = np.array(
        [35e-6, 50e-6, 70e-6],
        dtype=np.float64,
    )

    position = np.array(
        [
            [0.0 * reference_length / reference_radius],
            [0.08 * reference_length / reference_radius],
            [0.21 * reference_length / reference_radius],
        ],
        dtype=np.float64,
    )

    field = AcousticField(
        components=(UniformHarmonic(pressure_amplitude=0.35e5, frequency=25e3, temporal_phase=0.2),)
    )

    reference = ReferenceScales(radius=reference_radius, frequency=field.components[0].frequency)

    # ---------------------------------------------------------
    # Original ordering
    # ---------------------------------------------------------

    bubbles = CoupledBubblesParameters(
        equilibrium_radius=equilibrium_radius,
        position=position,
    )

    params = prepare_coupled_bubbles_parameters(material, bubbles, field, reference, 1)

    y0 = np.stack(
        (
            params.equilibrium_radius_ratio,
            np.zeros_like(params.equilibrium_radius_ratio),
        ),
        axis=-1,
    )

    _, y_original, _ = _integrate_coupled(
        params,
        y0,
        t_end=5.0,
        rtol=1e-10,
        atol=1e-12,
    )

    # ---------------------------------------------------------
    # Permuted ordering
    # ---------------------------------------------------------

    permutation = np.array(
        [2, 0, 1],
        dtype=np.int64,
    )

    permuted_bubbles = CoupledBubblesParameters(
        equilibrium_radius=equilibrium_radius[permutation],
        position=position[permutation],
    )

    permuted_params = prepare_coupled_bubbles_parameters(
        material, permuted_bubbles, field, reference, 1
    )

    y0_permuted = np.stack(
        (
            permuted_params.equilibrium_radius_ratio,
            np.zeros_like(permuted_params.equilibrium_radius_ratio),
        ),
        axis=-1,
    )

    _, y_permuted, _ = _integrate_coupled(
        permuted_params,
        y0_permuted,
        t_end=5.0,
        rtol=1e-10,
        atol=1e-12,
    )

    assert_allclose(
        y_permuted[0],
        y_original[0, permutation],
        rtol=1e-9,
        atol=1e-11,
    )


def test_solution_converges_with_tighter_rkck45_tolerances():
    material = MaterialProperties()

    frequency = 25e3
    reference_radius = 50e-6
    reference_length = material.sound_speed / frequency

    bubbles = CoupledBubblesParameters(
        equilibrium_radius=np.array(
            [40e-6, 55e-6],
            dtype=np.float64,
        ),
        position=np.array(
            [
                [0.0 * reference_length / reference_radius],
                [0.08 * reference_length / reference_radius],
            ],
            dtype=np.float64,
        ),
    )

    field = AcousticField(
        components=(
            UniformHarmonic(pressure_amplitude=0.30e5, frequency=frequency, temporal_phase=0.15),
        )
    )

    reference = ReferenceScales(radius=reference_radius, frequency=frequency)

    params = prepare_coupled_bubbles_parameters(material, bubbles, field, reference, 1)

    y0 = np.stack(
        (
            params.equilibrium_radius_ratio,
            np.zeros_like(params.equilibrium_radius_ratio),
        ),
        axis=-1,
    )

    # ---------------------------------------------------------
    # Coarse solution
    # ---------------------------------------------------------

    _, y_coarse, _ = _integrate_coupled(
        params,
        y0,
        t_end=5.0,
        rtol=1e-6,
        atol=1e-8,
    )

    # ---------------------------------------------------------
    # Fine solution
    # ---------------------------------------------------------

    _, y_fine, _ = _integrate_coupled(
        params,
        y0,
        t_end=5.0,
        rtol=1e-9,
        atol=1e-11,
    )

    # ---------------------------------------------------------
    # Reference solution
    # ---------------------------------------------------------

    _, y_reference, _ = _integrate_coupled(
        params,
        y0,
        t_end=5.0,
        rtol=1e-12,
        atol=1e-14,
    )

    error_coarse = np.linalg.norm(
        y_coarse - y_reference,
    )

    error_fine = np.linalg.norm(
        y_fine - y_reference,
    )

    assert np.all(np.isfinite(y_coarse))
    assert np.all(np.isfinite(y_fine))
    assert np.all(np.isfinite(y_reference))
    assert error_coarse > 0.0
    assert error_fine < error_coarse


def test_batched_environments_match_independent_coupled_integrations():
    material = MaterialProperties()

    frequency = np.array(
        [22e3, 25e3, 28e3],
        dtype=np.float64,
    )
    reference_radius = np.array(
        [50e-6, 55e-6, 60e-6],
        dtype=np.float64,
    )
    reference_length = material.sound_speed / frequency

    equilibrium_radius = np.array(
        [
            [35e-6, 50e-6],
            [40e-6, 55e-6],
            [45e-6, 60e-6],
        ],
        dtype=np.float64,
    )

    position = (
        np.array(
            [
                [
                    [0.0],
                    [0.12],
                ],
                [
                    [0.0],
                    [0.18],
                ],
                [
                    [0.0],
                    [0.25],
                ],
            ],
            dtype=np.float64,
        )
        * reference_length
        / reference_radius
    )

    pressure_amplitude = np.array(
        [0.20e5, 0.30e5, 0.40e5],
        dtype=np.float64,
    )

    phase = np.array(
        [0.1, 0.2, -0.15],
        dtype=np.float64,
    )

    bubbles = CoupledBubblesParameters(
        equilibrium_radius=equilibrium_radius,
        position=position,
    )

    field = AcousticField(
        components=(
            UniformHarmonic(
                pressure_amplitude=pressure_amplitude, frequency=frequency, temporal_phase=phase
            ),
        )
    )

    reference = ReferenceScales(radius=reference_radius, frequency=frequency)

    params = prepare_coupled_bubbles_parameters(material, bubbles, field, reference, 3)

    y0 = np.stack(
        (
            params.equilibrium_radius_ratio,
            np.zeros_like(params.equilibrium_radius_ratio),
        ),
        axis=-1,
    )

    _, y_batched, _ = _integrate_coupled(
        params,
        y0,
        t_end=4.0,
        rtol=1e-10,
        atol=1e-12,
    )

    for env in range(equilibrium_radius.shape[0]):
        independent_bubbles = CoupledBubblesParameters(
            equilibrium_radius=equilibrium_radius[env],
            position=position[env],
        )

        independent_field = AcousticField(
            components=(
                UniformHarmonic(
                    pressure_amplitude=pressure_amplitude[env],
                    frequency=frequency[env],
                    temporal_phase=phase[env],
                ),
            )
        )

        independet_reference = ReferenceScales(
            radius=reference_radius[env], frequency=frequency[env]
        )

        independent_params = prepare_coupled_bubbles_parameters(
            material, independent_bubbles, independent_field, independet_reference, 1
        )

        y0_independent = np.stack(
            (
                independent_params.equilibrium_radius_ratio,
                np.zeros_like(independent_params.equilibrium_radius_ratio),
            ),
            axis=-1,
        )

        _, y_independent, _ = _integrate_coupled(
            independent_params,
            y0_independent,
            t_end=4.0,
            rtol=1e-10,
            atol=1e-12,
        )

        assert_allclose(
            y_batched[env],
            y_independent[0],
            rtol=1e-8,
            atol=1e-10,
        )
