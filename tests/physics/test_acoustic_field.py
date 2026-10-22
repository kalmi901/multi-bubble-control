import numpy as np
import pytest
from numpy.testing import assert_allclose

from multibubble.physics.acoustic_field import (
    AcousticField,
    PlaneStandingWave,
    UniformHarmonic,
    evaluate_acoustic_field,
    prepare_acoustic_field,
)
from multibubble.physics.common import MaterialProperties


def test_prepare_uniform_harmonic():
    material = MaterialProperties()

    field = AcousticField(
        components=(
            UniformHarmonic(
                pressure_amplitude=2.0e5,
                frequency=50e3,
                temporal_phase=0.2,
            ),
        )
    )

    prepared = prepare_acoustic_field(
        material,
        field,
        reference_frequency=25e3,
    )

    assert prepared.pressure_amplitude.shape == (1, 1)
    assert prepared.direction.shape == (1, 1, 3)

    assert_allclose(
        prepared.pressure_amplitude,
        [[2.0e5]],
    )

    assert_allclose(
        prepared.dimensionless_angular_frequency,
        [[4.0 * np.pi]],
    )

    assert_allclose(
        prepared.physical_angular_frequency,
        [[2.0 * np.pi * 50e3]],
    )

    assert_allclose(
        prepared.dimensionless_wave_number,
        0.0,
    )

    assert_allclose(
        prepared.physical_wave_number,
        0.0,
    )

    assert_allclose(
        prepared.direction,
        0.0,
    )

    assert_allclose(
        prepared.fluid_velocity_amplitude,
        0.0,
    )


def test_prepare_standing_wave_normalizes_direction():
    material = MaterialProperties(
        density=1000.0,
        sound_speed=1500.0,
    )

    field = AcousticField(
        components=(
            PlaneStandingWave(
                pressure_amplitude=1.5e5,
                frequency=50e3,
                direction=np.array([1.0, 2.0, 2.0]),
            ),
        )
    )

    prepared = prepare_acoustic_field(
        material,
        field,
        reference_frequency=25e3,
    )

    assert_allclose(
        prepared.direction[0, 0],
        np.array([1.0, 2.0, 2.0]) / 3.0,
    )

    assert_allclose(
        prepared.dimensionless_wave_number[0, 0],
        4.0 * np.pi,
    )

    assert_allclose(
        prepared.physical_wave_number[0, 0],
        2.0 * np.pi * 50e3 / 1500.0,
    )

    assert_allclose(
        prepared.fluid_velocity_amplitude[0, 0],
        1.5e5 / (1000.0 * 1500.0),
    )


def test_prepare_rejects_zero_standing_wave_direction():
    field = AcousticField(
        components=(
            PlaneStandingWave(
                pressure_amplitude=1.0e5,
                frequency=25e3,
                direction=np.zeros(3),
            ),
        )
    )

    with pytest.raises(
        ValueError,
        match="direction must be non-zero",
    ):
        prepare_acoustic_field(
            MaterialProperties(),
            field,
            reference_frequency=25e3,
        )


def test_prepare_rejects_inconsistent_environment_counts():
    field = AcousticField(
        components=(
            UniformHarmonic(
                pressure_amplitude=np.array(
                    [1.0e5, 2.0e5],
                ),
                frequency=np.array(
                    [20e3, 25e3, 30e3],
                ),
            ),
        )
    )

    with pytest.raises(
        ValueError,
        match="inconsistent environment counts",
    ):
        prepare_acoustic_field(
            MaterialProperties(),
            field,
            reference_frequency=25e3,
        )


def test_uniform_harmonic_evaluation():
    pressure_amplitude = 1.7e5
    frequency = 25e3

    field = AcousticField(
        components=(
            UniformHarmonic(
                pressure_amplitude=pressure_amplitude,
                frequency=frequency,
            ),
        )
    )

    prepared = prepare_acoustic_field(
        MaterialProperties(),
        field,
        reference_frequency=frequency,
    )

    state = evaluate_acoustic_field(
        np.array([0.25]),
        np.zeros((1, 3)),
        prepared,
    )

    assert_allclose(
        state.pressure,
        pressure_amplitude,
        rtol=1e-14,
    )

    expected_pressure_dt = pressure_amplitude * 2.0 * np.pi * frequency * np.cos(2.0 * np.pi * 0.25)

    assert_allclose(
        state.pressure_time_derivative,
        expected_pressure_dt,
        rtol=1e-14,
        atol=1e-10,
    )

    assert_allclose(
        state.pressure_gradient,
        0.0,
        atol=0.0,
    )

    assert_allclose(
        state.fluid_velocity,
        0.0,
        atol=0.0,
    )


def test_standing_wave_antinode_and_node():
    material = MaterialProperties()

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

    prepared = prepare_acoustic_field(
        material,
        field,
        reference_frequency=frequency,
    )

    # tau = 1/4 -> sin(2*pi*tau) = 1
    tau = np.array([0.25])

    antinode = evaluate_acoustic_field(
        tau,
        np.array([[0.0, 0.0, 0.0]]),
        prepared,
    )

    assert_allclose(
        antinode.pressure,
        pressure_amplitude,
        rtol=1e-14,
    )

    assert_allclose(
        antinode.pressure_gradient,
        0.0,
        atol=1e-9,
    )

    # x* = 1/4 for f = f_ref:
    # k*x = 2*pi * 1/4 = pi/2 -> pressure node.
    node = evaluate_acoustic_field(
        tau,
        np.array([[0.25, 0.0, 0.0]]),
        prepared,
    )

    assert_allclose(
        node.pressure,
        0.0,
        atol=1e-9,
    )

    expected_gradient = np.array(
        [
            [
                -pressure_amplitude * (2.0 * np.pi * frequency / material.sound_speed),
                0.0,
                0.0,
            ]
        ]
    )

    assert_allclose(
        node.pressure_gradient,
        expected_gradient,
        rtol=1e-14,
        atol=1e-9,
    )


def test_pressure_gradient_matches_finite_difference():
    material = MaterialProperties(
        sound_speed=1500.0,
    )

    reference_frequency = 25e3

    field = AcousticField(
        components=(
            PlaneStandingWave(
                pressure_amplitude=1.2e5,
                frequency=32e3,
                direction=np.array([1.0, 2.0, 3.0]),
                temporal_phase=0.17,
                spatial_phase=-0.21,
            ),
        )
    )

    prepared = prepare_acoustic_field(
        material,
        field,
        reference_frequency,
    )

    tau = np.array([0.31])

    position = np.array(
        [[0.11, -0.07, 0.04]],
        dtype=np.float64,
    )

    state = evaluate_acoustic_field(
        tau,
        position,
        prepared,
    )

    reference_length = material.sound_speed / reference_frequency

    h = 1e-6

    gradient_fd = np.zeros(3)

    for axis in range(3):
        delta = np.zeros_like(position)
        delta[0, axis] = h

        p_plus = evaluate_acoustic_field(
            tau,
            position + delta,
            prepared,
        ).pressure[0]

        p_minus = evaluate_acoustic_field(
            tau,
            position - delta,
            prepared,
        ).pressure[0]

        gradient_fd[axis] = (p_plus - p_minus) / (2.0 * h * reference_length)

    assert_allclose(
        state.pressure_gradient[0],
        gradient_fd,
        rtol=1e-7,
        atol=1e-5,
    )


def test_standing_wave_gradient_is_parallel_to_direction():
    direction = np.array(
        [1.0, 2.0, 3.0],
        dtype=np.float64,
    )

    unit_direction = direction / np.linalg.norm(direction)

    field = AcousticField(
        components=(
            PlaneStandingWave(
                pressure_amplitude=1.0e5,
                frequency=25e3,
                direction=direction,
            ),
        )
    )

    prepared = prepare_acoustic_field(
        MaterialProperties(),
        field,
        reference_frequency=25e3,
    )

    state = evaluate_acoustic_field(
        np.array([0.25]),
        np.array([[0.1, 0.1, 0.1]]),
        prepared,
    )

    gradient = state.pressure_gradient[0]

    parallel_component = np.dot(gradient, unit_direction) * unit_direction

    assert_allclose(
        gradient,
        parallel_component,
        rtol=1e-14,
        atol=1e-8,
    )


def test_field_superposition_equals_sum_of_components():
    material = MaterialProperties()
    reference_frequency = 25e3

    component_a = UniformHarmonic(
        pressure_amplitude=0.4e5,
        frequency=25e3,
        temporal_phase=0.1,
    )

    component_b = PlaneStandingWave(
        pressure_amplitude=0.8e5,
        frequency=40e3,
        direction=np.array([1.0, 2.0, 3.0]),
        temporal_phase=-0.2,
        spatial_phase=0.3,
    )

    combined = prepare_acoustic_field(
        material,
        AcousticField(
            components=(
                component_a,
                component_b,
            )
        ),
        reference_frequency,
    )

    prepared_a = prepare_acoustic_field(
        material,
        AcousticField(
            components=(component_a,),
        ),
        reference_frequency,
    )

    prepared_b = prepare_acoustic_field(
        material,
        AcousticField(
            components=(component_b,),
        ),
        reference_frequency,
    )

    tau = np.array([0.37])

    position = np.array(
        [[0.13, -0.04, 0.08]],
    )

    state = evaluate_acoustic_field(
        tau,
        position,
        combined,
    )

    state_a = evaluate_acoustic_field(
        tau,
        position,
        prepared_a,
    )

    state_b = evaluate_acoustic_field(
        tau,
        position,
        prepared_b,
    )

    assert_allclose(
        state.pressure,
        state_a.pressure + state_b.pressure,
    )

    assert_allclose(
        state.pressure_time_derivative,
        (state_a.pressure_time_derivative + state_b.pressure_time_derivative),
    )

    assert_allclose(
        state.pressure_gradient,
        (state_a.pressure_gradient + state_b.pressure_gradient),
    )

    assert_allclose(
        state.fluid_velocity,
        (state_a.fluid_velocity + state_b.fluid_velocity),
    )


def test_empty_acoustic_field_evaluates_to_zero():
    prepared = prepare_acoustic_field(
        MaterialProperties(),
        AcousticField(
            components=(),
        ),
        reference_frequency=np.array([25e3, 25e3]),
    )

    state = evaluate_acoustic_field(
        np.array([0.1, 0.7]),
        np.zeros((2, 3)),
        prepared,
    )

    assert state.pressure.shape == (2,)
    assert state.pressure_gradient.shape == (2, 3)

    assert_allclose(state.pressure, 0.0)
    assert_allclose(state.pressure_time_derivative, 0.0)
    assert_allclose(state.pressure_gradient, 0.0)
    assert_allclose(state.fluid_velocity, 0.0)
