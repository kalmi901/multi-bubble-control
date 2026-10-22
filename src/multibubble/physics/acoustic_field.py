from dataclasses import dataclass

import numpy as np

from ._batching import (
    as_direction_matrix,
    as_env_vector,
    direction_environment_size,
    environment_size,
    resolve_num_envs,
)
from .common import FloatArray, MaterialProperties, Parameter


@dataclass(frozen=True)
class UniformHarmonic:
    pressure_amplitude: Parameter
    frequency: Parameter
    temporal_phase: Parameter = 0.0


@dataclass(frozen=True)
class PlaneStandingWave:
    pressure_amplitude: Parameter
    frequency: Parameter
    direction: FloatArray
    temporal_phase: Parameter = 0.0
    spatial_phase: Parameter = 0.0


AcousticComponent = UniformHarmonic | PlaneStandingWave


@dataclass(frozen=True)
class AcousticField:
    components: tuple[AcousticComponent, ...]


@dataclass(frozen=True)
class PreparedAcousticField:
    pressure_amplitude: FloatArray
    # (E, H)

    dimensionless_angular_frequency: FloatArray
    # (E, H)

    physical_angular_frequency: FloatArray
    # (E, H)

    dimensionless_wave_number: FloatArray
    # (E, H)

    physical_wave_number: FloatArray
    # (E, H)

    temporal_phase: FloatArray
    # (E, H)

    spatial_phase: FloatArray
    # (E, H)

    direction: FloatArray
    # (E, H, 3)

    fluid_velocity_amplitude: FloatArray
    # (E, H)


@dataclass(frozen=True)
class AcousticFieldState:
    """Evaluated prescribed acoustic field.

    Shapes for single-bubble evaluation:

    t        : (E,)
    position : (E, 3)

    pressure                 : (E,)
    pressure_time_derivative : (E,)
    pressure_gradient        : (E, 3)
    fluid_velocity           : (E, 3)
    """

    pressure: FloatArray
    pressure_time_derivative: FloatArray
    pressure_gradient: FloatArray
    fluid_velocity: FloatArray


def _infer_num_envs(
    material: MaterialProperties,
    field: AcousticField,
    reference_frequency: Parameter,
) -> int:
    """Infer and validate the number of independent environments."""

    candidates: list[tuple[str, int]] = []

    environment_parameters = (
        ("density", material.density),
        ("sound_speed", material.sound_speed),
        ("surface_tension", material.surface_tension),
        ("dynamic_viscosity", material.dynamic_viscosity),
        ("vapor_pressure", material.vapor_pressure),
        ("ambient_pressure", material.ambient_pressure),
        ("polytropic_exponent", material.polytropic_exponent),
        ("reference_frequency", reference_frequency),
    )

    for name, value in environment_parameters:
        size = environment_size(
            value,
            name=name,
        )

        if size is not None:
            candidates.append((name, size))

    for component_index, component in enumerate(field.components):
        component_parameters = (
            (
                f"components[{component_index}].pressure_amplitude",
                component.pressure_amplitude,
            ),
            (
                f"components[{component_index}].frequency",
                component.frequency,
            ),
            (
                f"components[{component_index}].temporal_phase",
                component.temporal_phase,
            ),
        )

        for name, value in component_parameters:
            size = environment_size(
                value,
                name=name,
            )

            if size is not None:
                candidates.append((name, size))

        if isinstance(component, PlaneStandingWave):
            size = environment_size(
                component.spatial_phase,
                name=f"components[{component_index}].spatial_phase",
            )

            if size is not None:
                candidates.append(
                    (
                        f"components[{component_index}].spatial_phase",
                        size,
                    )
                )

            direction_size = direction_environment_size(
                component.direction,
                name=f"components[{component_index}].direction",
            )

            if direction_size is not None:
                candidates.append(
                    (
                        f"components[{component_index}].direction",
                        direction_size,
                    )
                )

        elif not isinstance(component, UniformHarmonic):
            raise TypeError(f"Unsupported acoustic field component: {type(component).__name__}.")

    return resolve_num_envs(
        candidates,
        context="Acoustic-field parameters",
    )


def prepare_acoustic_field(
    material: MaterialProperties,
    field: AcousticField,
    reference_frequency: Parameter,
) -> PreparedAcousticField:
    """Prepare a prescribed acoustic field for repeated evaluation.

    The physical excitation frequencies belong to the field while
    ``reference_frequency`` defines the nondimensional time and
    position scales.

    Prepared arrays use shape ``(E, H)`` where E is the number of
    independent environments and H is the number of field components.
    """

    n_envs = _infer_num_envs(
        material,
        field,
        reference_frequency,
    )

    density = as_env_vector(
        material.density,
        n_envs,
        name="density",
    )

    sound_speed = as_env_vector(
        material.sound_speed,
        n_envs,
        name="sound_speed",
    )

    f_ref = as_env_vector(
        reference_frequency,
        n_envs,
        name="reference_frequency",
    )

    if np.any(density <= 0.0):
        raise ValueError("density must be positive.")

    if np.any(sound_speed <= 0.0):
        raise ValueError("sound_speed must be positive.")

    if np.any(f_ref <= 0.0):
        raise ValueError("reference_frequency must be positive.")

    n_components = len(field.components)

    shape = (n_envs, n_components)

    pressure_amplitude = np.zeros(
        shape,
        dtype=np.float64,
    )

    dimensionless_angular_frequency = np.zeros(
        shape,
        dtype=np.float64,
    )

    physical_angular_frequency = np.zeros(
        shape,
        dtype=np.float64,
    )

    dimensionless_wave_number = np.zeros(
        shape,
        dtype=np.float64,
    )

    physical_wave_number = np.zeros(
        shape,
        dtype=np.float64,
    )

    temporal_phase = np.zeros(
        shape,
        dtype=np.float64,
    )

    spatial_phase = np.zeros(
        shape,
        dtype=np.float64,
    )

    direction = np.zeros(
        (n_envs, n_components, 3),
        dtype=np.float64,
    )

    fluid_velocity_amplitude = np.zeros(
        shape,
        dtype=np.float64,
    )

    for component_index, component in enumerate(field.components):
        amplitude = as_env_vector(
            component.pressure_amplitude,
            n_envs,
            name=f"components[{component_index}].pressure_amplitude",
        )

        frequency = as_env_vector(
            component.frequency,
            n_envs,
            name=f"components[{component_index}].frequency",
        )

        phase = as_env_vector(
            component.temporal_phase,
            n_envs,
            name=f"components[{component_index}].temporal_phase",
        )

        if np.any(frequency <= 0.0):
            raise ValueError(f"components[{component_index}].frequency must be positive.")

        frequency_ratio = frequency / f_ref

        pressure_amplitude[:, component_index] = amplitude

        dimensionless_angular_frequency[:, component_index] = 2.0 * np.pi * frequency_ratio

        physical_angular_frequency[:, component_index] = 2.0 * np.pi * frequency

        temporal_phase[:, component_index] = phase

        if isinstance(component, UniformHarmonic):
            # Spatial quantities remain zero:
            #
            # k* = 0
            # k  = 0
            # direction = 0
            # u_ac amplitude = 0
            continue

        if isinstance(component, PlaneStandingWave):
            spatial_phase[:, component_index] = as_env_vector(
                component.spatial_phase,
                n_envs,
                name=f"components[{component_index}].spatial_phase",
            )

            component_direction = as_direction_matrix(
                component.direction,
                n_envs,
                name=f"components[{component_index}].direction",
            )

            direction_norm = np.linalg.norm(
                component_direction,
                axis=1,
            )

            if np.any(direction_norm == 0.0):
                raise ValueError(f"components[{component_index}].direction must be non-zero.")

            unit_direction = component_direction / direction_norm[:, None]

            direction[:, component_index, :] = unit_direction

            dimensionless_wave_number[:, component_index] = 2.0 * np.pi * frequency_ratio

            physical_wave_number[:, component_index] = 2.0 * np.pi * frequency / sound_speed

            fluid_velocity_amplitude[:, component_index] = amplitude / (density * sound_speed)

            continue

        raise TypeError(f"Unsupported acoustic field component: {type(component).__name__}.")

    return PreparedAcousticField(
        pressure_amplitude=pressure_amplitude,
        dimensionless_angular_frequency=dimensionless_angular_frequency,
        physical_angular_frequency=physical_angular_frequency,
        dimensionless_wave_number=dimensionless_wave_number,
        physical_wave_number=physical_wave_number,
        temporal_phase=temporal_phase,
        spatial_phase=spatial_phase,
        direction=direction,
        fluid_velocity_amplitude=fluid_velocity_amplitude,
    )


def evaluate_acoustic_field(
    t: FloatArray,
    position: FloatArray,
    field: PreparedAcousticField,
) -> AcousticFieldState:
    """Evaluate a prepared acoustic field.

    Parameters
    ----------
    t
        Dimensionless time tau = f_ref * t_phys, shape (E,).

    position
        Dimensionless position x / L_ref, shape (E, 3).

    field
        Prepared acoustic field.

    Returns
    -------
    AcousticFieldState
        Pressure and physical pressure derivative in SI units,
        together with physical pressure gradient and acoustic
        liquid velocity.
    """

    tau = np.asarray(
        t,
        dtype=np.float64,
    )

    position = np.asarray(
        position,
        dtype=np.float64,
    )

    n_envs = field.pressure_amplitude.shape[0]

    if tau.ndim == 0:
        tau = np.broadcast_to(
            tau.reshape(1),
            (n_envs,),
        )
    elif tau.shape != (n_envs,):
        raise ValueError(f"t must be a scalar or have shape ({n_envs},), got {tau.shape}.")

    if position.shape != (n_envs, 3):
        raise ValueError(f"position must have shape ({n_envs}, 3), got {position.shape}.")

    temporal_angle = field.dimensionless_angular_frequency * tau[:, None] + field.temporal_phase

    projected_position = np.einsum(
        "ehd,ed->eh",
        field.direction,
        position,
    )

    spatial_angle = field.dimensionless_wave_number * projected_position + field.spatial_phase

    sin_t = np.sin(temporal_angle)
    cos_t = np.cos(temporal_angle)

    sin_x = np.sin(spatial_angle)
    cos_x = np.cos(spatial_angle)

    pressure = np.sum(
        field.pressure_amplitude * sin_t * cos_x,
        axis=1,
    )

    pressure_time_derivative = np.sum(
        field.pressure_amplitude * field.physical_angular_frequency * cos_t * cos_x,
        axis=1,
    )

    gradient_amplitude = -field.pressure_amplitude * field.physical_wave_number * sin_t * sin_x

    pressure_gradient = np.einsum(
        "eh,ehd->ed",
        gradient_amplitude,
        field.direction,
    )

    velocity_amplitude = -field.fluid_velocity_amplitude * cos_t * sin_x

    fluid_velocity = np.einsum(
        "eh,ehd->ed",
        velocity_amplitude,
        field.direction,
    )

    return AcousticFieldState(
        pressure=pressure,
        pressure_time_derivative=pressure_time_derivative,
        pressure_gradient=pressure_gradient,
        fluid_velocity=fluid_velocity,
    )
