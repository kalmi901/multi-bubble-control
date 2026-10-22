from dataclasses import dataclass

import numpy as np

from ._batching import (
    as_direction_matrix,
    as_env_vector,
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


def prepare_acoustic_field(
    material: MaterialProperties, field: AcousticField, reference_frequency: Parameter, n_envs: int
) -> PreparedAcousticField:
    """Prepare a prescribed acoustic field for repeated evaluation.

    The physical excitation frequencies belong to the field while
    ``reference_frequency`` defines the nondimensional time and
    position scales.

    Prepared arrays use shape ``(E, H)`` where E is the number of
    independent environments and H is the number of field components.
    """

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
    """Evaluate a prepared prescribed acoustic field.

    Supported position shapes
    -------------------------
    Single point per environment:
        (E, D)

    Multiple points per environment:
        (E, N, D)

    where D is 1, 2, or 3.
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

    squeeze_point_axis = False

    if position.ndim == 2:
        if position.shape[0] != n_envs:
            raise ValueError(f"position must have {n_envs} environments, got {position.shape}.")

        if not 1 <= position.shape[1] <= 3:
            raise ValueError("position spatial dimension must be 1, 2, or 3.")

        # Canonical internal representation: (E, N, 3)
        position = position[:, None, :]
        squeeze_point_axis = True

    elif position.ndim == 3:
        if position.shape[0] != n_envs:
            raise ValueError(f"position must have {n_envs} environments, got {position.shape}.")

        if not 1 <= position.shape[2] <= 3:
            raise ValueError("position spatial dimension must be 1, 2, or 3.")

    else:
        raise ValueError("position must have shape (E, D) or (E, N, D).")

    spatial_dim = position.shape[-1]

    if spatial_dim < 3:
        pad_width = (
            (0, 0),
            (0, 0),
            (0, 3 - spatial_dim),
        )

        position = np.pad(
            position,
            pad_width,
            mode="constant",
        )

    # (E, H)
    temporal_angle = field.dimensionless_angular_frequency * tau[:, None] + field.temporal_phase

    # (E, N, H)
    projected_position = np.einsum(
        "ehd,end->enh",
        field.direction,
        position,
    )

    spatial_angle = (
        field.dimensionless_wave_number[:, None, :] * projected_position
        + field.spatial_phase[:, None, :]
    )

    sin_t = np.sin(temporal_angle)[:, None, :]
    cos_t = np.cos(temporal_angle)[:, None, :]

    sin_x = np.sin(spatial_angle)
    cos_x = np.cos(spatial_angle)

    pressure = np.sum(
        field.pressure_amplitude[:, None, :] * sin_t * cos_x,
        axis=-1,
    )

    pressure_time_derivative = np.sum(
        field.pressure_amplitude[:, None, :]
        * field.physical_angular_frequency[:, None, :]
        * cos_t
        * cos_x,
        axis=-1,
    )

    gradient_amplitude = (
        -field.pressure_amplitude[:, None, :]
        * field.physical_wave_number[:, None, :]
        * sin_t
        * sin_x
    )

    pressure_gradient = np.einsum(
        "enh,ehd->end",
        gradient_amplitude,
        field.direction,
    )

    velocity_amplitude = -field.fluid_velocity_amplitude[:, None, :] * cos_t * sin_x

    fluid_velocity = np.einsum(
        "enh,ehd->end",
        velocity_amplitude,
        field.direction,
    )

    if squeeze_point_axis:
        pressure = pressure[:, 0]
        pressure_time_derivative = pressure_time_derivative[:, 0]
        pressure_gradient = pressure_gradient[:, 0, :]
        fluid_velocity = fluid_velocity[:, 0, :]

    return AcousticFieldState(
        pressure=pressure,
        pressure_time_derivative=pressure_time_derivative,
        pressure_gradient=pressure_gradient,
        fluid_velocity=fluid_velocity,
    )
