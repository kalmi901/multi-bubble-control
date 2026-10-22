from dataclasses import dataclass

import numpy as np

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


def _environment_size(
    value: Parameter,
    *,
    name: str,
) -> int | None:
    """Return the environment count implied by a scalar or 1D parameter."""

    arr = np.asarray(
        value,
        dtype=np.float64,
    )

    if arr.ndim == 0:
        return None

    if arr.ndim == 1:
        return arr.shape[0]

    raise ValueError(f"{name} must be a scalar or have shape (n_envs,), got {arr.shape}.")


def _direction_environment_size(
    value: FloatArray,
    *,
    name: str,
) -> int | None:
    """Return the environment count implied by a direction array."""

    arr = np.asarray(
        value,
        dtype=np.float64,
    )

    if arr.shape == (3,):
        return None

    if arr.ndim == 2 and arr.shape[1] == 3:
        return arr.shape[0]

    raise ValueError(f"{name} must have shape (3,) or (n_envs, 3), got {arr.shape}.")


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
        ("reference_frequency", reference_frequency),
    )

    for name, value in environment_parameters:
        size = _environment_size(
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
            size = _environment_size(
                value,
                name=name,
            )

            if size is not None:
                candidates.append((name, size))

        if isinstance(component, PlaneStandingWave):
            size = _environment_size(
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

            direction_size = _direction_environment_size(
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

    if not candidates:
        return 1

    n_envs = candidates[0][1]

    inconsistent = [(name, size) for name, size in candidates if size != n_envs]

    if inconsistent:
        details = ", ".join(f"{name}={size}" for name, size in candidates)

        raise ValueError(
            f"Acoustic-field parameters have inconsistent environment counts: {details}."
        )

    return n_envs


def _as_env_vector(
    value: Parameter,
    n_envs: int,
    *,
    name: str,
) -> FloatArray:
    """Canonicalize an environment parameter to shape (E,)."""

    arr = np.asarray(
        value,
        dtype=np.float64,
    )

    if arr.ndim == 0:
        return np.broadcast_to(
            arr.reshape(1),
            (n_envs,),
        )

    if arr.shape == (n_envs,):
        return arr

    raise ValueError(f"{name} must be a scalar or have shape ({n_envs},), got {arr.shape}.")


def _as_direction_matrix(
    value: FloatArray,
    n_envs: int,
    *,
    name: str,
) -> FloatArray:
    """Canonicalize a direction vector to shape (E, 3)."""

    arr = np.asarray(
        value,
        dtype=np.float64,
    )

    if arr.shape == (3,):
        return np.broadcast_to(
            arr[None, :],
            (n_envs, 3),
        )

    if arr.shape == (n_envs, 3):
        return arr

    raise ValueError(f"{name} must have shape (3,) or ({n_envs}, 3), got {arr.shape}.")


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

    density = _as_env_vector(
        material.density,
        n_envs,
        name="density",
    )

    sound_speed = _as_env_vector(
        material.sound_speed,
        n_envs,
        name="sound_speed",
    )

    f_ref = _as_env_vector(
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
        amplitude = _as_env_vector(
            component.pressure_amplitude,
            n_envs,
            name=f"components[{component_index}].pressure_amplitude",
        )

        frequency = _as_env_vector(
            component.frequency,
            n_envs,
            name=f"components[{component_index}].frequency",
        )

        phase = _as_env_vector(
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
            spatial_phase[:, component_index] = _as_env_vector(
                component.spatial_phase,
                n_envs,
                name=f"components[{component_index}].spatial_phase",
            )

            component_direction = _as_direction_matrix(
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
