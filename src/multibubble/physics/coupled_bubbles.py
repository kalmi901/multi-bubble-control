from dataclasses import dataclass

import numpy as np

from .common import AcousticForcing, FloatArray, MaterialProperties, Parameter


@dataclass(frozen=True)
class CoupledBubblesParameters:
    equilibrium_radius: FloatArray
    position: FloatArray


@dataclass(frozen=True)
class PreparedCoupledBubblesParameters:
    # Bubble-scoped: (E, N)
    equilibrium_radius_ratio: FloatArray

    gas_pressure_coefficient: FloatArray
    gas_pressure_velocity_coefficient: FloatArray

    # Environment-scoped: (E, 1)
    static_pressure_coefficient: FloatArray
    surface_tension_coefficient: FloatArray
    viscous_coefficient: FloatArray

    acoustic_pressure_scale: FloatArray
    acoustic_pressure_derivative_scale: FloatArray
    wall_mach_scale: FloatArray

    gas_pressure_exponent: FloatArray

    forcing_amplitude: FloatArray
    forcing_angular_frequency: FloatArray
    forcing_phase: FloatArray

    # Fixed geometry: (E, N, D)
    position: FloatArray

    radial_coupling: FloatArray  # (E, N, N)


def _environment_size(
    value: Parameter,
    *,
    name: str,
) -> int | None:
    """Return environment count for an environment-scoped input.

    Scalars are shared by all environments and therefore do not
    determine the environment count.
    """

    arr = np.asarray(value, dtype=np.float64)

    if arr.ndim == 0:
        return None

    if arr.ndim == 1:
        return arr.shape[0]

    raise ValueError(f"{name} must be a scalar or have shape (n_envs,), got {arr.shape}.")


def _infer_num_envs(
    material: MaterialProperties,
    bubble: CoupledBubblesParameters,
    forcing: AcousticForcing,
    reference_radius: Parameter,
) -> int:
    """Infer and validate the number of independent environments."""

    candidates: list[tuple[str, int]] = []

    equilibrium_radius = np.asarray(
        bubble.equilibrium_radius,
        dtype=np.float64,
    )

    if equilibrium_radius.ndim == 2:
        candidates.append(("equilibrium_radius", equilibrium_radius.shape[0]))
    elif equilibrium_radius.ndim != 1:
        raise ValueError(
            "equilibrium_radius must have shape (n_bubbles,) "
            "or (n_envs, n_bubbles), "
            f"got {equilibrium_radius.shape}."
        )

    position = np.asarray(
        bubble.position,
        dtype=np.float64,
    )

    if position.ndim == 3:
        candidates.append(("position", position.shape[0]))
    elif position.ndim != 2:
        raise ValueError(
            "position must have shape (n_bubbles, spatial_dim) "
            "or (n_envs, n_bubbles, spatial_dim), "
            f"got {position.shape}."
        )

    environment_parameters = (
        ("density", material.density),
        ("sound_speed", material.sound_speed),
        ("surface_tension", material.surface_tension),
        ("dynamic_viscosity", material.dynamic_viscosity),
        ("vapor_pressure", material.vapor_pressure),
        ("ambient_pressure", material.ambient_pressure),
        ("polytropic_exponent", material.polytropic_exponent),
        ("pressure_amplitude", forcing.pressure_amplitude),
        ("frequency", forcing.frequency),
        ("phase", forcing.phase),
        ("reference_radius", reference_radius),
    )

    for name, value in environment_parameters:
        size = _environment_size(
            value,
            name=name,
        )

        if size is not None:
            candidates.append((name, size))

    if not candidates:
        return 1

    n_envs = candidates[0][1]

    inconsistent = [(name, size) for name, size in candidates if size != n_envs]

    if inconsistent:
        details = ", ".join(f"{name}={size}" for name, size in candidates)

        raise ValueError(
            f"Environment-specific parameters have inconsistent environment counts: {details}."
        )

    return n_envs


def _as_env_column(
    value: Parameter,
    n_envs: int,
    *,
    name: str,
) -> FloatArray:
    """Canonicalize an environment parameter to shape (E, 1)."""

    arr = np.asarray(
        value,
        dtype=np.float64,
    )

    if arr.ndim == 0:
        return np.broadcast_to(
            arr.reshape(1, 1),
            (n_envs, 1),
        )

    if arr.shape == (n_envs,):
        return arr[:, None]

    raise ValueError(f"{name} must be a scalar or have shape ({n_envs},), got {arr.shape}.")


def _as_bubble_matrix(
    value: FloatArray,
    n_envs: int,
    n_bubbles: int,
    *,
    name: str,
) -> FloatArray:
    """Canonicalize a bubble parameter to shape (E, N)."""

    arr = np.asarray(
        value,
        dtype=np.float64,
    )

    if arr.shape == (n_bubbles,):
        return np.broadcast_to(
            arr[None, :],
            (n_envs, n_bubbles),
        )

    if arr.shape == (n_envs, n_bubbles):
        return arr

    raise ValueError(
        f"{name} must have shape ({n_bubbles},) or ({n_envs}, {n_bubbles}), got {arr.shape}."
    )


def _as_position_tensor(
    value: FloatArray,
    n_envs: int,
    n_bubbles: int,
) -> FloatArray:
    """Canonicalize fixed bubble positions to shape (E, N, D)."""

    arr = np.asarray(
        value,
        dtype=np.float64,
    )

    if arr.ndim == 2:
        if arr.shape[0] != n_bubbles:
            raise ValueError(
                "Shared position array must have shape "
                f"({n_bubbles}, spatial_dim), got {arr.shape}."
            )

        return np.broadcast_to(
            arr[None, ...],
            (n_envs, *arr.shape),
        )

    if arr.ndim == 3:
        if arr.shape[0] != n_envs:
            raise ValueError(
                "Environment-specific position array must have "
                f"{n_envs} environments, got {arr.shape[0]}."
            )

        if arr.shape[1] != n_bubbles:
            raise ValueError(
                "Environment-specific position array must have "
                f"{n_bubbles} bubbles, got {arr.shape[1]}."
            )

        return arr

    raise ValueError(
        "position must have shape (n_bubbles, spatial_dim) "
        "or (n_envs, n_bubbles, spatial_dim), "
        f"got {arr.shape}."
    )


def _prepare_radial_coupling(
    position: FloatArray,
    reference_radius: FloatArray,
    sound_speed: FloatArray,
    frequency: FloatArray,
) -> FloatArray:
    """Return K_ij = R_ref / d_ij for fixed dimensionless geometry."""

    delta = position[:, :, None, :] - position[:, None, :, :]  # (E, N, N, D)

    distance = np.linalg.norm(
        delta,
        axis=-1,
    )  # (E, N, N)

    n_bubbles = distance.shape[1]

    diagonal = np.eye(
        n_bubbles,
        dtype=bool,
    )[None, :, :]

    if np.any((distance == 0.0) & ~diagonal):
        raise ValueError("Distinct bubbles must not occupy the same position.")

    inverse_distance = np.zeros_like(distance)

    np.divide(
        1.0,
        distance,
        out=inverse_distance,
        where=~diagonal,
    )

    reference_length = sound_speed / frequency  # (E, 1)

    coupling_scale = (reference_radius / reference_length)[:, :, None]  # (E, 1, 1)

    return coupling_scale * inverse_distance


def prepare_coupled_bubbles_parameters(
    material: MaterialProperties,
    bubble: CoupledBubblesParameters,
    forcing: AcousticForcing,
    reference_radius: Parameter,
) -> PreparedCoupledBubblesParameters:

    equilibrium_radius_input = np.asarray(
        bubble.equilibrium_radius,
        dtype=np.float64,
    )

    if equilibrium_radius_input.ndim not in (1, 2):
        raise ValueError("equilibrium_radius must have shape (n_bubbles,) or (n_envs, n_bubbles).")

    n_bubbles = equilibrium_radius_input.shape[-1]

    if n_bubbles == 0:
        raise ValueError("At least one bubble is required.")

    n_envs = _infer_num_envs(
        material,
        bubble,
        forcing,
        reference_radius,
    )

    # ---------------------------------------------------------
    # Canonical bubble-scoped parameters: (E, N)
    # ---------------------------------------------------------

    r0 = _as_bubble_matrix(
        equilibrium_radius_input,
        n_envs,
        n_bubbles,
        name="equilibrium_radius",
    )

    position = _as_position_tensor(
        bubble.position,
        n_envs,
        n_bubbles,
    )

    # ---------------------------------------------------------
    # Canonical environment-scoped parameters: (E, 1)
    # ---------------------------------------------------------

    rho = _as_env_column(
        material.density,
        n_envs,
        name="density",
    )

    c = _as_env_column(
        material.sound_speed,
        n_envs,
        name="sound_speed",
    )

    sigma = _as_env_column(
        material.surface_tension,
        n_envs,
        name="surface_tension",
    )

    mu = _as_env_column(
        material.dynamic_viscosity,
        n_envs,
        name="dynamic_viscosity",
    )

    p_vapor = _as_env_column(
        material.vapor_pressure,
        n_envs,
        name="vapor_pressure",
    )

    p_ambient = _as_env_column(
        material.ambient_pressure,
        n_envs,
        name="ambient_pressure",
    )

    gamma = _as_env_column(
        material.polytropic_exponent,
        n_envs,
        name="polytropic_exponent",
    )

    r_ref = _as_env_column(
        reference_radius,
        n_envs,
        name="reference_radius",
    )

    amplitude = _as_env_column(
        forcing.pressure_amplitude,
        n_envs,
        name="pressure_amplitude",
    )

    frequency = _as_env_column(
        forcing.frequency,
        n_envs,
        name="frequency",
    )

    phase = _as_env_column(
        forcing.phase,
        n_envs,
        name="phase",
    )

    # ---------------------------------------------------------
    # Nondimensionalization
    # ---------------------------------------------------------

    omega = 2.0 * np.pi * frequency

    gas_pressure_0 = 2.0 * sigma / r0 + p_ambient - p_vapor

    time_scale = 2.0 * np.pi / omega

    velocity_scale = r_ref * omega / (2.0 * np.pi)

    return PreparedCoupledBubblesParameters(
        equilibrium_radius_ratio=r0 / r_ref,
        gas_pressure_coefficient=(gas_pressure_0 * (time_scale / r_ref) ** 2 / rho),
        gas_pressure_velocity_coefficient=(
            (1.0 - 3.0 * gamma) * gas_pressure_0 * time_scale / (r_ref * c * rho)
        ),
        static_pressure_coefficient=((p_ambient - p_vapor) * (time_scale / r_ref) ** 2 / rho),
        surface_tension_coefficient=(2.0 * sigma / (r_ref * rho) * (time_scale / r_ref) ** 2),
        viscous_coefficient=(4.0 * mu / (rho * r_ref**2) * time_scale),
        acoustic_pressure_scale=((time_scale / r_ref) ** 2 / rho),
        acoustic_pressure_derivative_scale=(time_scale**2 / (c * rho * r_ref)),
        wall_mach_scale=velocity_scale / c,
        gas_pressure_exponent=3.0 * gamma,
        forcing_amplitude=amplitude,
        forcing_angular_frequency=omega,
        forcing_phase=phase,
        position=position,
        radial_coupling=_prepare_radial_coupling(
            position=position, reference_radius=r_ref, sound_speed=c, frequency=frequency
        ),
    )


def _homogeneous_pressure(
    t: FloatArray,
    params: PreparedCoupledBubblesParameters,
) -> tuple[FloatArray, FloatArray]:
    """Return acoustic pressure and its physical-time derivative.

    ``t`` is the dimensionless acoustic-cycle time tau = f * t_phys.
    """

    tau = np.asarray(
        t,
        dtype=np.float64,
    )

    n_envs = params.forcing_phase.shape[0]

    if tau.ndim == 0:
        tau = np.broadcast_to(
            tau.reshape(1, 1),
            (n_envs, 1),
        )

    elif tau.shape == (n_envs,):
        tau = tau[:, None]

    else:
        raise ValueError(f"t must be a scalar or have shape ({n_envs},), got {tau.shape}.")

    theta = 2.0 * np.pi * tau + params.forcing_phase

    pressure = params.forcing_amplitude * np.sin(theta)

    pressure_time_derivative = (
        params.forcing_amplitude * params.forcing_angular_frequency * np.cos(theta)
    )

    return pressure, pressure_time_derivative


def _assemble_radial_coupling_system(
    R: FloatArray,
    U: FloatArray,
    numerator: FloatArray,
    denominator: FloatArray,
    radial_coupling: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """Assemble A @ acceleration = rhs for fixed radial coupling.

    Shapes
    ------
    R, U, numerator, denominator : (E, N)
    radial_coupling              : (E, N, N)

    Returns
    -------
    matrix : (E, N, N)
    rhs    : (E, N)
    """

    inverse_denominator = 1.0 / denominator

    # A_eij = delta_ij + (1 / D_ei) K_eij R_ej^2
    coupling_matrix = np.einsum(
        "ei,eij,ej->eij",
        inverse_denominator,
        radial_coupling,
        R**2,
    )

    matrix = coupling_matrix + np.eye(R.shape[1], dtype=R.dtype)[None, :, :]

    # C_ei = sum_j K_eij * 2 R_ej U_ej^2
    explicit_coupling = np.einsum(
        "eij,ej->ei",
        radial_coupling,
        2.0 * R * U**2,
    )

    rhs = inverse_denominator * (numerator - explicit_coupling)

    return matrix, rhs


def coupled_bubbles_rhs(
    t: FloatArray,
    y: FloatArray,
    params: PreparedCoupledBubblesParameters,
) -> FloatArray:
    """Fixed-position coupled Keller--Miksis radial dynamics.

    State
    -----
    y[..., 0] : dimensionless bubble radius R / R_ref
    y[..., 1] : dimensionless wall velocity

    Expected state shape is ``(n_envs, n_bubbles, 2)``.

    Time is dimensionless and measured in acoustic cycles.
    """

    if y.ndim != 3 or y.shape[-1] != 2:
        raise ValueError(f"State must have shape (n_envs, n_bubbles, 2), got {y.shape}.")

    R = y[..., 0]
    U = y[..., 1]

    inv_R = 1.0 / R

    gas_term = (params.equilibrium_radius_ratio * inv_R) ** params.gas_pressure_exponent

    pressure, pressure_time_derivative = _homogeneous_pressure(
        t,
        params,
    )

    numerator = (
        (params.gas_pressure_coefficient + params.gas_pressure_velocity_coefficient * U) * gas_term
        - params.static_pressure_coefficient * (1.0 + params.wall_mach_scale * U)
        - params.surface_tension_coefficient * inv_R
        - params.viscous_coefficient * U * inv_R
        - 1.5 * (1.0 - params.wall_mach_scale * U / 3.0) * U**2
        - (1.0 + params.wall_mach_scale * U) * params.acoustic_pressure_scale * pressure
        - params.acoustic_pressure_derivative_scale * pressure_time_derivative * R
    )

    denominator = (
        R - params.wall_mach_scale * R * U + params.viscous_coefficient * params.wall_mach_scale
    )

    dydt = np.empty_like(y)

    matrix, rhs = _assemble_radial_coupling_system(
        R, U, numerator, denominator, params.radial_coupling
    )

    acceleration = np.linalg.solve(
        matrix,
        rhs[..., np.newaxis],
    ).squeeze(-1)

    dydt[..., 0] = U
    dydt[..., 1] = acceleration

    return dydt
