from dataclasses import dataclass

import numpy as np

from ._batching import (
    as_bubble_matrix,
    as_env_parameter_column,
    as_position_tensor,
)
from .acoustic_field import (
    AcousticField,
    PreparedAcousticField,
    evaluate_acoustic_field,
    prepare_acoustic_field,
)
from .common import FloatArray, MaterialProperties, Parameter, ReferenceScales


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

    # Shared scalar or environment-scoped: (E, 1)
    static_pressure_coefficient: Parameter
    surface_tension_coefficient: Parameter
    viscous_coefficient: Parameter

    acoustic_pressure_scale: Parameter
    acoustic_pressure_derivative_scale: Parameter
    wall_mach_scale: Parameter

    gas_pressure_exponent: Parameter

    acoustic_field: PreparedAcousticField

    # Shared scalar or environment-scoped: (E, 1)
    reference_radius: Parameter
    reference_frequency: Parameter

    # Fixed geometry: (E, N, D)
    position: FloatArray

    # (E, N, N)
    radial_coupling: FloatArray


def _prepare_radial_coupling(
    position: FloatArray,
    reference_radius: Parameter,
    sound_speed: Parameter,
    frequency: Parameter,
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

    # scalar or (E, 1)
    reference_length = sound_speed / frequency

    # scalar or (E, 1)
    coupling_scale = reference_radius / reference_length

    coupling_scale = np.asarray(
        coupling_scale,
        dtype=np.float64,
    )

    if coupling_scale.ndim == 0:
        return coupling_scale * inverse_distance

    # (E, 1) -> (E, 1, 1)
    return coupling_scale[..., None] * inverse_distance


def prepare_coupled_bubbles_parameters(
    material: MaterialProperties,
    bubble: CoupledBubblesParameters,
    field: AcousticField,
    reference: ReferenceScales,
    n_envs: int,
) -> PreparedCoupledBubblesParameters:

    if n_envs < 1:
        raise ValueError("n_envs must be positive.")

    equilibrium_radius_input = np.asarray(
        bubble.equilibrium_radius,
        dtype=np.float64,
    )

    if equilibrium_radius_input.ndim not in (1, 2):
        raise ValueError("equilibrium_radius must have shape (n_bubbles,) or (n_envs, n_bubbles).")

    n_bubbles = equilibrium_radius_input.shape[-1]

    if n_bubbles == 0:
        raise ValueError("At least one bubble is required.")

    # ---------------------------------------------------------
    # Bubble-scoped parameters: (E, N)
    # ---------------------------------------------------------

    r0 = as_bubble_matrix(
        equilibrium_radius_input,
        n_envs,
        n_bubbles,
        name="equilibrium_radius",
    )

    position = as_position_tensor(
        bubble.position,
        n_envs,
        n_bubbles,
    )

    # ---------------------------------------------------------
    # Shared scalar OR environment-scoped (E, 1)
    # ---------------------------------------------------------

    rho = as_env_parameter_column(
        material.density,
        n_envs,
        name="density",
    )

    c = as_env_parameter_column(
        material.sound_speed,
        n_envs,
        name="sound_speed",
    )

    sigma = as_env_parameter_column(
        material.surface_tension,
        n_envs,
        name="surface_tension",
    )

    mu = as_env_parameter_column(
        material.dynamic_viscosity,
        n_envs,
        name="dynamic_viscosity",
    )

    p_vapor = as_env_parameter_column(
        material.vapor_pressure,
        n_envs,
        name="vapor_pressure",
    )

    p_ambient = as_env_parameter_column(
        material.ambient_pressure,
        n_envs,
        name="ambient_pressure",
    )

    gamma = as_env_parameter_column(
        material.polytropic_exponent,
        n_envs,
        name="polytropic_exponent",
    )

    r_ref = as_env_parameter_column(
        reference.radius,
        n_envs,
        name="reference_radius",
    )

    f_ref = as_env_parameter_column(
        reference.frequency,
        n_envs,
        name="reference_frequency",
    )

    if np.any(np.asarray(r_ref) <= 0.0):
        raise ValueError("reference radius must be positive.")

    if np.any(np.asarray(f_ref) <= 0.0):
        raise ValueError("reference frequency must be positive.")

    # ---------------------------------------------------------
    # Nondimensionalization
    # ---------------------------------------------------------

    time_scale = 1.0 / f_ref
    velocity_scale = r_ref * f_ref

    # ---------------------------------------------------------
    # Equilibrium gas pressure
    #
    # r0:             (E, N)
    # sigma etc.:     scalar or (E, 1)
    # result:         (E, N)
    # ---------------------------------------------------------

    gas_pressure_0 = 2.0 * sigma / r0 + p_ambient - p_vapor

    # ---------------------------------------------------------
    # Acoustic field
    #
    # Important:
    # pass the ORIGINAL scalar/(E,) reference frequency,
    # not the coupled-model scalar/(E,1) representation.
    # ---------------------------------------------------------

    acoustic_field = prepare_acoustic_field(
        material,
        field,
        reference_frequency=reference.frequency,
        n_envs=n_envs,
    )

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
        position=position,
        radial_coupling=_prepare_radial_coupling(
            position=position,
            reference_radius=r_ref,
            sound_speed=c,
            frequency=f_ref,
        ),
        acoustic_field=acoustic_field,
        reference_radius=r_ref,
        reference_frequency=f_ref,
    )


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

    field_state = evaluate_acoustic_field(t, params.position, params.acoustic_field)

    pressure = field_state.pressure
    pressure_time_derivative = field_state.pressure_time_derivative

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
