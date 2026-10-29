from dataclasses import dataclass

import numpy as np

from ._batching import as_env_parameter, as_env_vector
from .acoustic_field import (
    AcousticField,
    PreparedAcousticField,
    evaluate_acoustic_field,
    prepare_acoustic_field,
)
from .common import FloatArray, MaterialProperties, Parameter, ReferenceScales
from .keller_miksis import keller_miksis_terms


@dataclass(frozen=True)
class BubbleParameters:
    equilibrium_radius: Parameter


@dataclass(frozen=True)
class PreparedSingleBubbleParameters:
    # Bubble-scoped: (E, )
    equilibrium_radius_ratio: FloatArray
    gas_pressure_coefficient: FloatArray
    gas_pressure_velocity_coefficient: FloatArray

    # Shared scalar or environment-scoped: (E, )
    static_pressure_coefficient: Parameter
    surface_tension_coefficient: Parameter
    viscous_coefficient: Parameter

    acoustic_pressure_scale: Parameter
    acoustic_pressure_derivative_scale: Parameter
    wall_mach_scale: Parameter

    gas_pressure_exponent: Parameter

    acoustic_field: PreparedAcousticField

    # Shared scalar or environment-scoped: (E, )
    reference_radius: Parameter
    reference_frequency: Parameter


def prepare_single_bubble_parameters(
    material: MaterialProperties,
    bubble: BubbleParameters,
    field: AcousticField,
    reference: ReferenceScales,
    n_envs: int,
) -> PreparedSingleBubbleParameters:

    if n_envs < 1:
        raise ValueError("n_envs must be positive.")

    equilibrium_radius_input = np.asarray(
        bubble.equilibrium_radius,
        dtype=np.float64,
    )

    if equilibrium_radius_input.ndim not in (0, 1):
        raise ValueError("equilibrium_radius must be a scalar or have shape (n_envs,).")

    # ---------------------------------------------------------
    # Bubble-scoped parameters: (E,)
    # ---------------------------------------------------------
    r0 = as_env_vector(
        equilibrium_radius_input,
        n_envs,
        name="equilibrium_radius",
    )

    # ---------------------------------------------------------
    # Shared scalar OR environment-scoped (E,)
    # ---------------------------------------------------------

    rho = as_env_parameter(
        material.density,
        n_envs,
        name="density",
    )

    c = as_env_parameter(
        material.sound_speed,
        n_envs,
        name="sound_speed",
    )

    sigma = as_env_parameter(
        material.surface_tension,
        n_envs,
        name="surface_tension",
    )

    mu = as_env_parameter(
        material.dynamic_viscosity,
        n_envs,
        name="dynamic_viscosity",
    )

    p_vapor = as_env_parameter(
        material.vapor_pressure,
        n_envs,
        name="vapor_pressure",
    )

    p_ambient = as_env_parameter(
        material.ambient_pressure,
        n_envs,
        name="ambient_pressure",
    )

    gamma = as_env_parameter(
        material.polytropic_exponent,
        n_envs,
        name="polytropic_exponent",
    )

    r_ref = as_env_parameter(
        reference.radius,
        n_envs,
        name="reference_radius",
    )

    f_ref = as_env_parameter(
        reference.frequency,
        n_envs,
        name="reference_frequency",
    )

    if np.any(r_ref <= 0.0):
        raise ValueError("reference radius must be positive.")

    if np.any(f_ref <= 0.0):
        raise ValueError("reference frequency must be positive.")

    # ---------------------------------------------------------
    # Reference scales
    #
    # tau = f_ref * t_phys
    #
    # R* = R / R_ref
    #
    # U* = Rdot / (R_ref * f_ref)
    # ---------------------------------------------------------

    time_scale = 1.0 / f_ref

    velocity_scale = r_ref * f_ref

    # ---------------------------------------------------------
    # Physical equilibrium gas pressure
    # ---------------------------------------------------------

    gas_pressure_0 = 2.0 * sigma / r0 + p_ambient - p_vapor

    # ---------------------------------------------------------
    # Prepare prescribed acoustic field using the SAME
    # reference-frequency scale as the bubble dynamics.
    # ---------------------------------------------------------

    acoustic_field = prepare_acoustic_field(material, field, reference, n_envs=n_envs)

    return PreparedSingleBubbleParameters(
        equilibrium_radius_ratio=(r0 / r_ref),
        gas_pressure_coefficient=(gas_pressure_0 * (time_scale / r_ref) ** 2 / rho),
        gas_pressure_velocity_coefficient=(
            (1.0 - 3.0 * gamma) * gas_pressure_0 * time_scale / (r_ref * c * rho)
        ),
        static_pressure_coefficient=((p_ambient - p_vapor) * (time_scale / r_ref) ** 2 / rho),
        surface_tension_coefficient=(2.0 * sigma / (r_ref * rho) * (time_scale / r_ref) ** 2),
        viscous_coefficient=(4.0 * mu / (rho * r_ref**2) * time_scale),
        acoustic_pressure_scale=((time_scale / r_ref) ** 2 / rho),
        acoustic_pressure_derivative_scale=(time_scale**2 / (c * rho * r_ref)),
        wall_mach_scale=(velocity_scale / c),
        gas_pressure_exponent=(3.0 * gamma),
        acoustic_field=acoustic_field,
        reference_radius=r_ref,
        reference_frequency=f_ref,
    )


def single_bubble_rhs(
    t: FloatArray,
    y: FloatArray,
    params: PreparedSingleBubbleParameters,
) -> FloatArray:
    """Uncoupled Keller--Miksis radial dynamics.

    State
    -----
    y[:, 0] : dimensionless bubble radius R / R_ref
    y[:, 1] : dimensionless wall velocity

    Time is dimensionless and measured in acoustic cycles.
    """

    R = y[:, 0]
    U = y[:, 1]

    field_state = evaluate_acoustic_field(
        t,
        np.zeros((R.shape[0], 3)),
        params.acoustic_field,
    )

    pressure = field_state.pressure
    pressure_time_derivative = field_state.pressure_time_derivative

    numerator, denominator = keller_miksis_terms(R, U, pressure, pressure_time_derivative, params)

    dydt = np.empty_like(y)

    dydt[:, 0] = U
    dydt[:, 1] = numerator / denominator

    return dydt
