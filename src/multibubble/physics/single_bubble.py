from dataclasses import dataclass

import numpy as np

from .acoustic_field import (
    AcousticField,
    PreparedAcousticField,
    evaluate_acoustic_field,
    prepare_acoustic_field,
)
from .common import FloatArray, MaterialProperties, Parameter, ReferenceScales


@dataclass(frozen=True)
class BubbleParameters:
    equilibrium_radius: Parameter


@dataclass(frozen=True)
class PreparedSingleBubbleParameters:
    equilibrium_radius_ratio: Parameter

    gas_pressure_coefficient: Parameter
    gas_pressure_velocity_coefficient: Parameter
    static_pressure_coefficient: Parameter
    surface_tension_coefficient: Parameter
    viscous_coefficient: Parameter

    acoustic_pressure_scale: Parameter
    acoustic_pressure_derivative_scale: Parameter
    wall_mach_scale: Parameter

    gas_pressure_exponent: Parameter

    acoustic_field: PreparedAcousticField

    reference_radius: FloatArray
    reference_frequency: FloatArray


def prepare_single_bubble_parameters(
    material: MaterialProperties,
    bubble: BubbleParameters,
    field: AcousticField,
    reference: ReferenceScales,
) -> PreparedSingleBubbleParameters:

    rho = np.asarray(material.density, dtype=np.float64)

    c = np.asarray(material.sound_speed, dtype=np.float64)

    sigma = np.asarray(material.surface_tension, dtype=np.float64)

    mu = np.asarray(material.dynamic_viscosity, dtype=np.float64)

    p_vapor = np.asarray(material.vapor_pressure, dtype=np.float64)

    p_ambient = np.asarray(material.ambient_pressure, dtype=np.float64)

    gamma = np.asarray(material.polytropic_exponent, dtype=np.float64)

    r0 = np.asarray(bubble.equilibrium_radius, dtype=np.float64)

    r_ref = np.asarray(reference.radius, dtype=np.float64)

    f_ref = np.asarray(reference.frequency, dtype=np.float64)

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

    acoustic_field = prepare_acoustic_field(
        material,
        field,
        reference_frequency=f_ref,
    )

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

    inv_R = 1.0 / R

    # Polytropic gas-pressure contribution.
    gas_term = (params.equilibrium_radius_ratio * inv_R) ** params.gas_pressure_exponent

    field_state = evaluate_acoustic_field(
        t,
        np.zeros((R.shape[0], 3)),
        params.acoustic_field,
    )

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

    dydt[:, 0] = U
    dydt[:, 1] = numerator / denominator

    return dydt
