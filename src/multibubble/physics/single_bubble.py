from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
Parameter = float | FloatArray


@dataclass(frozen=True)
class MaterialProperties:
    density: Parameter = 998.0
    sound_speed: Parameter = 1500.0
    surface_tension: Parameter = 0.0725
    dynamic_viscosity: Parameter = 0.001
    vapor_pressure: Parameter = 0.0
    ambient_pressure: Parameter = 1.013e5
    polytropic_exponent: Parameter = 1.4


@dataclass(frozen=True)
class BubbleParameters:
    equilibrium_radius: Parameter


@dataclass(frozen=True)
class AcousticForcing:
    pressure_amplitude: Parameter
    frequency: Parameter
    phase: Parameter = 0.0


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

    forcing_amplitude: Parameter
    forcing_angular_frequency: Parameter
    forcing_phase: Parameter


def prepare_single_bubble_parameters(
    material: MaterialProperties,
    bubble: BubbleParameters,
    forcing: AcousticForcing,
    reference_radius: Parameter,
) -> PreparedSingleBubbleParameters:
    rho = np.asarray(material.density, dtype=np.float64)
    c = np.asarray(material.sound_speed, dtype=np.float64)
    sigma = np.asarray(material.surface_tension, dtype=np.float64)
    mu = np.asarray(material.dynamic_viscosity, dtype=np.float64)
    p_vapor = np.asarray(material.vapor_pressure, dtype=np.float64)
    p_ambient = np.asarray(material.ambient_pressure, dtype=np.float64)
    gamma = np.asarray(material.polytropic_exponent, dtype=np.float64)

    r0 = np.asarray(bubble.equilibrium_radius, dtype=np.float64)

    r_ref = np.asarray(reference_radius, dtype=np.float64)

    amplitude = np.asarray(forcing.pressure_amplitude, dtype=np.float64)

    frequency = np.asarray(forcing.frequency, dtype=np.float64)

    phase = np.asarray(forcing.phase, dtype=np.float64)

    omega = 2.0 * np.pi * frequency

    gas_pressure_0 = 2.0 * sigma / r0 + p_ambient - p_vapor

    time_scale = 2.0 * np.pi / omega
    velocity_scale = r_ref * omega / (2.0 * np.pi)

    return PreparedSingleBubbleParameters(
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
    )


def _homogeneous_pressure(
    t: FloatArray,
    params: PreparedSingleBubbleParameters,
) -> tuple[FloatArray, FloatArray]:
    """Return acoustic pressure and its physical-time derivative.

    ``t`` is the dimensionless acoustic-cycle time tau = f * t_phys.
    """
    theta = 2.0 * np.pi * t + params.forcing_phase

    pressure = params.forcing_amplitude * np.sin(theta)

    pressure_time_derivative = (
        params.forcing_amplitude * params.forcing_angular_frequency * np.cos(theta)
    )

    return pressure, pressure_time_derivative


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

    dydt[:, 0] = U
    dydt[:, 1] = numerator / denominator

    return dydt
