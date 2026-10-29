from .common import FloatArray


def keller_miksis_terms(
    R: FloatArray, U: FloatArray, pressure: FloatArray, pressure_time_derivative: FloatArray, params
) -> tuple[FloatArray, FloatArray]:

    inv_R = 1.0 / R

    # Polytropic gas-pressure contribution.
    gas_term = (params.equilibrium_radius_ratio * inv_R) ** params.gas_pressure_exponent

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

    return numerator, denominator
