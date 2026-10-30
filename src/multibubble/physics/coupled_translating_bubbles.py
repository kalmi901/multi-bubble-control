from dataclasses import dataclass

import numpy as np

from ._batching import as_bubble_matrix, as_env_parameter_column
from .acoustic_field import (
    AcousticField,
    PreparedAcousticField,
    evaluate_acoustic_field,
    prepare_acoustic_field,
)
from .common import FloatArray, MaterialProperties, Parameter, ReferenceScales
from .keller_miksis import keller_miksis_terms
from .translation import translational_acceleration, translational_radial_correction


@dataclass(frozen=True)
class CoupledTranslatingBubbleParameters:
    equilibrium_radius: FloatArray


@dataclass(frozen=True)
class PreparedCoupledTranslatingBubbleParameters:
    # Bubble-scoped (E, N)
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

    net_force_coefficient: Parameter
    bjerknes_force_scale: Parameter
    drag_force_scale: Parameter
    translational_velocity_scale: Parameter
    translational_coupling_viscous_scale: Parameter

    acoustic_field: PreparedAcousticField

    # Shared scalar or environment-scoped: (E, 1)
    reference_radius: Parameter
    reference_frequency: Parameter


def prepare_coupled_translating_bubble_parameters(
    material: MaterialProperties,
    bubble: CoupledTranslatingBubbleParameters,
    field: AcousticField,
    reference: ReferenceScales,
    n_envs: int,
) -> PreparedCoupledTranslatingBubbleParameters:

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

    acoustic_field = prepare_acoustic_field(
        material,
        field,
        reference,
        n_envs=n_envs,
    )

    return PreparedCoupledTranslatingBubbleParameters(
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
        net_force_coefficient=1.0 / (2.0 * np.pi * rho * r_ref**4 * f_ref**2),
        bjerknes_force_scale=4.0 * np.pi / 3.0 * r_ref**3,
        drag_force_scale=12.0 * np.pi * mu * r_ref,
        translational_velocity_scale=velocity_scale,
        translational_coupling_viscous_scale=18.0 * mu / (rho * f_ref * r_ref**2),
        acoustic_field=acoustic_field,
        reference_radius=r_ref,
        reference_frequency=f_ref,
    )


def _coupling_geometry(X: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Return 1/d_ij and q_ij = (X_i - X_j) / d_ij^3.

    Parameters
    ----------
    X : (E, N, 3)
        Dimensionless bubble positions scaled by R_ref.

    Returns
    -------
    inverse_distance : (E, N, N)
    q                : (E, N, N, 3)
    """

    delta = X[:, :, None, :] - X[:, None, :, :]

    distance = np.linalg.norm(
        delta,
        axis=-1,
    )

    n_bubbles = X.shape[1]

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

    # q_ij = e_ij / d_ij^2 = (X_i - X_j) / d_ij^3
    q = delta * inverse_distance[..., np.newaxis] ** 3

    return inverse_distance, q


def _explicit_coupling(
    R: FloatArray,
    U: FloatArray,
    V: FloatArray,
    inverse_distance: FloatArray,
    q: FloatArray,
    translational_coupling_viscous_scale: Parameter,
) -> tuple[FloatArray, FloatArray]:
    """Return explicit radial and translational coupling terms."""

    inverse_distance_cubed = inverse_distance**3

    RU2 = R * U**2
    R2U = R**2 * U
    R3 = R**3
    V2 = np.sum(V * V, axis=-1)

    # =========================================================
    # Radial equation
    # =========================================================

    # Radial + radial -> radial
    radial_from_radial_motion = np.einsum(
        "eij,ej->ei",
        inverse_distance,
        2.0 * RU2,
    )

    # Translation + radial -> radial
    q_R2U = np.einsum(
        "eijk,ej->eik",
        q,
        R2U,
    )

    radial_from_mixed_motion = 0.5 * (
        np.einsum(
            "eik,eik->ei",
            V,
            q_R2U,
        )
        + 5.0
        * np.einsum(
            "eijk,ej,ejk->ei",
            q,
            R2U,
            V,
        )
    )

    # Translation + translation -> radial
    source_velocity = np.einsum(
        "eij,ej,ejk->eik",
        inverse_distance_cubed,
        R3,
        V,
    )

    radial_from_translational_motion = 0.5 * (
        np.einsum(
            "eik,eik->ei",
            V,
            source_velocity,
        )
        + 2.0
        * np.einsum(
            "eij,ej,ej->ei",
            inverse_distance_cubed,
            R3,
            V2,
        )
    )

    explicit_radial_coupling = (
        radial_from_radial_motion + radial_from_mixed_motion + radial_from_translational_motion
    )

    # =========================================================
    # Translational equation
    # =========================================================

    coupling_viscous_scale = np.asarray(
        translational_coupling_viscous_scale,
        dtype=np.float64,
    )

    g = 3.0 * U / R + coupling_viscous_scale / R**2

    # Directional d^-2 contribution
    translation_from_radial_motion = 6.0 * np.einsum(
        "eijk,ej->eik",
        q,
        RU2,
    ) + g[..., np.newaxis] * np.einsum(
        "eijk,ej->eik",
        q,
        R2U,
    )

    # Source translational velocity d^-3 contribution
    translation_from_translational_motion = 15.0 * np.einsum(
        "eij,ej,ejk->eik",
        inverse_distance_cubed,
        R2U,
        V,
    ) + g[..., np.newaxis] * np.einsum(
        "eij,ej,ejk->eik",
        inverse_distance_cubed,
        R3,
        V,
    )

    explicit_translation_coupling = (
        translation_from_radial_motion + translation_from_translational_motion
    )

    return explicit_radial_coupling, explicit_translation_coupling


def _assemble_coupling_system(
    R: FloatArray,
    U: FloatArray,
    X: FloatArray,
    V: FloatArray,
    numerator: FloatArray,
    denominator: FloatArray,
    translation_acceleration: FloatArray,
    params: PreparedCoupledTranslatingBubbleParameters,
) -> tuple[FloatArray, FloatArray]:
    """Assemble the dense coupled radial-translational acceleration system.

    Returns
    -------
    matrix : (E, 4N, 4N)
    rhs    : (E, 4N)
    """

    n_envs, n_bubbles = R.shape

    inverse_distance, q = _coupling_geometry(X)

    inverse_distance_cubed = inverse_distance**3
    inverse_denominator = 1.0 / denominator

    radial_explicit, translational_explicit = _explicit_coupling(
        R,
        U,
        V,
        inverse_distance,
        q,
        params.translational_coupling_viscous_scale,
    )

    # ---------------------------------------------------------
    # Matrix blocks
    # ---------------------------------------------------------

    identity = np.eye(
        n_bubbles,
        dtype=R.dtype,
    )[None, :, :]

    # R <- R
    radial_from_radial_acceleration = (
        np.einsum(
            "ei,eij,ej->eij",
            inverse_denominator,
            inverse_distance,
            R**2,
        )
        + identity
    )

    # R <- translation
    #
    # Shape: (E, N_receiver, N_source, 3)
    radial_from_translation_acceleration = 0.5 * np.einsum(
        "ei,eijk,ej->eijk",
        inverse_denominator,
        q,
        R**3,
    )

    # translation <- R
    translation_from_radial_acceleration = -3.0 * q * R[:, None, :, None] ** 2

    # translation_k <- translation_k
    translation_from_translation_acceleration = (
        -3.0 * inverse_distance_cubed * R[:, None, :] ** 3 + identity
    )

    # ---------------------------------------------------------
    # Dense 4N x 4N matrix
    # ---------------------------------------------------------

    size = 4 * n_bubbles

    matrix = np.zeros(
        (n_envs, size, size),
        dtype=R.dtype,
    )

    radial_slice = slice(
        0,
        n_bubbles,
    )

    translation_slices = (
        slice(n_bubbles, 2 * n_bubbles),
        slice(2 * n_bubbles, 3 * n_bubbles),
        slice(3 * n_bubbles, 4 * n_bubbles),
    )

    matrix[
        :,
        radial_slice,
        radial_slice,
    ] = radial_from_radial_acceleration

    for component, component_slice in enumerate(translation_slices):
        matrix[
            :,
            radial_slice,
            component_slice,
        ] = radial_from_translation_acceleration[..., component]

        matrix[
            :,
            component_slice,
            radial_slice,
        ] = translation_from_radial_acceleration[..., component]

        matrix[
            :,
            component_slice,
            component_slice,
        ] = translation_from_translation_acceleration

    # ---------------------------------------------------------
    # RHS
    # ---------------------------------------------------------

    rhs = np.empty(
        (n_envs, size),
        dtype=R.dtype,
    )

    rhs[:, radial_slice] = inverse_denominator * (numerator - radial_explicit)

    total_translation_rhs = translation_acceleration + translational_explicit

    for component, component_slice in enumerate(translation_slices):
        rhs[:, component_slice] = total_translation_rhs[..., component]

    return matrix, rhs


def coupled_translating_bubble_rhs(
    t: FloatArray, y: FloatArray, params: PreparedCoupledTranslatingBubbleParameters
) -> FloatArray:
    """Coupled Keller--Miksis radial dynamics with 3D translational motion.

    State
    -----
    y[..., 0]   : dimensionless bubble radius R / R_ref
    y[..., 1:4] : dimensionless position (x, y, z) / R_ref
    y[..., 4]   : dimensionless wall velocity Rdot / (R_ref * f_ref)
    y[..., 5:8] : dimensionless translational velocity / (R_ref * f_ref)

    Time is dimensionless: tau = f_ref * t_phys.
    """

    if y.ndim != 3 or y.shape[-1] != 8:
        raise ValueError(f"State must have shape (n_envs, n_bubbles, 8), got {y.shape}.")

    R = y[..., 0]
    X = y[..., 1:4]
    U = y[..., 4]
    V = y[..., 5:8]

    field_state = evaluate_acoustic_field(t, X, params.acoustic_field)

    pressure = field_state.pressure
    pressure_time_derivative = field_state.pressure_time_derivative
    pressure_gradient = field_state.pressure_gradient
    fluid_velocity = field_state.fluid_velocity

    numerator, denominator = keller_miksis_terms(R, U, pressure, pressure_time_derivative, params)

    numerator += translational_radial_correction(V)

    translation_acceleration = translational_acceleration(
        R,
        U,
        V,
        pressure_gradient,
        fluid_velocity,
        params,
    )

    dydt = np.zeros_like(y)

    matrix, rhs = _assemble_coupling_system(
        R,
        U,
        X,
        V,
        numerator,
        denominator,
        translation_acceleration,
        params,
    )

    acceleration = np.linalg.solve(
        matrix,
        rhs[..., np.newaxis],
    ).squeeze(-1)

    n_bubbles = R.shape[1]

    dydt = np.empty_like(y)

    dydt[..., 0] = U
    dydt[..., 1:4] = V

    dydt[..., 4] = acceleration[
        :,
        :n_bubbles,
    ]

    dydt[..., 5] = acceleration[
        :,
        n_bubbles : 2 * n_bubbles,
    ]

    dydt[..., 6] = acceleration[
        :,
        2 * n_bubbles : 3 * n_bubbles,
    ]

    dydt[..., 7] = acceleration[
        :,
        3 * n_bubbles : 4 * n_bubbles,
    ]

    return dydt
