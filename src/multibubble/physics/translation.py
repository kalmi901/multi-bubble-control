import numpy as np

from .common import FloatArray


def translational_acceleration(
    R: FloatArray,
    U: FloatArray,
    V: FloatArray,
    pressure_gradient: FloatArray,
    fluid_velocity: FloatArray,
    params,
) -> FloatArray:
    """Return dimensionless 3D translational acceleration."""

    inv_R = 1.0 / R

    net_force_coefficient = np.asarray(params.net_force_coefficient, dtype=np.float64)[
        ..., np.newaxis
    ]

    bjerknes_force_scale = np.asarray(params.bjerknes_force_scale, dtype=np.float64)[
        ..., np.newaxis
    ]

    drag_force_scale = np.asarray(params.drag_force_scale, dtype=np.float64)[..., np.newaxis]

    velocity_scale = np.asarray(
        params.translational_velocity_scale,
        dtype=np.float64,
    )[..., np.newaxis]

    R_column = R[:, np.newaxis]
    U_column = U[:, np.newaxis]
    inv_R_column = inv_R[:, np.newaxis]

    # V is dimensionless; fluid_velocity is physical [m/s].
    relative_velocity = V * velocity_scale - fluid_velocity

    bjerknes_force = -bjerknes_force_scale * R_column**3 * pressure_gradient

    drag_force = -drag_force_scale * R_column * relative_velocity

    return (
        3.0 * net_force_coefficient * (bjerknes_force + drag_force) * inv_R_column**3
        - 3.0 * U_column * inv_R_column * V
    )


def translational_radial_correction(
    V: FloatArray,
) -> FloatArray:
    return 0.25 * np.sum(V * V, axis=-1)
