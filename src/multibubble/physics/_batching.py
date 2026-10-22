import numpy as np

from .common import FloatArray, Parameter


def as_env_vector(
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


def as_env_parameter_column(
    value: Parameter,
    n_envs: int,
    *,
    name: str,
) -> Parameter:
    arr = np.asarray(value, dtype=np.float64)

    if arr.ndim == 0:
        return arr

    if arr.shape == (n_envs,):
        return arr[:, None]

    raise ValueError(f"{name} must be a scalar or have shape ({n_envs},), got {arr.shape}.")


def as_env_parameter(
    value: Parameter,
    n_envs: int,
    *,
    name: str,
) -> Parameter:
    arr = np.asarray(value, dtype=np.float64)

    if arr.ndim == 0:
        return arr

    if arr.shape == (n_envs,):
        return arr

    raise ValueError(f"{name} must be a scalar or have shape ({n_envs},), got {arr.shape}.")


def as_bubble_matrix(
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


def as_position_tensor(
    value: FloatArray,
    n_envs: int,
    n_bubbles: int,
    *,
    name: str = "position",
) -> FloatArray:
    """Canonicalize positions to shape (E, N, D)."""

    arr = np.asarray(
        value,
        dtype=np.float64,
    )

    if arr.ndim == 2:
        if arr.shape[0] != n_bubbles:
            raise ValueError(
                f"Shared {name} array must have shape ({n_bubbles}, spatial_dim), got {arr.shape}."
            )

        return np.broadcast_to(
            arr[None, ...],
            (n_envs, *arr.shape),
        )

    if arr.ndim == 3:
        if arr.shape[0] != n_envs:
            raise ValueError(
                f"Environment-specific {name} array must have "
                f"{n_envs} environments, got {arr.shape[0]}."
            )

        if arr.shape[1] != n_bubbles:
            raise ValueError(
                f"Environment-specific {name} array must have "
                f"{n_bubbles} bubbles, got {arr.shape[1]}."
            )

        return arr

    raise ValueError(
        f"{name} must have shape (n_bubbles, spatial_dim) "
        f"or (n_envs, n_bubbles, spatial_dim), "
        f"got {arr.shape}."
    )


def as_direction_matrix(
    value: FloatArray,
    n_envs: int,
    *,
    name: str,
) -> FloatArray:
    """Canonicalize 3D directions to shape (E, 3)."""

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
