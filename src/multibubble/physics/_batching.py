import numpy as np

from .common import FloatArray, Parameter


def environment_size(
    value: Parameter,
    *,
    name: str,
) -> int | None:
    """Return environment count for a scalar or environment vector."""

    arr = np.asarray(
        value,
        dtype=np.float64,
    )

    if arr.ndim == 0:
        return None

    if arr.ndim == 1:
        return arr.shape[0]

    raise ValueError(f"{name} must be a scalar or have shape (n_envs,), got {arr.shape}.")


def resolve_num_envs(
    candidates: list[tuple[str, int]],
    *,
    context: str = "Environment-specific parameters",
) -> int:
    """Resolve and validate a common environment count."""

    if not candidates:
        return 1

    n_envs = candidates[0][1]

    inconsistent = [(name, size) for name, size in candidates if size != n_envs]

    if inconsistent:
        details = ", ".join(f"{name}={size}" for name, size in candidates)

        raise ValueError(f"{context} have inconsistent environment counts: {details}.")

    return n_envs


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


def as_env_column(
    value: Parameter,
    n_envs: int,
    *,
    name: str,
) -> FloatArray:
    """Canonicalize an environment parameter to shape (E, 1)."""

    return as_env_vector(
        value,
        n_envs,
        name=name,
    )[:, None]


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


def direction_environment_size(
    value: FloatArray,
    *,
    name: str,
) -> int | None:
    """Return environment count implied by 3D direction vectors."""

    arr = np.asarray(
        value,
        dtype=np.float64,
    )

    if arr.shape == (3,):
        return None

    if arr.ndim == 2 and arr.shape[1] == 3:
        return arr.shape[0]

    raise ValueError(f"{name} must have shape (3,) or (n_envs, 3), got {arr.shape}.")


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
