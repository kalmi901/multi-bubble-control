import numpy as np

from .common import FloatArray

DEFAULT_COLLISION_SAFETY_FACTOR = 1.02


def _minimum_pairwise_collision_margin(
    radius: FloatArray,
    position: FloatArray,
    *,
    safety_factor: float = DEFAULT_COLLISION_SAFETY_FACTOR,
) -> FloatArray:
    """Return the minimum pairwise collision margin for each environment.

    The safe domain is

        d_ij > safety_factor * (R_i + R_j)

    for every bubble pair.

    Parameters
    ----------
    radius
        Bubble radii with shape ``(n_envs, n_bubbles)``.

    position
        Bubble-center positions with shape
        ``(n_envs, n_bubbles, n_dimensions)``.

    safety_factor
        Multiplicative safety factor applied to the contact distance.
        ``1.0`` corresponds to exact geometric contact.

    Returns
    -------
    margin
        Minimum collision margin for each environment,
        shape ``(n_envs,)``.
    """

    radius = np.asarray(
        radius,
        dtype=np.float64,
    )

    position = np.asarray(
        position,
        dtype=np.float64,
    )

    if radius.ndim != 2:
        raise ValueError("radius must have shape (n_envs, n_bubbles).")

    if position.ndim != 3 or position.shape[:2] != radius.shape:
        raise ValueError(
            "position must have shape (n_envs, n_bubbles, n_dimensions) matching radius."
        )

    if radius.shape[1] < 2:
        raise ValueError("Collision detection requires at least two bubbles.")

    if not np.isfinite(safety_factor):
        raise ValueError("safety_factor must be finite.")

    if safety_factor < 1.0:
        raise ValueError("safety_factor must be greater than or equal to 1.")

    i, j = np.triu_indices(
        radius.shape[1],
        k=1,
    )

    pair_distance = np.linalg.norm(
        position[:, i, :] - position[:, j, :],
        axis=-1,
    )

    safety_contact_distance = safety_factor * (radius[:, i] + radius[:, j])

    pair_margin = pair_distance - safety_contact_distance

    return np.min(
        pair_margin,
        axis=1,
    )
