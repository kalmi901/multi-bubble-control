import argparse
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np

from multibubble.integrators import solve_rkck45
from multibubble.physics import MaterialProperties
from multibubble.physics.acoustic_field import AcousticField, UniformHarmonic
from multibubble.physics.common import ReferenceScales
from multibubble.physics.coupled_translating_bubbles import (
    CoupledTranslatingBubbleParameters,
    coupled_translating_bubble_rhs,
    prepare_coupled_translating_bubble_parameters,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="32-bubble coupled radial-translational cluster example"
    )
    parser.add_argument("--frequency", type=float, default=25.0, help="Excitation frequency [kHz]")
    parser.add_argument(
        "--amplitude", type=float, default=0.30, help="Uniform pressure amplitude [bar]"
    )
    parser.add_argument(
        "--spacing", type=float, default=600.0, help="Initial lattice spacing [micron]"
    )
    parser.add_argument(
        "--reference-radius",
        type=float,
        default=50.0,
        help="Reference radius [micron]",
    )
    parser.add_argument(
        "--t-end",
        type=float,
        default=5.0,
        help="Integration end time in acoustic cycles",
    )
    parser.add_argument("--rtol", type=float, default=1e-6, help="RKCK45 relative tolerance")
    parser.add_argument("--atol", type=float, default=1e-8, help="RKCK45 absolute tolerance")
    parser.add_argument("--save", type=str, default=None, help="Optional output figure path")
    return parser.parse_args()


def _lattice_4x4x2(spacing_m: float, reference_radius_m: float) -> np.ndarray:
    """Return a centered 4 x 4 x 2 lattice with shape (32, 3)."""

    grid = np.stack(
        np.meshgrid(
            np.arange(4, dtype=np.float64),
            np.arange(4, dtype=np.float64),
            np.arange(2, dtype=np.float64),
            indexing="ij",
        ),
        axis=-1,
    ).reshape(-1, 3)

    grid -= np.mean(grid, axis=0, keepdims=True)
    return grid * spacing_m / reference_radius_m


def _equilibrium_radii() -> np.ndarray:
    """Deterministic non-uniform 32-bubble radius pattern [m]."""

    pattern_micron = np.array(
        [40.0, 50.0, 60.0, 70.0, 55.0, 45.0, 65.0, 50.0],
        dtype=np.float64,
    )
    return np.tile(pattern_micron, 4) * 1e-6


def _rms_cluster_radius(position: np.ndarray) -> np.ndarray:
    """Return RMS bubble-center distance from the cluster centroid."""

    center = np.mean(position, axis=1, keepdims=True)
    relative = position - center

    return np.sqrt(
        np.mean(
            np.sum(relative * relative, axis=-1),
            axis=1,
        )
    )


def main() -> None:
    args = _parse_args()

    if args.spacing <= 0.0:
        raise ValueError("spacing must be positive.")
    if args.reference_radius <= 0.0:
        raise ValueError("reference-radius must be positive.")
    if args.t_end <= 0.0:
        raise ValueError("t-end must be positive.")

    material = MaterialProperties()

    frequency = args.frequency * 1e3
    reference_radius = args.reference_radius * 1e-6
    spacing = args.spacing * 1e-6

    equilibrium_radius = _equilibrium_radii()
    n_bubbles = equilibrium_radius.size

    initial_position = _lattice_4x4x2(spacing, reference_radius)

    bubbles = CoupledTranslatingBubbleParameters(
        equilibrium_radius=equilibrium_radius,
    )

    field = AcousticField(
        components=(
            UniformHarmonic(
                pressure_amplitude=args.amplitude * 1e5,
                frequency=frequency,
            ),
        )
    )

    reference = ReferenceScales(
        radius=reference_radius,
        frequency=frequency,
    )

    params = prepare_coupled_translating_bubble_parameters(
        material,
        bubbles,
        field,
        reference,
        n_envs=1,
    )

    y0 = np.zeros((1, n_bubbles, 8), dtype=np.float64)
    y0[0, :, 0] = equilibrium_radius / reference_radius
    y0[0, :, 1:4] = initial_position

    start = perf_counter()

    t_final, _, info = solve_rkck45(
        coupled_translating_bubble_rhs,
        y0,
        t_span=(0.0, args.t_end),
        initial_step=1e-3,
        rtol=args.rtol,
        atol=np.full(8, args.atol, dtype=np.float64),
        args=(params,),
        store_trajectory=True,
        max_steps=int(1e6),
    )

    elapsed = perf_counter() - start

    if not np.allclose(t_final, args.t_end, rtol=0.0, atol=1e-13):
        raise RuntimeError("Integration did not reach the requested endpoint.")

    trajectory = info["trajectory"]
    if trajectory is None:
        raise RuntimeError("Trajectory storage was requested but no trajectory was returned.")

    t = trajectory["t"][0]
    y = trajectory["y"][0]

    position_micron = y[:, :, 1:4] * reference_radius * 1e6

    equilibrium_ratio = equilibrium_radius / reference_radius
    radius_over_equilibrium = y[:, :, 0] / equilibrium_ratio[None, :]

    rms_cluster_radius_micron = _rms_cluster_radius(position_micron)
    initial_rms_radius = rms_cluster_radius_micron[0]
    final_rms_radius = rms_cluster_radius_micron[-1]

    displacement_micron = position_micron[-1] - position_micron[0]
    displacement_norm_micron = np.linalg.norm(displacement_micron, axis=1)

    print(f"Bubbles             : {n_bubbles}")
    print(f"Acceleration system : {4 * n_bubbles} x {4 * n_bubbles}")
    print(f"Frequency [kHz]     : {args.frequency:.3f}")
    print(f"Amplitude [bar]     : {args.amplitude:.3f}")
    print(f"Spacing [micron]    : {args.spacing:.3f}")
    print(f"Cycles              : {args.t_end:.3f}")
    print(f"Elapsed time [s]    : {elapsed:.3f}")
    print(f"Accepted steps      : {int(info['n_accepted'][0])}")
    print(f"Rejected steps      : {int(info['n_rejected'][0])}")
    print(f"Initial Rc [micron] : {initial_rms_radius:.6f}")
    print(f"Final Rc [micron]   : {final_rms_radius:.6f}")
    print(
        f"Delta Rc / Rc0      : {(final_rms_radius - initial_rms_radius) / initial_rms_radius:.6e}"
    )
    print(f"Max displacement [micron] : {np.max(displacement_norm_micron):.6e}")

    fig = plt.figure(figsize=(13, 9), constrained_layout=True)

    ax_initial = fig.add_subplot(2, 2, 1, projection="3d")
    ax_final = fig.add_subplot(2, 2, 2, projection="3d")
    ax_radius = fig.add_subplot(2, 2, 3)
    ax_cluster = fig.add_subplot(2, 2, 4)

    initial = position_micron[0]
    final = position_micron[-1]

    size_scale = 18.0
    radius_micron = equilibrium_radius * 1e6
    marker_size = size_scale * (radius_micron / np.mean(radius_micron)) ** 2

    ax_initial.scatter(initial[:, 0], initial[:, 1], initial[:, 2], s=marker_size)  # type: ignore[arg-type]
    ax_final.scatter(final[:, 0], final[:, 1], final[:, 2], s=marker_size)  # type: ignore[arg-type]

    all_position = np.concatenate((initial, final), axis=0)
    mins = np.min(all_position, axis=0)
    maxs = np.max(all_position, axis=0)
    center = 0.5 * (mins + maxs)
    half_range = 0.55 * np.max(maxs - mins)

    if half_range == 0.0:
        half_range = 1.0

    for axis, title in ((ax_initial, "Initial cluster"), (ax_final, "Final cluster")):
        axis.set_xlim(center[0] - half_range, center[0] + half_range)
        axis.set_ylim(center[1] - half_range, center[1] + half_range)
        axis.set_zlim(center[2] - half_range, center[2] + half_range)
        axis.set_xlabel(r"$x\ [\mu\mathrm{m}]$")
        axis.set_ylabel(r"$y\ [\mu\mathrm{m}]$")
        axis.set_zlabel(r"$z\ [\mu\mathrm{m}]$")
        axis.set_title(title)
        axis.set_box_aspect((1.0, 1.0, 1.0))

    mesh = ax_radius.pcolormesh(
        t,
        np.arange(n_bubbles),
        radius_over_equilibrium.T,
        shading="auto",
    )
    ax_radius.set_xlabel(r"$\tau\ [-]$")
    ax_radius.set_ylabel("Bubble index")
    ax_radius.set_title(r"Radial response $R_i/R_{0,i}$")
    fig.colorbar(mesh, ax=ax_radius, label=r"$R_i/R_{0,i}$")

    ax_cluster.plot(
        t,
        rms_cluster_radius_micron / initial_rms_radius,
        linewidth=1.5,
    )
    ax_cluster.axhline(1.0, linestyle="--", linewidth=1.0)
    ax_cluster.set_xlabel(r"$\tau\ [-]$")
    ax_cluster.set_ylabel(r"$R_{\mathrm{rms}}/R_{\mathrm{rms},0}\ [-]$")
    ax_cluster.set_title("RMS cluster radius")
    ax_cluster.grid()

    fig.suptitle("32-bubble coupled radial-translational dynamics", fontsize=14)

    if args.save is not None:
        from tools.figure_io import save_figure_with_metadata

        metadata = {
            "example": "rkck45_coupled_translating_cluster",
            "model": "coupled radial and 3D translational multi-bubble dynamics",
            "num_bubbles": n_bubbles,
            "geometry": "4 x 4 x 2 centered Cartesian lattice",
            "equilibrium_radius_micron": radius_micron.tolist(),
            "frequency_khz": args.frequency,
            "amplitude_bar": args.amplitude,
            "spacing_micron": args.spacing,
            "reference_radius_micron": args.reference_radius,
            "integration_end_cycles": args.t_end,
            "solver": {
                "name": "RKCK45",
                "rtol": args.rtol,
                "atol": args.atol,
                "initial_step": 1e-3,
                "max_steps": int(1e6),
            },
            "statistics": {
                "elapsed_seconds": elapsed,
                "accepted_steps": int(info["n_accepted"][0]),
                "rejected_steps": int(info["n_rejected"][0]),
            },
            "response": {
                "initial_radius_of_gyration_micron": float(initial_rms_radius),
                "final_radius_of_gyration_micron": float(final_rms_radius),
                "relative_radius_of_gyration_change": float(
                    (final_rms_radius - initial_rms_radius) / initial_rms_radius
                ),
                "max_displacement_micron": float(np.max(displacement_norm_micron)),
            },
        }

        save_figure_with_metadata(fig, args.save, metadata)

    plt.show()


if __name__ == "__main__":
    main()
