import argparse
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np

from multibubble.integrators import solve_rkck45
from multibubble.physics import MaterialProperties
from multibubble.physics.acoustic_field import AcousticField, PlaneStandingWave
from multibubble.physics.common import ReferenceScales
from multibubble.physics.translating_bubble import (
    TranslatingBubbleParameters,
    prepare_translating_bubble_parameters,
    translating_bubble_rhs,
)
from tools.progress import print_progress


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Single translating-bubble migration benchmark in a plane standing wave"
    )
    p.add_argument(
        "--radius",
        type=float,
        default=50.0,
        help="Equilibrium bubble radius [micron]",
    )
    p.add_argument(
        "--frequency",
        type=float,
        default=25.0,
        help="Excitation frequency [kHz]",
    )
    p.add_argument(
        "--amplitude",
        type=float,
        default=0.3,
        help="Excitation pressure amplitude [bar]",
    )
    p.add_argument(
        "--ref-radius",
        type=float,
        default=None,
        help="Reference radius [micron] (default: equilibrium radius)",
    )
    p.add_argument(
        "--position-min",
        type=float,
        default=0.0,
        help="Minimum initial position x/lambda",
    )
    p.add_argument(
        "--position-max",
        type=float,
        default=0.5,
        help="Maximum initial position x/lambda",
    )
    p.add_argument(
        "--num-positions",
        type=int,
        default=33,
        help="Number of initial positions",
    )
    p.add_argument(
        "--transient-cycles",
        type=int,
        default=32,
        help="Number of cycles discarded before drift measurement",
    )
    p.add_argument(
        "--sample-cycles",
        type=int,
        default=128,
        help="Number of cycles used for migration statistics",
    )
    p.add_argument(
        "--direction",
        type=float,
        nargs=3,
        default=[1.0, 0.0, 0.0],
        help="Standing-wave propagation direction",
    )
    p.add_argument(
        "--save",
        type=str,
        default=None,
        help="Optional output figure path",
    )

    return p.parse_args()


def _projected_position_over_wavelength(
    y: np.ndarray,
    direction: np.ndarray,
    reference_radius: float,
    wavelength: float,
) -> np.ndarray:
    """Return bubble position projected onto the standing-wave direction as x/lambda."""

    position = y[:, 1:4]

    projected_position = position @ direction

    return projected_position * reference_radius / wavelength


def main() -> None:
    args = _parse_args()

    if args.num_positions < 2:
        raise ValueError("num_positions must be at least 2.")

    if args.transient_cycles < 0:
        raise ValueError("transient_cycles must be non-negative.")

    if args.sample_cycles < 1:
        raise ValueError("sample_cycles must be positive.")

    direction = np.asarray(
        args.direction,
        dtype=np.float64,
    )

    direction_norm = np.linalg.norm(direction)

    if direction_norm == 0.0:
        raise ValueError("direction must be non-zero.")

    direction /= direction_norm

    material = MaterialProperties()

    frequency = args.frequency * 1e3
    radius = args.radius * 1e-6

    reference_radius = radius if args.ref_radius is None else args.ref_radius * 1e-6

    wavelength = material.sound_speed / frequency

    initial_position_over_wavelength = np.linspace(
        args.position_min,
        args.position_max,
        args.num_positions,
        dtype=np.float64,
    )

    # Internal position scale is x / R_ref.
    initial_position = (
        initial_position_over_wavelength[:, None]
        * direction[None, :]
        * wavelength
        / reference_radius
    )

    bubble = TranslatingBubbleParameters(
        equilibrium_radius=radius,
    )

    field = AcousticField(
        components=(
            PlaneStandingWave(
                pressure_amplitude=args.amplitude * 1e5,
                frequency=frequency,
                direction=direction,
            ),
        )
    )

    reference = ReferenceScales(
        radius=reference_radius,
        frequency=frequency,
    )

    params = prepare_translating_bubble_parameters(
        material,
        bubble,
        field,
        reference,
        n_envs=args.num_positions,
    )

    y = np.zeros(
        (args.num_positions, 8),
        dtype=np.float64,
    )

    y[:, 0] = radius / reference_radius
    y[:, 1:4] = initial_position

    accepted_steps = np.zeros(
        args.num_positions,
        dtype=np.int64,
    )

    rejected_steps = np.zeros(
        args.num_positions,
        dtype=np.int64,
    )

    total_cycles = args.transient_cycles + args.sample_cycles

    # Store only cycle-boundary positions during the measurement window.
    position_samples = []

    start = perf_counter()

    for cycle in range(total_cycles):
        t_start = float(cycle)
        t_end = float(cycle + 1)

        t_final, y, info = solve_rkck45(
            translating_bubble_rhs,
            y,
            t_span=(t_start, t_end),
            initial_step=1e-3,
            rtol=1e-9,
            atol=np.full(8, 1e-10, dtype=np.float64),
            args=(params,),
            max_steps=int(1e6),
        )

        if not np.allclose(
            t_final,
            t_end,
            rtol=0.0,
            atol=1e-13,
        ):
            raise RuntimeError(f"Integration failed during cycle {cycle}.")

        accepted_steps += info["n_accepted"]
        rejected_steps += info["n_rejected"]

        # Store the position at the beginning of the measurement
        # interval and at every subsequent cycle boundary.
        if cycle == args.transient_cycles - 1:
            position_samples.append(
                _projected_position_over_wavelength(
                    y,
                    direction,
                    reference_radius,
                    wavelength,
                )
            )

        elif args.transient_cycles == 0 and cycle == 0 and not position_samples:
            # The initial state itself is required below.
            position_samples.append(initial_position_over_wavelength.copy())

            position_samples.append(
                _projected_position_over_wavelength(
                    y,
                    direction,
                    reference_radius,
                    wavelength,
                )
            )

        elif cycle >= args.transient_cycles:
            position_samples.append(
                _projected_position_over_wavelength(
                    y,
                    direction,
                    reference_radius,
                    wavelength,
                )
            )

        phase = "transient" if cycle < args.transient_cycles else "sampling"

        print_progress(
            cycle + 1,
            total_cycles,
            start,
            phase=phase,
        )

    print()

    elapsed = perf_counter() - start

    position_samples = np.asarray(
        position_samples,
        dtype=np.float64,
    )

    # Expected shape:
    #
    # (sample_cycles + 1, num_positions)
    expected_shape = (
        args.sample_cycles + 1,
        args.num_positions,
    )

    if position_samples.shape != expected_shape:
        raise RuntimeError(
            "Unexpected position-sample shape: "
            f"{position_samples.shape}, expected {expected_shape}."
        )

    # Cycle-to-cycle migration measured in wavelength per acoustic cycle.
    cycle_drift = np.diff(
        position_samples,
        axis=0,
    )

    mean_drift = np.mean(
        cycle_drift,
        axis=0,
    )

    drift_std = np.std(
        cycle_drift,
        axis=0,
    )

    spatial_shape = np.sin(4.0 * np.pi * initial_position_over_wavelength)

    scale = np.dot(mean_drift, spatial_shape) / np.dot(spatial_shape, spatial_shape)

    expected_shape = scale * spatial_shape

    shape_error = np.linalg.norm(mean_drift - expected_shape) / np.linalg.norm(mean_drift)

    net_displacement = position_samples[-1] - position_samples[0]

    # Since lambda * f = c:
    #
    # d(x/lambda)/d(tau) * c = dx/dt.
    mean_velocity_m_per_s = material.sound_speed * mean_drift

    print(f"Environments       : {args.num_positions}")
    print(f"Transient cycles   : {args.transient_cycles}")
    print(f"Sample cycles      : {args.sample_cycles}")
    print(f"Elapsed time [s]   : {elapsed:.3f}")
    print(f"Accepted steps     : {np.sum(accepted_steps)}")
    print(f"Rejected steps     : {np.sum(rejected_steps)}")

    print(f"max |mean drift|  : {np.max(np.abs(mean_drift)):.6e} lambda/cycle")

    print(f"max |mean speed|  : {np.max(np.abs(mean_velocity_m_per_s)):.6e} m/s")

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5),
    )

    axes[0].plot(
        initial_position_over_wavelength,
        mean_drift,
        marker="o",
        markersize=3,
        linewidth=1.5,
    )

    axes[0].plot(
        initial_position_over_wavelength,
        expected_shape,
        "--",
        label=r"$A\sin(4\pi x_0/\lambda)$",
    )

    axes[0].axhline(
        0.0,
        linestyle="--",
        linewidth=1.0,
    )

    for special_position in (0.0, 0.25, 0.5):
        if args.position_min <= special_position <= args.position_max:
            axes[0].axvline(
                special_position,
                linestyle=":",
                linewidth=1.0,
            )

    axes[0].set_xlabel(r"initial position $x_0/\lambda$ [-]")

    axes[0].set_ylabel(r"mean migration $\langle\Delta(x/\lambda)\rangle$ / cycle")

    axes[0].set_title("Cycle-Averaged Translational Migration")

    axes[0].grid()

    axes[1].plot(
        initial_position_over_wavelength,
        drift_std,
        marker="o",
        markersize=3,
        linewidth=1.5,
    )

    axes[1].set_xlabel(r"initial position $x_0/\lambda$ [-]")

    axes[1].set_ylabel(r"std$(\Delta(x/\lambda))$ / cycle")

    axes[1].set_title("Cycle-to-Cycle Drift Variation")

    axes[1].grid()

    plt.tight_layout()

    if args.save is not None:
        from tools.figure_io import save_figure_with_metadata

        metadata = {
            "benchmark": "translating_bubble_migration",
            "model": ("single-bubble Keller-Miksis radial dynamics with 3D translational motion"),
            "acoustic_field": "plane standing wave",
            "translation_model": ("primary Bjerknes force with Stokes drag"),
            "radius_micron": radius * 1e6,
            "frequency_khz": args.frequency,
            "amplitude_bar": args.amplitude,
            "reference_radius_micron": reference_radius * 1e6,
            "wavelength_m": wavelength,
            "standing_wave_direction": direction.tolist(),
            "position_sweep_over_wavelength": [
                args.position_min,
                args.position_max,
            ],
            "num_positions": args.num_positions,
            "transient_cycles": args.transient_cycles,
            "sample_cycles": args.sample_cycles,
            "solver": {
                "name": "RKCK45",
                "rtol": 1e-9,
                "atol": 1e-10,
                "initial_step": 1e-3,
                "max_steps": int(1e6),
            },
            "statistics": {
                "accepted_steps": int(np.sum(accepted_steps)),
                "rejected_steps": int(np.sum(rejected_steps)),
                "elapsed_seconds": elapsed,
            },
            "response": {
                "initial_position_over_wavelength": initial_position_over_wavelength.tolist(),
                "mean_drift_over_wavelength_per_cycle": mean_drift.tolist(),
                "drift_std_over_wavelength_per_cycle": drift_std.tolist(),
                "net_displacement_over_wavelength": net_displacement.tolist(),
                "migration_shape_relative_error": float(shape_error),
                "mean_velocity_m_per_s": mean_velocity_m_per_s.tolist(),
                "max_abs_mean_drift_over_wavelength_per_cycle": float(np.max(np.abs(mean_drift))),
                "max_abs_mean_velocity_m_per_s": float(np.max(np.abs(mean_velocity_m_per_s))),
            },
        }

        save_figure_with_metadata(
            fig,
            args.save,
            metadata,
        )

    plt.show()


if __name__ == "__main__":
    main()
