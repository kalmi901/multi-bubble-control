import argparse
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np

from multibubble.integrators import solve_rkck45
from multibubble.physics import AcousticForcing, MaterialProperties
from multibubble.physics.coupled_bubbles import (
    CoupledBubblesParameters,
    coupled_bubbles_rhs,
    prepare_coupled_bubbles_parameters,
)


def _parse_args():
    p = argparse.ArgumentParser(description="Coupled-bubbles Poincare / bifurcation benchmark")
    p.add_argument(
        "--radii",
        type=float,
        nargs="+",
        default=[41.3, 41.3],
        help="Equilibrium bubble radii [micron]",
    )

    p.add_argument(
        "--positions",
        type=float,
        nargs="+",
        default=[0.0, 0.02],
        help="Dimensionless position of bubbles [1D]",
    )

    p.add_argument("--frequency", type=float, default=25.0, help="Excitation frequency [kHz]")
    p.add_argument(
        "--p-min",
        type=float,
        default=0.2,
        help="Minimum pressure amplitude [bar]",
    )

    p.add_argument(
        "--p-max",
        type=float,
        default=1.9,
        help="Maximum pressure amplitude [bar]",
    )

    p.add_argument(
        "--num-amplitudes",
        type=int,
        default=128,
        help="Number of pressure amplitudes",
    )

    p.add_argument(
        "--transient-cycles",
        type=int,
        default=512,
        help="Discarded transient cycles",
    )

    p.add_argument(
        "--sample-cycles",
        type=int,
        default=32,
        help="Poincare samples retained per pressure amplitude",
    )

    return p.parse_args()


def _print_progress(
    current: int,
    total: int,
    start_time: float,
    transient_cycles: int,
    *,
    width: int = 30,
):
    elapsed = perf_counter() - start_time

    fraction = current / total
    filled = int(width * fraction)
    bar = "#" * filled + "-" * (width - filled)

    eta = elapsed * (total - current) / current if current > 0 else 0.0

    phase = "transient" if current <= transient_cycles else "sampling "

    print(
        f"\r[{bar}] "
        f"{current:4d}/{total} "
        f"({100.0 * fraction:5.1f}%) "
        f"{phase} | "
        f"elapsed {elapsed:7.1f}s | "
        f"ETA {eta:7.1f}s",
        end="",
        flush=True,
    )


def main():
    args = _parse_args()

    assert len(args.radii) == len(args.positions)

    material = MaterialProperties()

    bubbles = CoupledBubblesParameters(
        equilibrium_radius=np.asarray(args.radii, dtype=np.float64) * 1e-6,
        position=np.asarray(args.positions, dtype=np.float64)[..., np.newaxis],
    )

    frequency = args.frequency * 1e3

    pressure_amplitudes = (
        np.linspace(
            args.p_min,
            args.p_max,
            args.num_amplitudes,
            dtype=np.float64,
        )
        * 1e5
    )

    forcing = AcousticForcing(
        pressure_amplitude=pressure_amplitudes,
        frequency=frequency,
    )

    reference_radius = min(bubbles.equilibrium_radius)

    params = prepare_coupled_bubbles_parameters(material, bubbles, forcing, reference_radius)

    radius_ratio = bubbles.equilibrium_radius / reference_radius

    num_envs = pressure_amplitudes.size

    num_bubbles = len(args.radii)

    y = np.zeros((num_envs, num_bubbles, 2), dtype=np.float64)

    y[..., 0] = radius_ratio

    accepted_steps = np.zeros(
        num_envs,
        dtype=np.int64,
    )

    rejected_steps = np.zeros(
        num_envs,
        dtype=np.int64,
    )

    samples = []

    total_cycles = args.transient_cycles + args.sample_cycles

    start = perf_counter()
    for cycle in range(total_cycles):
        t_start = float(cycle)
        t_end = float(cycle + 1)

        t_final, y, info = solve_rkck45(
            coupled_bubbles_rhs,
            y,
            t_span=(t_start, t_end),
            initial_step=1e-3,
            rtol=1e-8,
            atol=np.array([1e-9, 1e-9]),
            args=(params,),
        )

        if not np.allclose(
            t_final,
            t_end,
            rtol=0.0,
            atol=1e-13,
        ):
            raise RuntimeError(f"Integration failed during cycle {cycle}")

        accepted_steps += info["n_accepted"]
        rejected_steps += info["n_rejected"]

        if cycle >= args.transient_cycles:
            samples.append(y[..., 0].copy())

        _print_progress(
            cycle + 1,
            total_cycles,
            start,
            args.transient_cycles,
        )

    print()

    elapsed = perf_counter() - start

    samples = np.asarray(samples)  # (S, E, N)

    pressure_bar = pressure_amplitudes / 1e5

    # Convert R / R_ref -> R / R0 for each bubble.
    samples_over_r0 = samples / params.equilibrium_radius_ratio[None, :, :]

    print(f"Samples shape      : {samples.shape}")
    print(f"Environments       : {num_envs}")
    print(f"Total cycles       : {total_cycles}")
    print(f"Elapsed time [s]   : {elapsed:.3f}")
    print(f"Accepted steps     : {np.sum(accepted_steps)}")
    print(f"Rejected steps     : {np.sum(rejected_steps)}")
    print(f"Mean accepted/env  : {np.mean(accepted_steps):.1f}")

    plt.figure(figsize=(9, 5))

    for bubble_idx in range(num_bubbles):
        bubble_samples = samples_over_r0[:, :, bubble_idx]  # (S, E)

        pressure_samples = np.broadcast_to(
            pressure_bar[None, :],
            bubble_samples.shape,
        )  # (S, E)

        plt.scatter(
            pressure_samples.ravel(),
            bubble_samples.ravel(),
            s=4,
            alpha=0.6,
            label=(f"Bubble {bubble_idx}: $R_0={args.radii[bubble_idx]:g}\\,\\mu$m"),
        )

    plt.xlabel(r"$p_A$ [bar]")
    plt.ylabel(r"$R/R_0$ at Poincaré section")
    plt.title("Coupled-bubble bifurcation diagram")

    if num_bubbles > 1:
        plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
