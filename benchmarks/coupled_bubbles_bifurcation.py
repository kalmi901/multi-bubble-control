import argparse
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np

from multibubble.integrators import solve_rkck45
from multibubble.physics import MaterialProperties
from multibubble.physics.acoustic_field import AcousticField, UniformHarmonic
from multibubble.physics.common import ReferenceScales
from multibubble.physics.coupled_bubbles import (
    CoupledBubblesParameters,
    coupled_bubbles_rhs,
    prepare_coupled_bubbles_parameters,
)
from tools.progress import print_progress


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

    p.add_argument(
        "--save",
        type=str,
        default=None,
        help="Optional output figure path",
    )

    return p.parse_args()


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

    field = AcousticField(
        components=(UniformHarmonic(pressure_amplitude=pressure_amplitudes, frequency=frequency),)
    )

    reference = ReferenceScales(radius=min(bubbles.equilibrium_radius), frequency=frequency)

    params = prepare_coupled_bubbles_parameters(
        material, bubbles, field, reference, args.num_amplitudes
    )

    radius_ratio = bubbles.equilibrium_radius / reference.radius

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

        phase = "transient" if cycle < args.transient_cycles else "sampling"

        print_progress(
            cycle + 1,
            total_cycles,
            start,
            phase=phase,
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

    if args.save is not None:
        from tools.figure_io import save_figure_with_metadata

        metadata = {
            "benchmark": "coupled_bubbles_bifurcation",
            "model": "fixed-position coupled Keller-Miksis radial dynamics",
            "interaction": "instantaneous pressure coupling",
            "radii_micron": np.asarray(args.radii).tolist(),
            "positions_dimensionless": np.asarray(args.positions).tolist(),
            "frequency_khz": args.frequency,
            "pressure_range_bar": [
                args.p_min,
                args.p_max,
            ],
            "num_amplitudes": args.num_amplitudes,
            "transient_cycles": args.transient_cycles,
            "sample_cycles": args.sample_cycles,
            "reference_radius_micron": reference.radius * 1e6,
            "solver": {
                "name": "RKCK45",
                "rtol": 1e-8,
                "atol": [1e-9, 1e-9],
                "initial_step": 1e-3,
            },
            "statistics": {
                "accepted_steps": int(np.sum(accepted_steps)),
                "rejected_steps": int(np.sum(rejected_steps)),
                "elapsed_seconds": elapsed,
            },
        }

        save_figure_with_metadata(plt.gcf(), args.save, metadata)

    plt.show()


if __name__ == "__main__":
    main()
