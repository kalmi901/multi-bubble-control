import argparse
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np

from multibubble.integrators import solve_rkck45
from multibubble.physics import AcousticForcing, MaterialProperties
from multibubble.physics.single_bubble import (
    BubbleParameters,
    prepare_single_bubble_parameters,
    single_bubble_rhs,
)
from tools.progress import print_progress


def _parse_args():
    p = argparse.ArgumentParser(description="Single-bubble Poincare / bifurcation benchmark")

    p.add_argument(
        "--radius",
        type=float,
        default=41.3,
        help="Equilibrium bubble radius [micron]",
    )

    p.add_argument(
        "--frequency",
        type=float,
        default=25.0,
        help="Driving frequency [kHz]",
    )

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

    material = MaterialProperties()

    equilibrium_radius = args.radius * 1e-6
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

    bubble = BubbleParameters(
        equilibrium_radius=equilibrium_radius,
    )

    forcing = AcousticForcing(
        pressure_amplitude=pressure_amplitudes,
        frequency=frequency,
    )

    reference_radius = equilibrium_radius

    params = prepare_single_bubble_parameters(
        material,
        bubble,
        forcing,
        reference_radius,
    )

    num_envs = pressure_amplitudes.size

    y = np.zeros(
        (num_envs, 2),
        dtype=np.float64,
    )

    y[:, 0] = 1.0

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
            single_bubble_rhs,
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
            samples.append(y[:, 0].copy())

        phase = "transient" if cycle < args.transient_cycles else "sampling"

        print_progress(
            cycle + 1,
            total_cycles,
            start,
            phase=phase,
        )

    print()

    elapsed = perf_counter() - start

    samples = np.asarray(samples)

    # R_ref = R0 here, so y[:, 0] is directly R/R0.
    pressure_bar = pressure_amplitudes / 1e5

    x = np.broadcast_to(
        pressure_bar,
        samples.shape,
    )

    print(f"Environments       : {num_envs}")
    print(f"Total cycles       : {total_cycles}")
    print(f"Elapsed time [s]   : {elapsed:.3f}")
    print(f"Accepted steps     : {np.sum(accepted_steps)}")
    print(f"Rejected steps     : {np.sum(rejected_steps)}")

    print(f"Mean accepted/env  : {np.mean(accepted_steps):.1f}")

    plt.figure(figsize=(9, 5))

    plt.scatter(
        x.ravel(),
        samples.ravel(),
        s=3,
        alpha=0.6,
    )

    plt.xlabel(r"$p_A$ [bar]")
    plt.ylabel(r"$R/R_0$ at Poincaré section")
    plt.title("Single-bubble bifurcation diagram")

    plt.tight_layout()

    if args.save is not None:
        from tools.figure_io import save_figure_with_metadata

        metadata = {
            "benchmark": "single_bubble_bifurcation",
            "model": "single-bubble Keller-Miksis radial dynamics",
            "interaction": "none",
            "radius_micron": args.radius,
            "frequency_khz": args.frequency,
            "pressure_range_bar": [
                args.p_min,
                args.p_max,
            ],
            "num_amplitudes": args.num_amplitudes,
            "transient_cycles": args.transient_cycles,
            "sample_cycles": args.sample_cycles,
            "reference_radius_micron": reference_radius * 1e6,
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
