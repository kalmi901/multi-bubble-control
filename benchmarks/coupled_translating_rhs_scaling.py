import argparse
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np

from multibubble.physics import MaterialProperties
from multibubble.physics.acoustic_field import AcousticField, UniformHarmonic
from multibubble.physics.common import ReferenceScales
from multibubble.physics.coupled_translating_bubbles import (
    CoupledTranslatingBubbleParameters,
    coupled_translating_bubble_rhs,
    prepare_coupled_translating_bubble_parameters,
)

DEFAULT_BUBBLE_COUNTS = (2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CPU RHS scaling benchmark for coupled radial-translational bubbles"
    )
    parser.add_argument(
        "--bubble-counts",
        type=int,
        nargs="+",
        default=list(DEFAULT_BUBBLE_COUNTS),
        help="Bubble counts used in the scaling sweep",
    )
    parser.add_argument("--frequency", type=float, default=25.0, help="Excitation frequency [kHz]")
    parser.add_argument(
        "--amplitude", type=float, default=0.20, help="Uniform pressure amplitude [bar]"
    )
    parser.add_argument(
        "--spacing", type=float, default=600.0, help="Nominal lattice spacing [micron]"
    )
    parser.add_argument(
        "--reference-radius",
        type=float,
        default=50.0,
        help="Reference radius [micron]",
    )
    parser.add_argument(
        "--warmup", type=int, default=5, help="Warmup RHS evaluations before timing"
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=25,
        help="Timed RHS evaluations for each bubble count",
    )
    parser.add_argument("--save", type=str, default=None, help="Optional output figure path")
    return parser.parse_args()


def _lattice_positions(
    n_bubbles: int,
    spacing_m: float,
    reference_radius_m: float,
) -> np.ndarray:
    """Return the first N points of a centered cubic lattice."""

    side = int(np.ceil(n_bubbles ** (1.0 / 3.0)))

    grid = np.stack(
        np.meshgrid(
            np.arange(side, dtype=np.float64),
            np.arange(side, dtype=np.float64),
            np.arange(side, dtype=np.float64),
            indexing="ij",
        ),
        axis=-1,
    ).reshape(-1, 3)

    position = grid[:n_bubbles].copy()
    position -= np.mean(position, axis=0, keepdims=True)
    return position * spacing_m / reference_radius_m


def _equilibrium_radii(n_bubbles: int) -> np.ndarray:
    """Return a deterministic radius pattern [m]."""

    pattern_micron = np.array(
        [40.0, 50.0, 60.0, 70.0, 55.0, 45.0, 65.0, 50.0],
        dtype=np.float64,
    )

    repetitions = int(np.ceil(n_bubbles / pattern_micron.size))
    return np.tile(pattern_micron, repetitions)[:n_bubbles] * 1e-6


def _make_state(
    equilibrium_radius: np.ndarray,
    position: np.ndarray,
    reference_radius: float,
) -> np.ndarray:
    """Construct one non-trivial state that exercises all coupling terms."""

    n_bubbles = equilibrium_radius.size
    y = np.zeros((1, n_bubbles, 8), dtype=np.float64)

    y[0, :, 0] = equilibrium_radius / reference_radius
    y[0, :, 1:4] = position

    index = np.arange(n_bubbles, dtype=np.float64)

    y[0, :, 4] = 0.02 * np.sin(0.7 * index)
    y[0, :, 5] = 0.01 * np.cos(0.3 * index)
    y[0, :, 6] = 0.01 * np.sin(0.5 * index + 0.2)
    y[0, :, 7] = 0.01 * np.cos(0.4 * index - 0.1)

    return y


def _benchmark_case(
    *,
    n_bubbles: int,
    material: MaterialProperties,
    frequency: float,
    amplitude_pa: float,
    spacing: float,
    reference_radius: float,
    warmup: int,
    repeats: int,
) -> dict[str, float]:
    equilibrium_radius = _equilibrium_radii(n_bubbles)
    position = _lattice_positions(n_bubbles, spacing, reference_radius)

    bubbles = CoupledTranslatingBubbleParameters(
        equilibrium_radius=equilibrium_radius,
    )

    field = AcousticField(
        components=(
            UniformHarmonic(
                pressure_amplitude=amplitude_pa,
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

    state = _make_state(equilibrium_radius, position, reference_radius)
    time = np.array([0.37], dtype=np.float64)

    result = None
    for _ in range(warmup):
        result = coupled_translating_bubble_rhs(time, state, params)

    if result is not None and not np.all(np.isfinite(result)):
        raise FloatingPointError(f"Non-finite RHS result during warmup for N={n_bubbles}.")

    samples = np.empty(repeats, dtype=np.float64)

    for repeat in range(repeats):
        start = perf_counter()
        result = coupled_translating_bubble_rhs(time, state, params)
        samples[repeat] = perf_counter() - start

    if result is None or not np.all(np.isfinite(result)):
        raise FloatingPointError(f"Non-finite RHS result during benchmark for N={n_bubbles}.")

    return {
        "median_seconds": float(np.median(samples)),
        "mean_seconds": float(np.mean(samples)),
        "min_seconds": float(np.min(samples)),
        "max_seconds": float(np.max(samples)),
    }


def main() -> None:
    args = _parse_args()

    bubble_counts = np.asarray(args.bubble_counts, dtype=np.int64)

    if bubble_counts.ndim != 1 or bubble_counts.size == 0:
        raise ValueError("bubble-counts must contain at least one value.")
    if np.any(bubble_counts < 2):
        raise ValueError("Every bubble count must be at least 2.")
    if len(np.unique(bubble_counts)) != bubble_counts.size:
        raise ValueError("bubble-counts must not contain duplicate values.")
    if args.warmup < 0:
        raise ValueError("warmup must be non-negative.")
    if args.repeats < 1:
        raise ValueError("repeats must be positive.")

    material = MaterialProperties()

    frequency = args.frequency * 1e3
    amplitude_pa = args.amplitude * 1e5
    spacing = args.spacing * 1e-6
    reference_radius = args.reference_radius * 1e-6

    median_seconds = []
    mean_seconds = []
    min_seconds = []
    max_seconds = []

    print(f"{'N':>5} {'4N':>6} {'median [ms]':>14} {'mean [ms]':>12} {'min [ms]':>11}")

    for n_bubbles in bubble_counts:
        result = _benchmark_case(
            n_bubbles=int(n_bubbles),
            material=material,
            frequency=frequency,
            amplitude_pa=amplitude_pa,
            spacing=spacing,
            reference_radius=reference_radius,
            warmup=args.warmup,
            repeats=args.repeats,
        )

        median_seconds.append(result["median_seconds"])
        mean_seconds.append(result["mean_seconds"])
        min_seconds.append(result["min_seconds"])
        max_seconds.append(result["max_seconds"])

        print(
            f"{n_bubbles:5d} "
            f"{4 * n_bubbles:6d} "
            f"{1e3 * result['median_seconds']:14.6f} "
            f"{1e3 * result['mean_seconds']:12.6f} "
            f"{1e3 * result['min_seconds']:11.6f}"
        )

    median_seconds = np.asarray(median_seconds, dtype=np.float64)
    mean_seconds = np.asarray(mean_seconds, dtype=np.float64)
    min_seconds = np.asarray(min_seconds, dtype=np.float64)
    max_seconds = np.asarray(max_seconds, dtype=np.float64)

    fig, axis = plt.subplots(1, 1, figsize=(7, 5))

    axis.plot(
        bubble_counts,
        1e3 * median_seconds,
        marker="o",
        linewidth=1.8,
        label="Measured RHS",
    )

    axis.set_xlabel("Number of bubbles N")
    axis.set_ylabel("Median RHS time [ms]")
    axis.set_xscale("log", base=2)
    axis.set_yscale("log")

    axis.set_xticks(bubble_counts)
    axis.set_xticklabels([str(n) for n in bubble_counts])

    axis.set_xlabel("Number of bubbles $N$")
    axis.set_title("Dense CPU coupled radial-translational RHS scaling")
    axis.grid(True, which="both", alpha=0.3)
    axis.legend()

    plt.tight_layout()

    if args.save is not None:
        from tools.figure_io import save_figure_with_metadata

        metadata = {
            "benchmark": "coupled_translating_rhs_scaling",
            "model": "coupled radial and 3D translational multi-bubble dynamics",
            "backend": "NumPy dense CPU reference",
            "bubble_counts": bubble_counts.tolist(),
            "acceleration_system_sizes": (4 * bubble_counts).tolist(),
            "frequency_khz": args.frequency,
            "amplitude_bar": args.amplitude,
            "spacing_micron": args.spacing,
            "reference_radius_micron": args.reference_radius,
            "warmup_evaluations": args.warmup,
            "timed_repeats": args.repeats,
            "timing_seconds": {
                "median": median_seconds.tolist(),
                "mean": mean_seconds.tolist(),
                "minimum": min_seconds.tolist(),
                "maximum": max_seconds.tolist(),
            },
        }

        save_figure_with_metadata(fig, args.save, metadata)

    plt.show()


if __name__ == "__main__":
    main()
