import argparse

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
    parser = argparse.ArgumentParser(description="Coupled radial-translational bubble-pair run")
    parser.add_argument(
        "--radii",
        type=float,
        nargs=2,
        default=[7.0, 5.0],
        metavar=("R0", "R1"),
        help="Equilibrium bubble radii [micron]",
    )
    parser.add_argument(
        "--positions",
        type=float,
        nargs=2,
        default=[0.0, 0.0026666666666666666],
        metavar=("X0", "X1"),
        help="Initial bubble positions as x/lambda [1D]",
    )
    parser.add_argument(
        "--frequency",
        type=float,
        default=20.0,
        help="Excitation frequency [kHz]",
    )
    parser.add_argument(
        "--amplitude",
        type=float,
        default=1.2,
        help="Uniform excitation amplitude [bar]",
    )
    parser.add_argument(
        "--ref_radius",
        type=float,
        default=None,
        help="Reference radius [micron] (default: min(equilibrium radii))",
    )
    parser.add_argument(
        "--t_end",
        type=float,
        default=10.0,
        help="Integration domain end in acoustic cycles",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Optional output figure path",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    radii_micron = np.asarray(args.radii, dtype=np.float64)
    positions_over_wavelength = np.asarray(args.positions, dtype=np.float64)

    if np.any(radii_micron <= 0.0):
        raise ValueError("Bubble radii must be positive.")

    if positions_over_wavelength[0] == positions_over_wavelength[1]:
        raise ValueError("Initial bubble positions must be distinct.")

    if args.t_end <= 0.0:
        raise ValueError("t_end must be positive.")

    material = MaterialProperties()

    frequency = args.frequency * 1e3
    wavelength = material.sound_speed / frequency
    equilibrium_radius = radii_micron * 1e-6

    reference_radius = (
        args.ref_radius * 1e-6 if args.ref_radius is not None else np.min(equilibrium_radius)
    )

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

    initial_position = np.zeros((2, 3), dtype=np.float64)
    initial_position[:, 0] = positions_over_wavelength * wavelength / reference.radius

    y0 = np.zeros((1, 2, 8), dtype=np.float64)
    y0[0, :, 0] = equilibrium_radius / reference.radius
    y0[0, :, 1:4] = initial_position

    _, _, info = solve_rkck45(
        coupled_translating_bubble_rhs,
        y0,
        t_span=(0.0, args.t_end),
        initial_step=1e-3,
        rtol=1e-9,
        atol=np.full(8, 1e-10, dtype=np.float64),
        args=(params,),
        store_trajectory=True,
        max_steps=int(1e6),
    )

    trajectory = info["trajectory"]
    if trajectory is None:
        raise RuntimeError("Trajectory storage was requested but no trajectory was returned.")

    env = 0
    t = trajectory["t"][env]
    y = trajectory["y"][env]

    radius_micron = y[:, :, 0] * reference_radius * 1e6
    radius_over_equilibrium = radius_micron / radii_micron[None, :]

    position_micron = y[:, :, 1:4] * reference_radius * 1e6

    pair_vector_micron = position_micron[:, 1, :] - position_micron[:, 0, :]
    center_distance_micron = np.linalg.norm(pair_vector_micron, axis=1)

    surface_gap_micron = center_distance_micron - radius_micron[:, 0] - radius_micron[:, 1]

    velocity_scale = reference_radius * frequency
    wall_velocity = y[:, :, 4] * velocity_scale
    translation_velocity = y[:, :, 5:8] * velocity_scale

    print(f"Accepted steps        : {int(info['n_accepted'][env])}")
    print(f"Rejected steps        : {int(info['n_rejected'][env])}")
    print(f"Wavelength [mm]       : {wavelength * 1e3:.6f}")
    print(f"Initial distance [um] : {center_distance_micron[0]:.6f}")
    print(f"Final distance [um]   : {center_distance_micron[-1]:.6f}")
    print(f"Minimum gap [um]      : {np.min(surface_gap_micron):.6f}")
    print("R_min / R0 : " + np.array2string(np.min(radius_over_equilibrium, axis=0), precision=6))
    print("R_max / R0 : " + np.array2string(np.max(radius_over_equilibrium, axis=0), precision=6))
    print(
        "max |Rdot| [m/s] : " + np.array2string(np.max(np.abs(wall_velocity), axis=0), precision=6)
    )
    print(
        "max |V| [m/s] : "
        + np.array2string(
            np.max(np.linalg.norm(translation_velocity, axis=-1), axis=0),
            precision=6,
        )
    )

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(9, 10),
        sharex=True,
    )

    for bubble_index in range(2):
        axes[0].plot(
            t,
            radius_over_equilibrium[:, bubble_index],
            linewidth=1.5,
            label=f"Bubble {bubble_index}: R0={radii_micron[bubble_index]:g} μm",
        )

    axes[0].set_ylabel(r"$R_i/R_{0,i}\ [-]$")
    axes[0].set_title("Radial response")
    axes[0].legend()
    axes[0].grid()

    for bubble_index in range(2):
        axes[1].plot(
            t,
            position_micron[:, bubble_index, 0],
            linewidth=1.5,
            label=f"Bubble {bubble_index}",
        )

    axes[1].set_ylabel(r"$x_i\ [\mu\mathrm{m}]$")
    axes[1].set_title("Translational position")
    axes[1].legend()
    axes[1].grid()

    axes[2].plot(
        t,
        center_distance_micron,
        linewidth=1.5,
        label="Center distance",
    )
    axes[2].plot(
        t,
        surface_gap_micron,
        linewidth=1.5,
        label="Surface gap",
    )
    axes[2].axhline(0.0, linestyle="--", linewidth=1.0)

    axes[2].set_xlabel(r"$\tau\ [-]$")
    axes[2].set_ylabel(r"Distance $[\mu\mathrm{m}]$")
    axes[2].set_title("Pair separation")
    axes[2].legend()
    axes[2].grid()

    fig.suptitle("Coupled radial-translational bubble pair", fontsize=14)
    fig.tight_layout()

    if args.save is not None:
        from tools.figure_io import save_figure_with_metadata

        metadata = {
            "example": "rkck45_coupled_translating_pair",
            "model": "coupled radial and 3D translational two-bubble dynamics",
            "acoustic_field": "uniform harmonic",
            "radii_micron": radii_micron.tolist(),
            "positions_x_over_wavelength": positions_over_wavelength.tolist(),
            "initial_center_distance_micron": float(center_distance_micron[0]),
            "frequency_khz": args.frequency,
            "amplitude_bar": args.amplitude,
            "reference_radius_micron": reference_radius * 1e6,
            "integration_end_cycles": args.t_end,
            "solver": {
                "name": "RKCK45",
                "rtol": 1e-9,
                "atol": 1e-10,
                "initial_step": 1e-3,
                "max_steps": int(1e6),
            },
            "statistics": {
                "accepted_steps": int(info["n_accepted"][env]),
                "rejected_steps": int(info["n_rejected"][env]),
            },
            "response": {
                "radius_min_over_equilibrium": np.min(
                    radius_over_equilibrium,
                    axis=0,
                ).tolist(),
                "radius_max_over_equilibrium": np.max(
                    radius_over_equilibrium,
                    axis=0,
                ).tolist(),
                "final_center_distance_micron": float(center_distance_micron[-1]),
                "minimum_surface_gap_micron": float(np.min(surface_gap_micron)),
            },
        }

        save_figure_with_metadata(fig, args.save, metadata)

    plt.show()


if __name__ == "__main__":
    main()
