import argparse

import matplotlib.pyplot as plt
import numpy as np

from multibubble.integrators import solve_rkck45
from multibubble.physics import AcousticForcing, MaterialProperties
from multibubble.physics.single_bubble import (
    BubbleParameters,
    prepare_single_bubble_parameters,
    single_bubble_rhs,
)


def _parse_args():
    p = argparse.ArgumentParser(description="Single-bubble single run")
    p.add_argument("--radius", type=float, default=50.0, help="Equilibrium bubble radius [micron]")
    p.add_argument("--frequency", type=float, default=25.0, help="Excitation frequency [kHz]")
    p.add_argument("--amplitude", type=float, default=0.3, help="Excitation amplitude [bar]")
    p.add_argument(
        "--ref_radius",
        type=float,
        default=None,
        help="Reference radius [micron] (default: Equilibrium radius)",
    )
    p.add_argument(
        "--t_end", type=float, default=10.0, help="Integration domain end (dimensionless time)"
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

    bubble = BubbleParameters(equilibrium_radius=args.radius * 1e-6)

    forcing = AcousticForcing(
        pressure_amplitude=args.amplitude * 1e5, frequency=args.frequency * 1e3
    )

    reference_radius = (args.ref_radius if args.ref_radius is not None else args.radius) * 1e-6
    y0 = np.array([[1.0 * bubble.equilibrium_radius / reference_radius, 0.0]])  # Initial Conditions

    params = prepare_single_bubble_parameters(material, bubble, forcing, reference_radius)

    _, _, info = solve_rkck45(
        single_bubble_rhs,
        y0,
        t_span=(0.0, args.t_end),
        initial_step=1e-3,
        rtol=1e-9,
        atol=np.array([1e-10, 1e-10]),
        args=(params,),
        store_trajectory=True,
    )

    trajectory = info["trajectory"]
    assert trajectory is not None

    t = trajectory["t"][0]
    y = trajectory["y"][0]

    print(f"Accepted steps : {info['n_accepted'][0]}")
    print(f"Rejected steps : {info['n_rejected'][0]}")
    velocity_scale = reference_radius * forcing.frequency

    radius_min_over_equilibrium = np.min(y[:, 0]) / params.equilibrium_radius_ratio
    radius_max_over_equilibrium = np.max(y[:, 0]) / params.equilibrium_radius_ratio

    print(f"R_min / R0 : {radius_min_over_equilibrium:.6f}")
    print(f"R_max / R0 : {radius_max_over_equilibrium:.6f}")

    wall_velocity = y[:, 1] * velocity_scale

    max_abs_wall_velocity_m_per_s = np.max(np.abs(wall_velocity))
    max_abs_wall_mach = max_abs_wall_velocity_m_per_s / material.sound_speed

    print(f"max |Rdot| [m/s] : {max_abs_wall_velocity_m_per_s:.6e}")
    print(f"max wall Mach [-] : {max_abs_wall_mach:.6e}")

    fig = plt.figure()
    plt.plot(t, y[:, 0], linewidth=1.5)
    plt.xlabel(r"$\tau\, [-]$")
    plt.ylabel(r"$R/R_{ref}\, [-]$")
    plt.title("Dimensionless Radius-Time Curve")
    plt.tight_layout()

    plt.figure()
    plt.plot(y[:, 0], y[:, 1], linewidth=1.5)
    plt.xlabel(r"$R/R_{ref}\, [-]$")
    plt.ylabel(r"$d(R/R_{ref})/d\tau\, [-]$")
    plt.title("Phase portrait")
    plt.tight_layout()

    plt.figure()
    plt.plot(t[1:], np.diff(t), linewidth=1.5)
    plt.xlabel(r"$\tau\, [-]$")
    plt.ylabel("Accepted dt")
    plt.title("Adaptive RKCK45 step sizes")
    plt.tight_layout()

    if args.save is not None:
        from tools.figure_io import save_figure_with_metadata

        metadata = {
            "example": "rkck45_single_bubble",
            "model": "single-bubble Keller-Miksis radial dynamics",
            "interaction": "none",
            "radius_micron": args.radius,
            "frequency_khz": args.frequency,
            "amplitude_bar": args.amplitude,
            "reference_radius_micron": reference_radius * 1e6,
            "solver": {
                "name": "RKCK45",
                "rtol": 1e-9,
                "atol": [1e-10, 1e-10],
                "initial_step": 1e-3,
            },
            "statistics": {
                "accepted_steps": int(info["n_accepted"][0]),
                "rejected_steps": int(info["n_rejected"][0]),
            },
            "response": {
                "radius_min_over_equilibrium": radius_min_over_equilibrium,
                "radius_max_over_equilibrium": radius_max_over_equilibrium,
                "max_abs_wall_velocity_m_per_s": max_abs_wall_velocity_m_per_s,
                "max_abs_wall_mach": max_abs_wall_mach,
            },
        }

        save_figure_with_metadata(fig, args.save, metadata)

    plt.show()


if __name__ == "__main__":
    main()
