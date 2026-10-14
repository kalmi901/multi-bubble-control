import argparse

import matplotlib.pyplot as plt
import numpy as np

from multibubble.integrators import solve_rkck45
from multibubble.physics.single_bubble import (
    AcousticForcing,
    BubbleParameters,
    MaterialProperties,
    prepare_single_bubble_parameters,
    single_bubble_rhs,
)


def _parse_args():
    p = argparse.ArgumentParser(description="Single Run")
    p.add_argument("--radius", type=float, default=50.0, help="Equilibrum bubble radius [micron]")
    p.add_argument("--frequency", type=float, default=25.0, help="Excitation frequency [kHz]")
    p.add_argument("--amplitude", type=float, default=0.3, help="Excitation amplitued [bar]")
    p.add_argument(
        "--ref_radius",
        type=float,
        default=None,
        help="Reference radius [micron] (default: Equilibrium radius)",
    )
    p.add_argument(
        "--t_end", type=float, default=10.0, help="Integration domain end (dimensionless time)"
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

    print(f"R_min / R0 : {np.min(y[:, 0]) / params.equilibrium_radius_ratio:.6f}")
    print(f"R_max / R0 : {np.max(y[:, 0]) / params.equilibrium_radius_ratio:.6f}")

    wall_velocity = y[:, 1] * velocity_scale

    print(f"max |Rdot| [m/s] : {np.max(np.abs(wall_velocity)):.6e}")
    print(f"max wall Mach [-] : {np.max(np.abs(wall_velocity)) / material.sound_speed:.6e}")

    plt.figure()
    plt.plot(t, y[:, 0], linewidth=1.5)
    plt.xlabel(r"$\tau\, [-]$")
    plt.ylabel(r"$R/R_{ref}\, [-]$")
    plt.title("Dimensionless Radius-Time Curve")
    plt.tight_layout()

    plt.figure()
    plt.plot(y[:, 0], y[:, 1], linewidth=1.5)
    plt.xlabel(r"$R/R_{ref}\, [-]$")
    plt.ylabel(r"$U/U_{ref}\, [-]$")
    plt.title("Phase portrait")
    plt.tight_layout()

    plt.figure()
    plt.plot(t[1:], np.diff(t))
    plt.xlabel("t")
    plt.ylabel("Accepted dt")
    plt.title("Adaptive RKCK45 step sizes")
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
