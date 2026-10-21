import argparse

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
    p = argparse.ArgumentParser(description="Coupled-bubbles single run")
    p.add_argument(
        "--radii",
        type=float,
        nargs="+",
        default=[50.0, 50.0],
        help="Equilibrium bubble radii [micron]",
    )

    p.add_argument(
        "--positions",
        type=float,
        nargs="+",
        default=[0.0, 1.0],
        help="Dimensionless position of bubbles [1D]",
    )
    p.add_argument("--frequency", type=float, default=25.0, help="Excitation frequency [kHz]")
    p.add_argument("--amplitude", type=float, default=0.3, help="Excitation amplitude [bar]")
    p.add_argument(
        "--ref_radius",
        type=float,
        default=None,
        help="Reference radius [micron] (default: min(Equilibrium radius))",
    )
    p.add_argument(
        "--t_end", type=float, default=10.0, help="Integration domain end (dimensionless time)"
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

    forcing = AcousticForcing(
        pressure_amplitude=args.amplitude * 1e5, frequency=args.frequency * 1e3
    )

    reference_radius = (args.ref_radius if args.ref_radius is not None else min(args.radii)) * 1e-6

    radius_ratio = bubbles.equilibrium_radius / reference_radius

    y0 = np.stack(
        (
            radius_ratio,
            np.zeros_like(radius_ratio),
        ),
        axis=-1,
    )[None, ...]

    params = prepare_coupled_bubbles_parameters(material, bubbles, forcing, reference_radius)

    _, _, info = solve_rkck45(
        coupled_bubbles_rhs,
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

    env = 0

    t = trajectory["t"][env]  # (K,)
    y = trajectory["y"][env]  # (K, N, 2)

    print(f"Accepted steps : {info['n_accepted'][env]}")
    print(f"Rejected steps : {info['n_rejected'][env]}")

    radius = y[:, :, 0]  # (K, N)
    r0_ratio = params.equilibrium_radius_ratio[env]  # (N,)

    r_min = np.min(radius, axis=0) / r0_ratio
    r_max = np.max(radius, axis=0) / r0_ratio

    print("R_min / R0 :", np.array2string(r_min, precision=6))
    print("R_max / R0 :", np.array2string(r_max, precision=6))

    velocity_scale = reference_radius * forcing.frequency
    wall_velocity = y[:, :, 1] * velocity_scale  # (K, N)

    sound_speed = np.asarray(material.sound_speed, dtype=np.float64)
    sound_speed_env = sound_speed if sound_speed.ndim == 0 else sound_speed[env]
    max_wall_velocity = np.max(np.abs(wall_velocity), axis=0)
    max_wall_mach = max_wall_velocity / sound_speed_env

    print("max |Rdot| [m/s] :", np.array2string(max_wall_velocity, precision=6))
    print("max wall Mach [-] :", np.array2string(max_wall_mach, precision=6))

    n_bubbles = y.shape[1]
    plt.figure()
    for i in range(n_bubbles):
        plt.plot(
            t,
            y[:, i, 0],
            linewidth=1.5,
            label=f"Bubble {i}",
        )

    plt.xlabel(r"$\tau\, [-]$")
    plt.ylabel(r"$R/R_{\mathrm{ref}}\, [-]$")
    plt.title("Dimensionless Radius-Time Curves")

    if n_bubbles > 1:
        plt.legend()

    plt.tight_layout()

    plt.figure()

    for i in range(n_bubbles):
        plt.plot(
            y[:, i, 0],
            y[:, i, 1],
            linewidth=1.5,
            label=f"Bubble {i}",
        )

    plt.xlabel(r"$R/R_{\mathrm{ref}}\, [-]$")
    plt.ylabel(r"$d(R/R_{\mathrm{ref}})/d\tau\, [-]$")
    plt.title("Phase Portrait")

    if n_bubbles > 1:
        plt.legend()

    plt.tight_layout()

    plt.figure()
    plt.plot(t[1:], np.diff(t), linewidth=1.5)
    plt.xlabel(r"$\tau\, [-]$")
    plt.ylabel(r"$\Delta\tau\, [-]$")
    plt.title("Adaptive RKCK45 Step Sizes")
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
