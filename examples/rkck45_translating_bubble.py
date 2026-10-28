import argparse

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


def eigen_frequency(radius_m: float, material):
    """Return the linear resonance angular frequency [rad/s]."""
    return (
        (3 * material.polytropic_exponent * material.ambient_pressure / material.density)
        + 2
        * (3.0 * material.polytropic_exponent - 1)
        * material.surface_tension
        / material.density
        / radius_m
    ) ** 0.5 / radius_m


def get_radius(relative_frequency: float, excitation_frequency: float, material):
    """Return R0 [m] for a prescribed excitation-to-resonance frequency ratio."""
    from scipy.optimize import fsolve

    return fsolve(
        func=lambda R0, wr, w: (w / wr - eigen_frequency(R0, material)) ** 2,
        x0=10e-6,
        args=(relative_frequency, 2 * np.pi * excitation_frequency),
    )[0]


def _parse_args():
    p = argparse.ArgumentParser(description="Single-bubble with translation in 3D space")
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
        "--relative_frequency",
        type=float,
        default=None,
        help="Excitation-to-resonance frequency ratio omega/omega_0. "
        "If provided, the equilibrium radius is determined from this ratio "
        "and --radius is ignored.",
    )
    p.add_argument(
        "--t_end", type=float, default=100.0, help="Integration domain end (dimensionless time)"
    )
    p.add_argument(
        "--displacement",
        type=float,
        default=0.01,
        help="Initial bubble displacement as x/lambda along a given direction",
    )
    p.add_argument(
        "--direction",
        type=float,
        nargs="+",
        default=[1.0, 0.0, 0.0],
        help="Direction vector of the Standing Wave",
    )
    p.add_argument("--save", type=str, default=None, help="Optional output figure path")

    return p.parse_args()


def main():

    args = _parse_args()

    assert len(args.direction) == 3

    # Normalize direction vector
    direction = np.asarray(args.direction, dtype=np.float64)
    direction /= np.linalg.norm(direction)

    material = MaterialProperties()

    frequency = args.frequency * 1e3

    radius = (
        args.radius * 1e-6
        if args.relative_frequency is None
        else get_radius(args.relative_frequency, frequency, material)
    )

    reference_radius = radius if args.ref_radius is None else args.ref_radius * 1e-6
    reference_length = material.sound_speed / frequency

    bubble = TranslatingBubbleParameters(equilibrium_radius=radius)

    field = AcousticField(
        components=(
            PlaneStandingWave(
                pressure_amplitude=args.amplitude * 1e5,
                frequency=frequency,
                temporal_phase=0,
                spatial_phase=0,
                direction=direction,
            ),
        )
    )

    reference = ReferenceScales(radius=reference_radius, frequency=frequency)

    initial_position = args.displacement * direction * reference_length / reference_radius

    radius_ratio = bubble.equilibrium_radius / reference_radius

    y0 = np.concatenate(([radius_ratio], initial_position, np.zeros(4)), dtype=np.float64)[
        np.newaxis, :
    ]

    params = prepare_translating_bubble_parameters(material, bubble, field, reference, 1)

    _, _, info = solve_rkck45(
        translating_bubble_rhs,
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
    assert trajectory is not None

    t = trajectory["t"][0]
    y = trajectory["y"][0]

    print(f"Accepted steps : {info['n_accepted'][0]}")
    print(f"Rejected steps : {info['n_rejected'][0]}")
    velocity_scale = reference_radius * reference.frequency

    print(f"R0 [µm] : {radius * 1e6:.2f}")
    print(f"λ  [cm] : {reference_length * 1e2:.2f}")

    radius_min_over_equilibrium = np.min(y[:, 0]) / params.equilibrium_radius_ratio[0]
    radius_max_over_equilibrium = np.max(y[:, 0]) / params.equilibrium_radius_ratio[0]

    print(f"R_min / R0 : {radius_min_over_equilibrium:.6f}")
    print(f"R_max / R0 : {radius_max_over_equilibrium:.6f}")

    wall_velocity = y[:, 4] * velocity_scale

    max_abs_wall_velocity_m_per_s = np.max(np.abs(wall_velocity))
    max_abs_wall_mach = max_abs_wall_velocity_m_per_s / material.sound_speed

    print(f"max |Rdot| [m/s] : {max_abs_wall_velocity_m_per_s:.6e}")
    print(f"max wall Mach [-] : {max_abs_wall_mach:.6e}")

    position_over_lambda = y[:, 1:4] * reference_radius / reference_length

    projected_position = position_over_lambda @ direction

    transverse_position = position_over_lambda - projected_position[:, None] * direction[None, :]

    max_transverse_displacement = np.max(np.linalg.norm(transverse_position, axis=1))

    print(f"max transverse x/lambda [-] : {max_transverse_displacement:.6e}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].plot(t, y[:, 0], linewidth=1.5)
    axes[0].set_xlabel(r"$\tau\, [-]$")
    axes[0].set_ylabel(r"$R/R_{ref}\, [-]$")
    axes[0].set_title("Dimensionless Radius-Time Curve")

    axes[1].plot(
        t, y[:, 1] * reference_radius / reference_length, linewidth=1.5, label="x-direction"
    )
    axes[1].plot(
        t, y[:, 2] * reference_radius / reference_length, linewidth=1.5, label="y-direction"
    )
    axes[1].plot(
        t, y[:, 3] * reference_radius / reference_length, linewidth=1.5, label="z-direction"
    )
    axes[1].plot(
        t,
        projected_position,
        linewidth=1.5,
        label="Agg",
    )
    axes[1].set_xlabel(r"$\tau\, [-]$")
    axes[1].set_ylabel(r"$\mathbf{x}/\lambda\, [-]$")
    axes[1].legend()
    axes[1].grid()

    plt.tight_layout()

    if args.save is not None:
        from tools.figure_io import save_figure_with_metadata

        metadata = {
            "example": "rkck45_translating_bubble",
            "model": ("single-bubble Keller-Miksis radial dynamics with 3D translational motion"),
            "acoustic_field": "plane standing wave",
            "translation_model": "primary Bjerknes force with Stokes drag",
            "radius_micron": radius * 1e6,
            "frequency_khz": args.frequency,
            "amplitude_bar": args.amplitude,
            "relative_frequency": (2.0 * np.pi * frequency / eigen_frequency(radius, material)),
            "reference_radius_micron": reference_radius * 1e6,
            "wavelength_m": reference_length,
            "standing_wave_direction": direction.tolist(),
            "initial_displacement_over_wavelength": args.displacement,
            "initial_position_over_wavelength": (args.displacement * direction).tolist(),
            "integration_end_cycles": args.t_end,
            "solver": {
                "name": "RKCK45",
                "rtol": 1e-9,
                "atol": 1e-10,
                "initial_step": 1e-3,
                "max_steps": int(1e6),
            },
            "statistics": {
                "accepted_steps": int(info["n_accepted"][0]),
                "rejected_steps": int(info["n_rejected"][0]),
            },
            "response": {
                "radius_min_over_equilibrium": float(radius_min_over_equilibrium),
                "radius_max_over_equilibrium": float(radius_max_over_equilibrium),
                "max_abs_wall_velocity_m_per_s": float(max_abs_wall_velocity_m_per_s),
                "max_abs_wall_mach": float(max_abs_wall_mach),
                "final_position_over_wavelength": position_over_lambda[-1].tolist(),
                "final_projected_position_over_wavelength": float(projected_position[-1]),
                "max_transverse_displacement_over_wavelength": float(max_transverse_displacement),
            },
        }

        save_figure_with_metadata(fig, args.save, metadata)

    plt.show()


if __name__ == "__main__":
    main()
