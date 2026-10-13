import matplotlib.pyplot as plt
import numpy as np

from multibubble.integrators import solve_rkck45


def main():
    omega = 1.0
    t_end = 100.0 * np.pi

    y0 = np.array([[1.0, 0.0]])

    def rhs(t, y, omega):
        dydt = np.empty_like(y)
        dydt[:, 0] = y[:, 1]
        dydt[:, 1] = -(omega**2) * y[:, 0]
        return dydt

    _, _, info = solve_rkck45(
        rhs,
        y0,
        t_span=(0.0, t_end),
        initial_step=0.2,
        rtol=1e-8,
        atol=np.array([1e-10, 1e-10]),
        args=(omega,),
        store_trajectory=True,
    )

    trajectory = info["trajectory"]
    assert trajectory is not None

    t = trajectory["t"][0]
    y = trajectory["y"][0]

    x_exact = np.cos(omega * t)
    # v_exact = -omega * np.sin(omega * t)

    x_error = np.abs(y[:, 0] - x_exact)
    nonzero = x_error > 0.0

    energy = 0.5 * (y[:, 1] ** 2 + omega**2 * y[:, 0] ** 2)
    energy_exact = energy[0]
    relative_energy_error = (energy - energy_exact) / energy_exact

    phase_num = np.unwrap(np.arctan2(-y[:, 1] / omega, y[:, 0]))
    phase_exact = omega * t
    phase_error = phase_num - phase_exact

    print(f"Accepted steps : {info['n_accepted'][0]}")
    print(f"Rejected steps : {info['n_rejected'][0]}")
    print(f"Max |x error|  : {np.max(x_error):.3e}")

    print("Max relative energy error:", np.max(np.abs(relative_energy_error)))
    print("Final relative energy error:", relative_energy_error[-1])
    print("Final phase error [rad]:", phase_error[-1])

    plt.figure()
    plt.plot(t, y[:, 0], "o", markersize=3, label="RKCK45")
    plt.plot(t, x_exact, "-", label="Exact")
    plt.xlabel("t")
    plt.ylabel("x")
    plt.title("Harmonic oscillator")
    plt.legend()
    plt.tight_layout()

    plt.figure()
    plt.plot(y[:, 0], y[:, 1])
    plt.xlabel("x")
    plt.ylabel("v")
    plt.title("Phase portrait")
    plt.tight_layout()

    plt.figure()
    plt.semilogy(t[nonzero], x_error[nonzero])
    plt.xlabel("t")
    plt.ylabel("|x - x_exact|")
    plt.title("Position error")
    plt.tight_layout()

    plt.figure()
    plt.plot(t[1:], np.diff(t))
    plt.xlabel("t")
    plt.ylabel("Accepted dt")
    plt.title("Adaptive RKCK45 step sizes")
    plt.tight_layout()

    plt.figure()
    plt.plot(t, relative_energy_error)
    plt.xlabel("t")
    plt.ylabel("(E - E0) / E0")
    plt.title("Relative energy error")
    plt.tight_layout()

    plt.figure()
    plt.plot(t, phase_error)
    plt.xlabel("t")
    plt.ylabel("Phase error [rad]")
    plt.title("Phase error")
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
