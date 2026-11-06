import matplotlib.pyplot as plt
import numpy as np

from multibubble.integrators import solve_rkck45


def main():
    omega = 1.0
    t_end = 2.0 * np.pi

    # Start at x=0 with positive velocity so the first velocity zero
    # corresponds to the first positive displacement maximum.
    y0 = np.array([[0.0, 1.0]])

    def rhs(t, y, omega):
        dydt = np.empty_like(y)
        dydt[:, 0] = y[:, 1]
        dydt[:, 1] = -(omega**2) * y[:, 0]
        return dydt

    def turning_point_event(t, y, omega):
        # Positive velocity is the valid domain. The first v <= 0
        # crossing terminates the integration.
        return y[:, 1]

    _t_final, y_final, info = solve_rkck45(
        rhs,
        y0,
        t_span=(0.0, t_end),
        initial_step=0.2,
        rtol=1e-9,
        atol=np.array([1e-11, 1e-11]),
        args=(omega,),
        event=turning_point_event,
        event_time_tolerance=1e-10,
        store_trajectory=True,
    )

    trajectory = info["trajectory"]
    assert trajectory is not None

    t = trajectory["t"][0]
    y = trajectory["y"][0]

    event_time = info["event_time"][0]
    exact_event_time = 0.5 * np.pi / omega

    print(f"Event triggered   : {info['event_triggered'][0]}")
    print(f"Event time        : {event_time:.12f}")
    print(f"Exact event time  : {exact_event_time:.12f}")
    print(f"Event time error  : {abs(event_time - exact_event_time):.3e}")
    print(f"Final position    : {y_final[0, 0]:.12f}")
    print(f"Final velocity    : {y_final[0, 1]:.3e}")

    t_exact = np.linspace(
        0.0,
        exact_event_time,
        400,
    )
    x_exact = np.sin(omega * t_exact)
    v_exact = np.cos(omega * t_exact)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(8, 7),
        sharex=True,
    )

    axes[0].plot(
        t_exact,
        x_exact,
        label="Exact",
    )
    axes[0].plot(
        t,
        y[:, 0],
        "o",
        markersize=3,
        label="RKCK45",
    )
    axes[0].axvline(
        event_time,
        linestyle="--",
        label="Terminal event",
    )
    axes[0].set_ylabel("x")
    axes[0].set_title("Harmonic oscillator terminal event")
    axes[0].legend()
    axes[0].grid()

    axes[1].plot(
        t_exact,
        v_exact,
        label="Exact",
    )
    axes[1].plot(
        t,
        y[:, 1],
        "o",
        markersize=3,
        label="RKCK45",
    )
    axes[1].axhline(
        0.0,
        linestyle="--",
        linewidth=1.0,
    )
    axes[1].axvline(
        event_time,
        linestyle="--",
        label="Terminal event",
    )
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("v")
    axes[1].legend()
    axes[1].grid()

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
