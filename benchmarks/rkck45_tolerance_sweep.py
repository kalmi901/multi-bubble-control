from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np

from multibubble.integrators import solve_rkck45


def main():
    omega = 1.0
    t_end = 20.0 * np.pi

    y0 = np.array([[1.0, 0.0]])

    def rhs(t, y, omega):
        dydt = np.empty_like(y)
        dydt[:, 0] = y[:, 1]
        dydt[:, 1] = -(omega**2) * y[:, 0]
        return dydt

    rtols = np.array(
        [
            1e-3,
            1e-4,
            1e-5,
            1e-6,
            1e-7,
            1e-8,
            1e-9,
        ]
    )

    errors = []
    accepted_steps = []
    rejected_steps = []
    runtimes_ms = []

    x_exact = np.cos(omega * t_end)
    v_exact = -omega * np.sin(omega * t_end)

    print(f"{'rtol':>10} {'error':>14} {'accepted':>10} {'rejected':>10} {'time [ms]':>12}")

    for rtol in rtols:
        atol = rtol * 1e-2

        start = perf_counter()

        _, y, info = solve_rkck45(
            rhs,
            y0,
            t_span=(0.0, t_end),
            initial_step=0.2,
            rtol=rtol,
            atol=atol,
            args=(omega,),
        )

        runtime_ms = (perf_counter() - start) * 1e3

        error = np.sqrt((y[0, 0] - x_exact) ** 2 + (y[0, 1] - v_exact) ** 2)

        n_accepted = info["n_accepted"][0]
        n_rejected = info["n_rejected"][0]

        errors.append(error)
        accepted_steps.append(n_accepted)
        rejected_steps.append(n_rejected)
        runtimes_ms.append(runtime_ms)

        print(f"{rtol:10.1e} {error:14.6e} {n_accepted:10d} {n_rejected:10d} {runtime_ms:12.3f}")

    errors = np.asarray(errors)
    accepted_steps = np.asarray(accepted_steps)

    plt.figure()
    plt.loglog(rtols, errors, "o-")
    plt.gca().invert_xaxis()
    plt.xlabel("Relative tolerance")
    plt.ylabel("Final-state error")
    plt.title("RKCK45 tolerance convergence")
    plt.tight_layout()

    plt.figure()
    plt.loglog(accepted_steps, errors, "o-")
    plt.xlabel("Accepted steps")
    plt.ylabel("Final-state error")
    plt.title("Accuracy vs computational work")
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
