import argparse
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np

from multibubble.integrators import solve_rkck45


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RKCK45 tolerance sweep")
    p.add_argument("--omega", type=float, default=1.0, help="Excitation frequency [-]")
    p.add_argument(
        "--t_end",
        type=float,
        default=20.0,
        help="Integration domain end in periods",
    )
    p.add_argument("--save", type=str, default=None, help="Optional output figure path")

    return p.parse_args()


def main():
    args = _parse_args()

    omega = args.omega
    t_end = args.t_end * np.pi

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

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].loglog(rtols, errors, "o-")
    axes[0].invert_xaxis()
    axes[0].set_xlabel("Relative tolerance")
    axes[0].set_ylabel("Final-state error")
    axes[0].set_title("RKCK45 tolerance convergence")
    axes[0].grid()

    axes[1].loglog(accepted_steps, errors, "o-")
    axes[1].set_xlabel("Accepted steps")
    axes[1].set_ylabel("Final-state error")
    axes[1].set_title("Accuracy vs computational work")
    axes[1].grid()
    plt.tight_layout()

    if args.save is not None:
        from tools.figure_io import save_figure_with_metadata

        metadata = {
            "benchmark": "rkck45_tolerance_sweep",
            "omega": args.omega,
            "t_end": args.t_end,
            "solver": {"name": "RKCK45", "initial_step": 0.2},
        }

        save_figure_with_metadata(fig, args.save, metadata)

    plt.show()


if __name__ == "__main__":
    main()
