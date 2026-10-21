from time import perf_counter


def print_progress(
    current: int,
    total: int,
    start_time: float,
    *,
    phase: str | None = None,
    width: int = 30,
) -> None:
    elapsed = perf_counter() - start_time

    fraction = current / total
    filled = int(width * fraction)
    bar = "#" * filled + "-" * (width - filled)

    eta = elapsed * (total - current) / current if current > 0 else 0.0

    phase_text = f"{phase} | " if phase is not None else ""

    print(
        f"\r[{bar}] "
        f"{current:4d}/{total} "
        f"({100.0 * fraction:5.1f}%) "
        f"{phase_text}"
        f"elapsed {elapsed:7.1f}s | "
        f"ETA {eta:7.1f}s",
        end="",
        flush=True,
    )
