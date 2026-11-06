#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON:-python}"

# Prevent matplotlib from opening interactive windows.
export MPLBACKEND=Agg

EXAMPLE_FIG_DIR="examples/figs"
BENCHMARK_RADIAL_DIR="benchmarks/figs/radial"
BENCHMARK_TRANSLATIONAL_DIR="benchmarks/figs/translational"
BENCHMARK_PERFORMANCE_DIR="benchmarks/figs/performance"

mkdir -p \
    "${EXAMPLE_FIG_DIR}" \
    "${BENCHMARK_RADIAL_DIR}" \
    "${BENCHMARK_TRANSLATIONAL_DIR}" \
    "${BENCHMARK_PERFORMANCE_DIR}"

run() {
    echo
    echo "================================================================"
    echo "Running: $*"
    echo "================================================================"
    "$@"
}


echo "Generating example figures..."

# ---------------------------------------------------------------------------
# Single-bubble radial dynamics
# ---------------------------------------------------------------------------

run "${PYTHON_BIN}" -m examples.rkck45_single_bubble \
    --save "${EXAMPLE_FIG_DIR}/rkck45_single_bubble.png"


# ---------------------------------------------------------------------------
# Fixed-position coupled bubbles
#
# Weak interaction:
#   identical bubbles separated by one wavelength
# ---------------------------------------------------------------------------

run "${PYTHON_BIN}" -m examples.rkck45_coupled_bubbles \
    --radii 50.0 50.0 \
    --positions 0.0 1.0 \
    --frequency 25.0 \
    --amplitude 1.5 \
    --save "${EXAMPLE_FIG_DIR}/rkck45_coupled_bubbles_weak.png"


# ---------------------------------------------------------------------------
# Strong interaction:
#   identical bubbles at 0.02 lambda separation
# ---------------------------------------------------------------------------

run "${PYTHON_BIN}" -m examples.rkck45_coupled_bubbles \
    --radii 50.0 50.0 \
    --positions 0.0 0.02 \
    --frequency 25.0 \
    --amplitude 0.3 \
    --save "${EXAMPLE_FIG_DIR}/rkck45_coupled_bubbles_strong.png"


# ---------------------------------------------------------------------------
# Single translating bubble in a 3D standing-wave direction
# ---------------------------------------------------------------------------

run "${PYTHON_BIN}" -m examples.rkck45_translating_bubble \
    --frequency 25.0 \
    --amplitude 0.4 \
    --relative_frequency 0.87 \
    --displacement 0.01 \
    --direction 1.0 2.0 3.0 \
    --t_end 2000.0 \
    --save "${EXAMPLE_FIG_DIR}/rkck45_translating_bubble_3d.png"


# ---------------------------------------------------------------------------
# Coupled translating bubble pair
# ---------------------------------------------------------------------------

run "${PYTHON_BIN}" -m examples.rkck45_coupled_translating_bubbles \
    --save "${EXAMPLE_FIG_DIR}/rkck45_coupled_translating_bubbles.png"


# ---------------------------------------------------------------------------
# 32-bubble coupled radial-translational cluster
# ---------------------------------------------------------------------------

run "${PYTHON_BIN}" -m examples.rkck45_coupled_translating_cluster \
    --save "${EXAMPLE_FIG_DIR}/rkck45_coupled_translating_cluster.png"


echo
echo "Generating benchmark figures..."


# ---------------------------------------------------------------------------
# RKCK45 Tolerance sweep
# ---------------------------------------------------------------------------

run "${PYTHON_BIN}" -m benchmarks.rkck45_tolerance_sweep \
    --save "${BENCHMARK_PERFORMANCE_DIR}/rkck45_tolerance_sweep.png"


# ---------------------------------------------------------------------------
# Single-bubble bifurcation benchmark
# ---------------------------------------------------------------------------

run "${PYTHON_BIN}" -m benchmarks.single_bubble_bifurcation \
    --save "${BENCHMARK_RADIAL_DIR}/single_bubble_bifurcation.png"


# ---------------------------------------------------------------------------
# Weakly coupled bubble bifurcation benchmark
# ---------------------------------------------------------------------------

run "${PYTHON_BIN}" -m benchmarks.coupled_bubbles_bifurcation \
    --positions 0.0 1.0 \
    --save "${BENCHMARK_RADIAL_DIR}/coupled_bubbles_bifurcation_weak.png"


# ---------------------------------------------------------------------------
# Strongly coupled bubble bifurcation benchmark
# ---------------------------------------------------------------------------

run "${PYTHON_BIN}" -m benchmarks.coupled_bubbles_bifurcation \
    --positions 0.0 0.02 \
    --save "${BENCHMARK_RADIAL_DIR}/coupled_bubbles_bifurcation_strong.png"


# ---------------------------------------------------------------------------
# Translational migration / linear-theory benchmark
# ---------------------------------------------------------------------------

run "${PYTHON_BIN}" -m benchmarks.translating_bubble_migration \
    --save \
    "${BENCHMARK_TRANSLATIONAL_DIR}/translating_bubble_migration_linear_theory.png"


# ---------------------------------------------------------------------------
# Dense CPU radial-translational RHS scaling
#
# Limit this reproducibility figure to the project's primary 2--32 bubble
# range. Larger systems can be explored manually with the benchmark CLI.
# ---------------------------------------------------------------------------

run "${PYTHON_BIN}" -m benchmarks.coupled_translating_rhs_scaling \
    --bubble-counts 2 4 8 16 32 \
    --save "${BENCHMARK_PERFORMANCE_DIR}/coupled_translating_rhs_scaling.png"


echo
echo "All requested figures generated successfully."