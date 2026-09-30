# Multi-Bubble Control

**GPU-accelerated multi-bubble dynamics and reinforcement learning framework for collective acoustic bubble control.**

> **Development status:** Early-stage research software. The project infrastructure is operational, while the numerical solvers and reinforcement learning components are under development.

## Overview

This project aims to develop a computational framework for investigating and controlling the collective dynamics of acoustically driven bubble populations.

The framework combines numerical bubble dynamics, inter-bubble acoustic coupling, GPU acceleration, and reinforcement learning.

The initial implementation focuses on a delay-free coupled-bubble model, providing a computationally efficient baseline for collective control experiments.

The primary development goals are:

- Implement and validate single- and multi-bubble dynamics.
- Develop a GPU-accelerated solver for parallel bubble populations.
- Integrate the solver with PyTorch reinforcement learning environments.
- Investigate collective control strategies using multi-agent reinforcement learning and population-based training.
- Evaluate numerical accuracy and computational performance.

## Current status

The repository currently provides the initial software development infrastructure:

- Installable Python package using `pyproject.toml`.
- Automated Python tests using `pytest`.
- Code formatting and linting using Ruff and pre-commit.
- CPU-only continuous integration using GitHub Actions.

**The numerical simulation and RL components are not yet available in this repository.**

## Installation

Python 3.12 or later is required.

Clone the repository:

```bash
git clone git@github.com:kalmi901/multi-bubble-control.git
cd multi-bubble-control
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows (PowerShell):

```powershell
.\.venv\Scripts\Activate.ps1
```

Or on Linux:

```bash
source .venv/bin/activate
```

Install the package with development dependencies:

```bash
python -m pip install -e ".[dev]"
```

GPU dependencies and installation instructions will be added alongside the CUDA backend.

## Development and testing

Run the automated test suite:

```bash
python -m pytest -v
```

Run linting and formatting checks:

```bash
ruff check .
ruff format --check .
```

Install the pre-commit hooks:

```bash
pre-commit install
```

The repository also uses GitHub Actions to run automated checks on pushes and pull requests.

## Development roadmap

| Component | Status |
|---|---|
| Python package and CI infrastructure | Completed |
| NumPy single-bubble reference solver | Planned |
| Adaptive RKCK45 integrator | Planned |
| Generalized multi-bubble coupling | Planned |
| Numerical verification and regression tests | Planned |
| C++/CUDA acceleration | Planned |
| PyTorch RL environment | Planned |
| Multi-agent PPO and population-based training | Planned |
| Performance profiling and benchmarking | Planned |

The roadmap describes intended development milestones rather than implemented functionality.

## Numerical model

The initial numerical model will describe the radial oscillation of acoustically driven spherical bubbles using a Keller–Miksis-type formulation.

Multi-bubble interactions will be represented through pressure coupling between individual bubbles, initially neglecting finite acoustic propagation delays.

The initial scope excludes bubble translation, coalescence, fragmentation, and spatially resolved acoustic wave propagation.

These simplifications define the intended applicability of the initial control experiments.

## License

See [LICENSE](LICENSE) for licensing information.