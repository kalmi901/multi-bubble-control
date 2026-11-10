# Native CUDA Backend

This directory contains the native C++/CUDA backend for **multi-bubble-control**.

The existing Python/NumPy implementation remains the canonical physical and numerical reference. The native backend is developed independently with a focus on GPU performance, scalability, profiling, and future interoperability with PyTorch.

The CUDA backend is intentionally not a line-by-line translation of the Python implementation. Physical models and numerical results should remain consistent, while data layout and execution strategy may differ substantially.

## Current status

The native backend currently provides the initial CUDA build and test infrastructure.

The first smoke test verifies the complete native development path:

```text
CMake
  |
  v
C++ / CUDA compilation
  |
  v
mbc_cuda_core static library
  |
  v
CUDA kernel launch
  |
  v
asynchronous device-to-host copy
  |
  v
CTest validation
```

No production bubble dynamics are implemented in the native backend yet.

## Project structure

```text
native/
├── CMakeLists.txt
│
├── include/
│   └── mbc/
│       └── cuda/
│           ├── check.hpp
│           └── smoke.hpp
│
└── src/
    └── cuda/
        └── smoke.cu
```

Native CUDA tests are located in:

```text
tests/cuda/
```

The native backend is built as the static library:

```text
mbc_cuda_core
```

Future CUDA models, numerical methods, and execution infrastructure will be added to this library.

## Design principles

The native backend follows several architectural constraints from the beginning:

- Python/NumPy remains the physical and numerical reference implementation.
- CUDA code must not depend on Python or PyTorch.
- Solver APIs should operate on caller-owned device memory.
- CUDA execution should support caller-provided streams.
- Core solver functions should avoid implicit global synchronization.
- Temporary solver memory should eventually be managed through reusable workspaces.
- Future PyTorch integration should wrap the CUDA core rather than duplicate it.
- Independent simulation environments should remain independently schedulable to support future multi-stream and multi-GPU execution.

The current smoke-test API already follows the basic asynchronous execution model.

## Build system

The native backend uses modern CMake with C++ and CUDA as first-class project languages.

The root project defines:

```cmake
project(
    MultiBubbleControl
    LANGUAGES CXX CUDA
)
```

CUDA sources are therefore treated as regular CMake target sources rather than using the legacy `FindCUDA` workflow.

CTest is used for native test discovery and execution.

## Windows development environment

The initial development environment is:

```text
Windows 10 x64
Visual Studio 2022 Build Tools
MSVC v142 / 19.29
CUDA Toolkit 11.8
CMake 3.31+
NVIDIA GTX 1070
CUDA architecture: sm_61
```

CUDA 11.8 requires a compatible MSVC host compiler. On the current development system, the VS2019-compatible `v142` toolset is installed side-by-side with the newer Visual Studio 2022 toolchain.

The newer MSVC installation can remain installed for other projects.

### Required Visual Studio components

Install **Visual Studio 2022 Build Tools** with at least:

```text
Desktop development with C++

MSVC v142 - VS 2019 C++ x64/x86 build tools
Windows SDK
C++ CMake tools for Windows
```

CUDA Toolkit 11.8 must also be installed.

Verify the tools from an **x64 Native Tools Command Prompt for VS 2022**:

```cmd
cl
nvcc --version
cmake --version
```

The selected MSVC compiler for CUDA 11.8 should report the 19.29 compiler family.

## CUDA Visual Studio integration

CUDA's Visual Studio build customization files must be visible to MSBuild.

CUDA 11.8 installs the source files under:

```text
%CUDA_PATH%\extras\visual_studio_integration\MSBuildExtensions
```

The directory contains:

```text
CUDA 11.8.props
CUDA 11.8.targets
CUDA 11.8.xml
Nvda.Build.CudaTasks.v11.8.dll
```

When Visual Studio Build Tools are installed after CUDA, these files may not automatically appear inside the Visual Studio MSBuild directories.

For the current VS2022 + v142 configuration they must be available under both:

```text
C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\
MSBuild\Microsoft\VC\v170\BuildCustomizations
```

and:

```text
C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\
MSBuild\Microsoft\VC\v160\BuildCustomizations
```

If required, copy the CUDA build customization files from an elevated command prompt.

Example:

```cmd
copy /Y "%CUDA_PATH%\extras\visual_studio_integration\MSBuildExtensions\*" ^
"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\MSBuild\Microsoft\VC\v170\BuildCustomizations\"
```

and:

```cmd
copy /Y "%CUDA_PATH%\extras\visual_studio_integration\MSBuildExtensions\*" ^
"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\MSBuild\Microsoft\VC\v160\BuildCustomizations\"
```

This is normally only required once when preparing the development machine.

## CMake presets

The repository uses `CMakePresets.json` to store common build configurations.

A preset does not introduce a new build system. It simply stores the arguments that would otherwise have to be passed manually to CMake.

For example, the current Windows configuration corresponds approximately to:

```cmd
cmake -S . -B build/native ^
  -G "Visual Studio 17 2022" ^
  -A x64 ^
  -T "v142,host=x64,cuda=11.8" ^
  -DCMAKE_CUDA_ARCHITECTURES=61
```

The preset stores these settings under a short name.

Available presets can be listed with:

```cmd
cmake --list-presets
```

## Configure

Configure the native build with:

```cmd
cmake --preset windows-cuda
```

This creates the build tree under:

```text
build/native/
```

## Debug build

Build the Debug configuration with:

```cmd
cmake --build --preset windows-cuda-debug
```

Run the native tests with:

```cmd
ctest --preset windows-cuda-debug
```

Expected result:

```text
100% tests passed
```

## Release build

Build the Release configuration with:

```cmd
cmake --build --preset windows-cuda-release
```

Run the Release tests with:

```cmd
ctest --preset windows-cuda-release
```

## Running selected tests

CUDA tests are labeled using CTest labels.

For example:

```cmd
ctest --test-dir build/native -C Debug -L cuda --output-on-failure
```

The initial smoke test is labeled:

```text
cuda
gpu
smoke
```

## CUDA architecture

The current development GPU is a GTX 1070, which uses CUDA compute capability 6.1.

The Windows development preset therefore currently builds for:

```text
sm_61
```

The CUDA architecture is a build configuration property and should not be hard-coded into solver source code.

Future presets may target different GPUs, for example:

```text
local GTX 1070
cloud GPU
profiling GPU
CI GPU runner
```

without modifying the numerical implementation.

## Smoke test

The initial smoke kernel writes array indices to device memory.

It is intentionally simple. Its purpose is to verify:

- CUDA compilation;
- CUDA runtime linking;
- caller-owned device memory;
- caller-provided CUDA streams;
- asynchronous kernel execution;
- asynchronous memory copies;
- CUDA error propagation;
- CTest integration.

The smoke test uses the raw CUDA Runtime API directly so that the low-level allocation and execution mechanism remains visible.

Future production code will introduce higher-level C++ abstractions such as RAII-managed device buffers and streams.

## Development roadmap

The native backend will be developed incrementally.

The planned progression is approximately:

```text
CUDA build and test infrastructure
        |
        v
basic numerical integration on GPU
        |
        v
single-bubble Keller--Miksis solver
        |
        v
single-bubble profiling baseline
        |
        v
warp-sized coupled bubble solver
        |
        v
matrix-free coupled acceleration solve
        |
        v
larger cluster execution strategies
        |
        v
multi-stream / multi-GPU orchestration
        |
        v
PyTorch interoperability
```

Each major CUDA component should be validated against the Python/NumPy reference before additional optimization or parallelism is introduced.

## Future portability

The current Windows environment is primarily used for convenient local development.

A containerized Linux/CUDA development and benchmarking environment is planned for a later performance-analysis milestone. The container will provide a reproducible compiler, CUDA toolkit, CMake, and dependency environment for development across multiple machines and cloud GPUs.

Performance results will still record the underlying GPU, driver, host environment, compiler configuration, and relevant runtime settings separately.