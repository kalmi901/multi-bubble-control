# Native CUDA Backend

This directory contains the native C++/CUDA backend for **multi-bubble-control**.

The existing Python/NumPy implementation remains the canonical physical and numerical reference. The native backend is developed independently with a focus on GPU performance, scalability, profiling, and future interoperability with PyTorch.

The CUDA backend is intentionally not a line-by-line translation of the Python implementation. Physical models and numerical results should remain consistent, while data layout, memory management, and execution strategy may differ substantially.

## Current status

The native backend currently provides the initial CUDA build, runtime infrastructure, and test foundation.

The implemented infrastructure includes:

- first-class CUDA support through CMake;
- CUDA runtime error handling;
- RAII-managed device memory;
- RAII-managed non-blocking CUDA streams;
- scoped CUDA device selection;
- non-owning device memory views;
- raw CUDA smoke testing;
- unit tests for CUDA resource wrappers;
- an integration test executing a real CUDA kernel through the wrapper infrastructure.

No production bubble dynamics or numerical integration algorithms are implemented in the native backend yet.

The current execution path is approximately:

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
DeviceBuffer + DeviceView + Stream
  |
  v
CUDA kernel launch
  |
  v
asynchronous device-to-host copy
  |
  v
stream synchronization
  |
  v
CTest validation
```

## Project structure

```text
native/
├── CMakeLists.txt
│
├── include/
│   └── mbc/
│       └── cuda/
│           ├── check.hpp
│           ├── device_buffer.hpp
│           ├── device_guard.hpp
│           ├── device_view.hpp
│           ├── smoke.hpp
│           └── stream.hpp
│
└── src/
    └── cuda/
        └── smoke.cu
```

Native CUDA tests are located under:

```text
tests/cuda/
├── CMakeLists.txt
├── device_buffer_test.cpp
├── device_view_test.cpp
├── infrastructure_test.cpp
├── smoke_test.cpp
└── stream_test.cpp
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
- CUDA resources use explicit ownership semantics.
- Solver APIs should operate primarily on caller-owned device memory.
- CUDA execution should support caller-provided streams.
- Core solver functions should avoid implicit global synchronization.
- Device-local operations should preserve the caller's previously selected CUDA device.
- Non-owning memory views should allow future interoperability with external allocations such as PyTorch tensors.
- Temporary solver memory should eventually be managed through reusable workspaces.
- Future PyTorch integration should wrap the CUDA core rather than duplicate it.
- Independent simulation environments should remain independently schedulable to support future multi-stream and multi-GPU execution.

The current infrastructure is intentionally small and explicit so that later solver development can build on a clear ownership and execution model.

## CUDA infrastructure

### Error handling

CUDA runtime calls that are allowed to propagate errors use:

```cpp
MBC_CUDA_CHECK(...)
```

defined in:

```text
check.hpp
```

CUDA errors are converted to `std::runtime_error` exceptions containing the failing expression, CUDA error string, source file, and source line.

Destructors and cleanup paths are `noexcept` and therefore use explicit CUDA error-code handling instead of throwing exceptions.

### DeviceGuard

`DeviceGuard` temporarily selects a CUDA device and restores the previously active device when the guard leaves scope.

Conceptually:

```text
current device = A

{
    DeviceGuard guard(B);

    current device = B
    ...
}

current device = A
```

This prevents resource construction and explicit device-local operations from unexpectedly changing CUDA device state for the calling thread.

### DeviceBuffer

`DeviceBuffer<T>` is a move-only RAII wrapper around CUDA device memory.

It owns:

```text
device pointer
element count
device ID
```

Allocation is performed with `cudaMalloc`, and owned memory is released automatically when the object is destroyed.

Copy operations are disabled to prevent multiple objects from owning the same allocation.

Move construction and move assignment transfer ownership.

Example:

```cpp
mbc::cuda::DeviceBuffer<float> buffer(
    0,
    4096
);
```

### DeviceView

`DeviceView<T>` is a lightweight non-owning view of device memory.

It stores only:

```text
pointer
element count
```

It performs no allocation and no deallocation.

This distinction allows memory ownership to remain separate from solver APIs.

A view can be constructed safely from a `DeviceBuffer` using:

```cpp
auto view =
    mbc::cuda::make_device_view(buffer);
```

For const buffers, the helper produces:

```cpp
DeviceView<const T>
```

This provides const-correct read-only access where appropriate.

The same abstraction can later wrap externally owned memory such as:

```text
PyTorch CUDA tensor
external CUDA allocation
solver workspace slice
```

without transferring ownership.

### Stream

`Stream` is a move-only RAII wrapper around a non-blocking CUDA stream.

Example:

```cpp
mbc::cuda::Stream stream(0);
```

The underlying `cudaStream_t` can be passed to asynchronous CUDA operations using:

```cpp
stream.get()
```

Explicit synchronization is available through:

```cpp
stream.synchronize();
```

Resource destruction preserves the previously active CUDA device.

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

The initial local development environment is:

```text
Windows 10 x64
Visual Studio 2022 Build Tools
MSVC v142 / 19.29
CUDA Toolkit 11.8
CMake 3.31+
NVIDIA GTX 1070
CUDA architecture: sm_61
```

CUDA Toolkit 11.8 requires a compatible MSVC host compiler. On the current development system, the VS2019-compatible `v142` toolset is installed side-by-side with the newer Visual Studio 2022 toolchain.

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

A preset does not introduce a new build system. It stores the arguments that would otherwise have to be passed manually to CMake.

The current Windows configuration corresponds approximately to:

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

## Release build

Build the Release configuration with:

```cmd
cmake --build --preset windows-cuda-release
```

Run the Release tests with:

```cmd
ctest --preset windows-cuda-release
```

## Clean validation

Before completing a native development milestone, the build can be validated from a clean build directory:

```cmd
rmdir /S /Q build\native

cmake --preset windows-cuda

cmake --build --preset windows-cuda-debug
ctest --preset windows-cuda-debug

cmake --build --preset windows-cuda-release
ctest --preset windows-cuda-release
```

This avoids accidentally depending on stale CMake cache entries or previously generated build artifacts.

## Native CUDA tests

The current native test suite contains five CTest targets.

### `cuda_smoke`

The raw CUDA smoke test exercises the CUDA Runtime API directly:

```text
cudaStreamCreateWithFlags
cudaMalloc
cudaMallocHost
CUDA kernel launch
cudaMemcpyAsync
cudaStreamSynchronize
cudaFree
cudaFreeHost
cudaStreamDestroy
```

It intentionally retains the low-level CUDA API so that the underlying resource and execution model remains visible.

### `cuda_device_buffer`

Unit tests for `DeviceBuffer<T>` validate:

- device allocation;
- reported size and device ID;
- move construction;
- move assignment;
- transferred ownership;
- reset state of moved-from buffers.

### `cuda_stream`

Unit tests for `Stream` validate:

- stream creation;
- device association;
- move construction;
- move assignment;
- transferred ownership;
- explicit stream synchronization.

### `cuda_device_view`

Unit tests for `DeviceView<T>` validate:

- default empty views;
- pointer and size storage;
- copy behavior;
- creation from `DeviceBuffer<T>`;
- creation of const views from const buffers.

### `cuda_infrastructure`

The infrastructure integration test executes a real CUDA kernel through the new wrapper layer.

The tested path is:

```text
DeviceBuffer<float>
        |
        v
make_device_view()
        |
        v
DeviceView<float>
        |
        v
fill_index_async()
        |
        v
cudaMemcpyAsync()
        |
        v
Stream::synchronize()
        |
        v
host-side result validation
```

This test verifies that the individual infrastructure components work together in an actual CUDA execution path.

## Running selected tests

CUDA tests are labeled using CTest labels.

Run all CUDA tests:

```cmd
ctest --test-dir build/native -C Debug -L cuda --output-on-failure
```

Run unit tests:

```cmd
ctest --test-dir build/native -C Debug -L unit --output-on-failure
```

Run integration tests:

```cmd
ctest --test-dir build/native -C Debug -L integration --output-on-failure
```

The test labels currently distinguish:

```text
cuda
gpu
smoke
unit
integration
```

Systems without a usable CUDA device may skip GPU-dependent tests using return code `77`.

## CUDA architecture

The current development GPU is a GTX 1070, which uses CUDA compute capability 6.1.

The Windows development preset therefore currently builds for:

```text
sm_61
```

The CUDA architecture is a build configuration property and is not hard-coded into solver source code.

Future presets may target different GPUs, for example:

```text
local development GPU
profiling GPU
cloud GPU
CI GPU runner
```

without modifying numerical implementation code.

## Current milestone boundary

The current native infrastructure milestone intentionally stops before implementing production numerical methods.

Implemented:

```text
CUDA build system
CUDA error handling
device selection guard
device memory ownership
device memory views
CUDA stream ownership
raw CUDA smoke test
resource unit tests
wrapper-based kernel integration test
```

Not yet implemented:

```text
RKCK integration
Keller--Miksis dynamics
solver workspaces
coupled bubble dynamics
collision handling
performance kernels
multi-stream scheduling
multi-GPU orchestration
PyTorch integration
```

These belong to later CUDA backend milestones.

## Development roadmap

The native backend will be developed incrementally.

The planned progression is approximately:

```text
CUDA infrastructure
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

A containerized Linux/CUDA development and benchmarking environment is planned for a later performance-analysis milestone. The container will provide a reproducible compiler, CUDA toolkit, CMake, and dependency environment across multiple machines and cloud GPUs.

Performance results will still record the underlying GPU, driver, host environment, compiler configuration, and relevant runtime settings separately.