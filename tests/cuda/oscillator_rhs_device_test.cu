#include <cmath>
#include <cstddef>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

#include <cuda_runtime_api.h>

#include "mbc/cuda/check.hpp"
#include "mbc/cuda/device_buffer.hpp"
#include "mbc/cuda/stream.hpp"
#include "mbc/models/harmonic_oscillator.hpp"
#include "mbc/ode/state.hpp"

#include "cuda_test_utils.hpp"
#include <algorithm>

__global__ void evaluate_rhs_kernel(const double* d_position,
                                    const double* d_velocity,
                                    const double* d_omega,
                                    double* d_dxdt,
                                    double* d_dvdt,
                                    std::size_t size) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (index >= size) {
        return;
    }

    mbc::ode::State<2> state{};
    state[0] = d_position[index];
    state[1] = d_velocity[index];

    mbc::models::HarmonicOscillatorParams params{};
    params.omega = d_omega[index];

    mbc::ode::State<2> derivative{};

    mbc::models::HarmonicOscillatorRhs rhs;
    rhs(derivative, 0.0, state, params);

    d_dxdt[index] = derivative[0];
    d_dvdt[index] = derivative[1];
}

void initialize_test_cases(std::vector<double>& position,
                           std::vector<double>& velocity,
                           std::vector<double>& omega) {
    const std::size_t problem_count = position.size();

    for (std::size_t i = 0; i < problem_count; ++i) {
        const double x = 1.0 + 0.01 * static_cast<double>(i);
        const double v = -0.5 * static_cast<double>(i);
        const double w = 1.0 + 0.001 * static_cast<double>(i);

        position[i] = x;
        velocity[i] = v;
        omega[i] = w;
    }
}

void compute_analytic_reference(const std::vector<double>& position,
                                const std::vector<double>& velocity,
                                const std::vector<double>& omega,
                                std::vector<double>& reference_dxdt,
                                std::vector<double>& reference_dvdt) {
    const std::size_t problem_count = position.size();

    for (std::size_t i = 0; i < problem_count; ++i) {
        reference_dxdt[i] = velocity[i];
        reference_dvdt[i] = -omega[i] * omega[i] * position[i];
    }
}

void evaluate_rhs_on_cpu(const std::vector<double>& position,
                         const std::vector<double>& velocity,
                         const std::vector<double>& omega,
                         std::vector<double>& cpu_dxdt,
                         std::vector<double>& cpu_dvdt) {
    const std::size_t problem_count = position.size();

    mbc::models::HarmonicOscillatorRhs rhs;

    for (std::size_t i = 0; i < problem_count; ++i) {
        mbc::ode::State<2> state{};
        state[0] = position[i];
        state[1] = velocity[i];

        mbc::models::HarmonicOscillatorParams params{};
        params.omega = omega[i];

        mbc::ode::State<2> derivative{};

        rhs(derivative, 0.0, state, params);

        cpu_dxdt[i] = derivative[0];
        cpu_dvdt[i] = derivative[1];
    }
}

void evaluate_rhs_on_gpu(const std::vector<double>& position,
                         const std::vector<double>& velocity,
                         const std::vector<double>& omega,
                         std::vector<double>& gpu_dxdt,
                         std::vector<double>& gpu_dvdt) {
    // DeviceBuffer allocation
    std::size_t size = position.size();
    mbc::cuda::DeviceBuffer<double> d_position(0, size);
    mbc::cuda::DeviceBuffer<double> d_velocity(0, size);
    mbc::cuda::DeviceBuffer<double> d_omega(0, size);
    mbc::cuda::DeviceBuffer<double> d_gpu_dxdt(0, size);
    mbc::cuda::DeviceBuffer<double> d_gpu_dvdt(0, size);

    mbc::cuda::Stream stream(0);
    // host -> device copies
    MBC_CUDA_CHECK(cudaMemcpyAsync(d_position.data(),
                                   position.data(),
                                   size * sizeof(double),
                                   cudaMemcpyHostToDevice,
                                   stream.get()));

    MBC_CUDA_CHECK(cudaMemcpyAsync(d_velocity.data(),
                                   velocity.data(),
                                   size * sizeof(double),
                                   cudaMemcpyHostToDevice,
                                   stream.get()));

    MBC_CUDA_CHECK(cudaMemcpyAsync(d_omega.data(),
                                   omega.data(),
                                   size * sizeof(double),
                                   cudaMemcpyHostToDevice,
                                   stream.get()));
    // kernel launch
    constexpr int block_size = 256;

    const int grid_size = static_cast<int>((size + block_size - 1) / block_size);

    evaluate_rhs_kernel<<<grid_size, block_size, 0, stream.get()>>>(d_position.data(),
                                                                    d_velocity.data(),
                                                                    d_omega.data(),
                                                                    d_gpu_dxdt.data(),
                                                                    d_gpu_dvdt.data(),
                                                                    size);

    MBC_CUDA_CHECK(cudaGetLastError());
    // device -> host copies
    MBC_CUDA_CHECK(cudaMemcpyAsync(gpu_dxdt.data(),
                                   d_gpu_dxdt.data(),
                                   size * sizeof(double),
                                   cudaMemcpyDeviceToHost,
                                   stream.get()));

    MBC_CUDA_CHECK(cudaMemcpyAsync(gpu_dvdt.data(),
                                   d_gpu_dvdt.data(),
                                   size * sizeof(double),
                                   cudaMemcpyDeviceToHost,
                                   stream.get()));
    // stream synchronization
    stream.synchronize();
}

void check_close(double actual,
                 double expected,
                 std::size_t index,
                 const char* quantity,
                 double rtol = 1.0e-12,
                 double atol = 1.0e-12) {
    const double error = std::abs(actual - expected);
    const double tol = atol + rtol * std::max(std::abs(actual), std::abs(expected));

    if (error > tol) {
        std::ostringstream message;

        message << quantity << " mismatch at index " << index << ": expected "
                << expected << ", got " << actual << ", error " << error
                << ", tolerance " << tol;

        throw std::runtime_error(message.str());
    }
}

void validate_results(const std::vector<double>& reference_dxdt,
                      const std::vector<double>& reference_dvdt,
                      const std::vector<double>& cpu_dxdt,
                      const std::vector<double>& cpu_dvdt,
                      const std::vector<double>& gpu_dxdt,
                      const std::vector<double>& gpu_dvdt) {
    const std::size_t problem_count = reference_dxdt.size();

    for (std::size_t i = 0; i < problem_count; ++i) {
        check_close(
            cpu_dxdt[i], reference_dxdt[i], i, "CPU dx/dt vs analytic reference");

        check_close(
            cpu_dvdt[i], reference_dvdt[i], i, "CPU dv/dt vs analytic reference");

        check_close(
            gpu_dxdt[i], reference_dxdt[i], i, "GPU dx/dt vs analytic reference");

        check_close(
            gpu_dvdt[i], reference_dvdt[i], i, "GPU dv/dt vs analytic reference");

        check_close(gpu_dxdt[i], cpu_dxdt[i], i, "GPU dx/dt vs CPU RHS");

        check_close(gpu_dvdt[i], cpu_dvdt[i], i, "GPU dv/dt vs CPU RHS");
    }
}

void test_harmonic_oscillator_rhs_gpu() {
    constexpr std::size_t problem_count = 1024;

    std::vector<double> position(problem_count);
    std::vector<double> velocity(problem_count);
    std::vector<double> omega(problem_count);

    std::vector<double> reference_dxdt(problem_count);
    std::vector<double> reference_dvdt(problem_count);

    std::vector<double> cpu_dxdt(problem_count);
    std::vector<double> cpu_dvdt(problem_count);

    std::vector<double> gpu_dxdt(problem_count);
    std::vector<double> gpu_dvdt(problem_count);

    initialize_test_cases(position, velocity, omega);

    compute_analytic_reference(
        position, velocity, omega, reference_dxdt, reference_dvdt);

    evaluate_rhs_on_cpu(position, velocity, omega, cpu_dxdt, cpu_dvdt);

    evaluate_rhs_on_gpu(position, velocity, omega, gpu_dxdt, gpu_dvdt);

    validate_results(
        reference_dxdt, reference_dvdt, cpu_dxdt, cpu_dvdt, gpu_dxdt, gpu_dvdt);
}

int main() {
    try {
        if (!mbc::test::cuda_device_available()) {
            std::cout << "CUDA device unavailable; skipping test.\n";
            return 77;
        }

        test_harmonic_oscillator_rhs_gpu();

        std::cout << "Harmonic oscillator RHS GPU test passed.\n";

        return 0;

    } catch (const std::exception& error) {
        std::cerr << "Harmonic oscillator RHS GPU test failed:\n"
                  << error.what() << '\n';

        return 1;
    }
}