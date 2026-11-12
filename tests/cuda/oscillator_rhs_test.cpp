#include <cstddef>
#include <iostream>
#include <stdexcept>

#include "mbc/models/harmonic_oscillator.hpp"
#include "mbc/ode/state.hpp"

void test_state() {
    mbc::ode::State<2> state{};
    static_assert(mbc::ode::State<2>::size() == 2);

    if (state[0] != 0.0 || state[1] != 0.0) {
        throw std::runtime_error("State was not zero initialized");
    }

    state[0] = 2.0;
    state[1] = -3.0;

    if (state[0] != 2 || state[1] != -3.0) {
        throw std::runtime_error("State access failed");
    }
}

void test_rhs() {
    mbc::ode::State<2> state{2.0, -3.0};

    mbc::models::HarmonicOscillatorParams params{};
    params.omega = 4.0;

    mbc::ode::State<2> derivative{};
    mbc::models::HarmonicOscillatorRhs rhs;
    rhs(derivative, 0.0, state, params);

    if (derivative[0] != -3.0) {
        throw std::runtime_error("Incorrect position derivative");
    }

    if (derivative[1] != -32.0) {
        throw std::runtime_error("Incorrect velocity derivative");
    }

    params.omega = 0.0;
    rhs(derivative, 0.0, state, params);

    if (derivative[0] != -3.0) {
        throw std::runtime_error("Incorrect position derivative");
    }

    if (derivative[1] != 0.0) {
        throw std::runtime_error("Incorrect velocity derivative");
    }
}

int main() {
    try {
        test_state();
        test_rhs();

        std::cout << "Harmonic Oscillator RHS CPU test passed.\n";

        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Harmonic Oscillator RHS CPU test failed:\n"
                  << error.what() << '\n';

        return 1;
    }
}
