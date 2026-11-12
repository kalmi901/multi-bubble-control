#pragma once

#include <cstddef>

#include "mbc/cuda/qualifiers.hpp"

namespace mbc::ode {

template <std::size_t N> struct State {
    static_assert(N > 0, "State dimension must be positive");

    double values[N]{};

    MBC_HOST_DEVICE double& operator[](std::size_t index) noexcept {
        return values[index];
    }

    MBC_HOST_DEVICE const double& operator[](std::size_t index) const noexcept {
        return values[index];
    }

    static constexpr std::size_t size() noexcept {
        return N;
    }
};

} // namespace mbc::ode
