#pragma once

#include "mbc/cuda/qualifiers.hpp"
#include "mbc/ode/state.hpp"

namespace mbc::models {

struct HarmonicOscillatorParams {
    double omega;
};

struct HarmonicOscillatorRhs {
    MBC_HOST_DEVICE void
    operator()(mbc::ode::State<2>& derivative,
               [[maybe_unused]] double time,
               const mbc::ode::State<2>& state,
               const HarmonicOscillatorParams& params) const noexcept {
        derivative[0] = state[1];
        derivative[1] = -params.omega * params.omega * state[0];
    }
};

} // namespace mbc::models