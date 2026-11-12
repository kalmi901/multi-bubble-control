#pragma once

#if defined(__CUDACC__)
#define MBC_HOST_DEVICE __host__ __device__
#else
#define MBC_HOST_DEVICE
#endif