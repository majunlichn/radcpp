#pragma once

#include <rad/Common/Platform.h>
#include <rad/Common/Integer.h>
#include <random>

namespace rad
{

// https://www.jcgt.org/published/0009/03/02/
// https://www.pcg-random.org/
inline uint32_t pcg(uint32_t v)
{
    uint32_t state = v * 747796405u + 2891336453u;
    uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

} // namespace rad
