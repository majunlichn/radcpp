#pragma once

#include <rad/Core/Integer.h>

#include <xxhash.h>

namespace rad
{

// For general-purpose, non-cryptographic hashing.
// https://www.jcgt.org/published/0009/03/02/
// https://www.pcg-random.org/
inline uint32_t pcg_hash32(uint32_t v)
{
    uint32_t state = v * 747796405u + 2891336453u;
    uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// For general-purpose, non-cryptographic hashing.
inline uint64_t pcg_hash64(uint64_t v)
{
    uint64_t state = v * 6364136223846793005ull + 1442695040888963407ull;
    uint64_t word = ((state >> ((state >> 59u) + 5u)) ^ state) * 12605985483714917081ull;
    return (word >> 43u) ^ word;
}

} // namespace rad
