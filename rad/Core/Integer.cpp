#include <rad/Core/Integer.h>

namespace rad
{

uint32_t BitScanReverse32(uint32_t mask)
{
    assert(mask != 0);
#if defined(__cpp_lib_bitops)
    return static_cast<uint32_t>(31 - std::countl_zero(mask));
#elif defined(_WIN32)
    static_assert(sizeof(uint32_t) == sizeof(unsigned long));
    unsigned long index = 0;
    _BitScanReverse(&index, mask);
    return static_cast<uint32_t>(index);
#elif defined(RAD_COMPILER_GCC)
    static_assert(sizeof(uint32_t) == sizeof(unsigned long));
    return static_cast<uint32_t>(31 - __builtin_clzl(mask));
#else // fallback
    uint32_t index = 31u;
    mask |= 0x1; // index=0 if mask=0
    while (((mask >> index) & 0x1) == 0)
    {
        --index;
    }
    return index;
#endif
}

uint32_t BitScanReverse64(uint64_t mask)
{
    assert(mask != 0);
#if defined(__cpp_lib_bitops)
    return static_cast<uint32_t>(63 - std::countl_zero(mask));
#elif defined(_WIN64)
    static_assert(sizeof(uint64_t) == sizeof(unsigned long long));
    unsigned long index = 0;
    _BitScanReverse64(&index, mask);
    return static_cast<uint32_t>(index);
#elif defined(_WIN32)
    static_assert(sizeof(uint32_t) == sizeof(unsigned long));
    static_assert(sizeof(uint64_t) == sizeof(unsigned long long));
    unsigned long index = 0;
    const uint32_t highPart = HighPart64(mask);
    _BitScanReverse(&index, (highPart != 0) ? highPart : LowPart64(mask));
    return (highPart != 0) ? (index + 32u) : index;
#elif defined(RAD_COMPILER_GCC)
    static_assert(sizeof(uint64_t) == sizeof(unsigned long long));
    return static_cast<uint32_t>(63 - __builtin_clzll(static_cast<uint64_t>(mask)));
#else // fallback
    uint32_t index = 63u;
    mask |= 0x1; // index=0 if mask=0
    while (((mask >> index) & 0x1) == 0)
    {
        --index;
    }
    return index;
#endif
}

uint32_t CountBits32(uint32_t x)
{
#if defined(__cpp_lib_bitops)
    return std::popcount(x);
#else // fallback
    // https://en.wikipedia.org/wiki/Hamming_weight
    x = x - ((x >> 1) & 0x55555555);
    x = (x & 0x33333333) + ((x >> 2) & 0x33333333);
    x = (((x + (x >> 4)) & 0x0F0F0F0F) * 0x01010101) >> ((sizeof(uint32_t) - 1) << 3);
    return x;
#endif
}

uint32_t CountBits64(uint64_t x)
{
#if defined(__cpp_lib_bitops)
    return std::popcount(x);
#else // fallback
    // https://en.wikipedia.org/wiki/Hamming_weight
    const uint64_t m1 = 0x5555555555555555;     // binary: 0101...
    const uint64_t m2 = 0x3333333333333333;     // binary: 00110011..
    const uint64_t m4 = 0x0f0f0f0f0f0f0f0f;     // binary:  4 zeros,  4 ones ...
    const uint64_t m8 = 0x00ff00ff00ff00ff;     // binary:  8 zeros,  8 ones ...
    const uint64_t m16 = 0x0000ffff0000ffff;    // binary: 16 zeros, 16 ones ...
    const uint64_t m32 = 0x00000000ffffffff;    // binary: 32 zeros, 32 ones
    const uint64_t h01 = 0x0101010101010101;    // the sum of 256 to the power of 0,1,2,3,...
    x = x - (x >> 1) & m1;          // put count of each 2 bits into those 2 bits
    x = (x & m2) + ((x >> 2) & m2); // put count of each 4 bits into those 4 bits
    x = (x + (x >> 4)) & m4;        // put count of each 8 bits into those 8 bits
    x = (x * h01) >> 56;            // returns left 8 bits of x + (x<<8) + (x<<16) + (x<<24) + ...
    return static_cast<uint32_t>(x);
#endif
}

} // namespace rad
