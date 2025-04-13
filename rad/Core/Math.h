#pragma once

#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif

#include <rad/Core/Platform.h>
#include <cassert>
#include <cmath>
#include <numbers>
// e = e_v<double>;
// log2e = log2e_v<double>;
// log10e = log10e_v<double>;
// pi = pi_v<double>;
// inv_pi = inv_pi_v<double>; // 1/pi
// inv_sqrtpi = inv_sqrtpi_v<double>; // 1/sqrt(pi)
// ln2 = ln2_v<double>;
// ln10 = ln10_v<double>;
// sqrt2 = sqrt2_v<double>;
// sqrt3 = sqrt3_v<double>;
// inv_sqrt3 = inv_sqrt3_v<double>; // 1/sqrt(3)
// egamma = egamma_v<double>; // the Euler-Mascheroni constant: https://en.wikipedia.org/wiki/Euler%27s_constant
// phi = phi_v<double>; // the golden ratio: (1+sqrt(5))/2 = 1.618033988749...

namespace rad
{

// The Right Way to Calculate Stuff: https://www.plunk.org/~hatch/rightway.html

template <typename T>
inline constexpr T Sqr(T x)
{
    return x * x;
}

// Finds solutions of the quadratic equation at^2 + bt + c = 0; return true if solutions were found.
// https://pbr-book.org/3ed-2018/Utilities/Mathematical_Routines
// https://github.com/mmp/pbrt-v4/blob/master/src/pbrt/util/math.h
template<typename Float>
inline bool SolveQuadratic(Float a, Float b, Float c, Float& t0, Float& t1)
{
    // Handle case of $a=0$ for quadratic solution
    if (a == 0) [[unlikely]]
    {
        if (b == 0) [[unlikely]]
        {
            return false;
        }
        t0 = t1 = -c / b;
        return true;
    }
    // Find quadratic discriminant: b^2 - 4ac
    Float discrim = b * b - 4 * a * c;
    if (discrim < 0)
    {
        return false;
    }
    Float rootDiscrim = std::sqrt(discrim);
    // Compute quadratic _t_ values
    Float q = -0.5f * (b + std::copysign(rootDiscrim, b));
    t0 = q / a;
    t1 = c / q;
    if (t0 > t1)
    {
        std::swap(t0, t1);
    }
    return true;
}

} // namespace rad
