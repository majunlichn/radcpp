#pragma once

#include <radcpp/Core/Platform.h>
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

// The Right Way to Calculate Stuff: https://www.plunk.org/~hatch/rightway.html
template<typename T>
inline T OneMinusCosx(T x)
{
    return T(2) * Sqr(std::sin(x / T(2)));
}

template<typename T>
inline T OneMinusCosx_OverX(T x)
{
    if (T(1) + x * x == T(1)) [[unlikely]]
    {
        return T(0.5) * x;
    }
    else [[likely]]
    {
        return OneMinusCosx(x) / x;
    }
}

template<typename T>
inline T SinXOverX(T x)
{
    if (T(1) + x * x == T(1)) [[unlikely]]
    {
        return T(1);
    }
    else [[likely]]
    {
        return std::sin(x) / x;
    }
}

template<typename T>
T Sinh(T x)
{
    T u = std::expm1(x);
    return T(0.5) * u / (u + T(1)) * (u + T(2));
}

template<typename T>
T SinhInverse(T x)
{
    return std::log1p(x * (T(1) + x / (std::sqrt(x * x + T(1)) + T(1))));
}

// https://stackoverflow.com/a/10792321
// max. rel. error <= 1.73e-3 on [-87,88]
inline float FastExp(float x)
{
    volatile union {
        float f;
        unsigned int i;
    } cvt;

    /* exp(x) = 2^i * 2^f; i = floor (log2(e) * x), 0 <= f <= 1 */
    float t = x * 1.442695041f;
    float fi = floorf(t);
    float f = t - fi;
    int i = (int)fi;
    cvt.f = (0.3371894346f * f + 0.657636276f) * f + 1.00172476f; /* compute 2^f */
    cvt.i += (i << 23);                                          /* scale by 2^i */
    return cvt.f;
}

// https://en.wikipedia.org/wiki/Gaussian_function
// https://github.com/mmp/pbrt-v4/blob/master/src/pbrt/util/math.h
template <typename Float>
inline Float Gaussian(Float x, Float mu = 0, Float sigma = 1)
{
    constexpr Float Pi = std::numbers::pi_v<Float>;
    return 1 / std::sqrt(2 * Pi * sigma * sigma) *
        std::exp(-Sqr(x - mu) / (2 * sigma * sigma));
}

template <typename Float>
inline Float GaussianIntegral(Float x0, Float x1, Float mu = 0, Float sigma = 1)
{
    assert(sigma > 0);
    Float sigmaRoot2 = sigma * Float(1.414213562373095);
    return Float(0.5) * (std::erf((mu - x0) / sigmaRoot2) - std::erf((mu - x1) / sigmaRoot2));
}

} // namespace rad
