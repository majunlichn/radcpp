#pragma once

#include <bit>
#include <concepts>
#include <cstdint>
#include <iosfwd>
#include <limits>
#include <type_traits>

namespace rad
{

// https://github.com/pytorch/pytorch/blob/main/torch/headeronly/util/Float8_e4m3fn.h
// https://github.com/pytorch/pytorch/blob/main/torch/headeronly/util/Float8_e5m2.h

[[nodiscard]] constexpr float fp8e4m3fn_to_fp32_value(std::uint8_t input) noexcept
{
    const std::uint32_t sign = static_cast<std::uint32_t>(input & 0x80u) << 24u;
    const std::uint32_t magnitude = input & 0x7Fu;
    const std::uint32_t exponent = magnitude >> 3u;
    const std::uint32_t mantissa = magnitude & 0x07u;

    if (magnitude == 0)
    {
        return std::bit_cast<float>(sign);
    }

    if (magnitude == 0x7Fu)
    {
        return std::bit_cast<float>(sign | 0x7F800000u | (mantissa << 20u));
    }

    if (exponent == 0)
    {
        const float value = static_cast<float>(mantissa) * 0x1p-9f;
        return sign == 0 ? value : -value;
    }

    const std::uint32_t result = sign | ((exponent + 120u) << 23u) | (mantissa << 20u);
    return std::bit_cast<float>(result);
}

[[nodiscard]] constexpr std::uint8_t fp8e4m3fn_from_fp32_value(float value) noexcept
{
    constexpr std::uint32_t fp32Infinity = 0x7F800000u;
    constexpr std::uint32_t overflowThreshold = 1087u << 20u;
    constexpr std::uint32_t minimumNormal = 121u << 23u;
    constexpr std::uint32_t denormalBias = 141u << 23u;
    constexpr std::uint32_t exponentBiasAdjustment = 0xC4000000u;

    std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    const std::uint8_t sign = static_cast<std::uint8_t>((bits >> 24u) & 0x80u);
    bits &= 0x7FFFFFFFu;

    std::uint8_t result = 0;
    if (bits >= overflowThreshold)
    {
        result = bits > fp32Infinity ? 0x7Fu : 0x7Eu;
    }
    else if (bits < minimumNormal)
    {
        const float biasedValue = std::bit_cast<float>(bits) + std::bit_cast<float>(denormalBias);
        result =
            static_cast<std::uint8_t>(std::bit_cast<std::uint32_t>(biasedValue) - denormalBias);
    }
    else
    {
        const std::uint32_t mantissaOdd = (bits >> 20u) & 1u;
        bits += exponentBiasAdjustment + 0x7FFFFu + mantissaOdd;
        result = static_cast<std::uint8_t>(bits >> 20u);
        if (result == 0x7Fu)
        {
            result = 0x7Eu;
        }
    }

    return static_cast<std::uint8_t>(result | sign);
}

[[nodiscard]] constexpr float fp8e5m2_to_fp32_value(std::uint8_t input) noexcept
{
    const std::uint32_t sign = static_cast<std::uint32_t>(input & 0x80u) << 24u;
    const std::uint32_t magnitude = input & 0x7Fu;
    const std::uint32_t exponent = magnitude >> 2u;
    const std::uint32_t mantissa = magnitude & 0x03u;

    if (magnitude == 0)
    {
        return std::bit_cast<float>(sign);
    }

    if (exponent == 0x1Fu)
    {
        if (mantissa == 0)
        {
            return std::bit_cast<float>(sign | 0x7F800000u);
        }

        return std::bit_cast<float>(sign | 0x7F800000u | (mantissa << 21u));
    }

    if (exponent == 0)
    {
        const float value = static_cast<float>(mantissa) * 0x1p-16f;
        return sign == 0 ? value : -value;
    }

    const std::uint32_t result = sign | ((exponent + 112u) << 23u) | (mantissa << 21u);
    return std::bit_cast<float>(result);
}

[[nodiscard]] constexpr std::uint8_t fp8e5m2_from_fp32_value(float value) noexcept
{
    constexpr std::uint32_t fp32Infinity = 0x7F800000u;
    constexpr std::uint32_t overflowThreshold = 143u << 23u;
    constexpr std::uint32_t minimumNormal = 113u << 23u;
    constexpr std::uint32_t denormalBias = 134u << 23u;
    constexpr std::uint32_t exponentBiasAdjustment = 0xC8000000u;

    std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    const std::uint8_t sign = static_cast<std::uint8_t>((bits >> 24u) & 0x80u);
    bits &= 0x7FFFFFFFu;

    std::uint8_t result = 0;
    if (bits >= overflowThreshold)
    {
        result = bits > fp32Infinity ? 0x7Fu : 0x7Cu;
    }
    else if (bits < minimumNormal)
    {
        const float biasedValue = std::bit_cast<float>(bits) + std::bit_cast<float>(denormalBias);
        result =
            static_cast<std::uint8_t>(std::bit_cast<std::uint32_t>(biasedValue) - denormalBias);
    }
    else
    {
        const std::uint32_t mantissaOdd = (bits >> 21u) & 1u;
        bits += exponentBiasAdjustment + 0xFFFFFu + mantissaOdd;
        result = static_cast<std::uint8_t>(bits >> 21u);
    }

    return static_cast<std::uint8_t>(result | sign);
}

class alignas(1) Float8E4M3 final
{
public:
    constexpr Float8E4M3() noexcept = default;
    constexpr Float8E4M3(float value) noexcept :
        m_bits(fp8e4m3fn_from_fp32_value(value))
    {
    }

    [[nodiscard]] static constexpr Float8E4M3 FromBits(std::uint8_t bits) noexcept
    {
        Float8E4M3 result;
        result.m_bits = bits;
        return result;
    }

    [[nodiscard]] constexpr std::uint8_t Bits() const noexcept { return m_bits; }

    [[nodiscard]] constexpr operator float() const noexcept
    {
        return fp8e4m3fn_to_fp32_value(m_bits);
    }

    [[nodiscard]] constexpr bool IsNaN() const noexcept { return (m_bits & 0x7Fu) == 0x7Fu; }

    [[nodiscard]] constexpr bool IsInfinite() const noexcept { return false; }

private:
    std::uint8_t m_bits = 0;
}; // class Float8E4M3

class alignas(1) Float8E5M2 final
{
public:
    constexpr Float8E5M2() noexcept = default;
    constexpr Float8E5M2(float value) noexcept :
        m_bits(fp8e5m2_from_fp32_value(value))
    {
    }

    [[nodiscard]] static constexpr Float8E5M2 FromBits(std::uint8_t bits) noexcept
    {
        Float8E5M2 result;
        result.m_bits = bits;
        return result;
    }

    [[nodiscard]] constexpr std::uint8_t Bits() const noexcept { return m_bits; }

    [[nodiscard]] constexpr operator float() const noexcept
    {
        return fp8e5m2_to_fp32_value(m_bits);
    }

    [[nodiscard]] constexpr bool IsNaN() const noexcept { return (m_bits & 0x7Fu) > 0x7Cu; }

    [[nodiscard]] constexpr bool IsInfinite() const noexcept { return (m_bits & 0x7Fu) == 0x7Cu; }

private:
    std::uint8_t m_bits = 0;
}; // class Float8E5M2

static_assert(sizeof(Float8E4M3) == sizeof(std::uint8_t));
static_assert(sizeof(Float8E5M2) == sizeof(std::uint8_t));

namespace detail
{

template <typename T>
concept Float8Type =
    std::same_as<std::remove_cv_t<T>, Float8E4M3> || std::same_as<std::remove_cv_t<T>, Float8E5M2>;

} // namespace detail

template <detail::Float8Type T>
[[nodiscard]] constexpr T operator+(T lhs, T rhs) noexcept
{
    return static_cast<float>(lhs) + static_cast<float>(rhs);
}

template <detail::Float8Type T>
[[nodiscard]] constexpr T operator-(T lhs, T rhs) noexcept
{
    return static_cast<float>(lhs) - static_cast<float>(rhs);
}

template <detail::Float8Type T>
[[nodiscard]] constexpr T operator*(T lhs, T rhs) noexcept
{
    return static_cast<float>(lhs) * static_cast<float>(rhs);
}

template <detail::Float8Type T>
[[nodiscard]] constexpr T operator/(T lhs, T rhs) noexcept
{
    return static_cast<float>(lhs) / static_cast<float>(rhs);
}

template <detail::Float8Type T>
[[nodiscard]] constexpr T operator-(T value) noexcept
{
    return T::FromBits(static_cast<std::uint8_t>(value.Bits() ^ 0x80u));
}

template <detail::Float8Type T>
constexpr T& operator+=(T& lhs, T rhs) noexcept
{
    return lhs = lhs + rhs;
}

template <detail::Float8Type T>
constexpr T& operator-=(T& lhs, T rhs) noexcept
{
    return lhs = lhs - rhs;
}

template <detail::Float8Type T>
constexpr T& operator*=(T& lhs, T rhs) noexcept
{
    return lhs = lhs * rhs;
}

template <detail::Float8Type T>
constexpr T& operator/=(T& lhs, T rhs) noexcept
{
    return lhs = lhs / rhs;
}

template <detail::Float8Type T, std::floating_point F>
[[nodiscard]] constexpr F operator+(T lhs, F rhs) noexcept
{
    return static_cast<F>(static_cast<float>(lhs)) + rhs;
}

template <detail::Float8Type T, std::floating_point F>
[[nodiscard]] constexpr F operator-(T lhs, F rhs) noexcept
{
    return static_cast<F>(static_cast<float>(lhs)) - rhs;
}

template <detail::Float8Type T, std::floating_point F>
[[nodiscard]] constexpr F operator*(T lhs, F rhs) noexcept
{
    return static_cast<F>(static_cast<float>(lhs)) * rhs;
}

template <detail::Float8Type T, std::floating_point F>
[[nodiscard]] constexpr F operator/(T lhs, F rhs) noexcept
{
    return static_cast<F>(static_cast<float>(lhs)) / rhs;
}

template <std::floating_point F, detail::Float8Type T>
[[nodiscard]] constexpr F operator+(F lhs, T rhs) noexcept
{
    return lhs + static_cast<F>(static_cast<float>(rhs));
}

template <std::floating_point F, detail::Float8Type T>
[[nodiscard]] constexpr F operator-(F lhs, T rhs) noexcept
{
    return lhs - static_cast<F>(static_cast<float>(rhs));
}

template <std::floating_point F, detail::Float8Type T>
[[nodiscard]] constexpr F operator*(F lhs, T rhs) noexcept
{
    return lhs * static_cast<F>(static_cast<float>(rhs));
}

template <std::floating_point F, detail::Float8Type T>
[[nodiscard]] constexpr F operator/(F lhs, T rhs) noexcept
{
    return lhs / static_cast<F>(static_cast<float>(rhs));
}

template <detail::Float8Type T, std::integral I>
    requires(!std::same_as<std::remove_cv_t<I>, bool>)
[[nodiscard]] constexpr T operator+(T lhs, I rhs) noexcept
{
    return lhs + T{static_cast<float>(rhs)};
}

template <detail::Float8Type T, std::integral I>
    requires(!std::same_as<std::remove_cv_t<I>, bool>)
[[nodiscard]] constexpr T operator-(T lhs, I rhs) noexcept
{
    return lhs - T{static_cast<float>(rhs)};
}

template <detail::Float8Type T, std::integral I>
    requires(!std::same_as<std::remove_cv_t<I>, bool>)
[[nodiscard]] constexpr T operator*(T lhs, I rhs) noexcept
{
    return lhs * T{static_cast<float>(rhs)};
}

template <detail::Float8Type T, std::integral I>
    requires(!std::same_as<std::remove_cv_t<I>, bool>)
[[nodiscard]] constexpr T operator/(T lhs, I rhs) noexcept
{
    return lhs / T{static_cast<float>(rhs)};
}

template <std::integral I, detail::Float8Type T>
    requires(!std::same_as<std::remove_cv_t<I>, bool>)
[[nodiscard]] constexpr T operator+(I lhs, T rhs) noexcept
{
    return T{static_cast<float>(lhs)} + rhs;
}

template <std::integral I, detail::Float8Type T>
    requires(!std::same_as<std::remove_cv_t<I>, bool>)
[[nodiscard]] constexpr T operator-(I lhs, T rhs) noexcept
{
    return T{static_cast<float>(lhs)} - rhs;
}

template <std::integral I, detail::Float8Type T>
    requires(!std::same_as<std::remove_cv_t<I>, bool>)
[[nodiscard]] constexpr T operator*(I lhs, T rhs) noexcept
{
    return T{static_cast<float>(lhs)} * rhs;
}

template <std::integral I, detail::Float8Type T>
    requires(!std::same_as<std::remove_cv_t<I>, bool>)
[[nodiscard]] constexpr T operator/(I lhs, T rhs) noexcept
{
    return T{static_cast<float>(lhs)} / rhs;
}

std::ostream& operator<<(std::ostream& stream, Float8E4M3 value);
std::ostream& operator<<(std::ostream& stream, Float8E5M2 value);

} // namespace rad

namespace std
{

template <>
class numeric_limits<rad::Float8E4M3>
{
public:
    static constexpr bool is_specialized = true;
    static constexpr bool is_signed = true;
    static constexpr bool is_integer = false;
    static constexpr bool is_exact = false;
    static constexpr bool has_infinity = false;
    static constexpr bool has_quiet_NaN = true;
    static constexpr bool has_signaling_NaN = false;
    static constexpr float_denorm_style has_denorm = denorm_present;
    static constexpr bool has_denorm_loss = true;
    static constexpr float_round_style round_style = round_to_nearest;
    static constexpr bool is_iec559 = false;
    static constexpr bool is_bounded = true;
    static constexpr bool is_modulo = false;
    static constexpr bool traps = false;
    static constexpr bool tinyness_before = false;
    static constexpr int digits = 4;
    static constexpr int digits10 = 0;
    static constexpr int max_digits10 = 3;
    static constexpr int radix = 2;
    static constexpr int min_exponent = -5;
    static constexpr int min_exponent10 = -1;
    static constexpr int max_exponent = 9;
    static constexpr int max_exponent10 = 2;

    [[nodiscard]] static constexpr rad::Float8E4M3 min() noexcept
    {
        return rad::Float8E4M3::FromBits(0x08);
    }

    [[nodiscard]] static constexpr rad::Float8E4M3 lowest() noexcept
    {
        return rad::Float8E4M3::FromBits(0xFE);
    }

    [[nodiscard]] static constexpr rad::Float8E4M3 max() noexcept
    {
        return rad::Float8E4M3::FromBits(0x7E);
    }

    [[nodiscard]] static constexpr rad::Float8E4M3 epsilon() noexcept
    {
        return rad::Float8E4M3::FromBits(0x20);
    }

    [[nodiscard]] static constexpr rad::Float8E4M3 round_error() noexcept
    {
        return rad::Float8E4M3::FromBits(0x30);
    }

    [[nodiscard]] static constexpr rad::Float8E4M3 infinity() noexcept { return {}; }

    [[nodiscard]] static constexpr rad::Float8E4M3 quiet_NaN() noexcept
    {
        return rad::Float8E4M3::FromBits(0x7F);
    }

    [[nodiscard]] static constexpr rad::Float8E4M3 signaling_NaN() noexcept { return {}; }

    [[nodiscard]] static constexpr rad::Float8E4M3 denorm_min() noexcept
    {
        return rad::Float8E4M3::FromBits(0x01);
    }
};

template <>
class numeric_limits<rad::Float8E5M2>
{
public:
    static constexpr bool is_specialized = true;
    static constexpr bool is_signed = true;
    static constexpr bool is_integer = false;
    static constexpr bool is_exact = false;
    static constexpr bool has_infinity = true;
    static constexpr bool has_quiet_NaN = true;
    static constexpr bool has_signaling_NaN = false;
    static constexpr float_denorm_style has_denorm = denorm_present;
    static constexpr bool has_denorm_loss = true;
    static constexpr float_round_style round_style = round_to_nearest;
    static constexpr bool is_iec559 = false;
    static constexpr bool is_bounded = true;
    static constexpr bool is_modulo = false;
    static constexpr bool traps = false;
    static constexpr bool tinyness_before = false;
    static constexpr int digits = 3;
    static constexpr int digits10 = 0;
    static constexpr int max_digits10 = 2;
    static constexpr int radix = 2;
    static constexpr int min_exponent = -13;
    static constexpr int min_exponent10 = -4;
    static constexpr int max_exponent = 16;
    static constexpr int max_exponent10 = 4;

    [[nodiscard]] static constexpr rad::Float8E5M2 min() noexcept
    {
        return rad::Float8E5M2::FromBits(0x04);
    }

    [[nodiscard]] static constexpr rad::Float8E5M2 lowest() noexcept
    {
        return rad::Float8E5M2::FromBits(0xFB);
    }

    [[nodiscard]] static constexpr rad::Float8E5M2 max() noexcept
    {
        return rad::Float8E5M2::FromBits(0x7B);
    }

    [[nodiscard]] static constexpr rad::Float8E5M2 epsilon() noexcept
    {
        return rad::Float8E5M2::FromBits(0x34);
    }

    [[nodiscard]] static constexpr rad::Float8E5M2 round_error() noexcept
    {
        return rad::Float8E5M2::FromBits(0x38);
    }

    [[nodiscard]] static constexpr rad::Float8E5M2 infinity() noexcept
    {
        return rad::Float8E5M2::FromBits(0x7C);
    }

    [[nodiscard]] static constexpr rad::Float8E5M2 quiet_NaN() noexcept
    {
        return rad::Float8E5M2::FromBits(0x7F);
    }

    [[nodiscard]] static constexpr rad::Float8E5M2 signaling_NaN() noexcept { return {}; }

    [[nodiscard]] static constexpr rad::Float8E5M2 denorm_min() noexcept
    {
        return rad::Float8E5M2::FromBits(0x01);
    }
};

} // namespace std
