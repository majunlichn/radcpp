#pragma once

#include <bit>
#include <concepts>
#include <cstdint>
#include <iosfwd>
#include <limits>
#include <type_traits>

namespace rad
{

// The bfloat16 format was developed by Google Brain, an artificial intelligence research group at Google.
// https://en.wikipedia.org/wiki/Bfloat16_floating-point_format

// https://github.com/pytorch/pytorch/blob/main/torch/headeronly/util/BFloat16.h

[[nodiscard]] constexpr float bf16_to_fp32(std::uint16_t bits) noexcept
{
    return std::bit_cast<float>(static_cast<std::uint32_t>(bits) << 16u);
}

[[nodiscard]] constexpr std::uint16_t bf16_from_fp32_round_to_zero(float value) noexcept
{
    const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    const std::uint32_t magnitude = bits & 0x7FFFFFFFu;

    // Canonicalize NaNs so truncation cannot turn a NaN into an infinity.
    if (magnitude > 0x7F800000u)
    {
        return 0x7FC0u;
    }

    return static_cast<std::uint16_t>(bits >> 16u);
}

[[nodiscard]] constexpr std::uint16_t bf16_from_fp32_round_to_nearest_even(float value) noexcept
{
    const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    const std::uint32_t magnitude = bits & 0x7FFFFFFFu;

    // Canonicalize NaNs so rounding cannot turn a NaN into an infinity.
    if (magnitude > 0x7F800000u)
    {
        return 0x7FC0u;
    }

    const std::uint32_t roundingBias = 0x7FFFu + ((bits >> 16u) & 1u);
    return static_cast<std::uint16_t>((bits + roundingBias) >> 16u);
}

class alignas(2) BFloat16 final
{
public:
    constexpr BFloat16() noexcept = default;
    constexpr BFloat16(float value) noexcept :
        m_bits(bf16_from_fp32_round_to_nearest_even(value))
    {
    }

    [[nodiscard]] static constexpr BFloat16 FromBits(std::uint16_t bits) noexcept
    {
        BFloat16 result;
        result.m_bits = bits;
        return result;
    }

    [[nodiscard]] constexpr std::uint16_t Bits() const noexcept { return m_bits; }

    [[nodiscard]] constexpr operator float() const noexcept { return bf16_to_fp32(m_bits); }

    constexpr BFloat16& operator+=(BFloat16 rhs) noexcept;
    constexpr BFloat16& operator-=(BFloat16 rhs) noexcept;
    constexpr BFloat16& operator*=(BFloat16 rhs) noexcept;
    constexpr BFloat16& operator/=(BFloat16 rhs) noexcept;

private:
    std::uint16_t m_bits = 0;
}; // class BFloat16

static_assert(sizeof(BFloat16) == sizeof(std::uint16_t));
static_assert(alignof(BFloat16) == alignof(std::uint16_t));

[[nodiscard]] constexpr BFloat16 operator+(BFloat16 lhs, BFloat16 rhs) noexcept
{
    return static_cast<float>(lhs) + static_cast<float>(rhs);
}

[[nodiscard]] constexpr BFloat16 operator-(BFloat16 lhs, BFloat16 rhs) noexcept
{
    return static_cast<float>(lhs) - static_cast<float>(rhs);
}

[[nodiscard]] constexpr BFloat16 operator*(BFloat16 lhs, BFloat16 rhs) noexcept
{
    return static_cast<float>(lhs) * static_cast<float>(rhs);
}

[[nodiscard]] constexpr BFloat16 operator/(BFloat16 lhs, BFloat16 rhs) noexcept
{
    return static_cast<float>(lhs) / static_cast<float>(rhs);
}

[[nodiscard]] constexpr BFloat16 operator-(BFloat16 value) noexcept
{
    return BFloat16::FromBits(static_cast<std::uint16_t>(value.Bits() ^ 0x8000u));
}

constexpr BFloat16& BFloat16::operator+=(BFloat16 rhs) noexcept
{
    return *this = *this + rhs;
}

constexpr BFloat16& BFloat16::operator-=(BFloat16 rhs) noexcept
{
    return *this = *this - rhs;
}

constexpr BFloat16& BFloat16::operator*=(BFloat16 rhs) noexcept
{
    return *this = *this * rhs;
}

constexpr BFloat16& BFloat16::operator/=(BFloat16 rhs) noexcept
{
    return *this = *this / rhs;
}

template <std::floating_point T>
[[nodiscard]] constexpr T operator+(BFloat16 lhs, T rhs) noexcept
{
    return static_cast<T>(static_cast<float>(lhs)) + rhs;
}

template <std::floating_point T>
[[nodiscard]] constexpr T operator-(BFloat16 lhs, T rhs) noexcept
{
    return static_cast<T>(static_cast<float>(lhs)) - rhs;
}

template <std::floating_point T>
[[nodiscard]] constexpr T operator*(BFloat16 lhs, T rhs) noexcept
{
    return static_cast<T>(static_cast<float>(lhs)) * rhs;
}

template <std::floating_point T>
[[nodiscard]] constexpr T operator/(BFloat16 lhs, T rhs) noexcept
{
    return static_cast<T>(static_cast<float>(lhs)) / rhs;
}

template <std::floating_point T>
[[nodiscard]] constexpr T operator+(T lhs, BFloat16 rhs) noexcept
{
    return lhs + static_cast<T>(static_cast<float>(rhs));
}

template <std::floating_point T>
[[nodiscard]] constexpr T operator-(T lhs, BFloat16 rhs) noexcept
{
    return lhs - static_cast<T>(static_cast<float>(rhs));
}

template <std::floating_point T>
[[nodiscard]] constexpr T operator*(T lhs, BFloat16 rhs) noexcept
{
    return lhs * static_cast<T>(static_cast<float>(rhs));
}

template <std::floating_point T>
[[nodiscard]] constexpr T operator/(T lhs, BFloat16 rhs) noexcept
{
    return lhs / static_cast<T>(static_cast<float>(rhs));
}

template <std::integral T>
    requires(!std::same_as<std::remove_cv_t<T>, bool>)
[[nodiscard]] constexpr BFloat16 operator+(BFloat16 lhs, T rhs) noexcept
{
    return lhs + BFloat16{static_cast<float>(rhs)};
}

template <std::integral T>
    requires(!std::same_as<std::remove_cv_t<T>, bool>)
[[nodiscard]] constexpr BFloat16 operator-(BFloat16 lhs, T rhs) noexcept
{
    return lhs - BFloat16{static_cast<float>(rhs)};
}

template <std::integral T>
    requires(!std::same_as<std::remove_cv_t<T>, bool>)
[[nodiscard]] constexpr BFloat16 operator*(BFloat16 lhs, T rhs) noexcept
{
    return lhs * BFloat16{static_cast<float>(rhs)};
}

template <std::integral T>
    requires(!std::same_as<std::remove_cv_t<T>, bool>)
[[nodiscard]] constexpr BFloat16 operator/(BFloat16 lhs, T rhs) noexcept
{
    return lhs / BFloat16{static_cast<float>(rhs)};
}

template <std::integral T>
    requires(!std::same_as<std::remove_cv_t<T>, bool>)
[[nodiscard]] constexpr BFloat16 operator+(T lhs, BFloat16 rhs) noexcept
{
    return BFloat16{static_cast<float>(lhs)} + rhs;
}

template <std::integral T>
    requires(!std::same_as<std::remove_cv_t<T>, bool>)
[[nodiscard]] constexpr BFloat16 operator-(T lhs, BFloat16 rhs) noexcept
{
    return BFloat16{static_cast<float>(lhs)} - rhs;
}

template <std::integral T>
    requires(!std::same_as<std::remove_cv_t<T>, bool>)
[[nodiscard]] constexpr BFloat16 operator*(T lhs, BFloat16 rhs) noexcept
{
    return BFloat16{static_cast<float>(lhs)} * rhs;
}

template <std::integral T>
    requires(!std::same_as<std::remove_cv_t<T>, bool>)
[[nodiscard]] constexpr BFloat16 operator/(T lhs, BFloat16 rhs) noexcept
{
    return BFloat16{static_cast<float>(lhs)} / rhs;
}

[[nodiscard]] constexpr bool operator==(BFloat16 lhs, BFloat16 rhs) noexcept
{
    return static_cast<float>(lhs) == static_cast<float>(rhs);
}

[[nodiscard]] constexpr bool operator!=(BFloat16 lhs, BFloat16 rhs) noexcept
{
    return !(lhs == rhs);
}

[[nodiscard]] constexpr bool operator<(BFloat16 lhs, BFloat16 rhs) noexcept
{
    return static_cast<float>(lhs) < static_cast<float>(rhs);
}

[[nodiscard]] constexpr bool operator<=(BFloat16 lhs, BFloat16 rhs) noexcept
{
    return static_cast<float>(lhs) <= static_cast<float>(rhs);
}

[[nodiscard]] constexpr bool operator>(BFloat16 lhs, BFloat16 rhs) noexcept
{
    return static_cast<float>(lhs) > static_cast<float>(rhs);
}

[[nodiscard]] constexpr bool operator>=(BFloat16 lhs, BFloat16 rhs) noexcept
{
    return static_cast<float>(lhs) >= static_cast<float>(rhs);
}

std::ostream& operator<<(std::ostream& stream, BFloat16 value);

} // namespace rad

namespace std
{

template <>
class numeric_limits<rad::BFloat16>
{
public:
    static constexpr bool is_specialized = true;
    static constexpr bool is_signed = true;
    static constexpr bool is_integer = false;
    static constexpr bool is_exact = false;
    static constexpr bool has_infinity = true;
    static constexpr bool has_quiet_NaN = true;
    static constexpr bool has_signaling_NaN = true;
    static constexpr float_denorm_style has_denorm = denorm_present;
    static constexpr bool has_denorm_loss = false;
    static constexpr bool is_iec559 = false;
    static constexpr bool is_bounded = true;
    static constexpr bool is_modulo = false;
    static constexpr bool traps = false;
    static constexpr bool tinyness_before = false;
    static constexpr float_round_style round_style = round_to_nearest;
    static constexpr int digits = 8;
    static constexpr int digits10 = 2;
    static constexpr int max_digits10 = 4;
    static constexpr int radix = 2;
    static constexpr int min_exponent = -125;
    static constexpr int min_exponent10 = -37;
    static constexpr int max_exponent = 128;
    static constexpr int max_exponent10 = 38;

    [[nodiscard]] static constexpr rad::BFloat16 min() noexcept
    {
        return rad::BFloat16::FromBits(0x0080);
    }

    [[nodiscard]] static constexpr rad::BFloat16 lowest() noexcept
    {
        return rad::BFloat16::FromBits(0xFF7F);
    }

    [[nodiscard]] static constexpr rad::BFloat16 max() noexcept
    {
        return rad::BFloat16::FromBits(0x7F7F);
    }

    [[nodiscard]] static constexpr rad::BFloat16 epsilon() noexcept
    {
        return rad::BFloat16::FromBits(0x3C00);
    }

    [[nodiscard]] static constexpr rad::BFloat16 round_error() noexcept
    {
        return rad::BFloat16::FromBits(0x3F00);
    }

    [[nodiscard]] static constexpr rad::BFloat16 infinity() noexcept
    {
        return rad::BFloat16::FromBits(0x7F80);
    }

    [[nodiscard]] static constexpr rad::BFloat16 quiet_NaN() noexcept
    {
        return rad::BFloat16::FromBits(0x7FC0);
    }

    [[nodiscard]] static constexpr rad::BFloat16 signaling_NaN() noexcept
    {
        return rad::BFloat16::FromBits(0x7F81);
    }

    [[nodiscard]] static constexpr rad::BFloat16 denorm_min() noexcept
    {
        return rad::BFloat16::FromBits(0x0001);
    }
};

} // namespace std
