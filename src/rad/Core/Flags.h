#pragma once

#include <rad/Core/TypeTraits.h>

#include <cassert>
#include <compare>
#include <type_traits>

namespace rad
{

// Specialize for enum types that represent a bitmask. allFlags must contain every valid bit;
// it bounds operator~ and validates FromMask. The specialization also enables bitwise
// operators directly between enum values.
//
// enum class Access : unsigned { eRead = 1, eWrite = 2 };
//
// template <>
// struct FlagTraits<Access>
// {
//     static constexpr bool isBitmask = true;
//     static constexpr Flags<Access> allFlags =
//         Flags<Access>(Access::eRead) | Access::eWrite;
// };
template <typename BitType>
struct FlagTraits
{
    static constexpr bool isBitmask = false;
};

template <typename BitType>
concept Bitmask = Enumeration<BitType> && FlagTraits<BitType>::isBitmask &&
                  requires { FlagTraits<BitType>::allFlags; };

// Type-safe set of values from an enum used as individual bits.
//
// Flags can be constructed from a single enum value and combined with other Flags or enum
// values. Specializing FlagTraits additionally permits expressions such as
// Access::eRead | Access::eWrite, bounded complement, and validated raw-mask construction.
// Use GetMask only when interoperating with an API that expects the underlying integer mask.
template <Enumeration BitType>
class [[nodiscard]] Flags
{
public:
    using Bit = BitType;
    using Mask = std::underlying_type_t<BitType>;

    constexpr Flags() noexcept = default;

    constexpr Flags(BitType bit) noexcept :
        m_mask(static_cast<Mask>(bit))
    {
    }

    constexpr Flags(const Flags&) noexcept = default;

    constexpr explicit Flags(Mask flags) noexcept :
        m_mask(flags)
    {
    }

    [[nodiscard]] static constexpr Flags FromMask(Mask mask) noexcept
        requires Bitmask<BitType>
    {
        assert(FlagTraits<BitType>::allFlags.HasBits(Flags(mask)));
        return Flags(mask);
    }

    [[nodiscard]] constexpr Mask GetMask() const noexcept { return m_mask; }

    constexpr Flags& operator=(const Flags&) noexcept = default;

    constexpr auto operator<=>(const Flags&) const noexcept = default;

    [[nodiscard]] constexpr bool operator!() const noexcept { return !m_mask; }

    [[nodiscard]] constexpr Flags operator&(Flags rhs) const noexcept
    {
        return Flags(m_mask & rhs.m_mask);
    }

    [[nodiscard]] constexpr Flags operator|(Flags rhs) const noexcept
    {
        return Flags(m_mask | rhs.m_mask);
    }

    [[nodiscard]] constexpr Flags operator^(Flags rhs) const noexcept
    {
        return Flags(m_mask ^ rhs.m_mask);
    }

    [[nodiscard]] constexpr Flags operator~() const noexcept
        requires Bitmask<BitType>
    {
        return Flags(static_cast<Mask>(~m_mask) & FlagTraits<BitType>::allFlags.GetMask());
    }

    [[nodiscard]] constexpr bool Empty() const noexcept { return m_mask == 0; }

    [[nodiscard]] constexpr bool HasBits(Flags bits) const noexcept
    {
        return (m_mask & bits.m_mask) == bits.m_mask;
    }

    [[nodiscard]] constexpr bool HasAnyBits(Flags bits) const noexcept
    {
        return (m_mask & bits.m_mask) != 0;
    }

    [[nodiscard]] constexpr bool HasNoBits(Flags bits) const noexcept
    {
        return (m_mask & bits.m_mask) == 0;
    }

    constexpr Flags& operator|=(Flags rhs) noexcept
    {
        m_mask |= rhs.m_mask;
        return *this;
    }

    constexpr Flags& operator&=(Flags rhs) noexcept
    {
        m_mask &= rhs.m_mask;
        return *this;
    }

    constexpr Flags& operator^=(Flags rhs) noexcept
    {
        m_mask ^= rhs.m_mask;
        return *this;
    }

    constexpr explicit operator bool() const noexcept { return m_mask != 0; }

    constexpr explicit operator Mask() const noexcept { return m_mask; }

private:
    Mask m_mask = 0;
};

template <typename BitType>
constexpr Flags<BitType> operator&(BitType bit, Flags<BitType> flags) noexcept
{
    return flags & bit;
}

template <typename BitType>
constexpr Flags<BitType> operator|(BitType bit, Flags<BitType> flags) noexcept
{
    return flags | bit;
}

template <typename BitType>
constexpr Flags<BitType> operator^(BitType bit, Flags<BitType> flags) noexcept
{
    return flags ^ bit;
}

template <typename BitType>
    requires Bitmask<BitType>
constexpr Flags<BitType> operator&(BitType lhs, BitType rhs) noexcept
{
    return Flags<BitType>(lhs) & rhs;
}

template <typename BitType>
    requires Bitmask<BitType>
constexpr Flags<BitType> operator|(BitType lhs, BitType rhs) noexcept
{
    return Flags<BitType>(lhs) | rhs;
}

template <typename BitType>
    requires Bitmask<BitType>
constexpr Flags<BitType> operator^(BitType lhs, BitType rhs) noexcept
{
    return Flags<BitType>(lhs) ^ rhs;
}

template <typename BitType>
    requires Bitmask<BitType>
constexpr Flags<BitType> operator~(BitType bit) noexcept
{
    return ~Flags<BitType>(bit);
}

} // namespace rad
