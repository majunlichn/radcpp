#pragma once

#include <rad/Core/Integer.h>

#include <boost/crc.hpp>

#include <cassert>
#include <cstddef>
#include <ranges>
#include <type_traits>

namespace rad
{

// Generic, table-driven CRC calculator.
template <std::size_t Bits, Uint64 Polynomial, Uint64 InitialRemainder = 0,
          bool ReflectInput = false, bool ReflectRemainder = false, Uint64 FinalXor = 0>
class Crc
{
    static_assert(Bits >= 1 && Bits <= 64, "Crc supports widths from 1 to 64 bits");

    using Engine = boost::crc_optimal<Bits, Polynomial, InitialRemainder, FinalXor, ReflectInput,
                                      ReflectRemainder>;

public:
    using ValueType = std::conditional_t<(Bits <= 32), Uint32, Uint64>;

    Crc() noexcept = default;

    void Reset() noexcept { m_engine.reset(); }

    void Update(const void* data, std::size_t size) noexcept
    {
        assert(data != nullptr || size == 0);
        if (size != 0)
        {
            m_engine.process_bytes(data, size);
        }
    }

    // Processes each element's raw representation; use serialized bytes for portable CRCs.
    template <std::ranges::contiguous_range R>
        requires std::is_trivially_copyable_v<std::ranges::range_value_t<R>>
    void Update(const R& values) noexcept
    {
        using Element = std::ranges::range_value_t<R>;
        Update(std::ranges::data(values), std::ranges::size(values) * sizeof(Element));
    }

    // Processes the object's raw representation; padding and byte order are platform-dependent.
    template <typename T>
        requires std::is_trivially_copyable_v<std::remove_cvref_t<T>> &&
                 (!std::is_pointer_v<std::remove_cvref_t<T>>) &&
                 (!std::ranges::range<std::remove_cvref_t<T>>)
    void Update(const T& value) noexcept
    {
        Update(&value, sizeof(value));
    }

    [[nodiscard]] ValueType Value() const noexcept
    {
        return static_cast<ValueType>(m_engine.checksum());
    }

    [[nodiscard]] static ValueType Compute(const void* data, std::size_t size) noexcept
    {
        Crc crc;
        crc.Update(data, size);
        return crc.Value();
    }

    template <typename T>
        requires requires(Crc& crc, const T& value) { crc.Update(value); }
    [[nodiscard]] static ValueType Compute(const T& value) noexcept
    {
        Crc crc;
        crc.Update(value);
        return crc.Value();
    }

private:
    Engine m_engine;
}; // class Crc

using Crc8Smbus = Crc<8, 0x07>;
using Crc16Arc = Crc<16, 0x8005, 0x0000, true, true>;
using Crc16Ibm3740 = Crc<16, 0x1021, 0xFFFF, false, false>;
using Crc16Kermit = Crc<16, 0x1021, 0x0000, true, true>;
using Crc16Modbus = Crc<16, 0x8005, 0xFFFF, true, true>;
using Crc16Usb = Crc<16, 0x8005, 0xFFFF, true, true, 0xFFFF>;
using Crc16Xmodem = Crc<16, 0x1021>;
using Crc24OpenPgp = Crc<24, 0x864CFB, 0xB704CE>;
using Crc32Bzip2 = Crc<32, 0x04C11DB7, 0xFFFFFFFF, false, false, 0xFFFFFFFF>;
using Crc32IsoHdlc = Crc<32, 0x04C11DB7, 0xFFFFFFFF, true, true, 0xFFFFFFFF>;
using Crc32Iscsi = Crc<32, 0x1EDC6F41, 0xFFFFFFFF, true, true, 0xFFFFFFFF>;
using Crc32Mpeg2 = Crc<32, 0x04C11DB7, 0xFFFFFFFF>;
using Crc64Ecma182 = Crc<64, 0x42F0E1EBA9EA3693, 0x0000000000000000, false, false>;
using Crc64Xz = Crc<64, 0x42F0E1EBA9EA3693, 0xFFFFFFFFFFFFFFFF, true, true, 0xFFFFFFFFFFFFFFFF>;

using Crc32 = Crc32IsoHdlc;

} // namespace rad
