#pragma once

#include <rad/Common/Algorithm.h>
#include <rad/Common/Float.h>
#include <rad/Common/Integer.h>
#include <rad/Common/Memory.h>
#include <rad/Common/RefCounted.h>
#include <rad/Common/String.h>
#include <rad/Container/ArrayRef.h>
#include <rad/Container/SmallVector.h>
#include <variant>

namespace ML
{

class Device;
class Context;
class Tensor;

enum class DataType
{
    Unknown,
    Float16,
    Float32,
    Float64,
    Sint8,
    Sint16,
    Sint32,
    Sint64,
    Uint8,
    Uint16,
    Uint32,
    Uint64,
    Bool,
    Complex32,  // Float16x2
    Complex64,  // Float32x2
    Complex128, // Float64x2
    BFloat16,
    Float8E4M3,
    Float8E5M2,
    Count
};

uint32_t GetElementSize(DataType type);
bool IsFloatingPointType(DataType type);
bool IsSignedIntegerType(DataType type);
bool IsUnsignedIntegerType(DataType type);
bool IsIntegerType(DataType type);

const char* GetDataTypeName(DataType dataType);

std::string ToStringFixedWidthDec(const void* data, DataType dataType);
std::string ToStringFixedWidthHex(const void* data, DataType dataType);

class Scalar
{
public:
    enum class Type
    {
        Undefined,
        Float,
        Sint,
        Uint,
        Bool,
        Complex,
    } m_type;

    std::variant<double, int64_t, uint64_t, bool, rad::Complex128> m_value;

    Scalar() : m_type(Type::Undefined)
    {
    }

    Scalar(float value) : m_type(Type::Float)
    {
        m_value = value;
    }

    Scalar(double value) : m_type(Type::Float)
    {
        m_value = value;
    }

    Scalar(int32_t value) : m_type(Type::Sint)
    {
        m_value = static_cast<int64_t>(value);
    }

    Scalar(int64_t value) : m_type(Type::Sint)
    {
        m_value = value;
    }

    Scalar(uint32_t value) : m_type(Type::Uint)
    {
        m_value = static_cast<uint64_t>(value);
    }

    Scalar(uint64_t value) : m_type(Type::Uint)
    {
        m_value = value;
    }

    Scalar(bool value) : m_type(Type::Bool)
    {
        m_value = value;
    }

    Scalar(const rad::Complex128& value) : m_type(Type::Complex)
    {
        m_value = value;
    }

    bool IsFloatingPoint() const
    {
        return m_type == Type::Float;
    }

    bool IsInteger() const
    {
        return (m_type == Type::Sint) || (m_type == Type::Uint);
    }

    bool IsSignedInteger() const
    {
        return m_type == Type::Sint;
    }

    bool IsUnsignedInteger() const
    {
        return m_type == Type::Uint;
    }

    bool IsBool() const
    {
        return m_type == Type::Bool;
    }

    bool IsComplex() const
    {
        return m_type == Type::Complex;
    }

    operator float() const
    {
        switch (m_type)
        {
        case Type::Undefined: return 0;
        case Type::Float: return static_cast<float>(std::get<double>(m_value));
        case Type::Sint: return static_cast<float>(std::get<int64_t>(m_value));
        case Type::Uint: return static_cast<float>(std::get<uint64_t>(m_value));
        }
        RAD_UNREACHABLE();
    }

    operator double() const
    {
        switch (m_type)
        {
        case Type::Undefined: return 0;
        case Type::Float: return static_cast<double>(std::get<double>(m_value));
        case Type::Sint: return static_cast<double>(std::get<int64_t>(m_value));
        case Type::Uint: return static_cast<double>(std::get<uint64_t>(m_value));
        }
        RAD_UNREACHABLE();
    }

    operator int8_t() const
    {
        switch (m_type)
        {
        case Type::Undefined: return 0;
        case Type::Float: return static_cast<int8_t>(std::get<double>(m_value));
        case Type::Sint: return static_cast<int8_t>(std::get<int64_t>(m_value));
        case Type::Uint: return static_cast<int8_t>(std::get<uint64_t>(m_value));
        }
        RAD_UNREACHABLE();
    }

    operator int16_t() const
    {
        switch (m_type)
        {
        case Type::Undefined: return 0;
        case Type::Float: return static_cast<int16_t>(std::get<double>(m_value));
        case Type::Sint: return static_cast<int16_t>(std::get<int64_t>(m_value));
        case Type::Uint: return static_cast<int16_t>(std::get<uint64_t>(m_value));
        }
        RAD_UNREACHABLE();
    }

    operator int32_t() const
    {
        switch (m_type)
        {
        case Type::Undefined: return 0;
        case Type::Float: return static_cast<int32_t>(std::get<double>(m_value));
        case Type::Sint: return static_cast<int32_t>(std::get<int64_t>(m_value));
        case Type::Uint: return static_cast<int32_t>(std::get<uint64_t>(m_value));
        }
        RAD_UNREACHABLE();
    }

    operator int64_t() const
    {
        switch (m_type)
        {
        case Type::Undefined: return 0;
        case Type::Float: return static_cast<int64_t>(std::get<double>(m_value));
        case Type::Sint: return static_cast<int64_t>(std::get<int64_t>(m_value));
        case Type::Uint: return static_cast<int64_t>(std::get<uint64_t>(m_value));
        }
        RAD_UNREACHABLE();
    }

    operator uint8_t() const
    {
        switch (m_type)
        {
        case Type::Undefined: return 0;
        case Type::Float: return static_cast<uint8_t>(std::get<double>(m_value));
        case Type::Sint: return static_cast<uint8_t>(std::get<int64_t>(m_value));
        case Type::Uint: return static_cast<uint8_t>(std::get<uint64_t>(m_value));
        }
        RAD_UNREACHABLE();
    }

    operator uint16_t() const
    {
        switch (m_type)
        {
        case Type::Undefined: return 0;
        case Type::Float: return static_cast<uint16_t>(std::get<double>(m_value));
        case Type::Sint: return static_cast<uint16_t>(std::get<int64_t>(m_value));
        case Type::Uint: return static_cast<uint16_t>(std::get<uint64_t>(m_value));
        }
        RAD_UNREACHABLE();
    }

    operator uint32_t() const
    {
        switch (m_type)
        {
        case Type::Undefined: return 0;
        case Type::Float: return static_cast<uint32_t>(std::get<double>(m_value));
        case Type::Sint: return static_cast<uint32_t>(std::get<int64_t>(m_value));
        case Type::Uint: return static_cast<uint32_t>(std::get<uint64_t>(m_value));
        }
        RAD_UNREACHABLE();
    }

    operator uint64_t() const
    {
        switch (m_type)
        {
        case Type::Undefined: return 0;
        case Type::Float: return static_cast<uint64_t>(std::get<double>(m_value));
        case Type::Sint: return static_cast<uint64_t>(std::get<int64_t>(m_value));
        case Type::Uint: return static_cast<uint64_t>(std::get<uint64_t>(m_value));
        }
        RAD_UNREACHABLE();
    }

    operator bool() const
    {
        assert(m_type == Type::Bool);
        switch (m_type)
        {
        case Type::Undefined: return false;
        case Type::Bool: return std::get<bool>(m_value);
        }
        RAD_UNREACHABLE();
    }

    operator rad::Complex32() const
    {
        assert(m_type == Type::Complex);
        const auto& c = std::get<rad::Complex128>(m_value);
        switch (m_type)
        {
        case Type::Undefined: return rad::Complex32{ rad::Float16(0), rad::Float16(0) };
        case Type::Complex: return rad::Complex64{ static_cast<rad::Float32>(c.real()), static_cast<rad::Float32>(c.imag()) };
        }
        RAD_UNREACHABLE();
    }

    operator rad::Complex64() const
    {
        assert(m_type == Type::Complex);
        const auto& c = std::get<rad::Complex128>(m_value);
        switch (m_type)
        {
        case Type::Undefined: return rad::Complex64{ rad::Float32(0), rad::Float32(0) };
        case Type::Complex: return rad::Complex64{ static_cast<rad::Float32>(c.real()), static_cast<rad::Float32>(c.imag()) };
        }
        RAD_UNREACHABLE();
    }

    operator rad::Complex128() const
    {
        assert(m_type == Type::Complex);
        switch (m_type)
        {
        case Type::Undefined: return rad::Complex128{ rad::Float64(0), rad::Float64(0) };
        case Type::Complex: return std::get<rad::Complex128>(m_value);
        }
        RAD_UNREACHABLE();
    }

}; // class Scalar

struct TensorOptions
{
    std::vector<size_t> m_strides;

    TensorOptions& SetStrides(rad::ArrayRef<size_t> strides)
    {
        m_strides = strides;
        return *this;
    }

}; // struct TensorOptions

} // namespace ML
