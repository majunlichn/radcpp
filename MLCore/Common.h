#pragma once

#include <rad/Common/Algorithm.h>
#include <rad/Common/Float.h>
#include <rad/Common/Integer.h>
#include <rad/Common/Memory.h>
#include <rad/Common/RefCounted.h>
#include <rad/Common/String.h>
#include <rad/Container/ArrayRef.h>
#include <rad/Container/SmallVector.h>

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

    BFloat16,
    Float8E4M3,
    Float8E5M2,

    Complex32,  // Float16x2
    Complex64,  // Float32x2
    Complex128, // Float64x2

    Count,
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
        Float,
        Sint,
        Uint,
    } m_type;

    union Value
    {
        double f;
        int64_t i;
        uint64_t u;
    } m_value;

    Scalar() : m_type(Type::Uint)
    {
        m_value.u = 0;
    }

    Scalar(float value) : m_type(Type::Float)
    {
        m_value.f = value;
    }

    Scalar(double value) : m_type(Type::Float)
    {
        m_value.f = value;
    }

    Scalar(int32_t value) : m_type(Type::Sint)
    {
        m_value.i = value;
    }

    Scalar(int64_t value) : m_type(Type::Sint)
    {
        m_value.i = value;
    }

    Scalar(uint32_t value) : m_type(Type::Uint)
    {
        m_value.u = value;
    }

    Scalar(uint64_t value) : m_type(Type::Uint)
    {
        m_value.u = value;
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

    operator float() const
    {
        switch (m_type)
        {
        case ML::Scalar::Type::Float: return static_cast<float>(m_value.f);
            break;
        case ML::Scalar::Type::Sint: return static_cast<float>(m_value.i);
            break;
        case ML::Scalar::Type::Uint: return static_cast<float>(m_value.u);
            break;
        }
        RAD_UNREACHABLE();
    }

    operator double() const
    {
        switch (m_type)
        {
        case ML::Scalar::Type::Float: return static_cast<double>(m_value.f);
            break;
        case ML::Scalar::Type::Sint: return static_cast<double>(m_value.i);
            break;
        case ML::Scalar::Type::Uint: return static_cast<double>(m_value.u);
            break;
        }
        RAD_UNREACHABLE();
    }

    operator int8_t() const
    {
        switch (m_type)
        {
        case ML::Scalar::Type::Float: return static_cast<int8_t>(m_value.f);
            break;
        case ML::Scalar::Type::Sint: return static_cast<int8_t>(m_value.i);
            break;
        case ML::Scalar::Type::Uint: return static_cast<int8_t>(m_value.u);
            break;
        }
        RAD_UNREACHABLE();
    }

    operator int16_t() const
    {
        switch (m_type)
        {
        case ML::Scalar::Type::Float: return static_cast<int16_t>(m_value.f);
            break;
        case ML::Scalar::Type::Sint: return static_cast<int16_t>(m_value.i);
            break;
        case ML::Scalar::Type::Uint: return static_cast<int16_t>(m_value.u);
            break;
        }
        RAD_UNREACHABLE();
    }

    operator int32_t() const
    {
        switch (m_type)
        {
        case ML::Scalar::Type::Float: return static_cast<int32_t>(m_value.f);
            break;
        case ML::Scalar::Type::Sint: return static_cast<int32_t>(m_value.i);
            break;
        case ML::Scalar::Type::Uint: return static_cast<int32_t>(m_value.u);
            break;
        }
        RAD_UNREACHABLE();
    }

    operator int64_t() const
    {
        switch (m_type)
        {
        case ML::Scalar::Type::Float: return static_cast<int64_t>(m_value.f);
            break;
        case ML::Scalar::Type::Sint: return static_cast<int64_t>(m_value.i);
            break;
        case ML::Scalar::Type::Uint: return static_cast<int64_t>(m_value.u);
            break;
        }
        RAD_UNREACHABLE();
    }

    operator uint8_t() const
    {
        switch (m_type)
        {
        case ML::Scalar::Type::Float: return static_cast<uint8_t>(m_value.f);
            break;
        case ML::Scalar::Type::Sint: return static_cast<uint8_t>(m_value.i);
            break;
        case ML::Scalar::Type::Uint: return static_cast<uint8_t>(m_value.u);
            break;
        }
        RAD_UNREACHABLE();
    }

    operator uint16_t() const
    {
        switch (m_type)
        {
        case ML::Scalar::Type::Float: return static_cast<uint16_t>(m_value.f);
            break;
        case ML::Scalar::Type::Sint: return static_cast<uint16_t>(m_value.i);
            break;
        case ML::Scalar::Type::Uint: return static_cast<uint16_t>(m_value.u);
            break;
        }
        RAD_UNREACHABLE();
    }

    operator uint32_t() const
    {
        switch (m_type)
        {
        case ML::Scalar::Type::Float: return static_cast<uint32_t>(m_value.f);
            break;
        case ML::Scalar::Type::Sint: return static_cast<uint32_t>(m_value.i);
            break;
        case ML::Scalar::Type::Uint: return static_cast<uint32_t>(m_value.u);
            break;
        }
        RAD_UNREACHABLE();
    }

    operator uint64_t() const
    {
        switch (m_type)
        {
        case ML::Scalar::Type::Float: return static_cast<uint64_t>(m_value.f);
            break;
        case ML::Scalar::Type::Sint: return static_cast<uint64_t>(m_value.i);
            break;
        case ML::Scalar::Type::Uint: return static_cast<uint64_t>(m_value.u);
            break;
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
