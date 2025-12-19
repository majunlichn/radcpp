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
    Count,
};

uint32_t GetElementSize(DataType type);
bool IsFloatingPointType(DataType type);
bool IsSignedIntegerType(DataType type);
bool IsUnsignedIntegerType(DataType type);
bool IsIntegerType(DataType type);

std::string ToStringFixedWidthDec(const void* data, DataType dataType);
std::string ToStringFixedWidthHex(const void* data, DataType dataType);

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
