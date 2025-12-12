#pragma once

#include <rad/Common/Platform.h>
#include <rad/Common/Float.h>
#include <rad/Common/Integer.h>
#include <rad/Common/String.h>

namespace rad
{

enum class MLDataType
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
};

uint32_t GetElementSize(MLDataType type);
bool IsFloatingPointType(MLDataType type);
bool IsSignedIntegerType(MLDataType type);
bool IsUnsignedIntegerType(MLDataType type);
bool IsIntegerType(MLDataType type);

std::string ToStringFixedWidthDec(const void* data, MLDataType dataType);
std::string ToStringFixedWidthHex(const void* data, MLDataType dataType);

} // namespace rad
