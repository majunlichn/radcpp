#include <rad/ML/MLDataType.h>

namespace rad
{

uint32_t GetElementSize(MLDataType type)
{
    switch (type)
    {
    case MLDataType::Float16: return 2;
    case MLDataType::Float32: return 4;
    case MLDataType::Float64: return 8;
    case MLDataType::Sint8: return 1;
    case MLDataType::Sint16: return 2;
    case MLDataType::Sint32: return 4;
    case MLDataType::Sint64: return 8;
    case MLDataType::Uint8: return 1;
    case MLDataType::Uint16: return 2;
    case MLDataType::Uint32: return 4;
    case MLDataType::Uint64: return 8;
    case MLDataType::BFloat16: return 2;
    case MLDataType::Float8E4M3: return 1;
    case MLDataType::Float8E5M2: return 1;
    default: RAD_UNREACHABLE();
    }
}

bool IsFloatingPointType(MLDataType type)
{
    if ((type == MLDataType::Float16) ||
        (type == MLDataType::Float32) ||
        (type == MLDataType::Float64) ||
        (type == MLDataType::BFloat16) ||
        (type == MLDataType::Float8E4M3) ||
        (type == MLDataType::Float8E5M2))
    {
        return true;
    }
    else
    {
        return false;
    }
}

bool IsSignedIntegerType(MLDataType type)
{
    if ((type == MLDataType::Sint8) ||
        (type == MLDataType::Sint16) ||
        (type == MLDataType::Sint32) ||
        (type == MLDataType::Sint64))
    {
        return true;
    }
    else
    {
        return false;
    }
}

bool IsUnsignedIntegerType(MLDataType type)
{
    if ((type == MLDataType::Uint8) ||
        (type == MLDataType::Uint16) ||
        (type == MLDataType::Uint32) ||
        (type == MLDataType::Uint64))
    {
        return true;
    }
    else
    {
        return false;
    }
}

bool IsIntegerType(MLDataType type)
{
    return IsSignedIntegerType(type) || IsUnsignedIntegerType(type);
}

std::string FormatValueFixedWidthDec(const void* data, MLDataType dataType)
{
    if (dataType == MLDataType::Float16)
    {
        uint16_t bits = *reinterpret_cast<const uint16_t*>(data);
        return std::format("{:11.4f}", fp16_ieee_to_fp32_value(bits));
    }
    else if (dataType == MLDataType::Float32)
    {
        float value = *reinterpret_cast<const float*>(data);
        if (std::abs(value) < 1000000)
        {
            return std::format("{:14.6f}", value);
        }
        else
        {
            return std::format("{:14.6e}", value);
        }
    }
    else if (dataType == MLDataType::Float64)
    {
        double value = *reinterpret_cast<const double*>(data);
        if (std::abs(value) < 1000000)
        {
            return std::format("{:14.6f}", value);
        }
        else
        {
            return std::format("{:14.6e}", value);
        }
    }
    else if (dataType == MLDataType::BFloat16)
    {
        uint16_t bits = *reinterpret_cast<const uint16_t*>(data);
        float value = bf16_to_fp32(bits);
        if (value < 1000000.0f)
        {
            return std::format("{:12.4f}", value);
        }
        else
        {
            return std::format("{:12.4e}", value);
        }
    }
    else if (dataType == MLDataType::Float8E4M3)
    {
        uint8_t bits = *reinterpret_cast<const uint8_t*>(data);
        float value = fp8e4m3fn_to_fp32_value(bits);
        return std::format("{:8.2f}", value);
    }
    else if (dataType == MLDataType::Float8E4M3)
    {
        uint8_t bits = *reinterpret_cast<const uint8_t*>(data);
        float value = fp8e5m2_to_fp32_value(bits);
        return std::format("{:8.2f}", value);
    }
    else if (dataType == MLDataType::Sint8)
    {
        int8_t value = *reinterpret_cast<const int8_t*>(data);
        return std::format("{:4d}", value);
    }
    else if (dataType == MLDataType::Sint16)
    {
        int16_t value = *reinterpret_cast<const int16_t*>(data);
        return std::format("{:6d}", value);
    }
    else if (dataType == MLDataType::Sint32)
    {
        int32_t value = *reinterpret_cast<const int32_t*>(data);
        return std::format("{:11d}", value);
    }
    else if (dataType == MLDataType::Sint64)
    {
        int64_t value = *reinterpret_cast<const int64_t*>(data);
        return std::format("{:20d}", value);
    }
    else if (dataType == MLDataType::Uint8)
    {
        uint8_t value = *reinterpret_cast<const uint8_t*>(data);
        return std::format("{:3d}", value);
    }
    else if (dataType == MLDataType::Uint16)
    {
        uint16_t value = *reinterpret_cast<const uint16_t*>(data);
        return std::format("{:5d}", value);
    }
    else if (dataType == MLDataType::Uint32)
    {
        uint32_t value = *reinterpret_cast<const uint32_t*>(data);
        return std::format("{:10d}", value);
    }
    else if (dataType == MLDataType::Uint64)
    {
        uint64_t value = *reinterpret_cast<const uint64_t*>(data);
        return std::format("{:20d}", value);
    }
    else
    {
        RAD_UNREACHABLE();
        return {};
    }
}

std::string FormatValueFixedWidthHex(const void* data, MLDataType dataType)
{
    size_t elementSize = GetElementSize(dataType);
    if (elementSize == 1)
    {
        uint8_t value = *reinterpret_cast<const uint8_t*>(data);
        return std::format("0x{:02X}", value);
    }
    else if (elementSize == 2)
    {
        uint16_t value = *reinterpret_cast<const uint16_t*>(data);
        return std::format("0x{:04X}", value);
    }
    else if (elementSize == 4)
    {
        uint32_t value = *reinterpret_cast<const uint32_t*>(data);
        return std::format("0x{:08X}", value);
    }
    else if (elementSize == 8)
    {
        uint64_t value = *reinterpret_cast<const uint64_t*>(data);
        return std::format("0x{:016X}", value);
    }
    else
    {
        RAD_UNREACHABLE();
        return {};
    }
}

} // namespace rad
