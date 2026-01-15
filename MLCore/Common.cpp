#include <MLCore/Common.h>

namespace ML
{

uint32_t GetElementSize(DataType type)
{
    switch (type)
    {
    case DataType::Float16: return 2;
    case DataType::Float32: return 4;
    case DataType::Float64: return 8;
    case DataType::Sint8: return 1;
    case DataType::Sint16: return 2;
    case DataType::Sint32: return 4;
    case DataType::Sint64: return 8;
    case DataType::Uint8: return 1;
    case DataType::Uint16: return 2;
    case DataType::Uint32: return 4;
    case DataType::Uint64: return 8;
    case DataType::Bool: return 1;
    case DataType::Complex32: return 4;
    case DataType::Complex64: return 8;
    case DataType::Complex128: return 16;
    case DataType::BFloat16: return 2;
    case DataType::Float8E4M3: return 1;
    case DataType::Float8E5M2: return 1;
    default: RAD_UNREACHABLE();
    }
}

bool IsFloatingPointType(DataType type)
{
    if ((type == DataType::Float16) ||
        (type == DataType::Float32) ||
        (type == DataType::Float64) ||
        (type == DataType::BFloat16) ||
        (type == DataType::Float8E4M3) ||
        (type == DataType::Float8E5M2))
    {
        return true;
    }
    else
    {
        return false;
    }
}

bool IsSignedIntegerType(DataType type)
{
    if ((type == DataType::Sint8) ||
        (type == DataType::Sint16) ||
        (type == DataType::Sint32) ||
        (type == DataType::Sint64))
    {
        return true;
    }
    else
    {
        return false;
    }
}

bool IsUnsignedIntegerType(DataType type)
{
    if ((type == DataType::Uint8) ||
        (type == DataType::Uint16) ||
        (type == DataType::Uint32) ||
        (type == DataType::Uint64))
    {
        return true;
    }
    else
    {
        return false;
    }
}

bool IsIntegerType(DataType type)
{
    return IsSignedIntegerType(type) || IsUnsignedIntegerType(type);
}

bool IsComplexType(DataType type)
{
    return (type == DataType::Complex32) ||
           (type == DataType::Complex64) ||
        (type == DataType::Complex128);
}

const char* GetDataTypeName(DataType dataType)
{
    switch (dataType)
    {
    case DataType::Unknown:     return "Unknown";
    case DataType::Float16:     return "Float16";
    case DataType::Float32:     return "Float32";
    case DataType::Float64:     return "Float64";
    case DataType::Sint8:       return "Sint8";
    case DataType::Sint16:      return "Sint16";
    case DataType::Sint32:      return "Sint32";
    case DataType::Sint64:      return "Sint64";
    case DataType::Uint8:       return "Uint8";
    case DataType::Uint16:      return "Uint16";
    case DataType::Uint32:      return "Uint32";
    case DataType::Uint64:      return "Uint64";
    case DataType::Bool:        return "Bool";
    case DataType::Complex32:   return "Complex32";
    case DataType::Complex64:   return "Complex64";
    case DataType::Complex128:  return "Complex128";
    case DataType::BFloat16:    return "BFloat16";
    case DataType::Float8E4M3:  return "Float8E4M3";
    case DataType::Float8E5M2:  return "Float8E5M2";
    }
    RAD_UNREACHABLE();
    return nullptr;
}

std::string ToStringFixedWidthDec(const void* data, DataType dataType)
{
    if (dataType == DataType::Float16)
    {
        uint16_t bits = *reinterpret_cast<const uint16_t*>(data);
        float value = rad::fp16_ieee_to_fp32_value(bits);
        return std::format("{:11.4e}", value);
    }
    else if (dataType == DataType::Float32)
    {
        float value = *reinterpret_cast<const float*>(data);
        return std::format("{:11.4e}", value);
    }
    else if (dataType == DataType::Float64)
    {
        double value = *reinterpret_cast<const double*>(data);
        return std::format("{:12.4e}", value);
    }
    else if (dataType == DataType::BFloat16)
    {
        uint16_t bits = *reinterpret_cast<const uint16_t*>(data);
        float value = rad::bf16_to_fp32(bits);
        return std::format("{:11.4e}", value);
    }
    else if (dataType == DataType::Float8E4M3)
    {
        uint8_t bits = *reinterpret_cast<const uint8_t*>(data);
        float value = rad::fp8e4m3fn_to_fp32_value(bits);
        return std::format("{:9.2e}", value);
    }
    else if (dataType == DataType::Float8E5M2)
    {
        uint8_t bits = *reinterpret_cast<const uint8_t*>(data);
        float value = rad::fp8e5m2_to_fp32_value(bits);
        return std::format("{:9.2e}", value);
    }
    else if (dataType == DataType::Sint8)
    {
        int8_t value = *reinterpret_cast<const int8_t*>(data);
        return std::format("{:4d}", value);
    }
    else if (dataType == DataType::Sint16)
    {
        int16_t value = *reinterpret_cast<const int16_t*>(data);
        return std::format("{:6d}", value);
    }
    else if (dataType == DataType::Sint32)
    {
        int32_t value = *reinterpret_cast<const int32_t*>(data);
        return std::format("{:11d}", value);
    }
    else if (dataType == DataType::Sint64)
    {
        int64_t value = *reinterpret_cast<const int64_t*>(data);
        return std::format("{:20d}", value);
    }
    else if (dataType == DataType::Uint8)
    {
        uint8_t value = *reinterpret_cast<const uint8_t*>(data);
        return std::format("{:3d}", value);
    }
    else if (dataType == DataType::Uint16)
    {
        uint16_t value = *reinterpret_cast<const uint16_t*>(data);
        return std::format("{:5d}", value);
    }
    else if (dataType == DataType::Uint32)
    {
        uint32_t value = *reinterpret_cast<const uint32_t*>(data);
        return std::format("{:10d}", value);
    }
    else if (dataType == DataType::Uint64)
    {
        uint64_t value = *reinterpret_cast<const uint64_t*>(data);
        return std::format("{:20d}", value);
    }
    else if (dataType == DataType::Bool)
    {
        Bool value = *reinterpret_cast<const Bool*>(data);
        return std::format("{}", value ? '1' : '0');
    }
    else if (dataType == DataType::Complex32)
    {
        rad::Complex32 value = *reinterpret_cast<const rad::Complex32*>(data);
        return std::format("({:11.4e}, {:11.4e})", float(value.real()), float(value.imag()));
    }
    else if (dataType == DataType::Complex64)
    {
        rad::Complex64 value = *reinterpret_cast<const rad::Complex64*>(data);
        return std::format("({:11.4e}, {:11.4e})", float(value.real()), float(value.imag()));
    }
    else if (dataType == DataType::Complex128)
    {
        rad::Complex128 value = *reinterpret_cast<const rad::Complex128*>(data);
        return std::format("({:12.4e}, {:12.4e})", value.real(), value.imag());
    }
    else
    {
        RAD_UNREACHABLE();
        return {};
    }
}

std::string ToStringFixedWidthHex(const void* data, DataType dataType)
{
    size_t elementSize = GetElementSize(dataType);
    if (dataType == DataType::Complex32)
    {
        const uint16_t* value = reinterpret_cast<const uint16_t*>(data);
        return std::format("(0x{:04X}, 0x{:04X})", value[0], value[1]);
    }
    else if (dataType == DataType::Complex64)
    {
        const uint32_t* value = reinterpret_cast<const uint32_t*>(data);
        return std::format("(0x{:08X}, 0x{:08X})", value[0], value[1]);
    }
    else if (dataType == DataType::Complex128)
    {
        const uint64_t* value = reinterpret_cast<const uint64_t*>(data);
        return std::format("(0x{:016X}, 0x{:016X})", value[0], value[1]);
    }
    else if (elementSize == 1)
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

} // namespace ML
