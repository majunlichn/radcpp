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

} // namespace rad
