#define VMA_IMPLEMENTATION 1
#include <vkpp/Core/Common.h>
#include <vulkan/utility/vk_format_utils.h>

VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE;

namespace vkpp
{

VKAPI_ATTR VkBool32 VKAPI_CALL DebugUtilsMessengerCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT severity,
    VkDebugUtilsMessageTypeFlagsEXT type,
    const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
    void* pUserData)
{
    switch (severity)
    {
    case VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT:
        VKPP_LOG(debug, "[{}] {}",
            pCallbackData->pMessageIdName, pCallbackData->pMessage);
        break;
    case VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT:
        VKPP_LOG(info, "[{}] {}",
            pCallbackData->pMessageIdName, pCallbackData->pMessage);
        break;
    case VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT:
        VKPP_LOG(warn, "[{}] {}",
            pCallbackData->pMessageIdName, pCallbackData->pMessage);
        break;
    case VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT:
        VKPP_LOG(err, "[{}] {}",
            pCallbackData->pMessageIdName, pCallbackData->pMessage);
#if defined(_DEBUG)
#if defined(RAD_COMPILER_MSVC)
        __debugbreak();
#endif
#endif
        break;
    }

    return VK_FALSE;
}

vk::ImageAspectFlags GetDefaultImageAspectFlags(vk::Format format)
{
    if (vkuFormatIsColor(VkFormat(format))) [[likely]]
    {
        return vk::ImageAspectFlagBits::eColor;
    }
    else if (vkuFormatIsDepthOnly(VkFormat(format)))
    {
        return vk::ImageAspectFlagBits::eDepth;
    }
    else if (vkuFormatIsDepthAndStencil(VkFormat(format)))
    {
        return (vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil);
    }
    else
    {
        return vk::ImageAspectFlagBits::eNone;
    }
}

spdlog::logger* GetLogger()
{
    static std::shared_ptr<spdlog::logger> logger = rad::CreateLogger("Vulkan");
    return logger.get();
}

void ReportError(vk::Result result, const char* call, std::source_location sourceLoc)
{
    VKPP_LOG(err, "{} failed with {} ({}, line {}, function {}).",
        call, string_VkResult(static_cast<VkResult>(result)),
        sourceLoc.file_name(), sourceLoc.line(), sourceLoc.function_name());
    throw vk::SystemError(vk::make_error_code(result), string_VkResult(static_cast<VkResult>(result)));
}

VkDeviceSize GetComponentSizeInBytes(vk::ComponentTypeKHR type)
{
    switch (type)
    {
    case vk::ComponentTypeKHR::eFloat16: return 2;
    case vk::ComponentTypeKHR::eFloat32: return 4;
    case vk::ComponentTypeKHR::eFloat64: return 8;
    case vk::ComponentTypeKHR::eSint8: return 1;
    case vk::ComponentTypeKHR::eSint16: return 2;
    case vk::ComponentTypeKHR::eSint32: return 4;
    case vk::ComponentTypeKHR::eSint64: return 8;
    case vk::ComponentTypeKHR::eUint8: return 1;
    case vk::ComponentTypeKHR::eUint16: return 2;
    case vk::ComponentTypeKHR::eUint32: return 4;
    case vk::ComponentTypeKHR::eUint64: return 8;
    case vk::ComponentTypeKHR::eSint8PackedNV: return 4;
    case vk::ComponentTypeKHR::eUint8PackedNV: return 4;
    case vk::ComponentTypeKHR::eFloatE4M3: return 1;
    case vk::ComponentTypeKHR::eFloatE5M2: return 1;
    default: RAD_UNREACHABLE();
    }
}

bool IsFloatingPointType(vk::ComponentTypeKHR type)
{
    if ((type == vk::ComponentTypeKHR::eFloat16) ||
        (type == vk::ComponentTypeKHR::eFloat32) ||
        (type == vk::ComponentTypeKHR::eFloat64) ||
        (type == vk::ComponentTypeKHR::eFloatE4M3) ||
        (type == vk::ComponentTypeKHR::eFloatE5M2))
    {
        return true;
    }
    else
    {
        return false;
    }
}

bool IsSignedIntegerType(vk::ComponentTypeKHR type)
{
    if ((type == vk::ComponentTypeKHR::eSint8) ||
        (type == vk::ComponentTypeKHR::eSint16) ||
        (type == vk::ComponentTypeKHR::eSint32) ||
        (type == vk::ComponentTypeKHR::eSint64) ||
        (type == vk::ComponentTypeKHR::eSint8PackedNV))
    {
        return true;
    }
    else
    {
        return false;
    }
}

bool IsUnsignedIntegerType(vk::ComponentTypeKHR type)
{
    if ((type == vk::ComponentTypeKHR::eUint8) ||
        (type == vk::ComponentTypeKHR::eUint16) ||
        (type == vk::ComponentTypeKHR::eUint32) ||
        (type == vk::ComponentTypeKHR::eUint64) ||
        (type == vk::ComponentTypeKHR::eUint8PackedNV))
    {
        return true;
    }
    else
    {
        return false;
    }
}

bool IsIntegerType(vk::ComponentTypeKHR type)
{
    return IsSignedIntegerType(type) || IsUnsignedIntegerType(type);
}

vk::ImageAspectFlags GetImageAspectFromFormat(vk::Format format)
{
    if (vkuFormatIsColor(static_cast<VkFormat>(format)))
    {
        return vk::ImageAspectFlagBits::eColor;
    }
    else if (vkuFormatIsDepthOnly(static_cast<VkFormat>(format)))
    {
        return vk::ImageAspectFlagBits::eDepth;
    }
    else if (vkuFormatIsStencilOnly(static_cast<VkFormat>(format)))
    {
        return vk::ImageAspectFlagBits::eStencil;
    }
    else if (vkuFormatIsDepthAndStencil(static_cast<VkFormat>(format)))
    {
        return vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil;
    }
    else
    {
        return vk::ImageAspectFlagBits::eNone;
    }
}

std::string FormatValueFixedWidthDec(vk::ComponentTypeKHR dataType, const void* data)
{
    if (dataType == vk::ComponentTypeKHR::eFloat16)
    {
        uint16_t value = *reinterpret_cast<const uint16_t*>(data);
        return std::format("{:11.4f}", rad::fp16_ieee_to_fp32_value(value));
    }
    else if (dataType == vk::ComponentTypeKHR::eFloat32)
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
    else if (dataType == vk::ComponentTypeKHR::eFloat64)
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
    else if (dataType == vk::ComponentTypeKHR::eSint8)
    {
        int8_t value = *reinterpret_cast<const int8_t*>(data);
        return std::format("{:4d}", value);
    }
    else if (dataType == vk::ComponentTypeKHR::eSint16)
    {
        int16_t value = *reinterpret_cast<const int16_t*>(data);
        return std::format("{:6d}", value);
    }
    else if (dataType == vk::ComponentTypeKHR::eSint32)
    {
        int32_t value = *reinterpret_cast<const int32_t*>(data);
        return std::format("{:11d}", value);
    }
    else if (dataType == vk::ComponentTypeKHR::eSint64)
    {
        int64_t value = *reinterpret_cast<const int64_t*>(data);
        return std::format("{:20d}", value);
    }
    else if (dataType == vk::ComponentTypeKHR::eUint8)
    {
        uint8_t value = *reinterpret_cast<const uint8_t*>(data);
        return std::format("{:3d}", value);
    }
    else if (dataType == vk::ComponentTypeKHR::eUint16)
    {
        uint16_t value = *reinterpret_cast<const uint16_t*>(data);
        return std::format("{:5d}", value);
    }
    else if (dataType == vk::ComponentTypeKHR::eUint32)
    {
        uint32_t value = *reinterpret_cast<const uint32_t*>(data);
        return std::format("{:10d}", value);
    }
    else if (dataType == vk::ComponentTypeKHR::eUint64)
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

std::string FormatValueFixedWidthDecNA(vk::ComponentTypeKHR dataType)
{
    if (dataType == vk::ComponentTypeKHR::eFloat16)
    {
        return std::format("{:>11}", "NA");
    }
    else if (dataType == vk::ComponentTypeKHR::eFloat32)
    {
        return std::format("{:>14}", "NA");
    }
    else if (dataType == vk::ComponentTypeKHR::eFloat64)
    {
        return std::format("{:>14}", "NA");
    }
    else if (dataType == vk::ComponentTypeKHR::eSint8)
    {
        return std::format("{:>4}", "NA");
    }
    else if (dataType == vk::ComponentTypeKHR::eSint16)
    {
        return std::format("{:>6}", "NA");
    }
    else if (dataType == vk::ComponentTypeKHR::eSint32)
    {
        return std::format("{:>11}", "NA");
    }
    else if (dataType == vk::ComponentTypeKHR::eSint64)
    {
        return std::format("{:>20}", "NA");
    }
    else if (dataType == vk::ComponentTypeKHR::eUint8)
    {
        return std::format("{:>3}", "NA");
    }
    else if (dataType == vk::ComponentTypeKHR::eUint16)
    {
        return std::format("{:>5}", "NA");
    }
    else if (dataType == vk::ComponentTypeKHR::eUint32)
    {
        return std::format("{:>10}", "NA");
    }
    else if (dataType == vk::ComponentTypeKHR::eUint64)
    {
        return std::format("{:>20}", "NA");
    }
    else
    {
        RAD_UNREACHABLE();
        return {};
    }
}

std::string FormatValueFixedWidthHex(vk::ComponentTypeKHR dataType, const void* data)
{
    size_t elementSize = GetComponentSizeInBytes(dataType);
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

std::string FormatValueFixedWidthHexNA(vk::ComponentTypeKHR dataType)
{
    size_t elementSize = GetComponentSizeInBytes(dataType);
    if (elementSize == 1)
    {
        return std::format("{:>4}", "NA");
    }
    else if (elementSize == 2)
    {
        return std::format("{:>6}", "NA");
    }
    else if (elementSize == 4)
    {
        return std::format("{:>10}", "NA");
    }
    else if (elementSize == 8)
    {
        return std::format("{:>18}", "NA");
    }
    else
    {
        RAD_UNREACHABLE();
        return {};
    }
}

} // namespace vkpp
