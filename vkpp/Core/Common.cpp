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
    static std::shared_ptr<spdlog::logger> VulkanLogger = rad::CreateLogger("Vulkan");
    return VulkanLogger.get();
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
    case vk::ComponentTypeKHR::eFloatE4M3NV: return 1;
    case vk::ComponentTypeKHR::eFloatE5M2NV: return 1;
    default: RAD_UNREACHABLE();
    }
}

bool IsFloatingPointType(vk::ComponentTypeKHR type)
{
    if ((type == vk::ComponentTypeKHR::eFloat16) ||
        (type == vk::ComponentTypeKHR::eFloat32) ||
        (type == vk::ComponentTypeKHR::eFloat64) ||
        (type == vk::ComponentTypeKHR::eFloatE4M3NV) ||
        (type == vk::ComponentTypeKHR::eFloatE5M2NV))
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

} // namespace vkpp
