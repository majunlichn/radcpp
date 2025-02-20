#define VMA_IMPLEMENTATION 1
#include <radcpp/GPU/VulkanCommon.h>
#include <vulkan/utility/vk_format_utils.h>
#include <set>

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
        LOG_VULKAN(debug, "[{}] {}",
            pCallbackData->pMessageIdName, pCallbackData->pMessage);
        break;
    case VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT:
        LOG_VULKAN(info, "[{}] {}",
            pCallbackData->pMessageIdName, pCallbackData->pMessage);
        break;
    case VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT:
        LOG_VULKAN(warn, "[{}] {}",
            pCallbackData->pMessageIdName, pCallbackData->pMessage);
        break;
    case VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT:
        LOG_VULKAN(err, "[{}] {}",
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

} // namespace vkpp

namespace rad
{

spdlog::logger* GetVulkanLogger()
{
    static std::shared_ptr<spdlog::logger> VulkanLogger = rad::CreateLogger("Vulkan");
    return VulkanLogger.get();
}

void ReportVulkanError(VkResult result, const char* expr, std::source_location sourceLoc)
{
    if (result < 0)
    {
        LOG_VULKAN(err, "{} failed with {} ({}, line {}, function {}).",
            expr, string_VkResult(result),
            sourceLoc.file_name(), sourceLoc.line(), sourceLoc.function_name());
        throw vk::SystemError(std::error_code(static_cast<int>(result), vk::errorCategory()), string_VkResult(result));
    }
}

} // namespace rad
