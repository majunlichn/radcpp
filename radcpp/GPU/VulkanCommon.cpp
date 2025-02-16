#define VMA_IMPLEMENTATION 1
#include <radcpp/GPU/VulkanCommon.h>

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

} // namespace vkpp

namespace rad
{

spdlog::logger* GetVulkanLogger()
{
    static std::shared_ptr<spdlog::logger> VulkanLogger = rad::CreateLogger("Vulkan");
    return VulkanLogger.get();
}

void ReportVulkanError(VkResult result, const char* function, const char* file, uint32_t line)
{
    if (result < 0)
    {
        LOG_VULKAN(err, "{} failed with VkResult={} in {} ({}, line {}).",
            function, string_VkResult(result), function, file, line);
        throw vk::SystemError(std::error_code(static_cast<int>(result), vk::errorCategory()), string_VkResult(result));
    }
}

} // namespace rad
