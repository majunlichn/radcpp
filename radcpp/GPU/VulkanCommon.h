#pragma once

#define VK_NO_PROTOTYPES 1
#define VK_ENABLE_BETA_EXTENSIONS 1
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1

#include <radcpp/Core/Platform.h>
#include <radcpp/Core/Integer.h>
#include <radcpp/Core/String.h>
#include <radcpp/Container/SmallVector.h>
#include <radcpp/Container/Span.h>
#include <radcpp/IO/Logging.h>

#include <vulkan/vulkan_raii.hpp>
#include <vulkan/vk_enum_string_helper.h>

#include <cmath>

namespace vkpp
{

inline bool IsVersionMatchOrGreater(uint32_t version, uint32_t major, uint32_t minor, uint32_t patch)
{
    return (VK_VERSION_MAJOR(version) >= major) &&
        (VK_VERSION_MINOR(version) >= minor) &&
        (VK_VERSION_PATCH(version) >= patch);
}

template<typename Layers>
bool HasLayer(Layers layers, std::string_view name)
{
    for (const VkLayerProperties& layer : layers)
    {
        if (rad::StrEqual(layer.layerName, name))
        {
            return true;
        }
    }
    return false;
}

template<typename Extensions>
bool HasExtension(Extensions extensions, std::string_view name)
{
    for (const VkExtensionProperties& extension : extensions)
    {
        if (rad::StrEqual(extension.extensionName, name))
        {
            return true;
        }
    }
    return false;
}

VKAPI_ATTR VkBool32 VKAPI_CALL DebugUtilsMessengerCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT severity,
    VkDebugUtilsMessageTypeFlagsEXT type,
    const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
    void* pUserData);

inline uint32_t GetMaxMipLevel(uint32_t width, uint32_t height)
{
    uint32_t maxExtent = std::max(width, height);
    return (uint32_t)std::log2f(float(maxExtent)) + 1;
}

inline uint32_t GetMaxMipLevel(uint32_t width, uint32_t height, uint32_t depth)
{
    uint32_t maxExtent = std::max(std::max(width, height), depth);
    return (uint32_t)std::log2f(float(maxExtent)) + 1;
}

} // namespace vkpp

namespace rad
{

spdlog::logger* GetVulkanLogger();
#define LOG_VULKAN(LogLevel, ...) RAD_LOGGER_CALL(rad::GetVulkanLogger(), LogLevel, __VA_ARGS__)
void ReportVulkanError(VkResult result, const char* function, const char* file, uint32_t line);
#define VK_CHECK_RETURN(Function) \
    do { const VkResult result = Function; if (result < 0) { ReportVulkanError(result, #Function, __FILE__, __LINE__); } } while(0)

} // namespace rad
