#pragma once

#define VK_NO_PROTOTYPES 1
#define VK_ENABLE_BETA_EXTENSIONS 1
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
#define VULKAN_HPP_FLAGS_MASK_TYPE_AS_PUBLIC 1

#include <radcpp/Core/Platform.h>
#include <radcpp/Core/Integer.h>
#include <radcpp/Core/Memory.h>
#include <radcpp/Core/RefCounted.h>
#include <radcpp/Core/String.h>
#include <radcpp/Core/TypeTraits.h>
#include <radcpp/Container/SmallVector.h>
#include <radcpp/Container/Span.h>
#include <radcpp/IO/Logging.h>

#include <vulkan/vulkan_raii.hpp>
#include <vulkan/vk_enum_string_helper.h>

#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1
#include <vma/vk_mem_alloc.h>

#include <cmath>
#include <source_location>

namespace vkpp
{

#define VK_STRUCTURE_CHAIN_BEGIN(Head) auto Head##ChainNext = &Head.pNext;
#define VK_STRUCTURE_CHAIN_ADD(Head, Next) do { *Head##ChainNext = &Next; Head##ChainNext = &Next.pNext; } while(0)
#define VK_STRUCTURE_CHAIN_END(Head) do { *Head##ChainNext = nullptr; } while(0)

template<typename T, typename Head, typename = std::enable_if_t<std::is_const_v<Head>>>
const T* GetFromStructureChain(Head& head, vk::StructureType type)
{
    const vk::BaseInStructure* pStructure = reinterpret_cast<const vk::BaseInStructure*>(&head);
    do {
        if (pStructure->sType == type)
        {
            return reinterpret_cast<const T*>(pStructure);
        }
        pStructure = pStructure->pNext;
    } while (pStructure != nullptr);
    return nullptr;
}

template<typename T, typename Head, typename = std::enable_if_t<!std::is_const_v<Head>>>
T* GetFromStructureChain(Head& head, vk::StructureType type)
{
    vk::BaseOutStructure* pStructure = reinterpret_cast<vk::BaseOutStructure*>(&head);
    do {
        if (pStructure->sType == type)
        {
            return reinterpret_cast<T*>(pStructure);
        }
        pStructure = pStructure->pNext;
    } while (pStructure != nullptr);
    return nullptr;
}

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

vk::ImageAspectFlags GetDefaultImageAspectFlags(vk::Format format);

} // namespace vkpp

namespace rad
{

enum class VulkanQueueFamily
{
    Graphics,
    Compute,    // Async Compute Engine (ACE)
    Transfer,   // DMA
    Count
};

spdlog::logger* GetVulkanLogger();
#define LOG_VULKAN(LogLevel, ...) RAD_LOGGER_CALL(rad::GetVulkanLogger(), LogLevel, __VA_ARGS__)
void ReportVulkanError(VkResult result, const char* expr, std::source_location sourceLoc = std::source_location::current());
#define VK_CHECK(Expr) \
    do { const VkResult result = Expr; if (result < 0) { ReportVulkanError(result, #Expr); } } while(0)

} // namespace rad
