#pragma once

#define VK_NO_PROTOTYPES 1
#define VK_ENABLE_BETA_EXTENSIONS 1
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
#define VULKAN_HPP_FLAGS_MASK_TYPE_AS_PUBLIC 1

#include <rad/Core/Platform.h>

#include <rad/Core/Integer.h>
#include <rad/Core/Memory.h>
#include <rad/Core/RefCounted.h>
#include <rad/Core/String.h>
#include <rad/Core/TypeTraits.h>
#include <rad/Container/SmallVector.h>
#include <rad/Container/Span.h>
#include <rad/IO/File.h>
#include <rad/IO/Logging.h>

#include <vulkan/vulkan_raii.hpp>
#include <vulkan/vk_enum_string_helper.h>

#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1
#include <vma/vk_mem_alloc.h>

#include <cmath>
#include <source_location>

namespace vkpp
{

using DeviceDispatcher = vk::raii::detail::DeviceDispatcher;

enum class QueueFamily
{
    Universal,
    Graphics,
    Compute,    // Async Compute Engine (ACE)
    Transfer,   // Async Transfer (DMA)
    Count
};

struct SubmitWaitInfo
{
    vk::Semaphore semaphore;
    vk::PipelineStageFlagBits dstStageMask;
};

#define VK_STRUCTURE_CHAIN_CREATE(Head) auto Head##ChainNext = &Head.pNext;
#define VK_STRUCTURE_CHAIN_APPEND(Head, Next) do { *Head##ChainNext = &Next; Head##ChainNext = &Next.pNext; } while(0)
#define VK_STRUCTURE_CHAIN_END(Head) do { *Head##ChainNext = nullptr; } while(0)

template<typename T, typename Head>
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

spdlog::logger* GetLogger();
#define VKPP_LOG(LogLevel, ...) SPDLOG_LOGGER_CALL(vkpp::GetLogger(), spdlog::level::LogLevel, __VA_ARGS__)
void ReportError(VkResult result, const char* expr, std::source_location sourceLoc = std::source_location::current());
#define VK_CHECK(Expr) \
    do { const VkResult result_ = static_cast<VkResult>(Expr); if (result_ < 0) { vkpp::ReportError(result_, #Expr); } } while(0)

VkDeviceSize GetComponentSizeInBytes(vk::ComponentTypeKHR type);
bool IsFloatingPointType(vk::ComponentTypeKHR type);
bool IsSignedIntegerType(vk::ComponentTypeKHR type);
bool IsUnsignedIntegerType(vk::ComponentTypeKHR type);
bool IsIntegerType(vk::ComponentTypeKHR type);

vk::ImageAspectFlags GetImageAspectFromFormat(vk::Format format);

} // namespace vkpp
