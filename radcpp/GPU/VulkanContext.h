#pragma once

#include <radcpp/GPU/VulkanCommon.h>

namespace rad
{

spdlog::logger* GetVulkanLogger();
#define LOG_VULKAN(LogLevel, ...) RAD_LOGGER_CALL(rad::GetVulkanLogger(), LogLevel, __VA_ARGS__)
void ReportVulkanError(VkResult result, const char* function, const char* file, uint32_t line);
#define VK_CHECK_RETURN(Function) \
    do { const VkResult result = Function; if (result < 0) { ReportVulkanError(result, #Function, __FILE__, __LINE__); } } while(0)

class VulkanContext
{
public:
    VulkanContext() {}
    ~VulkanContext() {}

    std::vector<VkLayerProperties> EnumerateInstanceLayers();
    std::vector<VkExtensionProperties> EnumerateInstanceExtensions(const char* layerName);

    bool Init(std::string_view appName, uint32_t appVersion, std::string_view engineName, uint32_t engineVersion,
        const std::set<std::string>& instanceLayers, const std::set<std::string>& instanceExtensions);
    // Init with common instance layers and extensions.
    bool Init(std::string_view appName, uint32_t appVersion, std::string_view engineName, uint32_t engineVersion);
    bool Init(std::string_view appName, uint32_t appVersion) { return Init(appName, appVersion, appName, appVersion); }

    uint32_t GetApiVersion() const { return m_apiVersion; }
    bool IsInstanceLayerEnabled(std::string_view name) const { return (m_enabledLayers.find(name) != m_enabledLayers.end()); }
    bool IsInstanceExtensionEnabled(std::string_view name) const { return (m_enabledExtensions.find(name) != m_enabledExtensions.end()); }

    vk::raii::Context m_context = {};
    uint32_t m_apiVersion = 0;
    std::set<std::string, rad::StringLess> m_enabledLayers;
    std::set<std::string, rad::StringLess> m_enabledExtensions;
    vk::raii::Instance m_instance = { nullptr };
    vk::raii::DebugUtilsMessengerEXT m_debugUtilsMessenger = { nullptr };
    vk::raii::PhysicalDevices m_physicalDevices = { nullptr };

}; // class VulkanContext

} // namespace rad
