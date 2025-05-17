#pragma once

#include <vkpp/Core/Common.h>

namespace vkpp
{

class Device;

class Instance : public rad::RefCounted<Instance>
{
public:
    Instance();
    ~Instance();

    vk::Instance GetHandle() const { return static_cast<vk::Instance>(m_wrapper); }

    std::vector<VkLayerProperties> EnumerateInstanceLayers();
    std::vector<VkExtensionProperties> EnumerateInstanceExtensions(const char* layerName);

    bool Init(std::string_view appName, uint32_t appVersion,
        std::string_view engineName, uint32_t engineVersion,
        const std::set<std::string>& requiredLayers, const std::set<std::string>& requiredExtensions);
    // Init with common instance layers and extensions.
    bool Init(std::string_view appName, uint32_t appVersion,
        std::string_view engineName, uint32_t engineVersion);
    bool Init(std::string_view appName, uint32_t appVersion)
    {
        return Init(appName, appVersion, appName, appVersion);
    }

    uint32_t GetApiVersion() const { return m_apiVersion; }

    bool IsLayerEnabled(std::string_view name) const
    {
        return (m_enabledLayers.find(name) != m_enabledLayers.end());
    }

    bool IsExtensionEnabled(std::string_view name) const
    {
        return (m_enabledExtensions.find(name) != m_enabledExtensions.end());
    }

    // Select physical device automatically, prefer the first discrete one.
    rad::Ref<Device> CreateDevice();
    // Create device with all KHR and EXT extensions available.
    rad::Ref<Device> CreateDevice(vk::raii::PhysicalDevice& physicalDevice);
    rad::Ref<Device> CreateDevice(
        vk::raii::PhysicalDevice& physicalDevice, const std::set<std::string>& requiredExtensions);

    vk::raii::Context m_context = {};
    uint32_t m_apiVersion = 0;
    std::set<std::string, rad::StringLess> m_enabledLayers;
    std::set<std::string, rad::StringLess> m_enabledExtensions;
    vk::raii::Instance m_wrapper = { nullptr };
    vk::raii::DebugUtilsMessengerEXT m_debugUtilsMessenger = { nullptr };
    vk::raii::PhysicalDevices m_physicalDevices = { nullptr };

}; // class Instance

} // namespace vkpp
