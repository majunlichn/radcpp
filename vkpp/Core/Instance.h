#pragma once

#include <vkpp/Core/Common.h>

namespace vkpp
{

class Instance : public rad::RefCounted<Instance>
{
public:
    Instance();
    ~Instance();

    vk::Instance GetHandle() const { return static_cast<vk::Instance>(m_wrapper); }
    const InstanceDispatcher* GetDispatcher() const { return m_wrapper.getDispatcher(); }

    PFN_vkVoidFunction GetProcAddr(const char* name) const
    {
        return m_wrapper.getProcAddr(name);
    }

    std::vector<vk::LayerProperties> EnumerateInstanceLayers();
    std::vector<vk::ExtensionProperties> EnumerateInstanceExtensions(vk::Optional<const std::string> layerName = nullptr);

    bool Init(std::string_view appName, uint32_t appVersion,
        std::string_view engineName, uint32_t engineVersion,
        const InstanceConfig& config);
    // Init with common instance layers and extensions.
    bool Init(std::string_view appName, uint32_t appVersion,
        std::string_view engineName, uint32_t engineVersion);
    bool Init(std::string_view appName, uint32_t appVersion);

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
    rad::Ref<Device> CreateDevice(vk::raii::PhysicalDevice& physicalDevice, const DeviceConfig& config);

    vk::raii::Context m_context = {};
    uint32_t m_apiVersion = 0;
    InstanceConfig m_config = {};
    std::set<std::string, rad::StringLess> m_enabledLayers;
    std::set<std::string, rad::StringLess> m_enabledExtensions;
    vk::raii::Instance m_wrapper = { nullptr };
    vk::raii::DebugUtilsMessengerEXT m_debugUtilsMessenger = { nullptr };
    std::vector<vk::raii::PhysicalDevice> m_physicalDevices;

}; // class Instance

} // namespace vkpp
