#include <radcpp/GPU/VulkanInstance.h>
#include <radcpp/GPU/VulkanDevice.h>
#if defined(_WIN32)
#include <Windows.h>
#include <vulkan/vulkan_win32.h>
#endif

#include <set>

namespace rad
{

VulkanInstance::VulkanInstance()
{
    LOG_VULKAN(trace, __func__);
    m_context.getDispatcher()->vkEnumerateInstanceVersion(&m_apiVersion);
    LOG_VULKAN(info, "Instance Version: {}.{}.{}",
        VK_VERSION_MAJOR(m_apiVersion), VK_VERSION_MINOR(m_apiVersion), VK_VERSION_PATCH(m_apiVersion));
}

VulkanInstance::~VulkanInstance()
{
    LOG_VULKAN(trace, __func__);
}

std::vector<VkLayerProperties> VulkanInstance::EnumerateInstanceLayers()
{
    std::vector<VkLayerProperties> layers;
    uint32_t count = 0;
    VK_CHECK(m_context.getDispatcher()->
        vkEnumerateInstanceLayerProperties(&count, nullptr));
    if (count > 0)
    {
        layers.resize(count);
        VK_CHECK(m_context.getDispatcher()->
            vkEnumerateInstanceLayerProperties(&count, layers.data()));
    }
    return layers;
}

std::vector<VkExtensionProperties> VulkanInstance::EnumerateInstanceExtensions(const char* layerName)
{
    std::vector<VkExtensionProperties> extensions;
    uint32_t count = 0;
    VK_CHECK(m_context.getDispatcher()->
        vkEnumerateInstanceExtensionProperties(layerName, &count, nullptr));
    if (count > 0)
    {
        extensions.resize(count);
        VK_CHECK(m_context.getDispatcher()->
            vkEnumerateInstanceExtensionProperties(layerName, &count, extensions.data()));
    }
    return extensions;
}

bool VulkanInstance::Init(std::string_view appName, uint32_t appVersion, std::string_view engineName, uint32_t engineVersion,
    const std::set<std::string>& requiredLayers, const std::set<std::string>& requiredExtensions)
{
    vk::ApplicationInfo applicationInfo(appName.data(), appVersion, engineName.data(), engineVersion, m_apiVersion);
#if !defined(_DEBUG)
    vk::StructureChain<vk::InstanceCreateInfo> instanceCreateInfoChain = { {{}, &applicationInfo} };
#else
    vk::DebugUtilsMessengerCreateInfoEXT debugUtilsMessengerCreateInfo =
    {
        {}, // flags
        vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
        vk::DebugUtilsMessageSeverityFlagBitsEXT::eError,
        vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
        vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance |
        vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation,
        vkpp::DebugUtilsMessengerCallback
    };

    auto supportedLayers = EnumerateInstanceLayers();
    auto supportedExtensions = EnumerateInstanceExtensions(nullptr);

    std::vector<const char*> enabledLayers;
    for (const std::string& requiredLayer : requiredLayers)
    {
        if (vkpp::HasLayer(supportedLayers, requiredLayer))
        {
            auto [iter, inserted] = m_enabledLayers.insert(requiredLayer);
            if (inserted)
            {
                enabledLayers.push_back(iter->c_str());
            }
        }
        else
        {
            LOG_VULKAN(warn, "Instance dones't support layer {}", requiredLayer);
        }
    }

    std::vector<const char*> enabledExtensions;
    for (const std::string& requiredExtension : requiredExtensions)
    {
        if (vkpp::HasExtension(supportedExtensions, requiredExtension))
        {
            auto [iter, inserted] = m_enabledExtensions.insert(requiredExtension);
            if (inserted)
            {
                enabledExtensions.push_back(iter->c_str());
            }
        }
        else
        {
            LOG_VULKAN(warn, "Instance extension not supported: {}", requiredExtension);
        }
    }

    vk::StructureChain<vk::InstanceCreateInfo, vk::DebugUtilsMessengerCreateInfoEXT> instanceCreateInfoChain =
    {
        vk::InstanceCreateInfo{{}, &applicationInfo, enabledLayers, enabledExtensions },
        debugUtilsMessengerCreateInfo,
    };
#endif
    m_handle = vk::raii::Instance(m_context, instanceCreateInfoChain.get());
    for (const std::string& layer : m_enabledLayers)
    {
        LOG_VULKAN(info, "Instance layer enabled: {}", layer);
    }
    for (const std::string& extension : m_enabledExtensions)
    {
        LOG_VULKAN(info, "Instance extension enabled: {}", extension);
    }

#if defined(_DEBUG)
    vk::raii::DebugUtilsMessengerEXT debugUtilsMessenger(m_handle, debugUtilsMessengerCreateInfo);
#endif
    m_physicalDevices = vk::raii::PhysicalDevices(m_handle);
    for (size_t physicalDeviceIndex = 0; physicalDeviceIndex < m_physicalDevices.size(); physicalDeviceIndex++)
    {
        auto& physicalDevice = m_physicalDevices[physicalDeviceIndex];
        LOG_VULKAN(info, "PhysicalDevice#{}: {}", physicalDeviceIndex, physicalDevice.getProperties().deviceName.data());
        auto queueFamilies = physicalDevice.getQueueFamilyProperties();
        const uint32_t& apiVersion = physicalDevice.getProperties().apiVersion;
        LOG_VULKAN(info, "API Version: {}.{}.{}",
            VK_VERSION_MAJOR(apiVersion), VK_VERSION_MINOR(apiVersion), VK_VERSION_PATCH(apiVersion));
        for (size_t queueFamilyIndex = 0; queueFamilyIndex < queueFamilies.size(); ++queueFamilyIndex)
        {
            const auto& queueFamily = queueFamilies[queueFamilyIndex];
            LOG_VULKAN(info, "QueueFamily#{}: {}", queueFamilyIndex, vk::to_string(queueFamily.queueFlags));
        }
    }

    return !m_physicalDevices.empty();
}

bool VulkanInstance::Init(std::string_view appName, uint32_t appVersion, std::string_view engineName, uint32_t engineVersion)
{
    std::set<std::string> instanceLayers;
#if defined(_DEBUG)
    instanceLayers.insert("VK_LAYER_KHRONOS_validation");
#endif
    std::set<std::string> requiredExtension;
    requiredExtension.insert(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
    requiredExtension.insert(VK_KHR_SURFACE_EXTENSION_NAME);
    requiredExtension.insert(VK_KHR_GET_SURFACE_CAPABILITIES_2_EXTENSION_NAME);
#if defined(_WIN32)
    requiredExtension.insert(VK_KHR_WIN32_SURFACE_EXTENSION_NAME);
#endif
#if defined(_DEBUG)
    requiredExtension.insert(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
#endif
    return Init(appName, appVersion, engineName, engineVersion, instanceLayers, requiredExtension);
}

Ref<VulkanDevice> VulkanInstance::CreateDevice()
{
    if (m_physicalDevices.empty())
    {
        return nullptr;
    }
    vk::raii::PhysicalDevice physicalDevicePreferred = { nullptr };
    physicalDevicePreferred = m_physicalDevices[0];
    for (auto& physicalDevice : m_physicalDevices)
    {
        const auto props = physicalDevice.getProperties();
        if (props.deviceType == vk::PhysicalDeviceType::eDiscreteGpu)
        {
            physicalDevicePreferred = physicalDevice;
        }
    }
    return CreateDevice(physicalDevicePreferred);
}

Ref<VulkanDevice> VulkanInstance::CreateDevice(vk::raii::PhysicalDevice& physicalDevice)
{
    const auto supportedExtensions = physicalDevice.enumerateDeviceExtensionProperties();
    std::set<std::string> requiredExtensions;
    for (const auto& extension : supportedExtensions)
    {
        if (std::string_view(extension.extensionName).starts_with("VK_KHR") ||
            std::string_view(extension.extensionName).starts_with("VK_EXT"))
        {
            requiredExtensions.insert(extension.extensionName);
        }
    }

    if (!IsExtensionEnabled("VK_KHR_get_physical_device_properties2") ||
        !IsExtensionEnabled("VK_KHR_surface") ||
        !IsExtensionEnabled("VK_KHR_get_surface_capabilities2") ||
        !IsExtensionEnabled("VK_KHR_swapchain"))
    {
        requiredExtensions.erase("VK_EXT_full_screen_exclusive");
    }

    if (!requiredExtensions.contains("VK_EXT_surface_maintenance1"))
    {
        requiredExtensions.erase("VK_EXT_swapchain_maintenance1");
    }

    if (requiredExtensions.contains("VK_KHR_buffer_device_address"))
    {
        requiredExtensions.erase("VK_EXT_buffer_device_address");
    }

    return CreateDevice(physicalDevice, requiredExtensions);
}

Ref<VulkanDevice> VulkanInstance::CreateDevice(vk::raii::PhysicalDevice& physicalDevice, const std::set<std::string>& requiredExtensions)
{
    return RAD_NEW VulkanDevice(this, physicalDevice, requiredExtensions);
}

} // namespace rad
