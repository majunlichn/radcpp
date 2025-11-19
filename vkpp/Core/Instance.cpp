#include <vkpp/Core/Instance.h>
#include <vkpp/Core/Device.h>
#if defined(_WIN32)
#include <Windows.h>
#include <vulkan/vulkan_win32.h>
#endif

#include <set>

namespace vkpp
{

Instance::Instance()
{
    VKPP_LOG(trace, __func__);
    m_apiVersion = m_context.enumerateInstanceVersion();
    VKPP_LOG(info, "Instance Version: {}.{}.{}",
        VK_VERSION_MAJOR(m_apiVersion), VK_VERSION_MINOR(m_apiVersion), VK_VERSION_PATCH(m_apiVersion));
}

Instance::~Instance()
{
    VKPP_LOG(trace, __func__);
}

std::vector<vk::LayerProperties> Instance::EnumerateInstanceLayers()
{
    return m_context.enumerateInstanceLayerProperties();
}

std::vector<vk::ExtensionProperties> Instance::EnumerateInstanceExtensions(vk::Optional<const std::string> layerName)
{
    return m_context.enumerateInstanceExtensionProperties(layerName);
}

bool Instance::Init(
    std::string_view appName, uint32_t appVersion,
    std::string_view engineName, uint32_t engineVersion,
    const std::set<std::string>& requiredLayers, const std::set<std::string>& requiredExtensions)
{
    vk::ApplicationInfo appInfo(appName.data(), appVersion, engineName.data(), engineVersion, m_apiVersion);

    vk::InstanceCreateInfo instanceCreateInfo = {};
    instanceCreateInfo.pApplicationInfo = &appInfo;
    VK_STRUCTURE_CHAIN_BEGIN(instanceCreateInfo);

    auto supportedLayers = EnumerateInstanceLayers();
    auto supportedExtensions = EnumerateInstanceExtensions(nullptr);

    for (const std::string& requiredLayer : requiredLayers)
    {
        if (HasLayer(supportedLayers, requiredLayer))
        {
            m_enabledLayers.insert(requiredLayer);
        }
        else
        {
            VKPP_LOG(warn, "Instance dones't support layer {}", requiredLayer);
        }
    }

    for (const std::string& requiredExtension : requiredExtensions)
    {
        if (HasExtension(supportedExtensions, requiredExtension))
        {
            m_enabledExtensions.insert(requiredExtension);
        }
        else
        {
            VKPP_LOG(warn, "Instance extension not supported: {}", requiredExtension);
        }
    }

    // Layers/extensions that should be always enabled:
    if (HasExtension(supportedExtensions, VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME))
    {
        m_enabledExtensions.insert(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
    }
    if (HasExtension(supportedExtensions, VK_KHR_SURFACE_EXTENSION_NAME))
    {
        m_enabledExtensions.insert(VK_KHR_SURFACE_EXTENSION_NAME);
    }
    if (HasExtension(supportedExtensions, VK_KHR_GET_SURFACE_CAPABILITIES_2_EXTENSION_NAME))
    {
        m_enabledExtensions.insert(VK_KHR_GET_SURFACE_CAPABILITIES_2_EXTENSION_NAME);
    }
#if defined(_WIN32)
    if (HasExtension(supportedExtensions, VK_KHR_WIN32_SURFACE_EXTENSION_NAME))
    {
        m_enabledExtensions.insert(VK_KHR_WIN32_SURFACE_EXTENSION_NAME);
    }
#endif

#if defined(_DEBUG)
    bool enableValidation = true;
#else
    bool enableValidation = false;
#endif

    if (const char* envVulkanSDKPath = std::getenv("VULKAN_SDK"))
    {
        VKPP_LOG(info, "VulkanSDK Path: {}", envVulkanSDKPath);
    }
    else
    {
        VKPP_LOG(warn, "VulkanSDK is not available!");
    }

    if (const char* envEnableValidation = std::getenv("VKPP_ENABLE_VALIDATION"))
    {
        VKPP_LOG(info, "VKPP_ENABLE_VALIDATION={}.", envEnableValidation);
        enableValidation = rad::StrToBool(envEnableValidation);
    }

    if (enableValidation)
    {
        if (HasLayer(supportedLayers, "VK_LAYER_KHRONOS_validation") &&
            HasExtension(supportedExtensions, VK_EXT_DEBUG_UTILS_EXTENSION_NAME))
        {
            m_enabledLayers.insert("VK_LAYER_KHRONOS_validation");
            m_enabledExtensions.insert(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        }
        else
        {
            VKPP_LOG(warn, "Cannot enable validation due to missing layer VK_LAYER_KHRONOS_validation or extension VK_EXT_debug_utils!");
        }
    }

    vk::DebugUtilsMessengerCreateInfoEXT debugUtilsMessengerCreateInfo =
    {
        {}, // flags
        vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose |
        vk::DebugUtilsMessageSeverityFlagBitsEXT::eInfo |
        vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
        vk::DebugUtilsMessageSeverityFlagBitsEXT::eError,
        vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
        vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation |
        vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance,
        DebugUtilsMessengerCallback
    };

    if (m_enabledLayers.contains("VK_LAYER_KHRONOS_validation") &&
        m_enabledExtensions.contains(VK_EXT_DEBUG_UTILS_EXTENSION_NAME))
    {
        VK_STRUCTURE_CHAIN_ADD(instanceCreateInfo, debugUtilsMessengerCreateInfo);
        enableValidation = true;
    }
    else
    {
        enableValidation = false;
    }

    std::vector<const char*> enabledLayers;
    for (const std::string& layer : m_enabledLayers)
    {
        enabledLayers.push_back(layer.c_str());
    }
    instanceCreateInfo.enabledLayerCount = static_cast<uint32_t>(enabledLayers.size());
    instanceCreateInfo.ppEnabledLayerNames = enabledLayers.data();

    std::vector<const char*> enabledExtensions;
    for (const std::string& extension : m_enabledExtensions)
    {
        enabledExtensions.push_back(extension.c_str());
    }
    instanceCreateInfo.enabledExtensionCount = static_cast<uint32_t>(enabledExtensions.size());
    instanceCreateInfo.ppEnabledExtensionNames = enabledExtensions.data();

    VK_STRUCTURE_CHAIN_END(instanceCreateInfo);
    m_wrapper = m_context.createInstance(instanceCreateInfo);
    for (const std::string& layer : m_enabledLayers)
    {
        VKPP_LOG(info, "Instance layer enabled: {}", layer);
    }
    for (const std::string& extension : m_enabledExtensions)
    {
        VKPP_LOG(info, "Instance extension enabled: {}", extension);
    }

    if (enableValidation)
    {
        m_debugUtilsMessenger = m_wrapper.createDebugUtilsMessengerEXT(debugUtilsMessengerCreateInfo);
    }

    m_physicalDevices = m_wrapper.enumeratePhysicalDevices();
    for (size_t physicalDeviceIndex = 0; physicalDeviceIndex < m_physicalDevices.size(); physicalDeviceIndex++)
    {
        auto& physicalDevice = m_physicalDevices[physicalDeviceIndex];
        VKPP_LOG(info, "GPU#{}: {}", physicalDeviceIndex, physicalDevice.getProperties().deviceName.data());
        auto queueFamilies = physicalDevice.getQueueFamilyProperties();
        const uint32_t& apiVersion = physicalDevice.getProperties().apiVersion;
        VKPP_LOG(info, "API Version: {}.{}.{}",
            VK_VERSION_MAJOR(apiVersion), VK_VERSION_MINOR(apiVersion), VK_VERSION_PATCH(apiVersion));
        for (size_t queueFamilyIndex = 0; queueFamilyIndex < queueFamilies.size(); ++queueFamilyIndex)
        {
            const auto& queueFamily = queueFamilies[queueFamilyIndex];
            VKPP_LOG(info, "QueueFamily#{}: {}", queueFamilyIndex, vk::to_string(queueFamily.queueFlags));
        }
    }

    return !m_physicalDevices.empty();
}

rad::Ref<Device> Instance::CreateDevice()
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

rad::Ref<Device> Instance::CreateDevice(vk::raii::PhysicalDevice& physicalDevice)
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

    if (requiredExtensions.contains("VK_EXT_debug_utils"))
    {
        requiredExtensions.erase("VK_EXT_debug_report");
        requiredExtensions.erase("VK_EXT_debug_marker");
    }

    if (!requiredExtensions.contains("VK_EXT_debug_report"))
    {
        requiredExtensions.erase("VK_EXT_debug_marker");
    }

    if (!requiredExtensions.contains("VK_KHR_surface_maintenance1"))
    {
        requiredExtensions.erase("VK_KHR_swapchain_maintenance1");
    }

    return CreateDevice(physicalDevice, requiredExtensions);
}

rad::Ref<Device> Instance::CreateDevice(vk::raii::PhysicalDevice& physicalDevice, const std::set<std::string>& requiredExtensions)
{
    return RAD_NEW Device(this, physicalDevice, requiredExtensions);
}

} // namespace vkpp
