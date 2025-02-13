#include <radcpp/GPU/VulkanContext.h>
#if defined(_WIN32)
#include <Windows.h>
#include <vulkan/vulkan_win32.h>
#endif

#include <set>

namespace rad
{

std::vector<VkLayerProperties> VulkanContext::EnumerateInstanceLayers()
{
    std::vector<VkLayerProperties> layers;
    uint32_t count = 0;
    VK_CHECK_RETURN(m_context.getDispatcher()->
        vkEnumerateInstanceLayerProperties(&count, nullptr));
    if (count > 0)
    {
        layers.resize(count);
        VK_CHECK_RETURN(m_context.getDispatcher()->
            vkEnumerateInstanceLayerProperties(&count, layers.data()));
    }
    return layers;
}

std::vector<VkExtensionProperties> VulkanContext::EnumerateInstanceExtensions(const char* layerName)
{
    std::vector<VkExtensionProperties> extensions;
    uint32_t count = 0;
    VK_CHECK_RETURN(m_context.getDispatcher()->
        vkEnumerateInstanceExtensionProperties(layerName, &count, nullptr));
    if (count > 0)
    {
        extensions.resize(count);
        VK_CHECK_RETURN(m_context.getDispatcher()->
            vkEnumerateInstanceExtensionProperties(layerName, &count, extensions.data()));
    }
    return extensions;
}

bool VulkanContext::Init(std::string_view appName, uint32_t appVersion, std::string_view engineName, uint32_t engineVersion,
    const std::set<std::string>& instanceLayers, const std::set<std::string>& instanceExtensions)
{
    m_context.getDispatcher()->vkEnumerateInstanceVersion(&m_apiVersion);
    LOG_VULKAN(info, "Instance Version: {}.{}.{}",
        VK_VERSION_MAJOR(m_apiVersion), VK_VERSION_MINOR(m_apiVersion), VK_VERSION_PATCH(m_apiVersion));
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

    std::vector<VkLayerProperties> supportedLayers = EnumerateInstanceLayers();
    std::vector<VkExtensionProperties> supportedExtensions = EnumerateInstanceExtensions(nullptr);

    std::set<std::string> requiredLayers = instanceLayers;
    std::set<std::string> requiredExtensions = instanceExtensions;

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
            LOG_VULKAN(warn, "Instance dones't support extension {}", requiredExtension);
        }
    }

    vk::StructureChain<vk::InstanceCreateInfo, vk::DebugUtilsMessengerCreateInfoEXT> instanceCreateInfoChain =
    {
        vk::InstanceCreateInfo{{}, &applicationInfo, enabledLayers, enabledExtensions },
        debugUtilsMessengerCreateInfo,
    };
#endif
    m_instance = vk::raii::Instance(m_context, instanceCreateInfoChain.get());
#if defined(_DEBUG)
    vk::raii::DebugUtilsMessengerEXT debugUtilsMessenger(m_instance, debugUtilsMessengerCreateInfo);
#endif
    m_physicalDevices = vk::raii::PhysicalDevices(m_instance);
    for (size_t i = 0; i < m_physicalDevices.size(); ++i)
    {
        auto& physicalDevice = m_physicalDevices[i];
        LOG_VULKAN(info, "PhysicalDevice#{}: {}", i, physicalDevice.getProperties().deviceName.data());
    }

    return !m_physicalDevices.empty();
}

bool VulkanContext::Init(std::string_view appName, uint32_t appVersion, std::string_view engineName, uint32_t engineVersion)
{
    std::set<std::string> instanceLayers;
#if defined(_DEBUG)
    instanceLayers.insert("VK_LAYER_KHRONOS_validation");
#endif
    std::set<std::string> instanceExtensions;
    m_enabledExtensions.insert(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
    m_enabledExtensions.insert(VK_KHR_SURFACE_EXTENSION_NAME);
    m_enabledExtensions.insert(VK_KHR_GET_SURFACE_CAPABILITIES_2_EXTENSION_NAME);
#if defined(_WIN32)
    m_enabledExtensions.insert(VK_KHR_WIN32_SURFACE_EXTENSION_NAME);
#endif
#if defined(_DEBUG)
    instanceExtensions.insert(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
#endif
    return Init(appName, appVersion, engineName, engineVersion, instanceLayers, instanceExtensions);
}

} // namespace rad
