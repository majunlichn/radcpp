#include <radcpp/GPU/VulkanDevice.h>
#include <radcpp/GPU/VulkanInstance.h>

namespace rad
{

VulkanDevice::VulkanDevice(Ref<VulkanInstance> instance, vk::raii::PhysicalDevice physicalDevice,
    const std::set<std::string>& requiredExtensions) :
    m_instance(std::move(instance)),
    m_physicalDevice(physicalDevice)
{
    m_properties = m_physicalDevice.getProperties();
    const uint32_t& apiVersion = m_properties.apiVersion;
    if (vkpp::IsVersionMatchOrGreater(apiVersion, 1, 1, 0))
    {
        VK_STRUCTURE_CHAIN_BEGIN(m_properties2);
        if (vkpp::IsVersionMatchOrGreater(apiVersion, 1, 2, 0))
        {
            VK_STRUCTURE_CHAIN_ADD(m_properties2, m_vk11Properties);
        }
        if (vkpp::IsVersionMatchOrGreater(apiVersion, 1, 2, 0))
        {
            VK_STRUCTURE_CHAIN_ADD(m_properties2, m_vk12Properties);
        }
        if (vkpp::IsVersionMatchOrGreater(apiVersion, 1, 3, 0))
        {
            VK_STRUCTURE_CHAIN_ADD(m_properties2, m_vk13Properties);
        }
        VK_STRUCTURE_CHAIN_END(m_properties2);
        m_physicalDevice.getDispatcher()->vkGetPhysicalDeviceProperties2(
            static_cast<vk::PhysicalDevice>(m_physicalDevice),
            reinterpret_cast<VkPhysicalDeviceProperties2*>(&m_properties2));
    }

    m_queueFamilies = m_physicalDevice.getQueueFamilyProperties();
    m_memoryProperties = m_physicalDevice.getMemoryProperties();

    m_features = m_physicalDevice.getFeatures();
    if (vkpp::IsVersionMatchOrGreater(apiVersion, 1, 1, 0))
    {
        VK_STRUCTURE_CHAIN_BEGIN(m_features2);
        if (vkpp::IsVersionMatchOrGreater(apiVersion, 1, 2, 0))
        {
            VK_STRUCTURE_CHAIN_ADD(m_features2, m_Vulkan11Features);
        }
        if (vkpp::IsVersionMatchOrGreater(apiVersion, 1, 2, 0))
        {
            VK_STRUCTURE_CHAIN_ADD(m_features2, m_Vulkan12Features);
        }
        if (vkpp::IsVersionMatchOrGreater(apiVersion, 1, 3, 0))
        {
            VK_STRUCTURE_CHAIN_ADD(m_features2, m_Vulkan13Features);
        }
        VK_STRUCTURE_CHAIN_END(m_features2);
        m_physicalDevice.getDispatcher()->vkGetPhysicalDeviceFeatures2(
            static_cast<vk::PhysicalDevice>(m_physicalDevice),
            reinterpret_cast<VkPhysicalDeviceFeatures2*>(&m_features2));
    }

    std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos;
    m_queueFamilies = m_physicalDevice.getQueueFamilyProperties();
    m_queueFamilyIndices.fill(VK_QUEUE_FAMILY_IGNORED);
    float priority = 1.0f;
    for (uint32_t i = 0; i < m_queueFamilies.size(); i++)
    {
        if (!HasQueueFamily(VulkanQueueFamily::Graphics) &&
            (m_queueFamilies[i].queueFlags & vk::QueueFlagBits::eGraphics))
        {
            vk::DeviceQueueCreateInfo queueInfo = {};
            queueInfo.flags = {};
            queueInfo.queueFamilyIndex = i;
            queueInfo.queueCount = 1;
            queueInfo.pQueuePriorities = &priority;
            queueCreateInfos.emplace_back(queueInfo);
            m_queueFamilyIndices[size_t(VulkanQueueFamily::Graphics)] = i;
        }
        // Async Compute Engine (ACE): no graphics bit, has compute bit.
        else if (!HasQueueFamily(VulkanQueueFamily::Compute) &&
            !(m_queueFamilies[i].queueFlags & vk::QueueFlagBits::eGraphics) &&
            (m_queueFamilies[i].queueFlags & vk::QueueFlagBits::eCompute))
        {
            vk::DeviceQueueCreateInfo queueInfo = {};
            queueInfo.flags = {};
            queueInfo.queueFamilyIndex = i;
            queueInfo.queueCount = 1;
            queueInfo.pQueuePriorities = &priority;
            queueCreateInfos.emplace_back(queueInfo);
            m_queueFamilyIndices[size_t(VulkanQueueFamily::Compute)] = i;
        }
        // DMA: no graphics or compute bit, has transfer bit.
        else if (!HasQueueFamily(VulkanQueueFamily::Transfer) &&
            !(m_queueFamilies[i].queueFlags &
                (vk::QueueFlagBits::eGraphics | vk::QueueFlagBits::eCompute)) &&
            (m_queueFamilies[i].queueFlags & vk::QueueFlagBits::eTransfer))
        {
            vk::DeviceQueueCreateInfo queueInfo = {};
            queueInfo.flags = {};
            queueInfo.queueFamilyIndex = i;
            queueInfo.queueCount = 1;
            queueInfo.pQueuePriorities = &priority;
            queueCreateInfos.emplace_back(queueInfo);
            m_queueFamilyIndices[size_t(VulkanQueueFamily::Transfer)] = i;
        }
    }

    assert(queueCreateInfos.size() == m_queueFamilyIndices.size());

    const auto& supportedExtensions = m_physicalDevice.enumerateDeviceExtensionProperties();
    std::vector<const char*> enabledExtensions;
    for (const std::string extension : requiredExtensions)
    {
        if (vkpp::HasExtension(supportedExtensions, extension))
        {
            auto [iter, inserted] = m_enabledExtensions.insert(extension);
            enabledExtensions.push_back(iter->c_str());
        }
    }

    vk::DeviceCreateInfo createInfo = {};
    createInfo.setPNext(&m_features2);
    createInfo.flags = {};
    createInfo.setQueueCreateInfos(queueCreateInfos);
    createInfo.setPEnabledExtensionNames(enabledExtensions);
    createInfo.pEnabledFeatures;
    m_handle = vk::raii::Device(m_physicalDevice, createInfo);

    for (int i = 0; i < ToUnderlying(VulkanQueueFamily::Count); i++)
    {
        m_queues[i] = m_handle.getQueue(m_queueFamilyIndices[i], 0);
    }

    // Vma Initialization
    // https://gpuopen-librariesandsdks.github.io/VulkanMemoryAllocator/html/quick_start.html#quick_start_initialization
    VmaAllocatorCreateInfo allocatorCreateInfo = {};
    allocatorCreateInfo.vulkanApiVersion = VK_API_VERSION_1_3;
    allocatorCreateInfo.instance = static_cast<vk::Instance>(m_instance->m_handle);
    allocatorCreateInfo.physicalDevice = static_cast<vk::PhysicalDevice>(m_physicalDevice);
    allocatorCreateInfo.device = static_cast<vk::Device>(m_handle);
    VmaVulkanFunctions vmaFunctions = {};
    vmaFunctions.vkGetInstanceProcAddr = m_physicalDevice.getDispatcher()->vkGetInstanceProcAddr;
    vmaFunctions.vkGetDeviceProcAddr = m_physicalDevice.getDispatcher()->vkGetDeviceProcAddr;
    allocatorCreateInfo.pVulkanFunctions = &vmaFunctions;
    if (m_Vulkan12Features.bufferDeviceAddress)
    {
        allocatorCreateInfo.flags |= VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
    }
    VK_CHECK_RETURN(vmaCreateAllocator(&allocatorCreateInfo, &m_allocator));
}

VulkanDevice::~VulkanDevice()
{
    if (m_allocator)
    {
        vmaDestroyAllocator(m_allocator);
        m_allocator = VK_NULL_HANDLE;
    }
}

} // namespace rad
