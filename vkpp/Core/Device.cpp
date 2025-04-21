#include <vkpp/Core/Device.h>
#include <vkpp/Core/Instance.h>
#include <vkpp/Core/Command.h>
#include <vkpp/Core/Descriptor.h>
#include <vkpp/Core/Buffer.h>
#include <vkpp/Core/Image.h>

namespace vkpp
{

Device::Device(
    rad::Ref<Instance> instance, vk::raii::PhysicalDevice physicalDevice,
    const std::set<std::string>& requiredExtensions) :
    m_instance(std::move(instance)),
    m_physicalDevice(physicalDevice)
{
    m_properties = m_physicalDevice.getProperties();
    const uint32_t& apiVersion = m_properties.apiVersion;
    if (vkpp::IsVersionMatchOrGreater(apiVersion, 1, 1, 0))
    {
        VK_STRUCTURE_CHAIN_CREATE(m_properties2);
        if (vkpp::IsVersionMatchOrGreater(apiVersion, 1, 2, 0))
        {
            VK_STRUCTURE_CHAIN_APPEND(m_properties2, m_vk11Properties);
        }
        if (vkpp::IsVersionMatchOrGreater(apiVersion, 1, 2, 0))
        {
            VK_STRUCTURE_CHAIN_APPEND(m_properties2, m_vk12Properties);
        }
        if (vkpp::IsVersionMatchOrGreater(apiVersion, 1, 3, 0))
        {
            VK_STRUCTURE_CHAIN_APPEND(m_properties2, m_vk13Properties);
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
        VK_STRUCTURE_CHAIN_CREATE(m_features2);
        if (vkpp::IsVersionMatchOrGreater(apiVersion, 1, 2, 0))
        {
            VK_STRUCTURE_CHAIN_APPEND(m_features2, m_Vulkan11Features);
        }
        if (vkpp::IsVersionMatchOrGreater(apiVersion, 1, 2, 0))
        {
            VK_STRUCTURE_CHAIN_APPEND(m_features2, m_Vulkan12Features);
        }
        if (vkpp::IsVersionMatchOrGreater(apiVersion, 1, 3, 0))
        {
            VK_STRUCTURE_CHAIN_APPEND(m_features2, m_Vulkan13Features);
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
        if (!HasQueueFamily(QueueFamily::Graphics) &&
            (m_queueFamilies[i].queueFlags & vk::QueueFlagBits::eGraphics))
        {
            vk::DeviceQueueCreateInfo queueInfo = {};
            queueInfo.flags = {};
            queueInfo.queueFamilyIndex = i;
            queueInfo.queueCount = 1;
            queueInfo.pQueuePriorities = &priority;
            queueCreateInfos.emplace_back(queueInfo);
            m_queueFamilyIndices[size_t(QueueFamily::Graphics)] = i;
        }
        // Async Compute Engine (ACE): no graphics bit, has compute bit.
        else if (!HasQueueFamily(QueueFamily::Compute) &&
            !(m_queueFamilies[i].queueFlags & vk::QueueFlagBits::eGraphics) &&
            (m_queueFamilies[i].queueFlags & vk::QueueFlagBits::eCompute))
        {
            vk::DeviceQueueCreateInfo queueInfo = {};
            queueInfo.flags = {};
            queueInfo.queueFamilyIndex = i;
            queueInfo.queueCount = 1;
            queueInfo.pQueuePriorities = &priority;
            queueCreateInfos.emplace_back(queueInfo);
            m_queueFamilyIndices[size_t(QueueFamily::Compute)] = i;
        }
        // DMA: no graphics or compute bit, has transfer bit.
        else if (!HasQueueFamily(QueueFamily::Transfer) &&
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
            m_queueFamilyIndices[size_t(QueueFamily::Transfer)] = i;
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

    for (int i = 0; i < rad::ToUnderlying(QueueFamily::Count); i++)
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
    VK_CHECK(vmaCreateAllocator(&allocatorCreateInfo, &m_allocator));
}

Device::~Device()
{
    if (m_allocator)
    {
        vmaDestroyAllocator(m_allocator);
        m_allocator = VK_NULL_HANDLE;
    }
}

rad::Ref<CommandPool> Device::CreateCommandPool(
    QueueFamily queueFamily, vk::CommandPoolCreateFlags flags)
{
    return RAD_NEW CommandPool(this, queueFamily, flags);
}

rad::Ref<DescriptorPool> Device::CreateDescriptorPool(
    uint32_t maxSets, rad::ArrayRef<vk::DescriptorPoolSize> poolSizes,
    vk::DescriptorPoolCreateFlags flags)
{
    vk::DescriptorPoolCreateInfo createInfo = {};
    createInfo.flags = flags;
    createInfo.maxSets = maxSets;
    createInfo.setPoolSizes(poolSizes);
    return RAD_NEW DescriptorPool(this, createInfo);
}

vk::raii::DescriptorSetLayout Device::CreateDescriptorSetLayout(
    rad::ArrayRef<vk::DescriptorSetLayoutBinding> bindings)
{
    vk::DescriptorSetLayoutCreateInfo createInfo = {};
    createInfo.setBindings(bindings);
    return vk::raii::DescriptorSetLayout(this->m_handle, createInfo);
}

vk::Format Device::FindFormat(
    rad::ArrayRef<vk::Format> candidates,
    vk::FormatFeatureFlags linearTilingFeatures,
    vk::FormatFeatureFlags optimalTilingFeatures,
    vk::FormatFeatureFlags bufferFeatures)
{
    for (const auto& candidate : candidates)
    {
        vk::FormatProperties props = m_physicalDevice.getFormatProperties(candidate);
        if (((linearTilingFeatures & props.linearTilingFeatures) == linearTilingFeatures) &&
            ((optimalTilingFeatures & props.optimalTilingFeatures) == optimalTilingFeatures) &&
            ((bufferFeatures & props.bufferFeatures) == bufferFeatures))
        {
            return candidate;
        }
    }
    return vk::Format::eUndefined;
}

rad::Ref<Image> Device::CreateImage2DColorAttachment(
    vk::Format format, uint32_t width, uint32_t height, vk::ImageUsageFlags usage)
{
    vk::ImageCreateInfo imageInfo;
    imageInfo.imageType = vk::ImageType::e2D;
    imageInfo.format = format;
    imageInfo.extent.width = width;
    imageInfo.extent.height = height;
    imageInfo.extent.depth = 1;
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.samples = vk::SampleCountFlagBits::e1;
    imageInfo.tiling = vk::ImageTiling::eOptimal;
    imageInfo.usage = usage;
    imageInfo.initialLayout = vk::ImageLayout::eUndefined;
    VmaAllocationCreateInfo allocCreateInfo = {};
    allocCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;
    return RAD_NEW Image(this, imageInfo, allocCreateInfo);
}

rad::Ref<Image> Device::CreateImage2DDepthStencilAttachment(
    vk::Format format, uint32_t width, uint32_t height, vk::ImageUsageFlags usage)
{
    vk::ImageCreateInfo imageInfo;
    imageInfo.imageType = vk::ImageType::e2D;
    imageInfo.format = format;
    imageInfo.extent.width = width;
    imageInfo.extent.height = height;
    imageInfo.extent.depth = 1;
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.samples = vk::SampleCountFlagBits::e1;
    imageInfo.tiling = vk::ImageTiling::eOptimal;
    imageInfo.usage = usage;
    imageInfo.initialLayout = vk::ImageLayout::eUndefined;
    VmaAllocationCreateInfo allocCreateInfo = {};
    allocCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;
    return RAD_NEW Image(this, imageInfo, allocCreateInfo);
}

} // namespace vkpp
