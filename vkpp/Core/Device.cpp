#include <vkpp/Core/Device.h>
#include <vkpp/Core/Instance.h>
#include <vkpp/Core/Command.h>
#include <vkpp/Core/Fence.h>
#include <vkpp/Core/Semaphore.h>
#include <vkpp/Core/Event.h>
#include <vkpp/Core/Descriptor.h>
#include <vkpp/Core/Buffer.h>
#include <vkpp/Core/Image.h>
#include <vkpp/Core/Sampler.h>
#include <vkpp/Core/RenderPass.h>
#include <vkpp/Core/Framebuffer.h>
#include <vkpp/Core/Pipeline.h>
#include <vkpp/Core/Surface.h>
#include <vkpp/Core/Swapchain.h>

namespace vkpp
{

Device::Device(
    rad::Ref<Instance> instance, vk::raii::PhysicalDevice physicalDevice, const DeviceConfig& config) :
    m_instance(std::move(instance)),
    m_physicalDevice(physicalDevice)
{
    m_config = config;

    const auto& supportedExtensions = m_physicalDevice.enumerateDeviceExtensionProperties();
    auto& requiredExtensions = m_config.requiredExtensions;

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

    m_queueFamilyProperties = m_physicalDevice.getQueueFamilyProperties();
    m_memoryProperties = m_physicalDevice.getMemoryProperties();

    m_features = m_physicalDevice.getFeatures();
    if (vkpp::IsVersionMatchOrGreater(apiVersion, 1, 1, 0))
    {
        VK_STRUCTURE_CHAIN_BEGIN(m_features2);
        if (m_config.enableVulkan11Features && vkpp::IsVersionMatchOrGreater(apiVersion, 1, 2, 0))
        {
            VK_STRUCTURE_CHAIN_ADD(m_features2, m_Vulkan11Features);
        }
        if (m_config.enableVulkan12Features && vkpp::IsVersionMatchOrGreater(apiVersion, 1, 2, 0))
        {
            VK_STRUCTURE_CHAIN_ADD(m_features2, m_Vulkan12Features);
        }
        if (m_config.enableVulkan13Features && vkpp::IsVersionMatchOrGreater(apiVersion, 1, 3, 0))
        {
            VK_STRUCTURE_CHAIN_ADD(m_features2, m_Vulkan13Features);
        }
        if (m_config.enableVulkan14Features && vkpp::IsVersionMatchOrGreater(apiVersion, 1, 4, 0))
        {
            VK_STRUCTURE_CHAIN_ADD(m_features2, m_Vulkan14Features);
        }
        if (m_config.enableBFloat16 && vkpp::HasExtension(supportedExtensions, "VK_KHR_shader_bfloat16"))
        {
            VK_STRUCTURE_CHAIN_ADD(m_features2, m_shaderBfloat16Features);
            requiredExtensions.insert("VK_KHR_shader_bfloat16");
        }
        if (m_config.enableFloat8 && vkpp::HasExtension(supportedExtensions, "VK_EXT_shader_float8"))
        {
            VK_STRUCTURE_CHAIN_ADD(m_features2, m_shaderFloat8Features);
            requiredExtensions.insert("VK_EXT_shader_float8");
        }
        VK_STRUCTURE_CHAIN_END(m_features2);
        m_physicalDevice.getDispatcher()->vkGetPhysicalDeviceFeatures2(
            static_cast<vk::PhysicalDevice>(m_physicalDevice),
            reinterpret_cast<VkPhysicalDeviceFeatures2*>(&m_features2));
    }

    std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos;
    m_queueFamilyProperties = m_physicalDevice.getQueueFamilyProperties();
    m_queueFamilyIndices.fill(VK_QUEUE_FAMILY_IGNORED);
    float priority = 1.0f;
    // Find the universal queue that support both graphics and compute.
    for (uint32_t i = 0; i < m_queueFamilyProperties.size(); i++)
    {
        if ((m_queueFamilyProperties[i].queueFlags & vk::QueueFlagBits::eGraphics) &&
            (m_queueFamilyProperties[i].queueFlags & vk::QueueFlagBits::eCompute))
        {
            vk::DeviceQueueCreateInfo queueInfo = {};
            queueInfo.flags = {};
            queueInfo.queueFamilyIndex = i;
            queueInfo.queueCount = 1;
            queueInfo.pQueuePriorities = &priority;
            queueCreateInfos.emplace_back(queueInfo);
            m_queueFamilyIndices[size_t(QueueFamily::Universal)] = i;
            break;
        }
    }
    // No queue support both graphics and compute, pick the first compute queue as the universal queue.
    if (!HasQueueFamily(QueueFamily::Universal))
    {
        for (uint32_t i = 0; i < m_queueFamilyProperties.size(); i++)
        {
            if (m_queueFamilyProperties[i].queueFlags & vk::QueueFlagBits::eCompute)
            {
                vk::DeviceQueueCreateInfo queueInfo = {};
                queueInfo.flags = {};
                queueInfo.queueFamilyIndex = i;
                queueInfo.queueCount = 1;
                queueInfo.pQueuePriorities = &priority;
                queueCreateInfos.emplace_back(queueInfo);
                m_queueFamilyIndices[size_t(QueueFamily::Universal)] = i;
                break;
            }
        }
    }
    assert(HasQueueFamily(QueueFamily::Universal));
    for (uint32_t i = 0; i < m_queueFamilyProperties.size(); i++)
    {
        if (!HasQueueFamily(QueueFamily::Graphics) &&
            (m_queueFamilyProperties[i].queueFlags & vk::QueueFlagBits::eGraphics))
        {
            if (i != GetQueueFamilyIndex(QueueFamily::Universal))
            {
                vk::DeviceQueueCreateInfo queueInfo = {};
                queueInfo.flags = {};
                queueInfo.queueFamilyIndex = i;
                queueInfo.queueCount = 1;
                queueInfo.pQueuePriorities = &priority;
                queueCreateInfos.emplace_back(queueInfo);
                m_queueFamilyIndices[size_t(QueueFamily::Graphics)] = i;
            }
            m_queueFamilyIndices[size_t(QueueFamily::Graphics)] = i;
        }
        // Async Compute Engine (ACE): different from the universal queue and only support compute.
        else if (!HasQueueFamily(QueueFamily::Compute) &&
            (i != m_queueFamilyIndices[size_t(QueueFamily::Universal)]) &&
            !(m_queueFamilyProperties[i].queueFlags & vk::QueueFlagBits::eGraphics) &&
            (m_queueFamilyProperties[i].queueFlags & vk::QueueFlagBits::eCompute))
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
            !(m_queueFamilyProperties[i].queueFlags & vk::QueueFlagBits::eGraphics) &&
            !(m_queueFamilyProperties[i].queueFlags & vk::QueueFlagBits::eCompute) &&
            (m_queueFamilyProperties[i].queueFlags & vk::QueueFlagBits::eTransfer))
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

    m_enabledExtensions.clear();
    std::vector<const char*> enabledExtensions;
    for (const std::string extension : requiredExtensions)
    {
        if (vkpp::HasExtension(supportedExtensions, extension))
        {
            auto [iter, inserted] = m_enabledExtensions.insert(extension);
            if (inserted)
            {
                enabledExtensions.push_back(iter->c_str());
            }
        }
    }

    vk::DeviceCreateInfo createInfo = {};
    createInfo.setPNext(&m_features2);
    createInfo.flags = {};
    createInfo.setQueueCreateInfos(queueCreateInfos);
    createInfo.setPEnabledExtensionNames(enabledExtensions);
    createInfo.pEnabledFeatures = nullptr;
    m_wrapper = m_physicalDevice.createDevice(createInfo);

    // Vma Initialization
    // https://gpuopen-librariesandsdks.github.io/VulkanMemoryAllocator/html/quick_start.html#quick_start_initialization
    VmaAllocatorCreateInfo allocatorCreateInfo = {};
    allocatorCreateInfo.vulkanApiVersion = VK_API_VERSION_1_3;
    allocatorCreateInfo.instance = static_cast<vk::Instance>(m_instance->m_wrapper);
    allocatorCreateInfo.physicalDevice = static_cast<vk::PhysicalDevice>(m_physicalDevice);
    allocatorCreateInfo.device = static_cast<vk::Device>(m_wrapper);
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

vk::SurfaceCapabilitiesKHR Device::GetCapabilities(vk::SurfaceKHR surface) const
{
    return m_physicalDevice.getSurfaceCapabilitiesKHR(surface);
}

std::vector<vk::SurfaceFormatKHR> Device::GetSurfaceFormats(vk::SurfaceKHR surface) const
{
    return m_physicalDevice.getSurfaceFormatsKHR(surface);
}

std::vector<vk::PresentModeKHR> Device::GetPresentModes(vk::SurfaceKHR surface) const
{
    return m_physicalDevice.getSurfacePresentModesKHR(surface);
}

rad::Ref<CommandPool> Device::CreateCommandPool(
    QueueFamily queueFamily, vk::CommandPoolCreateFlags flags)
{
    return RAD_NEW CommandPool(this, queueFamily, flags | vk::CommandPoolCreateFlagBits::eResetCommandBuffer);
}

rad::Ref<CommandStream> Device::CreateCommandStream(QueueFamily queueFamily)
{
    return RAD_NEW CommandStream(this, queueFamily);
}

rad::Ref<Fence> Device::CreateFence(vk::FenceCreateFlags flags)
{
    vk::FenceCreateInfo createInfo = {};
    createInfo.flags = flags;
    return RAD_NEW Fence(this, createInfo);
}

rad::Ref<Fence> Device::CreateFenceSignaled()
{
    return CreateFence(vk::FenceCreateFlagBits::eSignaled);
}

rad::Ref<Semaphore> Device::CreateSemaphore(vk::SemaphoreCreateFlags flags)
{
    vk::SemaphoreCreateInfo createInfo = {};
    createInfo.flags = flags;
    return RAD_NEW Semaphore(this, createInfo);
}

rad::Ref<Event> Device::CreateEvent(vk::EventCreateFlags flags)
{
    vk::EventCreateInfo createInfo = {};
    createInfo.flags = flags;
    return RAD_NEW Event(this, createInfo);
}

void Device::WaitIdle()
{
    m_wrapper.waitIdle();
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

rad::Ref<DescriptorSetLayout>  Device::CreateDescriptorSetLayout(
    rad::ArrayRef<vk::DescriptorSetLayoutBinding> bindings)
{
    vk::DescriptorSetLayoutCreateInfo createInfo = {};
    createInfo.setBindings(bindings);
    return RAD_NEW DescriptorSetLayout(this, createInfo);
}

rad::Ref<PipelineLayout> Device::CreatePipelineLayout(
    rad::ArrayRef<vk::DescriptorSetLayout> setLayouts,
    rad::ArrayRef<vk::PushConstantRange> pushConstantRanges)
{
    vk::PipelineLayoutCreateInfo createInfo = {};
    createInfo.flags = {};
    createInfo.setLayoutCount = setLayouts.size32();
    createInfo.pSetLayouts = setLayouts.data();
    createInfo.pushConstantRangeCount = pushConstantRanges.size32();
    createInfo.pPushConstantRanges = pushConstantRanges.data();
    return RAD_NEW PipelineLayout(this, createInfo);
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

rad::Ref<Image> Device::CreateImage2D(
    vk::Format format, uint32_t width, uint32_t height,
    uint32_t mipLevels, vk::ImageUsageFlags usage)
{
    vk::ImageCreateInfo imageInfo;
    imageInfo.imageType = vk::ImageType::e2D;
    imageInfo.format = format;
    imageInfo.extent.width = width;
    imageInfo.extent.height = height;
    imageInfo.extent.depth = 1;
    imageInfo.mipLevels = mipLevels;
    imageInfo.arrayLayers = 1;
    imageInfo.samples = vk::SampleCountFlagBits::e1;
    imageInfo.tiling = vk::ImageTiling::eOptimal;
    imageInfo.usage = usage;
    imageInfo.initialLayout = vk::ImageLayout::eUndefined;
    VmaAllocationCreateInfo allocCreateInfo = {};
    allocCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;
    rad::Ref<Image> image = RAD_NEW Image(this, imageInfo, allocCreateInfo);
    return image;
}

rad::Ref<Image> Device::CreateImage2D_Sampled(vk::Format format, uint32_t width, uint32_t height, uint32_t mipLevels)
{
    return CreateImage2D(format, width, height, mipLevels,
        vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled);
}

rad::Ref<Image> Device::CreateImage2D_ColorAttachment(vk::Format format, uint32_t width, uint32_t height)
{
    return CreateImage2D(format, width, height, 1,
        vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eColorAttachment);
}

rad::Ref<Image> Device::CreateImage2D_DepthStencilAttachment(vk::Format format, uint32_t width, uint32_t height)
{
    return CreateImage2D(format, width, height, 1,
        vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eDepthStencilAttachment);
}

rad::Ref<Sampler> Device::CreateSampler(const vk::SamplerCreateInfo& createInfo)
{
    return RAD_NEW Sampler(this, createInfo);
}

rad::Ref<RenderPass> Device::CreateRenderPass(const vk::RenderPassCreateInfo& createInfo)
{
    return RAD_NEW RenderPass(this, createInfo);
}

rad::Ref<Framebuffer> Device::CreateFramebuffer(const vk::FramebufferCreateInfo& createInfo)
{
    return RAD_NEW Framebuffer(this, createInfo);
}

rad::Ref<Framebuffer> Device::CreateFramebuffer(
    vk::RenderPass renderPass, rad::ArrayRef<vk::ImageView> attachments, uint32_t width, uint32_t height, uint32_t layers)
{
    vk::FramebufferCreateInfo createInfo = {};
    createInfo.renderPass = renderPass;
    createInfo.attachmentCount = attachments.size32();
    createInfo.pAttachments = attachments.data();
    createInfo.width = width;
    createInfo.height = height;
    createInfo.layers = layers;
    return CreateFramebuffer(createInfo);
}

rad::Ref<PipelineLayout> Device::CreateLayout(const vk::PipelineLayoutCreateInfo& createInfo)
{
    return RAD_NEW PipelineLayout(this, createInfo);
}

rad::Ref<PipelineLayout> Device::CreateLayout(vk::PipelineLayoutCreateFlags flags, rad::ArrayRef<vk::DescriptorSetLayout> setLayouts,
    rad::ArrayRef<vk::PushConstantRange> pushConstantRanges)
{
    vk::PipelineLayoutCreateInfo createInfo;
    createInfo.flags = flags;
    createInfo.setSetLayouts(setLayouts);
    createInfo.setPushConstantRanges(pushConstantRanges);
    return CreateLayout(createInfo);
}

rad::Ref<ShaderModule> Device::CreateShaderModule(rad::ArrayRef<uint32_t> code)
{
    vk::ShaderModuleCreateInfo createInfo = {};
    createInfo.setCode(code);
    return RAD_NEW ShaderModule(this, createInfo);
}

rad::Ref<GraphicsPipeline> Device::CreateGraphicsPipeline(const vk::GraphicsPipelineCreateInfo& createInfo)
{
    return RAD_NEW GraphicsPipeline(this, createInfo, nullptr);
}

rad::Ref<ComputePipeline> Device::CreateComputePipeline(
    rad::Ref<ShaderStageInfo> shaderStage, vk::PipelineLayout layout)
{
    vk::ComputePipelineCreateInfo pipelineInfo = {};
    pipelineInfo.flags = {};
    pipelineInfo.stage = *shaderStage;
    pipelineInfo.layout = layout;
    pipelineInfo.basePipelineHandle = nullptr;
    pipelineInfo.basePipelineIndex = 0;
    return RAD_NEW ComputePipeline(this, pipelineInfo, nullptr);
}

rad::Ref<Swapchain> Device::CreateSwapchain(const vk::SwapchainCreateInfoKHR& createInfo)
{
    return RAD_NEW Swapchain(this, createInfo);
}

} // namespace vkpp
