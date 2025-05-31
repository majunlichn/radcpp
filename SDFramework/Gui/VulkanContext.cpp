#include <SDFramework/Gui/VulkanContext.h>
#include <SDFramework/Gui/VulkanWindow.h>

#include <vkpp/Core/Instance.h>

// Implementation references:
// https://github.com/KhronosGroup/Vulkan-Tools/blob/main/cube/cube.c
// https://github.com/ocornut/imgui/blob/master/examples/example_sdl3_vulkan/main.cpp

namespace sdf
{

VulkanContext::VulkanContext(rad::Ref<vkpp::Instance> instance, rad::Ref<vkpp::Device> device, VulkanWindow* window) :
    m_instance(std::move(instance)),
    m_device(std::move(device)),
    m_window(window)
{
}

VulkanContext::~VulkanContext()
{
    Destroy();
}

static void CheckVulkanResult(VkResult result)
{
    if (result < 0)
    {
        VKPP_LOG(err, "ImGui: {}", string_VkResult(result));
    }
}

bool VulkanContext::Init()
{
    m_device->WaitIdle();

    vk::SamplerCreateInfo samplerInfo = {};
    samplerInfo.magFilter = vk::Filter::eNearest;
    samplerInfo.minFilter = vk::Filter::eNearest;;
    samplerInfo.mipmapMode = vk::SamplerMipmapMode::eNearest;
    samplerInfo.addressModeU = vk::SamplerAddressMode::eClampToEdge;
    samplerInfo.addressModeV = vk::SamplerAddressMode::eClampToEdge;;
    samplerInfo.addressModeW = vk::SamplerAddressMode::eClampToEdge;;
    samplerInfo.mipLodBias = 0;
    samplerInfo.minLod = 0;
    samplerInfo.maxLod = 0;
    m_samplerNearest = m_device->CreateSampler(samplerInfo);
    samplerInfo.magFilter = vk::Filter::eLinear;
    samplerInfo.minFilter = vk::Filter::eLinear;
    m_samplerLinear = m_device->CreateSampler(samplerInfo);

    m_cmdPool = m_device->CreateCommandPool(vkpp::QueueFamily::Graphics);
    m_guiPass.cmdBuffers = m_cmdPool->AllocatePrimary(sdf::MaxFrameLag);
    m_presentPass.cmdBuffers = m_cmdPool->AllocatePrimary(sdf::MaxFrameLag);

    for (size_t i = 0; i < MaxFrameLag; ++i)
    {
        m_frameThrottles[i] = m_device->CreateFence(vk::FenceCreateFlagBits::eSignaled);
        m_swapchainImageAcquiredSemaphores[i] = m_device->CreateSemaphore();
    }

    int width = 0;
    int height = 0;
    m_window->GetSizeInPixels(&width, &height);
    Resize(width, height);

    return true;
}

void VulkanContext::Destroy()
{
    m_device->WaitIdle();
    if (m_plot)
    {
        ImPlot::DestroyContext(m_plot);
        m_plot = nullptr;
    }
    if (m_gui)
    {
        ImGui_ImplVulkan_Shutdown();
        ImGui_ImplSDL3_Shutdown();
        ImGui::DestroyContext();
        m_gui = nullptr;
    }
}

void VulkanContext::Resize(uint32_t width, uint32_t height)
{
    m_swapchain = CreateSwapchain(width, height);
    VKPP_LOG(info, "Swapchain created: {}x{} ({}, {}, {})",
        width, height, vk::to_string(m_swapchain->GetFormat()),
        vk::to_string(m_swapchain->m_colorSpace), vk::to_string(m_presentMode));

    vk::Format colorFormat = vk::Format::eR8G8B8A8Unorm;    // TODO: support HDR?
    m_renderTarget = m_device->CreateImage2D_ColorAttachment(
        colorFormat, width, height, vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eColorAttachment);
    m_renderTargetView = m_renderTarget->CreateView2D();

    vk::Format overlayFormat = vk::Format::eR8G8B8A8Unorm;
    m_overlay = m_device->CreateImage2D_ColorAttachment(
        overlayFormat, width, height, vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eColorAttachment);
    m_overlayView = m_overlay->CreateView2D();

    std::string vertSource =
        R"(
#version 450 core

layout (location = 0) out vec2 out_TexCoord;

void main()
{
    const vec4 vertices[3] = vec4[3](
        vec4(-1.0f, 1.0f, 0.0f, 1.0f),
        vec4(-1.0f,-3.0f, 0.0f,-1.0f),
        vec4( 3.0f, 1.0f, 2.0f, 1.0f)
    );
    vec4 vertex = vertices[gl_VertexIndex & 3];
    gl_Position = vec4(vertex.xy, 0.0f, 1.0f);
    out_TexCoord = vertex.zw;
}
        )";
    std::string fragSource =
        R"(
#version 450 core

layout (set = 0, binding = 0) uniform sampler2D g_renderTargetSampler;
layout (set = 0, binding = 1) uniform sampler2D g_overlaySampler;

layout (location = 0) in vec2 in_TexCoord;
layout (location = 0) out vec4 out_FragColor;

void main()
{
    const vec4 color = textureLod(g_renderTargetSampler, in_TexCoord, 0);
    const vec4 overlay = textureLod(g_overlaySampler, in_TexCoord, 0);
    out_FragColor.rgb = color.rgb * (1.0f - overlay.a) + overlay.rgb;
    out_FragColor.a = 1.0f;
}
        )";
    m_presentPass.vertStage = vkpp::ShaderStageInfo::CreateFromGLSL(m_device,
        vk::ShaderStageFlagBits::eVertex, "Present.vert", vertSource);
    m_presentPass.fragStage = vkpp::ShaderStageInfo::CreateFromGLSL(m_device,
        vk::ShaderStageFlagBits::eFragment, "Present.frag", fragSource);

    m_presentPass.descSetLayout = m_device->CreateDescriptorSetLayout(
        {
            vk::DescriptorSetLayoutBinding( // RenderTarget
                0,  // binding
                vk::DescriptorType::eCombinedImageSampler,
                1,  // count
                vk::ShaderStageFlagBits::eFragment,
                nullptr // pImmutableSamplers
            ),
            vk::DescriptorSetLayoutBinding( // Overlay
                1,  // binding
                vk::DescriptorType::eCombinedImageSampler,
                1,  // count
                vk::ShaderStageFlagBits::eFragment,
                nullptr // pImmutableSamplers
            ),
        }
    );

    m_presentPass.pipelineLayout = m_device->CreatePipelineLayout(
        m_presentPass.descSetLayout->GetHandle()
    );

    vk::GraphicsPipelineCreateInfo pipelineInfo = {};
    VK_STRUCTURE_CHAIN_BEGIN(pipelineInfo);
    vk::PipelineShaderStageCreateInfo shaderStageInfos[2] =
    {
        *m_presentPass.vertStage,
        *m_presentPass.fragStage,
    };
    pipelineInfo.setStages(shaderStageInfos);
    vk::PipelineVertexInputStateCreateInfo vertexInputInfo = {};
    pipelineInfo.pVertexInputState = &vertexInputInfo;
    vk::PipelineInputAssemblyStateCreateInfo inputAssemblyInfo = {};
    inputAssemblyInfo.topology = vk::PrimitiveTopology::eTriangleList;
    inputAssemblyInfo.primitiveRestartEnable = vk::False;
    pipelineInfo.pInputAssemblyState = &inputAssemblyInfo;
    pipelineInfo.pTessellationState = nullptr;
    vk::PipelineViewportStateCreateInfo viewportInfo = {};
    viewportInfo.viewportCount = 1;
    viewportInfo.scissorCount = 1;
    pipelineInfo.pViewportState = &viewportInfo;
    vk::PipelineRasterizationStateCreateInfo rasterizationInfo = {};
    rasterizationInfo.polygonMode = vk::PolygonMode::eFill;
    rasterizationInfo.cullMode = vk::CullModeFlagBits::eNone;
    rasterizationInfo.frontFace = vk::FrontFace::eCounterClockwise;
    rasterizationInfo.lineWidth = 1.0f;
    pipelineInfo.pRasterizationState = &rasterizationInfo;
    vk::PipelineMultisampleStateCreateInfo multisampleInfo = {};
    pipelineInfo.pMultisampleState = &multisampleInfo;
    pipelineInfo.pDepthStencilState = nullptr;
    vk::PipelineColorBlendStateCreateInfo colorBlendInfo = {};
    vk::PipelineColorBlendAttachmentState colorBlendAttachment = {};
    colorBlendAttachment.blendEnable = vk::False;
    colorBlendAttachment.colorWriteMask =
        vk::ColorComponentFlagBits::eR |
        vk::ColorComponentFlagBits::eG |
        vk::ColorComponentFlagBits::eB |
        vk::ColorComponentFlagBits::eA;
    colorBlendInfo.setAttachments({ 1, &colorBlendAttachment });
    pipelineInfo.pColorBlendState = &colorBlendInfo;
    vk::PipelineDynamicStateCreateInfo dynamicStateInfo = {};
    std::vector<vk::DynamicState> dynamicStates =
    {
        vk::DynamicState::eViewport,
        vk::DynamicState::eScissor,
    };
    dynamicStateInfo.setDynamicStates(dynamicStates);
    pipelineInfo.pDynamicState = &dynamicStateInfo;
    pipelineInfo.layout = m_presentPass.pipelineLayout->GetHandle();
    pipelineInfo.renderPass = nullptr;
    pipelineInfo.subpass = 0;
    pipelineInfo.basePipelineHandle = nullptr;
    pipelineInfo.basePipelineIndex = 0;

    vk::PipelineRenderingCreateInfo renderingInfo = {};
    renderingInfo.colorAttachmentCount = 1;
    vk::Format swapchainImageFormat = m_swapchain->GetFormat();
    renderingInfo.pColorAttachmentFormats = &swapchainImageFormat;
    VK_STRUCTURE_CHAIN_ADD(pipelineInfo, renderingInfo);

    VK_STRUCTURE_CHAIN_END(pipelineInfo);
    m_presentPass.pipeline = m_device->CreateGraphicsPipeline(pipelineInfo);

    std::vector<vk::DescriptorPoolSize> descPoolSizes =
    {
        { vk::DescriptorType::eCombinedImageSampler, 2 },
    };
    m_presentPass.descPool = m_device->CreateDescriptorPool(1024, descPoolSizes);
    m_presentPass.descSet = m_presentPass.descPool->Allocate(m_presentPass.descSetLayout->GetHandle())[0];

    if (m_presentPass.descSet)
    {
        vk::DescriptorImageInfo renderTargetInfo = {};
        if ((m_renderTarget->GetWidth() <= m_swapchain->GetWidth()) &&
            (m_renderTarget->GetHeight() <= m_swapchain->GetHeight()))
        {
            renderTargetInfo.sampler = m_samplerNearest->GetHandle();
        }
        else
        {
            renderTargetInfo.sampler = m_samplerLinear->GetHandle();
        }
        renderTargetInfo.imageView = m_renderTargetView->GetHandle();
        renderTargetInfo.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
        m_presentPass.descSet->UpdateCombinedImageSamplers(0, 0, renderTargetInfo);

        vk::DescriptorImageInfo overlayInfo = {};
        overlayInfo.sampler = m_samplerNearest->GetHandle();
        overlayInfo.imageView = m_overlayView->GetHandle();
        overlayInfo.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
        m_presentPass.descSet->UpdateCombinedImageSamplers(1, 0, overlayInfo);
    }

    m_presentReady.resize(m_swapchain->GetImageCount());
    m_swapchainImageOwnershipSemaphores.resize(m_swapchain->GetImageCount());
    for (size_t i = 0; i < m_swapchain->GetImageCount(); ++i)
    {
        m_presentReady[i] = m_device->CreateSemaphore();
        m_swapchainImageOwnershipSemaphores[i] = m_device->CreateSemaphore();
    }

    if (m_gui)
    {
        ImGui_ImplVulkan_Shutdown();
        ImGui_ImplSDL3_Shutdown();
        ImGui::DestroyContext();
        m_gui = nullptr;
    }

    IMGUI_CHECKVERSION();
    m_gui = ImGui::CreateContext();
    m_plot = ImPlot::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;   // Enable Keyboard Controls
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;    // Enable Gamepad Controls

    ImGui::StyleColorsDark();

    ImGui_ImplSDL3_InitForVulkan(m_window->GetHandle());
    ImGui_ImplVulkan_InitInfo initInfo = {};
    initInfo.ApiVersion = m_device->m_properties.apiVersion;
    initInfo.Instance = m_device->m_instance->GetHandle();
    initInfo.PhysicalDevice = static_cast<vk::PhysicalDevice>(m_device->m_physicalDevice);
    initInfo.Device = m_device->GetHandle();
    initInfo.QueueFamily = m_device->GetQueueFamilyIndex(vkpp::QueueFamily::Graphics);
    initInfo.Queue = m_device->GetQueue(vkpp::QueueFamily::Graphics)->GetHandle();
    initInfo.DescriptorPool = VK_NULL_HANDLE;
    initInfo.RenderPass = VK_NULL_HANDLE;
    initInfo.MinImageCount = m_swapchain->GetImageCount();
    initInfo.ImageCount = m_swapchain->GetImageCount();
    initInfo.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    // (Optional)
    initInfo.PipelineCache;
    initInfo.Subpass;
    // (Optional) Set to create internal descriptor pool instead of using DescriptorPool
    initInfo.DescriptorPoolSize = 4096;
    // (Optional) Dynamic Rendering
    // Need to explicitly enable VK_KHR_dynamic_rendering extension to use this, even for Vulkan 1.3.
    initInfo.UseDynamicRendering = true;
    VkPipelineRenderingCreateInfo& guiRenderingInfo = initInfo.PipelineRenderingCreateInfo;
    guiRenderingInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    guiRenderingInfo.colorAttachmentCount = 1;
    guiRenderingInfo.pColorAttachmentFormats =
        reinterpret_cast<const VkFormat*>(&overlayFormat);
    // (Optional) Allocation, Debugging
    initInfo.Allocator = nullptr;
    initInfo.CheckVkResultFn = CheckVulkanResult;
    initInfo.MinAllocationSize = 1024 * 1024;

    ImGui_ImplVulkan_LoadFunctions(m_device->m_properties.apiVersion,
        [](const char* functionName, void* userData) {
            vkpp::Instance* instance = reinterpret_cast<vkpp::Instance*>(userData);
            return instance->GetProcAddr(functionName);
        },
        m_device->m_instance.get());
    ImGui_ImplVulkan_Init(&initInfo);

    SDL_DisplayID displayID = SDL_GetDisplayForWindow(m_window->GetHandle());
    SDL_Rect rect = {};
    float fontSize = 24.0f; // TODO: Calculate font size according to DPI.
#if defined(_WIN32)
    auto fonts = ImGui::GetIO().Fonts;
    fonts->AddFontFromFileTTF("C:\\Windows\\Fonts\\consola.ttf", fontSize);
#endif
}

rad::Ref<vkpp::Swapchain> VulkanContext::CreateSwapchain(uint32_t width, uint32_t height)
{
    vk::SurfaceKHR surfaceHandle = m_window->GetVulkanSurface()->GetHandle();
    vk::SurfaceCapabilitiesKHR surfaceCaps = m_device->GetCapabilities(surfaceHandle);
    uint32_t imageCount = 3;
    if (imageCount < surfaceCaps.minImageCount)
    {
        imageCount = surfaceCaps.minImageCount;
    }
    // If maxImageCount is 0, we can ask for as many images as we want,
    // otherwise we're limited to maxImageCount.
    if ((surfaceCaps.maxImageCount > 0) &&
        (imageCount > surfaceCaps.maxImageCount))
    {
        imageCount = surfaceCaps.maxImageCount;
    }

    vk::Format imageFormat = vk::Format::eUndefined;
    vk::ColorSpaceKHR imageColorSpace = vk::ColorSpaceKHR::eSrgbNonlinear;

    std::vector<vk::SurfaceFormatKHR> surfaceFormats = m_device->GetSurfaceFormats(surfaceHandle);
    if (surfaceFormats[0].format == vk::Format::eUndefined)
    {
        imageFormat = vk::Format::eR8G8B8A8Unorm;
    }
    for (const auto& surfaceFormat : surfaceFormats)
    {
        const vk::Format format = surfaceFormat.format;
        if ((format == vk::Format::eR8G8B8A8Unorm) || (format == vk::Format::eB8G8R8A8Unorm) ||
            (format == vk::Format::eA2R10G10B10UnormPack32) || (format == vk::Format::eA2B10G10R10UnormPack32) ||
            (format == vk::Format::eR16G16B16A16Sfloat))
        {
            imageFormat = surfaceFormat.format;
            imageColorSpace = surfaceFormat.colorSpace;
            break;
        }
    }

    // width and height are either both 0xFFFFFFFF, or both not 0xFFFFFFFF.
    if (surfaceCaps.currentExtent.width == 0xFFFFFFFF)
    {
        // If the surface size is undefined, the size is set to the size of the images requested,
        // which must fit within the minimum and maximum values.
        if (width < surfaceCaps.minImageExtent.width)
        {
            width = surfaceCaps.minImageExtent.width;
        }
        else if (width > surfaceCaps.maxImageExtent.width)
        {
            width = surfaceCaps.maxImageExtent.width;
        }

        if (height < surfaceCaps.minImageExtent.height)
        {
            height = surfaceCaps.minImageExtent.height;
        }
        else if (height > surfaceCaps.minImageExtent.height)
        {
            height = surfaceCaps.minImageExtent.height;
        }
    }
    else
    {
        // If the surface size is defined, the swap chain size must match.
        width = surfaceCaps.minImageExtent.width;
        height = surfaceCaps.minImageExtent.height;
    }

    if ((width == 0) || (height == 0))
    {
        return nullptr;
    }

    vk::SurfaceTransformFlagBitsKHR preTransform = vk::SurfaceTransformFlagBitsKHR::eIdentity;
    if (surfaceCaps.supportedTransforms & vk::SurfaceTransformFlagBitsKHR::eIdentity)
    {
        preTransform = vk::SurfaceTransformFlagBitsKHR::eIdentity;
    }
    else
    {
        preTransform = surfaceCaps.currentTransform;
    }

    vk::CompositeAlphaFlagBitsKHR compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque;
    std::array<vk::CompositeAlphaFlagBitsKHR, 4> compositeAlphaFlags =
    {
        vk::CompositeAlphaFlagBitsKHR::eOpaque,
        vk::CompositeAlphaFlagBitsKHR::ePreMultiplied,
        vk::CompositeAlphaFlagBitsKHR::ePostMultiplied,
        vk::CompositeAlphaFlagBitsKHR::eInherit,
    };
    for (const auto& compositeAlphaFlag : compositeAlphaFlags)
    {
        if (surfaceCaps.supportedCompositeAlpha & compositeAlphaFlag)
        {
            compositeAlpha = compositeAlphaFlag;
            break;
        }
    }

    std::vector<vk::PresentModeKHR> presentModes = m_device->GetPresentModes(surfaceHandle);
    if (std::ranges::find(presentModes, m_presentMode) == std::end(presentModes))
    {
        VKPP_LOG(warn, "PresentMode {} is not supported, fallback to {}!\n",
            vk::to_string(m_presentMode), vk::to_string(presentModes[0]));
        m_presentMode = presentModes[0];
    }

    vk::SwapchainCreateInfoKHR swapchainInfo = {};
    swapchainInfo.flags = {};
    swapchainInfo.surface = surfaceHandle;
    swapchainInfo.minImageCount = imageCount;
    swapchainInfo.imageFormat = imageFormat;
    swapchainInfo.imageColorSpace = imageColorSpace;
    swapchainInfo.imageExtent.width = width;
    swapchainInfo.imageExtent.height = height;
    swapchainInfo.imageArrayLayers = 1;
    swapchainInfo.imageUsage = vk::ImageUsageFlagBits::eColorAttachment;
    swapchainInfo.imageSharingMode = vk::SharingMode::eExclusive;
    swapchainInfo.queueFamilyIndexCount = 0;
    swapchainInfo.pQueueFamilyIndices = nullptr;
    swapchainInfo.preTransform = preTransform;
    swapchainInfo.compositeAlpha = compositeAlpha;
    swapchainInfo.presentMode = m_presentMode;
    swapchainInfo.clipped = vk::True;
    swapchainInfo.oldSwapchain = m_swapchain ? m_swapchain->GetHandle() : VK_NULL_HANDLE;

    return RAD_NEW vkpp::Swapchain(m_device, swapchainInfo);
}

bool VulkanContext::ProcessEvent(const SDL_Event& event)
{
    return ImGui_ImplSDL3_ProcessEvent(&event);
}

void VulkanContext::BeginFrame()
{
    m_frameThrottles[m_cmdBufferIndex]->Wait();
    vk::Result err = vk::Result::eSuccess;
    do {
        err = m_swapchain->AcquireNextImage(
            UINT64_MAX, m_swapchainImageAcquiredSemaphores[m_cmdBufferIndex].get(), nullptr, 0x1);
    } while (err != vk::Result::eSuccess);

    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplSDL3_NewFrame();
    ImGui::NewFrame();
}

void VulkanContext::Render()
{
    ImGui::Render();
    ImDrawData* drawData = ImGui::GetDrawData();
    const bool isMinimized = ((drawData->DisplaySize.x <= 0.0f) || (drawData->DisplaySize.y <= 0.0f));
    if (!isMinimized)
    {
        vkpp::CommandBuffer* cmdBuffer = m_guiPass.cmdBuffers[m_cmdBufferIndex].get();
        cmdBuffer->Begin();
        if (m_overlay->GetCurrentLayout() != vk::ImageLayout::eColorAttachmentOptimal)
        {
            cmdBuffer->TransitLayoutFromCurrent(m_overlay.get(),
                vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                vk::AccessFlagBits2::eColorAttachmentRead | vk::AccessFlagBits2::eColorAttachmentWrite,
                vk::ImageLayout::eColorAttachmentOptimal);
        }
        vk::RenderingInfo renderingInfo = {};
        renderingInfo.renderArea.offset = vk::Offset2D{ 0, 0 };
        renderingInfo.renderArea.extent = vk::Extent2D{ m_overlay->GetWidth(), m_overlay->GetHeight() };
        renderingInfo.layerCount = 1;
        renderingInfo.viewMask = 0;
        vk::RenderingAttachmentInfo colorAttachmentInfo = {};
        colorAttachmentInfo.imageView = m_overlayView->GetHandle();
        colorAttachmentInfo.imageLayout = m_overlay->GetCurrentLayout();
        colorAttachmentInfo.loadOp = vk::AttachmentLoadOp::eClear;
        colorAttachmentInfo.storeOp = vk::AttachmentStoreOp::eStore;
        colorAttachmentInfo.clearValue = {};
        renderingInfo.setColorAttachments({ 1, &colorAttachmentInfo });
        cmdBuffer->BeginRendering(renderingInfo);
        ImGui_ImplVulkan_RenderDrawData(drawData, cmdBuffer->GetHandle());
        cmdBuffer->EndRendering();
        cmdBuffer->End();
        m_device->GetQueue(vkpp::QueueFamily::Graphics)->
            Execute(cmdBuffer->GetHandle(), {}, {}, nullptr);
    }
}

void VulkanContext::EndFrame()
{
    vkpp::CommandBuffer* cmdBuffer = m_presentPass.cmdBuffers[m_cmdBufferIndex].get();
    cmdBuffer->Begin();
    if (m_renderTarget->GetCurrentLayout() != vk::ImageLayout::eShaderReadOnlyOptimal)
    {
        cmdBuffer->TransitLayoutFromCurrent(m_renderTarget.get(),
            vk::PipelineStageFlagBits2::eFragmentShader,
            vk::AccessFlagBits2::eShaderRead,
            vk::ImageLayout::eShaderReadOnlyOptimal);
    }
    if (m_overlay->GetCurrentLayout() != vk::ImageLayout::eShaderReadOnlyOptimal)
    {
        cmdBuffer->TransitLayoutFromCurrent(m_overlay.get(),
            vk::PipelineStageFlagBits2::eFragmentShader,
            vk::AccessFlagBits2::eShaderRead,
            vk::ImageLayout::eShaderReadOnlyOptimal);
    }

    uint32_t swapchainImageIndex = m_swapchain->GetCurrentImageIndex();
    vkpp::Image* swapchainImage = m_swapchain->GetCurrentImage();
    vkpp::ImageView* swapchainImageView = m_swapchain->GetCurrentImageView();
    if (swapchainImage->GetCurrentLayout() != vk::ImageLayout::eColorAttachmentOptimal)
    {
        cmdBuffer->TransitLayoutFromCurrent(swapchainImage,
            vk::PipelineStageFlagBits2::eColorAttachmentOutput,
            vk::AccessFlagBits2::eColorAttachmentWrite,
            vk::ImageLayout::eColorAttachmentOptimal);
    }

    vk::RenderingInfo renderingInfo = {};
    renderingInfo.renderArea.offset = vk::Offset2D{ 0, 0 };
    renderingInfo.renderArea.extent =
        vk::Extent2D{ swapchainImage->GetWidth(), swapchainImage->GetHeight() };
    renderingInfo.layerCount = 1;
    renderingInfo.viewMask = 0;
    vk::RenderingAttachmentInfo colorAttachmentInfo = {};
    colorAttachmentInfo.imageView = swapchainImageView->GetHandle();
    colorAttachmentInfo.imageLayout = swapchainImage->GetCurrentLayout();
    colorAttachmentInfo.loadOp = vk::AttachmentLoadOp::eDontCare;
    colorAttachmentInfo.storeOp = vk::AttachmentStoreOp::eStore;
    colorAttachmentInfo.clearValue = {};
    renderingInfo.setColorAttachments({ 1, &colorAttachmentInfo });
    cmdBuffer->BeginRendering(renderingInfo);

    cmdBuffer->BindPipeine(m_presentPass.pipeline.get());
    cmdBuffer->BindDescriptorSets(vk::PipelineBindPoint::eGraphics,
        m_presentPass.pipelineLayout->GetHandle(),
        0, { m_presentPass.descSet->GetHandle() }
    );

    vk::Viewport viewport = {};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = float(swapchainImage->GetWidth());
    viewport.height = float(swapchainImage->GetHeight());
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    cmdBuffer->SetViewport(0, viewport);

    vk::Rect2D scissor = {};
    scissor.extent.width = swapchainImage->GetWidth();
    scissor.extent.height = swapchainImage->GetHeight();
    cmdBuffer->SetScissor(0, scissor);

    cmdBuffer->Draw(3, 1, 0, 0);

    cmdBuffer->EndRendering();

    if (swapchainImage->GetCurrentLayout() != vk::ImageLayout::ePresentSrcKHR)
    {
        cmdBuffer->TransitLayoutFromCurrent(swapchainImage,
            vk::PipelineStageFlagBits2::eNone,
            vk::AccessFlagBits2::eNone,
            vk::ImageLayout::ePresentSrcKHR);
    }
    cmdBuffer->End();

    // Only reset right before submitting so we can't deadlock on an un-signalled fence
    // that has nothing submitted to it.
    m_frameThrottles[m_cmdBufferIndex]->Reset();

    m_device->GetQueue(vkpp::QueueFamily::Graphics)->Execute(
        cmdBuffer->GetHandle(),
        {   // waits
            vkpp::SubmitWaitInfo
            {
                .semaphore = m_swapchainImageAcquiredSemaphores[m_cmdBufferIndex]->GetHandle(),
                .dstStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput
            }
        },
        {   // signals
            m_presentReady[swapchainImageIndex]->GetHandle()
        },
        m_frameThrottles[m_cmdBufferIndex]->GetHandle()
    );

    vk::PresentInfoKHR presentInfo = {};
    presentInfo.waitSemaphoreCount = 1;
    vk::Semaphore presentReady = m_presentReady[swapchainImageIndex]->GetHandle();
    presentInfo.pWaitSemaphores = &presentReady;
    presentInfo.swapchainCount = 1;
    vk::SwapchainKHR swapchainHandle = m_swapchain->GetHandle();
    presentInfo.pSwapchains = &swapchainHandle;
    presentInfo.pImageIndices = &swapchainImageIndex;
    presentInfo.pResults = nullptr;
    m_device->Present(vkpp::QueueFamily::Graphics, presentInfo);

    m_cmdBufferIndex += 1;
    m_cmdBufferIndex %= MaxFrameLag;
}

} // namespace sdf
