#include "CubeDemo.h"
#include <glm/ext.hpp>

#include "MeshData.h"
#include "lunarg.ppm.h"


CubeDemo::CubeDemo()
{
}

CubeDemo::~CubeDemo()
{
}

void CubeDemo::ParseCommandLine(int argc, char* argv[])
{
    for (int i = 1; i < argc; i++)
    {
        if (rad::StrCaseEqual(argv[i], "--gpu-index") && (i < argc - 1))
        {
            m_gpuIndex = std::stoi(argv[i + 1]);
            if (m_gpuIndex < 0)
            {
                fprintf(stderr, "Invalid GPU index %d!\n", m_gpuIndex);
                m_gpuIndex = -1;
            }
            i++;
            continue;
        }

        if (rad::StrCaseEqual(argv[i], "--gpu-name") && (i < argc - 1))
        {
            m_gpuName = argv[i + 1];
            i++;
            continue;
        }

        if (rad::StrCaseEqual(argv[i], "--width"))
        {
            if (i < argc - 1)
            {
                int32_t widthInput = std::stoi(argv[i + 1]);
                if (widthInput > 0)
                {
                    m_width = static_cast<uint32_t>(widthInput);
                    i++;
                    continue;
                }
            }
        }
        if (rad::StrCaseEqual(argv[i], "--height"))
        {
            if (i < argc - 1)
            {
                int32_t heightInput = std::stoi(argv[i + 1]);
                if (heightInput > 0)
                {
                    m_height = static_cast<uint32_t>(heightInput);
                    i++;
                    continue;
                }
            }
        }

        if (rad::StrCaseEqual(argv[i], "--present-mode") && (i < argc - 1))
        {
            if (rad::StrCaseEqual(argv[i + 1], "Immediate"))
            {
                m_presentMode = vk::PresentModeKHR::eImmediate;
            }
            else if (rad::StrCaseEqual(argv[i + 1], "Mailbox"))
            {
                m_presentMode = vk::PresentModeKHR::eMailbox;
            }
            else if (rad::StrCaseEqual(argv[i + 1], "Fifo"))
            {
                m_presentMode = vk::PresentModeKHR::eFifo;
            }
            else if (rad::StrCaseEqual(argv[i + 1], "FifoRelaxed"))
            {
                m_presentMode = vk::PresentModeKHR::eFifoRelaxed;
            }
            else
            {
                fprintf(stderr, "Invalid present mode %s\n", argv[i + 1]);
            }
        }
    }
}

bool CubeDemo::Init(int argc, char* argv[])
{
    m_gpuIndex = -1;
    m_width = 1024;
    m_height = 1024;
    m_frameIndex = 0;
    m_frameCount = UINT32_MAX;

    ParseCommandLine(argc, argv);

    m_instance = RAD_NEW vkpp::Instance();
    std::set<std::string> instanceLayers = {};
    std::set<std::string> instanceExtensions = GetVulkanInstanceExtensionsRequired();
    if (!m_instance->Init(
        APP_NAME, APP_VERSION,
        APP_NAME, APP_VERSION,
        instanceLayers, instanceExtensions))
    {
        VKPP_LOG(err, "Failed to init the Vulkan Instance!");
    }

    glm::vec3 eye = { 0.0f, 3.0f, 5.0f };
    glm::vec3 origin = { 0, 0, 0 };
    glm::vec3 up = { 0.0f, 1.0f, 0.0f };

    float aspect = float(m_width) / float(m_height);
    m_projectionMatrix = glm::perspective(glm::radians(45.0f), aspect, 0.1f, 100.0f);
    m_projectionMatrix[1][1] *= -1; // Flip projection matrix from GL to Vulkan orientation.
    m_viewMatrix = glm::lookAt(eye, origin, up);
    m_modelMatrix = glm::identity<glm::mat4>();

    m_spinAngle = 4.0f;
    m_spinIncrement = 0.2f;
    m_spinPause = false;

    if (!VulkanWindow::Create("Vulkan Cube", m_width, m_height, SDL_WINDOW_VULKAN))
    {
        return false;
    }

    const auto& physicalDevices = m_instance->m_physicalDevices;
    if ((m_gpuIndex != -1) && (m_gpuIndex >= physicalDevices.size()))
    {
        VKPP_LOG(info, "Invalid GPU index {}!", m_gpuIndex);
        m_gpuIndex = -1;
    }
    if (!m_gpuName.empty())
    {
        for (uint32_t i = 0; i < physicalDevices.size(); i++)
        {
            const vk::PhysicalDeviceProperties deviceProps = physicalDevices[i].getProperties();
            std::string_view deviceName(deviceProps.deviceName);
            if (deviceName.find(m_gpuName) != deviceName.npos)
            {
                m_gpuIndex = i;
                break;
            }
        }
    }

    if (m_gpuIndex == -1)
    {
        int priorityPrev = 0;
        for (uint32_t i = 0; i < physicalDevices.size(); i++)
        {
            const vk::PhysicalDeviceProperties deviceProps = physicalDevices[i].getProperties();
            assert(deviceProps.deviceType <= vk::PhysicalDeviceType::eCpu);

            auto surfaceSupport = physicalDevices[i].getSurfaceSupportKHR(0, m_surface->GetHandle());
            if (surfaceSupport != vk::True)
            {
                continue;
            }

            std::map<vk::PhysicalDeviceType, int> deviceTypePriorities =
            {
                { vk::PhysicalDeviceType::eDiscreteGpu,     5 },
                { vk::PhysicalDeviceType::eIntegratedGpu,   4 },
                { vk::PhysicalDeviceType::eVirtualGpu,      3 },
                { vk::PhysicalDeviceType::eCpu,             2 },
                { vk::PhysicalDeviceType::eOther,           1 },
            };

            int priority = -1;
            if (deviceTypePriorities.find(deviceProps.deviceType) != deviceTypePriorities.end())
            {
                priority = deviceTypePriorities[deviceProps.deviceType];
            }

            if (priority > priorityPrev)
            {
                m_gpuIndex = i;
                priorityPrev = priority;
            }
        }
    }
    assert((m_gpuIndex >= 0) && (m_gpuIndex < static_cast<int>(physicalDevices.size())));
    vk::raii::PhysicalDevice physicalDevice = physicalDevices[m_gpuIndex];
    m_device = m_instance->CreateDevice(physicalDevice);

    VKPP_LOG(info, "Logical device created on GPU#{}: {}", m_gpuIndex, m_device->GetName());

    m_frame = RAD_NEW sdf::VulkanFrame(this, m_device);
    m_frame->m_presentMode = m_presentMode;
    if (!m_frame->Init())
    {
        return false;
    }

    vkpp::Image* renderTarget = m_frame->GetRenderTarget();

    m_depthImage = m_device->CreateImage2D_DepthStencilAttachment(
        vk::Format::eD32Sfloat, renderTarget->GetWidth(), renderTarget->GetHeight());
    m_depthImageView = m_depthImage->CreateView2D();

    m_shaderUniforms = {};
    for (unsigned int i = 0; i < 12 * 3; i++)
    {
        m_shaderUniforms.positions[i][0] = g_vertex_buffer_data[i * 3];
        m_shaderUniforms.positions[i][1] = g_vertex_buffer_data[i * 3 + 1];
        m_shaderUniforms.positions[i][2] = g_vertex_buffer_data[i * 3 + 2];
        m_shaderUniforms.positions[i][3] = 1.0f;
        m_shaderUniforms.attribs[i][0] = g_uv_buffer_data[2 * i];
        m_shaderUniforms.attribs[i][1] = g_uv_buffer_data[2 * i + 1];
        m_shaderUniforms.attribs[i][2] = 0;
        m_shaderUniforms.attribs[i][3] = 0;
    }

    for (uint32_t i = 0; i < sdf::MaxFrameLag; ++i)
    {
        m_uniformBuffers[i] = vkpp::Buffer::CreateUniform(m_device, sizeof(ShaderUniformData));
        m_uniforms[i] = m_uniformBuffers[i]->m_allocInfo.pMappedData;
        std::memcpy(m_uniforms[i], &m_shaderUniforms, sizeof(m_shaderUniforms));
    }

    m_textures.resize(1);
    m_textureViews.resize(1);
    m_samplers.resize(1);
    m_textures[0] = vkpp::CreateTextureFromMemory_R8G8B8A8_SRGB(m_device, lunarg_ppm, lunarg_ppm_len);
    m_textureViews[0] = m_textures[0]->CreateView2D();
    vk::SamplerCreateInfo samplerInfo = {};
    samplerInfo.flags = {};
    samplerInfo.magFilter = vk::Filter::eNearest;
    samplerInfo.minFilter = vk::Filter::eNearest;
    samplerInfo.mipmapMode = vk::SamplerMipmapMode::eNearest;
    samplerInfo.addressModeU = vk::SamplerAddressMode::eClampToEdge;
    samplerInfo.addressModeV = vk::SamplerAddressMode::eClampToEdge;
    samplerInfo.addressModeW = vk::SamplerAddressMode::eClampToEdge;
    samplerInfo.mipLodBias = 0.0f;
    samplerInfo.anisotropyEnable = vk::False;
    samplerInfo.maxAnisotropy = 1;
    samplerInfo.compareEnable = vk::False;
    samplerInfo.compareOp = vk::CompareOp::eNever;
    samplerInfo.minLod = 0.0f;
    samplerInfo.maxLod = 0.0f;
    samplerInfo.borderColor = vk::BorderColor::eFloatOpaqueWhite;
    samplerInfo.unnormalizedCoordinates = vk::False;
    m_samplers[0] = m_device->CreateSampler(samplerInfo);

    std::string vertShaderName = "cube.vert";
    std::string fragShaderName = "cube.frag";
    m_cubeVert = vkpp::ShaderStageInfo::CreateFromGLSL(m_device, vk::ShaderStageFlagBits::eVertex,
        vertShaderName, rad::File::ReadAll(vertShaderName));
    m_cubeFrag = vkpp::ShaderStageInfo::CreateFromGLSL(m_device, vk::ShaderStageFlagBits::eFragment,
        fragShaderName, rad::File::ReadAll(fragShaderName));

    m_descSetLayout = m_device->CreateDescriptorSetLayout(
        {   // binding, type, count, stageFlags, pImmutableSamplers
            vk::DescriptorSetLayoutBinding { 0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eVertex, nullptr },
            vk::DescriptorSetLayoutBinding { 1, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment, nullptr },
        }
    );

    m_pipelineLayout = m_device->CreatePipelineLayout({ m_descSetLayout->GetHandle() });

    vk::GraphicsPipelineCreateInfo pipelineInfo = {};
    VK_STRUCTURE_CHAIN_BEGIN(pipelineInfo);
    vk::PipelineShaderStageCreateInfo shaderStageInfos[2] =
    {
        *m_cubeVert,
        *m_cubeFrag,
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
    rasterizationInfo.cullMode = vk::CullModeFlagBits::eBack;
    rasterizationInfo.frontFace = vk::FrontFace::eCounterClockwise;
    rasterizationInfo.lineWidth = 1.0f;
    pipelineInfo.pRasterizationState = &rasterizationInfo;
    vk::PipelineMultisampleStateCreateInfo multisampleInfo = {};
    pipelineInfo.pMultisampleState = &multisampleInfo;
    vk::PipelineDepthStencilStateCreateInfo depthStencilInfo = {};
    depthStencilInfo.flags = {};
    depthStencilInfo.depthTestEnable = vk::True;
    depthStencilInfo.depthWriteEnable = vk::True;
    depthStencilInfo.depthCompareOp = vk::CompareOp::eLessOrEqual;
    depthStencilInfo.depthBoundsTestEnable = vk::False;
    depthStencilInfo.stencilTestEnable = vk::False;
    depthStencilInfo.front.failOp = vk::StencilOp::eKeep;
    depthStencilInfo.front.passOp = vk::StencilOp::eKeep;
    depthStencilInfo.front.depthFailOp = vk::StencilOp::eKeep;
    depthStencilInfo.front.compareOp = vk::CompareOp::eAlways;
    depthStencilInfo.front.compareMask = 0;
    depthStencilInfo.front.writeMask = 0;
    depthStencilInfo.front.reference = 0;
    depthStencilInfo.back = depthStencilInfo.front;
    depthStencilInfo.minDepthBounds = 0.0f;
    depthStencilInfo.maxDepthBounds = 0.0f;
    pipelineInfo.pDepthStencilState = &depthStencilInfo;
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
    pipelineInfo.layout = m_pipelineLayout->GetHandle();
    pipelineInfo.renderPass = nullptr;
    pipelineInfo.subpass = 0;
    pipelineInfo.basePipelineHandle = nullptr;
    pipelineInfo.basePipelineIndex = 0;

    vk::PipelineRenderingCreateInfo renderingInfo = {};
    renderingInfo.colorAttachmentCount = 1;
    vk::Format swapchainImageFormat = m_frame->GetRenderTarget()->GetFormat();
    renderingInfo.pColorAttachmentFormats = &swapchainImageFormat;
    renderingInfo.depthAttachmentFormat = m_depthImage->GetFormat();
    VK_STRUCTURE_CHAIN_ADD(pipelineInfo, renderingInfo);

    VK_STRUCTURE_CHAIN_END(pipelineInfo);
    m_pipeline = m_device->CreateGraphicsPipeline(pipelineInfo);

    std::vector<vk::DescriptorPoolSize> descPoolSizes =
    {
        { vk::DescriptorType::eUniformBuffer, sdf::MaxFrameLag },
        { vk::DescriptorType::eCombinedImageSampler, sdf::MaxFrameLag },
    };
    m_descPool = m_device->CreateDescriptorPool(sdf::MaxFrameLag, descPoolSizes);
    std::vector<vk::DescriptorSetLayout> setLayoutHandles(sdf::MaxFrameLag, m_descSetLayout->GetHandle());
    m_descSets = m_descPool->Allocate(setLayoutHandles);
    for (uint32_t i = 0; i < sdf::MaxFrameLag; ++i)
    {
        m_descSets[i]->UpdateBuffers(0, 0, vk::DescriptorType::eUniformBuffer, m_uniformBuffers[i].get());
        std::vector<vk::DescriptorImageInfo> textureInfos(m_textures.size());
        for (size_t i = 0; i < m_textures.size(); ++i)
        {
            textureInfos[i].sampler = m_samplers[i]->GetHandle();
            textureInfos[i].imageView = m_textureViews[i]->GetHandle();
            textureInfos[i].imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
        }
        m_descSets[i]->UpdateCombinedImageSamplers(1, 0, textureInfos);
    }

    m_cmdStream = m_device->CreateCommandStream(vkpp::QueueFamily::Graphics);
    m_cmdBuffers = m_cmdStream->m_cmdPool->AllocatePrimaries(sdf::MaxFrameLag);

    return true;
}

void CubeDemo::OnIdle()
{
    if (m_frame)
    {
        m_frame->BeginFrame();

        uint32_t cmdBufferIndex = m_frame->m_cmdBufferIndex;
        vkpp::CommandBuffer* cmdBuffer = m_cmdBuffers[cmdBufferIndex].get();

        // Update ShaderUniformData
        float aspect = float(m_width) / float(m_height);
        m_projectionMatrix = glm::perspective(glm::radians(45.0f), aspect, 0.1f, 100.0f);
        m_projectionMatrix[1][1] *= -1; // Flip projection matrix from GL to Vulkan orientation.
        glm::mat4 viewProjection = m_projectionMatrix * m_viewMatrix;
        // Rotate around the Y axis
        m_modelMatrix = glm::rotate(m_modelMatrix, glm::radians(m_spinAngle), glm::vec3(0, 1, 0));
        glm::mat4 modelViewProjection = viewProjection * m_modelMatrix;
        memcpy(m_uniforms[cmdBufferIndex], (const void*)&modelViewProjection, sizeof(modelViewProjection));

        cmdBuffer->Begin();
        vkpp::Image* renderTarget = m_frame->GetRenderTarget();
        vkpp::ImageView* renderTargetView = m_frame->GetRenderTargetView();
        if (renderTarget->GetCurrentLayout() != vk::ImageLayout::eColorAttachmentOptimal)
        {
            cmdBuffer->TransitLayoutFromCurrent(renderTarget,
                vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                vk::AccessFlagBits2::eColorAttachmentWrite,
                vk::ImageLayout::eColorAttachmentOptimal);
        }
        if (m_depthImage->GetCurrentLayout() != vk::ImageLayout::eDepthAttachmentOptimal)
        {
            cmdBuffer->TransitLayoutFromCurrent(m_depthImage.get(),
                vk::PipelineStageFlagBits2::eEarlyFragmentTests |
                vk::PipelineStageFlagBits2::eLateFragmentTests,
                vk::AccessFlagBits2::eDepthStencilAttachmentRead |
                vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
                vk::ImageLayout::eDepthAttachmentOptimal);
        }

        vk::RenderingInfo renderingInfo = {};
        renderingInfo.renderArea.offset = vk::Offset2D{ 0, 0 };
        renderingInfo.renderArea.extent =
            vk::Extent2D{ renderTarget->GetWidth(), renderTarget->GetHeight() };
        renderingInfo.layerCount = 1;
        renderingInfo.viewMask = 0;
        vk::RenderingAttachmentInfo colorAttachmentInfo = {};
        colorAttachmentInfo.imageView = renderTargetView->GetHandle();
        colorAttachmentInfo.imageLayout = renderTarget->GetCurrentLayout();
        colorAttachmentInfo.loadOp = vk::AttachmentLoadOp::eClear;
        colorAttachmentInfo.storeOp = vk::AttachmentStoreOp::eStore;
        colorAttachmentInfo.clearValue.color = { 0.2f, 0.2f, 0.2f, 0.2f };
        vk::RenderingAttachmentInfo depthAttachmentInfo = {};
        depthAttachmentInfo.imageView = m_depthImageView->GetHandle();
        depthAttachmentInfo.imageLayout = m_depthImage->GetCurrentLayout();
        depthAttachmentInfo.loadOp = vk::AttachmentLoadOp::eClear;
        depthAttachmentInfo.storeOp = vk::AttachmentStoreOp::eStore;
        depthAttachmentInfo.clearValue.depthStencil.depth = 1.0f;
        renderingInfo.setColorAttachments({ 1, &colorAttachmentInfo });
        renderingInfo.setPDepthAttachment(&depthAttachmentInfo);
        cmdBuffer->BeginRendering(renderingInfo);

        cmdBuffer->BindPipeine(m_pipeline.get());
        cmdBuffer->BindDescriptorSets(vk::PipelineBindPoint::eGraphics,
            m_pipelineLayout->GetHandle(),
            0, { m_descSets[cmdBufferIndex]->GetHandle() }
        );

        vk::Viewport viewport = {};
        viewport.x = 0.0f;
        viewport.y = 0.0f;
        viewport.width = float(renderTarget->GetWidth());
        viewport.height = float(renderTarget->GetHeight());
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;
        cmdBuffer->SetViewport(0, viewport);

        vk::Rect2D scissor = {};
        scissor.extent.width = renderTarget->GetWidth();
        scissor.extent.height = renderTarget->GetHeight();
        cmdBuffer->SetScissor(0, scissor);

        cmdBuffer->Draw(12 * 3, 1, 0, 0);

        cmdBuffer->EndRendering();
        cmdBuffer->End();

        m_cmdStream->Submit(m_cmdBuffers[cmdBufferIndex]->GetHandle(), {}, {}, nullptr);

        if (m_showDemoWindow)
        {
            ImGui::ShowDemoWindow(&m_showDemoWindow);
        }
        m_frame->Render();
        m_frame->EndFrame();
    }
}

void CubeDemo::OnResized(int width, int height)
{
    VulkanWindow::OnResized(width, height);
    m_depthImage = m_device->CreateImage2D_DepthStencilAttachment(
        vk::Format::eD32Sfloat, width, height);
    m_depthImageView = m_depthImage->CreateView2D();
    m_width = width;
    m_height = height;
}

void CubeDemo::OnKeyDown(const SDL_KeyboardEvent& keyDown)
{
    if (keyDown.key == SDLK_F1)
    {
        m_showDemoWindow = !m_showDemoWindow;
    }

    if (keyDown.key == SDLK_F11)
    {
        m_enableFullscreen = !m_enableFullscreen;
        SetFullscreen(m_enableFullscreen);
    }

    if (keyDown.key == SDLK_ESCAPE)
    {
        Destroy();
    }
}

void CubeDemo::OnKeyUp(const SDL_KeyboardEvent& keyUp)
{
}
