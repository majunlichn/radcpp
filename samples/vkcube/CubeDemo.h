#pragma once

#include <SDFramework/Gui/VulkanWindow.h>
#include <SDFramework/Gui/VulkanGuiContext.h>
#include <vkpp/Core/RenderPass.h>
#include <vkpp/Core/Framebuffer.h>
#include <vkpp/Core/ShaderCompiler.h>

#include <glm/glm.hpp>

class CubeDemo : public sdf::VulkanWindow
{
public:
    CubeDemo(rad::Ref<vkpp::Instance> instance);
    ~CubeDemo();

    bool Init(int argc, char* argv);

    std::string m_name = "vkcube";
    bool m_initialized = false;
    bool m_swapchainReady = false;
    bool m_isMinimized = false;
    bool m_useStagingBuffer = false;
    bool m_separatePresentQueue = false;
    bool m_invalidGpuSelection = false;
    int32_t m_gpuNumber = 0;

    // If true, the Demo renders on the protected memory.
    bool m_protectedOutput = false;

    uint32_t m_width = 0;
    uint32_t m_height = 0;
    vk::Format m_format = vk::Format::eUndefined;
    vk::ColorSpaceKHR m_colorSpace;

    rad::Ref<vkpp::Image> m_depthImage;
    rad::Ref<vkpp::ImageView> m_depthImageView;

    std::vector<rad::Ref<vkpp::Image>> m_textures;
    std::vector<rad::Ref<vkpp::ImageView>> m_textureViews;

    rad::Ref<vkpp::RenderPass> m_renderPass;

    rad::Ref<vkpp::PipelineLayout> m_pipelineLayout;
    rad::Ref<vkpp::Pipeline> m_pipeline;

    glm::mat4 m_projectionMatrix;
    glm::mat4 m_viewMatrix;
    glm::mat4 m_modelMatrix;

    float m_spinAngle = 0.0f;
    float m_spinIncrement = 0.0f;
    bool m_spinPause = false;

    rad::Ref<vkpp::ShaderStageInfo> m_vertexShaderStage;
    rad::Ref<vkpp::ShaderStageInfo> m_fragmentShaderStage;

    rad::Ref<vkpp::DescriptorPool> m_descPool;
    rad::Ref<vkpp::DescriptorSet> m_descSet;

    std::vector<rad::Ref<vkpp::Framebuffer>> m_framebuffers;

    uint32_t m_frameIndex = 0;
    uint32_t m_frameCount = 0;

}; // class CubeDemo
