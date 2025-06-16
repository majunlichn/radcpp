#pragma once

#include "Version.h"
#include <SDFramework/Gui/VulkanWindow.h>
#include <SDFramework/Gui/VulkanFrame.h>
#include <vkpp/Core/RenderPass.h>
#include <vkpp/Core/Framebuffer.h>
#include <vkpp/Core/ShaderCompiler.h>

#include <glm/glm.hpp>

class CubeDemo : public sdf::VulkanWindow
{
public:
    CubeDemo();
    ~CubeDemo();

    bool Init(int argc, char* argv[]);
    void ParseCommandLine(int argc, char* argv[]);

    virtual void OnIdle() override;

    virtual void OnResized(int width, int height) override;

    virtual void OnKeyDown(const SDL_KeyboardEvent& keyDown) override;
    virtual void OnKeyUp(const SDL_KeyboardEvent& keyUp) override;

    int32_t m_gpuIndex = -1;
    std::string m_gpuName;

    uint32_t m_width = 0;
    uint32_t m_height = 0;
    vk::PresentModeKHR m_presentMode = vk::PresentModeKHR::eFifo;
    bool m_enableFullscreen = false;

    glm::mat4 m_projectionMatrix = {};
    glm::mat4 m_viewMatrix = {};
    glm::mat4 m_modelMatrix = {};

    float m_spinAngle = 0.0f;
    float m_spinIncrement = 0.0f;
    bool m_spinPause = false;

    uint32_t m_frameIndex = 0;
    uint32_t m_frameCount = 0;

    bool m_showDemoWindow = false;

    struct ShaderUniformData
    {
        glm::mat4 modelViewProjection;
        glm::vec4 positions[12 * 3];
        glm::vec4 attribs[12 * 3];
    } m_shaderUniforms = {};

    rad::Ref<vkpp::Buffer> m_uniformBuffers[sdf::MaxFrameLag];
    void* m_uniforms[sdf::MaxFrameLag] = {};

    rad::Ref<vkpp::Image> m_depthImage;
    rad::Ref<vkpp::ImageView> m_depthImageView;

    std::vector<rad::Ref<vkpp::Image>> m_textures;
    std::vector<rad::Ref<vkpp::ImageView>> m_textureViews;
    std::vector<rad::Ref<vkpp::Sampler>> m_samplers;

    rad::Ref<vkpp::ShaderStageInfo> m_cubeVert;
    rad::Ref<vkpp::ShaderStageInfo> m_cubeFrag;
    rad::Ref<vkpp::DescriptorSetLayout> m_descSetLayout;
    rad::Ref<vkpp::PipelineLayout> m_pipelineLayout;
    rad::Ref<vkpp::Pipeline> m_pipeline;

    rad::Ref<vkpp::DescriptorPool> m_descPool;
    std::vector<rad::Ref<vkpp::DescriptorSet>> m_descSets;

    rad::Ref<vkpp::CommandStream> m_cmdStream;
    std::vector<rad::Ref<vkpp::CommandBuffer>> m_cmdBuffers;

}; // class CubeDemo
