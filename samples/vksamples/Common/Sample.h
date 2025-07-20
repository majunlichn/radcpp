#pragma once

#include <SDFramework/Gui/VulkanWindow.h>
#include <SDFramework/Gui/VulkanFrame.h>
#include <vkpp/Core/RenderPass.h>
#include <vkpp/Core/Framebuffer.h>
#include <vkpp/Core/ShaderCompiler.h>

#include <glm/glm.hpp>

class Sample : public sdf::VulkanWindow
{
public:
    Sample() {}
    ~Sample() {}

    virtual bool Init(int argc, char* argv[]);

}; // class Sample
