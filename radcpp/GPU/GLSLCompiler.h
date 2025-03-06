#pragma once

#include <radcpp/GPU/VulkanCommon.h>
#include <radcpp/GPU/ShaderMacro.h>
#include <shaderc/shaderc.hpp>
#include "ShaderIncluder.h"
#include <spirv-tools/libspirv.h>

namespace rad
{

shaderc_shader_kind GetShaderKind(vk::ShaderStageFlagBits stage);

// Runtime GLSL compiler that wraps google/shaderc targets the latest Vulkan profile.
class GLSLCompiler : public RefCounted<GLSLCompiler>
{
public:
    enum
    {
        DefaultDisassembleOptions =
            SPV_BINARY_TO_TEXT_OPTION_INDENT |
            SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES |
            SPV_BINARY_TO_TEXT_OPTION_COMMENT |
            SPV_BINARY_TO_TEXT_OPTION_NESTED_INDENT |
            SPV_BINARY_TO_TEXT_OPTION_REORDER_BLOCKS,
    };

    GLSLCompiler();
    ~GLSLCompiler();

    void AddIncludeDir(std::string includeDir)
    {
        m_fileFinder.search_path().push_back(std::move(includeDir));
    }

    const std::string& GetLog() const { return m_log; }

    std::string Preprocess(
        vk::ShaderStageFlagBits stage, const std::string& fileName, const std::string& source,
        const std::string& entryPoint = "main", rad::Span<ShaderMacro> macros = {});
    std::vector<uint32_t> Compile(
        vk::ShaderStageFlagBits stage, const std::string& fileName, const std::string& source,
        const std::string& entryPoint = "main", rad::Span<ShaderMacro> macros = {},
        shaderc_optimization_level opt = shaderc_optimization_level_zero);
    std::string CompileToAssembly(
        vk::ShaderStageFlagBits stage, const std::string& fileName, const std::string& source,
        const std::string& entryPoint = "main", rad::Span<ShaderMacro> macros = {},
        shaderc_optimization_level opt = shaderc_optimization_level_zero);

    // Default option for max
    std::string Disassemble(const uint32_t* binary, size_t binary_size,
        uint32_t options = DefaultDisassembleOptions);

    shaderc::Compiler m_compiler;
    std::string m_log;
    glslc::FileFinder m_fileFinder;

}; // class GLSLCompiler

} // namespace rad
