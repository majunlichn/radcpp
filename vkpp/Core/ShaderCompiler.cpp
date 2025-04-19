#include <vkpp/Core/ShaderCompiler.h>
#include <rad/IO/File.h>
#include <rad/System/OS.h>

namespace vkpp
{

shaderc_shader_kind GetShaderKind(vk::ShaderStageFlagBits stage)
{
    switch (stage)
    {
    case vk::ShaderStageFlagBits::eVertex: return shaderc_vertex_shader;
    case vk::ShaderStageFlagBits::eTessellationControl: return shaderc_tess_control_shader;
    case vk::ShaderStageFlagBits::eTessellationEvaluation: return shaderc_tess_evaluation_shader;
    case vk::ShaderStageFlagBits::eGeometry: return shaderc_geometry_shader;
    case vk::ShaderStageFlagBits::eFragment: return shaderc_fragment_shader;
    case vk::ShaderStageFlagBits::eCompute: return shaderc_compute_shader;
    case vk::ShaderStageFlagBits::eRaygenKHR: return shaderc_raygen_shader;
    case vk::ShaderStageFlagBits::eAnyHitKHR: return shaderc_anyhit_shader;
    case vk::ShaderStageFlagBits::eClosestHitKHR: return shaderc_closesthit_shader;
    case vk::ShaderStageFlagBits::eMissKHR: return shaderc_miss_shader;
    case vk::ShaderStageFlagBits::eIntersectionKHR: return shaderc_intersection_shader;
    case vk::ShaderStageFlagBits::eCallableKHR: return shaderc_callable_shader;
    case vk::ShaderStageFlagBits::eTaskEXT: return shaderc_task_shader;
    case vk::ShaderStageFlagBits::eMeshEXT: return shaderc_mesh_shader;
    }
    return shaderc_glsl_infer_from_source;
}

ShaderCompiler::ShaderCompiler()
{
}

ShaderCompiler::~ShaderCompiler()
{
}

std::string ShaderCompiler::Preprocess(
    vk::ShaderStageFlagBits stage, const std::string& fileName, const std::string& source,
    const std::string& entryPoint, rad::Span<ShaderMacro> macros)
{
    shaderc::CompileOptions options;

    for (const auto& macro : macros)
    {
        options.AddMacroDefinition(macro.m_name, macro.m_definition);
    }
    if (!rad::StrEqual(entryPoint, "main"))
    {
        options.AddMacroDefinition(entryPoint, "main");
    }

    shaderc::PreprocessedSourceCompilationResult result =
        m_compiler.PreprocessGlsl(source, GetShaderKind(stage), fileName.c_str(), options);
    if (result.GetCompilationStatus() == shaderc_compilation_status_success)
    {
        return { result.cbegin(), result.cend() };
    }
    else
    {
        m_log = result.GetErrorMessage();
        return {};
    }
}

std::vector<uint32_t> ShaderCompiler::CompileGLSL(
    vk::ShaderStageFlagBits stage, const std::string& fileName, const std::string& source,
    const std::string& entryPoint, rad::Span<ShaderMacro> macros, shaderc_optimization_level opt)
{
    shaderc::CompileOptions options;
    for (const ShaderMacro& macro : macros)
    {
        options.AddMacroDefinition(macro.m_name, macro.m_definition);
    }
    if (!rad::StrEqual(entryPoint, "main"))
    {
        options.AddMacroDefinition(entryPoint, "main");
    }

    std::unique_ptr<glslc::FileIncluder> includer(
        RAD_NEW glslc::FileIncluder(&m_fileFinder));
    options.SetIncluder(std::move(includer));
    options.SetTargetEnvironment(shaderc_target_env_vulkan, shaderc_env_version_vulkan_1_4);
    options.SetOptimizationLevel(opt);

    shaderc::SpvCompilationResult result =
        m_compiler.CompileGlslToSpv(source, GetShaderKind(stage), fileName.c_str(), options);

    if (result.GetCompilationStatus() == shaderc_compilation_status_success)
    {
        return { result.cbegin(), result.cend() };
    }
    else
    {
        m_log = result.GetErrorMessage();
        return {};
    }
}

std::string ShaderCompiler::Disassemble(const uint32_t* binary, size_t binary_size, uint32_t options)
{
    spv_text text = nullptr;
    spv_diagnostic diagnostic = nullptr;
    spv_context context = spvContextCreate(SPV_ENV_VULKAN_1_4);
    spv_result_t error =
        spvBinaryToText(context, binary, binary_size, options, &text, &diagnostic);
    if (error)
    {
        m_log = diagnostic->error;
        spvDiagnosticDestroy(diagnostic);
        return {};
    }
    return std::string(text->str, text->length);
    spvTextDestroy(text);
    spvContextDestroy(context);
}

} // namespace vkpp
