#include <vkpp/Core/SlangSession.h>
#include <mutex>

namespace vkpp
{

thread_local Slang::ComPtr<slang::IGlobalSession> g_slangGlobalSession;
thread_local std::once_flag g_slangGlobalSessionCreated;

slang::IGlobalSession* SlangSession::GetGlobalSession()
{
    return g_slangGlobalSession.get();
}

spdlog::logger* SlangSession::GetLogger()
{
    static std::shared_ptr<spdlog::logger> slangLogger = rad::CreateLogger("slang");
    return slangLogger.get();
}

SlangSession::SlangSession()
{
    // Currently, the global session type is not thread-safe.
    std::call_once(g_slangGlobalSessionCreated,
        []() {
            SlangGlobalSessionDesc desc = {};
            createGlobalSession(&desc, g_slangGlobalSession.writeRef());
        });
}

SlangSession::~SlangSession()
{
}

bool SlangSession::Init(const slang::SessionDesc& sessionDesc)
{
    SlangResult result = g_slangGlobalSession->createSession(sessionDesc, m_session.writeRef());
    if (result == SLANG_OK)
    {
        return true;
    }
    else
    {
        GetLogger()->error("Failed to create session: {}", result);
        return false;
    }
}

bool SlangSession::Init()
{
    slang::SessionDesc sessionDesc = {};
    slang::TargetDesc targetDesc = {};
    targetDesc.format = SLANG_SPIRV;
    targetDesc.profile = g_slangGlobalSession->findProfile("spirv_1_6");
    sessionDesc.targets = &targetDesc;
    sessionDesc.targetCount = 1;
    std::vector<const char*> searchPaths(m_searchPaths.size());
    for (size_t i = 0; i < m_searchPaths.size(); ++i)
    {
        searchPaths[i] = m_searchPaths[i].c_str();
    }
    sessionDesc.searchPaths = searchPaths.data();
    sessionDesc.searchPathCount = static_cast<SlangInt>(searchPaths.size());
    std::vector<slang::PreprocessorMacroDesc> macros(m_macros.size());
    for (size_t i = 0; i < m_macros.size(); ++i)
    {
        macros[i].name = m_macros[i].m_name.c_str();
        macros[i].value = m_macros[i].m_value.c_str();
    }
    sessionDesc.preprocessorMacros = macros.data();
    sessionDesc.preprocessorMacroCount = static_cast<SlangInt>(macros.size());
    return Init(sessionDesc);
}

slang::IModule* SlangSession::LoadModule(std::string_view moduleName)
{
    Slang::ComPtr<slang::IBlob> diagnostics;
    slang::IModule* module = m_session->loadModule(moduleName.data(), diagnostics.writeRef());
    if (module)
    {
        m_modules[std::string(moduleName)] = module;
        if (diagnostics)
        {
            GetLogger()->info("{} diagnostics:\n{}", moduleName, diagnostics->getBufferPointer());
        }
        return module;
    }
    else
    {
        if (diagnostics)
        {
            GetLogger()->error("Failed to load {}:\n{}", moduleName, diagnostics->getBufferPointer());
        }
        return nullptr;
    }
}

bool SlangSession::CreateProgram(rad::ArrayRef<ComponentDesc> descs)
{
    Slang::ComPtr<slang::IComponentType> program;
    Slang::Result result = 0;
    std::vector<Slang::ComPtr<slang::IEntryPoint>> slangEntryPoints;
    slangEntryPoints.reserve(descs.size());
    std::vector<slang::IComponentType*> components;
    components.reserve(descs.size());
    for (const ComponentDesc& desc : descs)
    {
        auto iter = m_modules.find(desc.moduleName);
        slang::IModule* module = nullptr;
        if (iter != m_modules.end())
        {
            module = iter->second;
        }
        else
        {
            module = LoadModule(desc.moduleName);
            if (!module)
            {
                GetLogger()->error("Failed to load module: {}", desc.moduleName);
                return false;
            }
        }

        auto& slangEntryPoint = slangEntryPoints.emplace_back();
        if (desc.entryPoint.empty())
        {
            components.push_back(module);
        }
        else
        {
            result = module->findEntryPointByName(desc.entryPoint.c_str(),
                slangEntryPoint.writeRef());
            if (result == SLANG_OK)
            {
                components.push_back(slangEntryPoint);
            }
            else
            {
                GetLogger()->error("Cannot find entry {} in {}!", desc.entryPoint, desc.moduleName);
                return false;
            }
        }
    }
    result = m_session->createCompositeComponentType(
        components.data(), static_cast<SlangInt>(components.size()),
        program.writeRef());
    if (result != SLANG_OK)
    {
        GetLogger()->error("Failed to create program: {}", result);
        return false;
    }
    Slang::ComPtr<ISlangBlob> diagnostics;
    result = program->link(m_linkedProgram.writeRef(), diagnostics.writeRef());
    if (result != SLANG_OK)
    {
        if (diagnostics)
        {
            GetLogger()->error("Failed to link program:\n{}", diagnostics->getBufferPointer());
        }
        return false;
    }
    return true;
}

Slang::ComPtr<slang::IBlob> SlangSession::GenerateTargetCode(int entryPointIndex, int targetIndex)
{
    Slang::ComPtr<slang::IBlob> codeBlob;
    Slang::ComPtr<ISlangBlob> diagnostics;
    Slang::Result result = m_linkedProgram->getEntryPointCode(
        entryPointIndex, targetIndex,
        codeBlob.writeRef(), diagnostics.writeRef());
    if (result == SLANG_OK)
    {
        return codeBlob;
    }
    else
    {
        if (diagnostics)
        {
            GetLogger()->error("Failed to get code for entryPoint#{} target#{}: {}",
                entryPointIndex, targetIndex, diagnostics->getBufferPointer());
        }
        return nullptr;
    }
}

} // namespace vkpp
