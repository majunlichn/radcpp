#pragma once

#include <vkpp/Core/Common.h>
#include <vkpp/Core/ShaderMacro.h>

#include <slang.h>
#include <slang-com-ptr.h>

#include <map>

namespace vkpp
{

class SlangSession : public rad::RefCounted<SlangSession>
{
public:
    static slang::IGlobalSession* GetGlobalSession();
    static spdlog::logger* GetLogger();

    SlangSession();
    ~SlangSession();

    bool Init(const slang::SessionDesc& sessionDesc);
    bool Init();
    slang::IModule* LoadModule(std::string_view moduleName);
    slang::IModule* GetModule(std::string_view moduleName) const
    {
        auto it = m_modules.find(moduleName);
        if (it != m_modules.end())
        {
            return it->second;
        }
        else
        {
            return nullptr;
        }
    }

    struct ComponentDesc
    {
        std::string moduleName;
        std::string entryPoint; // if empty, the entire module will be imported.
    };
    bool CreateProgram(rad::ArrayRef<ComponentDesc> descs);
    Slang::ComPtr<slang::IBlob> GenerateTargetCode(int entryPointIndex = 0, int targetIndex = 0);

    std::vector<std::string> m_searchPaths;
    std::vector<ShaderMacro> m_macros;

    Slang::ComPtr<slang::ISession> m_session;
    std::map<std::string, slang::IModule*, rad::StringLess> m_modules;
    Slang::ComPtr<slang::IComponentType> m_linkedProgram;

}; // class SlangSession

} // namespace vkpp
