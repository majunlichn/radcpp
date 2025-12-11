#include <rad/ML/MLDevice.h>
#include <rad/ML/MLContext.h>
#include <rad/ML/MLTensor.h>

#include <map>

namespace rad
{

static std::map<std::string, Ref<MLDevice>, StringLess> g_MLDevices;
thread_local static std::map<std::string, Ref<MLContext>, StringLess> g_MLContexts;

MLDevice* RegisterGlobalMLDevice(std::string_view name, Ref<MLDevice> device)
{
    return g_MLDevices.emplace(std::string(name), device).first->second.get();
}

MLDevice* GetGlobalMLDevice(std::string_view name)
{
    auto iter = g_MLDevices.find(name);
    if (iter != g_MLDevices.end())
    {
        return iter->second.get();
    }
    else
    {
        return nullptr;
    }
}

MLContext* RegisterPerThreadMLContext(std::string_view name, Ref<MLContext> context)
{
    return g_MLContexts.emplace(std::string(name), context).first->second.get();
}

MLContext* GetPerThreadMLContext(std::string_view name)
{
    auto iter = g_MLContexts.find(name);
    if (iter != g_MLContexts.end())
    {
        return iter->second.get();
    }
    else if (auto device = GetGlobalMLDevice(name))
    {
        return RegisterPerThreadMLContext(name, device->CreateContext());
    }
    else
    {
        return nullptr;
    }
}

} // namespace rad
