#include <MLCore/Global.h>
#include <MLCore/Backend.h>
#include <MLCore/Logging.h>

#include <MLCore/CPU/CpuBackend.h>

namespace ML
{

static std::map<std::string, rad::Ref<Backend>, rad::StringLess> g_backends;

std::map<Device*, rad::Ref<Context>> g_defaultContexts;

thread_local rad::Ref<Device> g_currentDevice;
thread_local rad::Ref<Context> g_currentContext;

bool Initialize()
{
    if (auto cpuBackend = CreateCpuBackend())
    {
        RegisterBackend("CPU", cpuBackend);
        SetCurrentDevice(cpuBackend->GetDevice(0));
    }
    return true;
}

void Finalize()
{
    g_currentContext.reset();
    g_currentDevice.reset();
    g_defaultContexts.clear();
    g_backends.clear();
}

bool RegisterBackend(std::string_view name, rad::Ref<Backend> backend)
{
    g_backends[std::string(name)] = backend;
    backend->m_name = std::string(name);
    return true;
}

void UnregisterBackend(std::string_view name)
{
    g_backends.erase(std::string(name));
}

Backend* GetBackend(std::string_view name)
{
    name = name.substr(0, name.find_first_of(':'));
    auto iter = g_backends.find(name);
    if (iter != g_backends.end())
    {
        return iter->second.get();
    }
    else
    {
        return nullptr;
    }
}

Device* GetDevice(std::string_view backendName, size_t deviceIndex)
{
    return GetBackend(backendName)->GetDevice(deviceIndex);
}

void SetDefaultContext(Device* device, rad::Ref<Context> context)
{
    g_defaultContexts[device] = context;
}

Context* GetDefaultContext(Device* device)
{
    auto iter = g_defaultContexts.find(device);
    if (iter != g_defaultContexts.end())
    {
        return iter->second.get();
    }
    else
    {
        auto [iter, inserted] = g_defaultContexts.emplace(device, device->CreateContext());
        return iter->second.get();
    }
}

Context* GetDefaultContext(std::string_view backendName, size_t deviceIndex)
{
    Device* device = GetDevice(backendName, deviceIndex);
    if (device)
    {
        return GetDefaultContext(device);
    }
    else
    {
        return nullptr;
    }
}

void SetCurrentDevice(rad::Ref<Device> device)
{
    g_currentDevice = std::move(device);
    SetCurrentContext(GetDefaultContext(g_currentDevice.get()));
}

Device* GetCurrentDevice()
{
    return g_currentDevice.get();
}

void SetCurrentContext(rad::Ref<Context> context)
{
    assert(context->m_device == g_currentDevice);
    g_currentContext = std::move(context);
}

Context* GetCurrentContext()
{
    return g_currentContext.get();
}

} // namespace ML
