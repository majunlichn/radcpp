#include <MLCore/Global.h>
#include <MLCore/Backend.h>
#include <MLCore/Logging.h>

namespace ML
{

static std::map<std::string, rad::Ref<Backend>, rad::StringLess> g_backends;

thread_local rad::Ref<Device> g_currentDevice;
thread_local rad::Ref<Context> g_currentContext;

rad::Ref<ContextPool> g_contextPool;

bool Initialize()
{
    if (!g_contextPool)
    {
        g_contextPool = RAD_NEW ContextPool();
    }
    return true;
}

void Finalize()
{
    g_currentContext.reset();
    g_currentDevice.reset();
    g_contextPool.reset();
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

void SetCurrentDevice(rad::Ref<Device> device)
{
    g_currentDevice = std::move(device);
    SetCurrentContext(g_contextPool->GetContext(g_currentDevice.get()));
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

ContextPool* GetGlobalContextPool()
{
    return g_contextPool.get();
}

ContextPool::ContextPool()
{
}

ContextPool::~ContextPool()
{
    Clear();
}

bool ContextPool::CreateContextsForBackend(Backend* backend)
{
    for (size_t i = 0; i < backend->GetDeviceCount(); ++i)
    {
        Device* device = backend->GetDevice(i);
        if (device != nullptr)
        {
            m_contexts[device] = device->CreateContext();
        }
    }
    return true;
}

bool ContextPool::SetDeviceContext(Device* device, rad::Ref<Context> context)
{
    m_contexts[device] = device->CreateContext();
    return true;
}

void ContextPool::Clear()
{
    m_contexts.clear();
}

Context* ContextPool::GetContext(Device* device)
{
    if (device == nullptr)
    {
        device = GetCurrentDevice();
    }
    auto iter = m_contexts.find(device);
    if (iter != m_contexts.end())
    {
        return iter->second.get();
    }
    else
    {
        auto [iter, inserted] = m_contexts.emplace(device, device->CreateContext());
        return iter->second.get();
    }
}

} // namespace ML
