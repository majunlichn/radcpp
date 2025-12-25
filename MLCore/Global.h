#pragma once

#include <MLCore/Common.h>
#include <map>

namespace ML
{

class Backend;
class Device;
class Context;

bool Initialize();
void Finalize();

bool RegisterBackend(std::string_view name, rad::Ref<Backend> backend);
void UnregisterBackend(std::string_view name);
Backend* GetBackend(std::string_view name);

class ContextPool : public rad::RefCounted<ContextPool>
{
public:
    std::map<Device*, rad::Ref<Context>> m_contexts;

    ContextPool();
    virtual ~ContextPool();

    // Create default contexts for all devices of the backend.
    bool CreateContextsForBackend(Backend* backend);
    bool SetDeviceContext(Device* device, rad::Ref<Context> context);

    void Clear();

    Context* GetContext(Device* device);

}; // class ContextPool

// The global context pool.
extern rad::Ref<ContextPool> g_contextPool;

// Set the default global device of the current thread.
void SetCurrentDevice(rad::Ref<Device> device);
// Get the default global device of the current thread.
Device* GetCurrentDevice();
// Set the default global context of the current thread (the context must be created with the current device).
void SetCurrentContext(rad::Ref<Context> context);
// Get the default global context of the current thread.
Context* GetCurrentContext();

} // namespace ML
