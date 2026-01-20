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

Device* GetDevice(std::string_view backendName, size_t deviceIndex);
void SetDefaultContext(Device* device, rad::Ref<Context> context);
Context* GetDefaultContext(Device* device);
Context* GetDefaultContext(std::string_view backendName, size_t deviceIndex);

// Set the default global device of the current thread.
void SetCurrentDevice(rad::Ref<Device> device);
// Get the default global device of the current thread.
Device* GetCurrentDevice();
// Set the default global context of the current thread (the context must be created with the current device).
void SetCurrentContext(rad::Ref<Context> context);
// Get the default global context of the current thread.
Context* GetCurrentContext();

} // namespace ML
