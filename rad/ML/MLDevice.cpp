#include <rad/ML/MLDevice.h>
#include <rad/ML/MLCpuDevice.h>
#include <rad/ML/MLContext.h>
#include <rad/ML/MLTensor.h>
#include <rad/IO/Logging.h>

#include <map>

namespace rad
{

static std::map<std::string, Ref<MLDevice>, StringLess> g_MLDevices;
thread_local static std::map<std::string, Ref<MLContext>, StringLess> g_MLContexts;

MLDevice* MLRegisterGlobalDevice(std::string_view backend, Ref<MLDevice> device)
{
    return g_MLDevices.emplace(std::string(backend), device).first->second.get();
}

MLDevice* MLGetGlobalDevice(std::string_view backend)
{
    if (g_MLDevices.empty())
    {
        // register CPU device for CPU backend as the default option.
        MLRegisterGlobalDevice("CPU", RAD_NEW MLCpuDevice());
    }
    auto iter = g_MLDevices.find(backend);
    if (iter != g_MLDevices.end())
    {
        return iter->second.get();
    }
    else
    {
        return nullptr;
    }
}

MLContext* MLRegisterPerThreadContext(std::string_view backend, Ref<MLContext> context)
{
    return g_MLContexts.emplace(std::string(backend), context).first->second.get();
}

MLContext* MLGetPerThreadContext(std::string_view backend)
{
    auto iter = g_MLContexts.find(backend);
    if (iter != g_MLContexts.end())
    {
        return iter->second.get();
    }
    else if (auto device = MLGetGlobalDevice(backend))
    {
        return MLRegisterPerThreadContext(backend, device->CreateContext());
    }
    else
    {
        return nullptr;
    }
}

Ref<MLTensor> MLCreateTensor(ArrayRef<size_t> sizes, MLDataType dataType, std::string_view backend, const MLTensorOptions& options)
{
    if (auto device = MLGetGlobalDevice(backend))
    {
        return device->CreateTensor(sizes, dataType, options);
    }
    else
    {
        RAD_LOG(err, "MLCreateTensor: no device registered for backend '{}'", backend);
        return nullptr;
    }
}

} // namespace rad
