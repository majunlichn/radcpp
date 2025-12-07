#pragma once

#include <rad/ML/MLContext.h>

namespace rad
{

class MLCpuDevice;

class MLCpuContext : public MLContext
{
public:
    MLCpuContext(Ref<MLCpuDevice> device);
    ~MLCpuContext() override;

    Ref<MLCpuDevice> m_device;

}; // class MLCpuContext

} // namespace rad
