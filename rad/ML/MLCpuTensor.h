#pragma once

#include <rad/ML/MLTensor.h>

namespace rad
{

class MLCpuDevice;

class MLCpuTensor : public MLTensor
{
public:
    MLCpuTensor(Ref<MLCpuDevice> device);
    ~MLCpuTensor();

    Ref<MLCpuDevice> m_device;
    std::vector<uint8_t> m_buffer;

}; // class MLCpuTensor

} // namespace rad
