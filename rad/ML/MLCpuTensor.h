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

    bool Init(MLDataType dataType, ArrayRef<size_t> sizes, ArrayRef<size_t> strides = {});

    Ref<MLCpuDevice> m_device;
    std::vector<uint8_t> m_buffer;
    bool m_isContiguous = false;

}; // class MLCpuTensor

} // namespace rad
