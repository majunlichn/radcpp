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

    virtual MLDevice* GetDevice() override;

    bool Init(ArrayRef<size_t> sizes, MLDataType dataType, const MLTensorOptions& options = {});

    virtual void* GetData() override { return m_buffer.data(); }
    virtual size_t GetDataSize() const override { return m_buffer.size(); }
    virtual void Read(void* data, size_t offset, size_t sizeInBytes) override;
    virtual void Write(const void* data, size_t offset, size_t sizeInBytes) override;

    Ref<MLCpuDevice> m_device;
    std::vector<uint8_t> m_buffer;
    bool m_isContiguous = false;

}; // class MLCpuTensor

} // namespace rad
