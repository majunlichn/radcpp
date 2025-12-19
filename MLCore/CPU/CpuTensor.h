#pragma once

#include <MLCore/Tensor.h>

namespace ML
{

class CpuDevice;

class CpuTensor : public Tensor
{
public:
    std::vector<uint8_t> m_buffer;
    bool m_isContiguous = false;

    CpuTensor(rad::Ref<CpuDevice> device);
    ~CpuTensor();

    CpuDevice* GetDevice();

    bool Init(rad::ArrayRef<size_t> sizes, DataType dataType, const TensorOptions& options = {});

    virtual void* GetData() override { return m_buffer.data(); }
    virtual size_t GetDataSize() const override { return m_buffer.size(); }
    virtual void Read(void* data, size_t offset, size_t sizeInBytes) override;
    virtual void Write(const void* data, size_t offset, size_t sizeInBytes) override;

}; // class CpuTensor

} // namespace ML
