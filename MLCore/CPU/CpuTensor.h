#pragma once

#include <MLCore/Tensor.h>

namespace ML
{

class CpuDevice;

class CpuTensorStorage : public TensorStorage
{
public:
    std::vector<uint8_t> m_buffer;

    CpuTensorStorage(rad::Ref<CpuDevice> device);
    ~CpuTensorStorage();

    CpuDevice* GetDevice();

    bool Init(rad::ArrayRef<size_t> sizes, DataType dataType, const TensorOptions& options = {});

    virtual void Read(void* data, size_t offset, size_t sizeInBytes) override;
    virtual void Write(const void* data, size_t offset, size_t sizeInBytes) override;

    virtual void* MapMemory(size_t offset, size_t size) override;
    virtual void UnmapMemory() override;

}; // class CpuTensor

} // namespace ML
