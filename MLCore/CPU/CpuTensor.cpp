#include <MLCore/CPU/CpuTensor.h>
#include <MLCore/CPU/CpuDevice.h>
#include <MLCore/CPU/CpuContext.h>

namespace ML
{

CpuTensorStorage::CpuTensorStorage(rad::Ref<CpuDevice> device) :
    TensorStorage(std::move(device))
{
}

CpuTensorStorage::~CpuTensorStorage()
{
}

CpuDevice* CpuTensorStorage::GetDevice()
{
    return static_cast<CpuDevice*>(m_device.get());
}

bool CpuTensorStorage::Init(rad::ArrayRef<size_t> sizes, DataType dataType, const TensorOptions& options)
{
    m_dataType = dataType;
    m_sizes = sizes;
    const auto& strides = options.m_strides;
    if (strides.empty())
    {
        m_strides = MakeTensorStrides(sizes);
    }
    else
    {
        assert(strides.size() == sizes.size());
        m_strides = strides;
    }

    size_t indexOfTheLastElement = 0;
    for (size_t i = 0; i < m_sizes.size(); ++i)
    {
        indexOfTheLastElement += (m_sizes[i] - 1) * m_strides[i];
    }
    m_buffer.resize((indexOfTheLastElement + 1) * GetElementSize(dataType));
    return true;
}

void CpuTensorStorage::Read(void* data, size_t offset, size_t sizeInBytes)
{
    std::memcpy(data, m_buffer.data() + offset, sizeInBytes);
}

void CpuTensorStorage::Write(const void* data, size_t offset, size_t sizeInBytes)
{
    std::memcpy(m_buffer.data() + offset, data, sizeInBytes);
}

void* CpuTensorStorage::MapMemory(size_t offset, size_t size)
{
    return m_buffer.data() + offset;
}

void CpuTensorStorage::UnmapMemory()
{
    // nothing to do
}

} // namespace ML
