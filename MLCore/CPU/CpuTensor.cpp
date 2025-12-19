#include <MLCore/CPU/CpuTensor.h>
#include <MLCore/CPU/CpuDevice.h>
#include <MLCore/CPU/CpuContext.h>

namespace ML
{

CpuTensor::CpuTensor(rad::Ref<CpuDevice> device) :
    Tensor(std::move(device))
{
}

CpuTensor::~CpuTensor()
{
}

CpuDevice* CpuTensor::GetDevice()
{
    return static_cast<CpuDevice*>(m_device.get());
}

bool CpuTensor::Init(rad::ArrayRef<size_t> sizes, DataType dataType, const TensorOptions& options)
{
    m_dataType = dataType;
    m_sizes = sizes;
    const auto& strides = options.m_strides;
    if (strides.empty())
    {
        m_strides = MakeStrides(sizes);
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
    size_t elementCount = GetElementCount();
    if (indexOfTheLastElement + 1 == elementCount)
    {
        m_isContiguous = true;
    }
    m_buffer.resize((indexOfTheLastElement + 1) * GetElementSize(dataType));
    return true;
}

void CpuTensor::Read(void* data, size_t offset, size_t sizeInBytes)
{
    std::memcpy(data, m_buffer.data() + offset, sizeInBytes);
}

void CpuTensor::Write(const void* data, size_t offset, size_t sizeInBytes)
{
    std::memcpy(m_buffer.data() + offset, data, sizeInBytes);
}

} // namespace ML
