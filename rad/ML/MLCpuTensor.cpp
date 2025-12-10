#include <rad/ML/MLCpuTensor.h>
#include <rad/ML/MLCpuDevice.h>

namespace rad
{

MLCpuTensor::MLCpuTensor(Ref<MLCpuDevice> device) :
    m_device(std::move(device))
{
}

MLCpuTensor::~MLCpuTensor()
{
}

MLDevice* MLCpuTensor::GetDevice()
{
    return m_device.get();
}

bool MLCpuTensor::Init(MLDataType dataType, ArrayRef<size_t> sizes, ArrayRef<size_t> strides)
{
    m_dataType = dataType;
    m_sizes = sizes;
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

} // namespace rad
