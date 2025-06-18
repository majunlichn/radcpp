#include <vkpp/Compute/Tensor.h>
#include <vkpp/Core/Command.h>

namespace vkpp
{

Tensor::Tensor(rad::Ref<Device> device) :
    m_device(std::move(device))
{
}

Tensor::~Tensor()
{
}

bool Tensor::Init(vk::ComponentTypeKHR dataType,
    rad::ArrayRef<size_t> sizes, rad::ArrayRef<size_t> strides,
    rad::Ref<Buffer> buffer, VkDeviceSize bufferOffset)
{
    assert(sizes.size() > 0);
    assert(sizes.size() <= MaxDimensionCount);
    assert((sizes.size() == strides.size()) || strides.empty());

    m_dataType = dataType;
    m_sizes = sizes;
    m_strides = strides;
    if (m_strides.empty())
    {
        m_strides = MakeStrides(m_sizes);
    }
    assert(m_sizes.size() == m_strides.size());

    size_t elementCount = GetElementCount();
    size_t indexOfLastElement = 0;
    for (size_t i = 0; i < m_sizes.size(); ++i)
    {
        indexOfLastElement += (m_sizes[i] - 1) * m_strides[i];
    }
    if (indexOfLastElement + 1 == elementCount)
    {
        m_isContiguous = true;
    }

    m_bufferSize = VkDeviceSize(indexOfLastElement + 1) * GetElementSizeInBytes();
    m_bufferSize = rad::Pow2AlignUp(m_bufferSize, VkDeviceSize(4));

    if (buffer)
    {
        m_buffer = buffer;
        m_bufferOffset = bufferOffset;
        assert(m_bufferSize < m_buffer->GetSize() - m_bufferOffset);
    }
    else
    {
        m_buffer = Buffer::CreateStorage(m_device, m_bufferSize);
        m_bufferOffset = 0;
    }

    m_cmdStream = m_device->CreateCommandStream(QueueFamily::Universal);
    rad::Ref<CommandBuffer> cmdBuffer = m_cmdStream->m_cmdPoolTransientAlloc->AllocatePrimary();
    cmdBuffer->Begin();
    cmdBuffer->FillBuffer(m_buffer->m_handle, m_bufferOffset, m_buffer->GetSize(), 0);
    cmdBuffer->End();
    m_cmdStream->SubmitAndWaitForCompletion(cmdBuffer->GetHandle(), {}, {});

    return true;
}

std::vector<size_t> Tensor::MakeStrides(rad::ArrayRef<size_t> sizes)
{
    std::vector<size_t> strides(sizes.size(), 0);
    strides.back() = 1;
    std::partial_sum(
        sizes.rbegin(), sizes.rend() - 1, strides.rbegin() + 1, std::multiplies<std::size_t>());
    return strides;
}

std::vector<size_t> MakeStridesByMemoryOrder(rad::ArrayRef<size_t> sizes, rad::ArrayRef<size_t> memoryOrder)
{
    std::vector<size_t> strides(sizes.size());
    size_t stride = 1;
    for (size_t i = 0; i < sizes.size(); ++i)
    {
        strides[memoryOrder[i]] = stride;
        stride *= sizes[memoryOrder[i]];
    }
    return strides;
}

std::vector<size_t> Tensor::PadSizes(rad::ArrayRef<size_t> sizes)
{
    std::vector<size_t> sizesPadded(MaxDimensionCount, 1);
    for (size_t i = 0; i < sizes.size(); ++i)
    {
        sizesPadded[i + MaxDimensionCount - sizes.size()] = sizes[i];
    }
    return sizesPadded;
}

std::vector<size_t> Tensor::PadStrides(rad::ArrayRef<size_t> strides)
{
    size_t maxStride = *std::max_element(strides.begin(), strides.end());
    std::vector<size_t> stridesPadded(MaxDimensionCount, maxStride);
    for (size_t i = 0; i < strides.size(); ++i)
    {
        stridesPadded[i + MaxDimensionCount - strides.size()] = strides[i];
    }
    return stridesPadded;
}

VkDeviceSize Tensor::GetBufferSizeInBytes(vk::ComponentTypeKHR dataType, rad::ArrayRef<size_t> sizes, rad::ArrayRef<size_t> strides)
{
    size_t indexOfLastElement = 0;
    for (size_t i = 0; i < sizes.size(); ++i)
    {
        indexOfLastElement += (sizes[i] - 1) * strides[i];
    }
    VkDeviceSize bufferSize = VkDeviceSize(indexOfLastElement + 1) * GetComponentSizeInBytes(dataType);
    return rad::Pow2AlignUp<VkDeviceSize>(bufferSize, VkDeviceSize(4));
}

size_t Tensor::GetElementCount() const
{
    size_t count = m_sizes[0];
    for (size_t i = 1; i < m_sizes.size(); ++i)
    {
        count *= m_sizes[i];
    }
    return count;
}

bool Tensor::IsNCHW() const
{
    if ((m_sizes.size() == 4) &&
        (m_strides[0] > m_strides[1]) &&
        (m_strides[1] > m_strides[2]) &&
        (m_strides[2] > m_strides[3]) &&
        (m_strides[3] == 1))
    {
        return true;
    }
    return false;
}

bool Tensor::IsNHWC() const
{
    if ((m_sizes.size() == 4) &&
        (m_strides[0] > m_strides[2]) &&
        (m_strides[2] > m_strides[3]) &&
        (m_strides[3] > m_strides[1]) &&
        (m_strides[1] == 1))
    {
        return true;
    }
    return false;
}

bool Tensor::IsNCDHW() const
{
    if ((m_sizes.size() == 5) &&
        (m_strides[0] > m_strides[1]) &&
        (m_strides[1] > m_strides[2]) &&
        (m_strides[2] > m_strides[3]) &&
        (m_strides[3] > m_strides[4]) &&
        (m_strides[4] == 1))
    {
        return true;
    }
    return false;
}

bool Tensor::IsNDHWC() const
{
    if ((m_sizes.size() == 5) &&
        (m_strides[0] > m_strides[2]) &&
        (m_strides[2] > m_strides[3]) &&
        (m_strides[3] > m_strides[4]) &&
        (m_strides[4] > m_strides[1]) &&
        (m_strides[1] == 1))
    {
        return true;
    }
    return false;
}

void Tensor::Read(void* data, vk::DeviceSize offset, vk::DeviceSize dataSize)
{
    m_buffer->Read(data, m_bufferOffset + offset, dataSize);
}

void Tensor::Write(const void* data, vk::DeviceSize offset, vk::DeviceSize dataSize)
{
    m_buffer->Write(data, m_bufferOffset + offset, dataSize);
}

void Tensor::FillUniformDistribution(float minValue, float maxValue)
{
    assert(IsFloatingPointType(m_dataType));
    assert(minValue < maxValue);
    std::uniform_real_distribution<float> dist(minValue, maxValue);
    FillRandomFloat(dist);
}

void Tensor::FillUniformDistribution(int minValue, int maxValue)
{
    assert(IsSignedIntegerType(m_dataType) ||
        (IsUnsignedIntegerType(m_dataType) && (minValue >= 0)));
    assert(minValue < maxValue);
    std::uniform_int_distribution<int> dist(minValue, maxValue);
    FillRandomInteger(dist);
}

void Tensor::FillNormalDistribution(float mean, float stddev)
{
    assert(IsFloatingPointType(m_dataType));
    assert(stddev > 0.0f);
    std::normal_distribution<float> dist(mean, stddev);
    FillRandomFloat(dist);
}

} // namespace vkpp
