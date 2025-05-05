#include <vkpp/Compute/Tensor.h>
#include <vkpp/Core/Command.h>
#include <rad/Core/Float.h>
#include <numeric>
#include <random>

namespace vkpp
{

Tensor::Tensor(rad::Ref<Device> device) :
    m_device(std::move(device))
{
}

Tensor::~Tensor()
{
}

bool Tensor::Init(vk::ComponentTypeKHR dataType, rad::ArrayRef<size_t> sizes, MemoryLayout layout, rad::ArrayRef<size_t> strides,
    rad::Ref<Buffer> buffer, VkDeviceSize bufferOffset)
{
    assert((sizes.size() == 4) || (sizes.size() == 5));

    m_dataType = dataType;
    m_sizes = sizes;
    m_layout = layout;
    m_strides = strides;
    if (m_strides.empty())
    {
        m_strides = GetContiguousStrides(m_sizes, m_layout);
    }
    assert(m_sizes.size() == m_strides.size());
#if defined(_DEBUG) // check strides
    std::vector<size_t> memoryOrder = GetMemoryOrder(layout);
    assert(m_strides.size() == memoryOrder.size());
    size_t stride = m_strides[memoryOrder[0]];
    assert(stride >= 1);
    for (size_t i = 1; i < memoryOrder.size(); ++i)
    {
        assert(stride <= m_strides[memoryOrder[i]]);
        stride = m_strides[memoryOrder[i]];
    }
#endif

    size_t elementCount = GetElementCount();
    size_t indexOfLastElement = 0;
    for (size_t i = 0; i < m_sizes.size(); ++i)
    {
        if (m_sizes[i] > 1)
        {
            indexOfLastElement += (m_sizes[i] - 1) * m_strides[i];
        }
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

    vk::raii::CommandBuffer cmdBuffer = m_device->AllocateTemporaryCommandBuffer(QueueFamily::Universal);
    CommandRecorder(cmdBuffer).Begin();
    cmdBuffer.fillBuffer(m_buffer->m_handle, m_bufferOffset, m_buffer->GetSize(), 0);
    CommandRecorder(cmdBuffer).End();
    m_device->ExecuteSync({}, { cmdBuffer }, {});

    return true;
}

std::vector<size_t> GetContiguousStridesByMemoryOrder(rad::ArrayRef<size_t> sizes, rad::ArrayRef<size_t> memoryOrder)
{
    if (sizes.empty())
    {
        return {};
    }
    std::vector<size_t> strides(sizes.size());
    size_t stride = 1;
    for (size_t i = 0; i < sizes.size(); ++i)
    {
        strides[memoryOrder[i]] = stride;
        stride *= sizes[memoryOrder[i]];
    }
    return strides;
}

std::vector<size_t> Tensor::GetContiguousStrides(rad::ArrayRef<size_t> sizes, MemoryLayout layout)
{
    assert(layout != MemoryLayout::Undefined);
    if (sizes.empty())
    {
        return {};
    }
    std::vector<size_t> strides;
    if (sizes.size() == 4)
    {
        assert((layout == MemoryLayout::NCHW) || (layout == MemoryLayout::NHWC));
        if (layout == MemoryLayout::NCHW)
        {
            return GetContiguousStridesByMemoryOrder(sizes, { 3, 2, 1, 0 });
        }
        else if (layout == MemoryLayout::NHWC)
        {
            return GetContiguousStridesByMemoryOrder(sizes, { 1, 3, 2, 0 });
        }
    }
    else if (sizes.size() == 5)
    {
        assert((layout == MemoryLayout::NCDHW) || (layout == MemoryLayout::NDHWC));
        if (layout == MemoryLayout::NCDHW)
        {
            return GetContiguousStridesByMemoryOrder(sizes, { 4, 3, 2, 1, 0 });
        }
        else if (layout == MemoryLayout::NDHWC)
        {
            return GetContiguousStridesByMemoryOrder(sizes, { 1, 4, 3, 2, 0 });
        }
    }
    return {};
}

std::vector<size_t> Tensor::GetMemoryOrder(MemoryLayout layout)
{
    assert(layout != MemoryLayout::Undefined);
    std::vector<size_t> memoryOrder;
    if (layout == MemoryLayout::NCHW)
    {
        memoryOrder = { 3, 2, 1, 0 };
    }
    else if (layout == MemoryLayout::NHWC)
    {
        memoryOrder = { 1, 3, 2, 0 };
    }
    else if (layout == MemoryLayout::NCDHW)
    {
        memoryOrder = { 4, 3, 2, 1, 0 };
    }
    else if (layout == MemoryLayout::NDHWC)
    {
        memoryOrder = { 1, 4, 3, 2, 0 };
    }
    return memoryOrder;
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

size_t Tensor::GetBufferElementCount() const
{
    return static_cast<size_t>(m_bufferSize / GetElementSizeInBytes());
}

void Tensor::Read(void* data, vk::DeviceSize offset, vk::DeviceSize dataSize)
{
    m_buffer->Read(data, m_bufferOffset + offset, dataSize);
}

void Tensor::Write(const void* data, vk::DeviceSize offset, vk::DeviceSize dataSize)
{
    m_buffer->Write(data, m_bufferOffset + offset, dataSize);
}

void Tensor::FillRandom(float minValue, float maxValue)
{
    assert(IsFloatingPointType(m_dataType));
    assert(minValue < maxValue);
    std::random_device rd;
    std::default_random_engine eng(rd());
    std::uniform_real_distribution<float> dist(minValue, maxValue);
    if (m_dataType == vk::ComponentTypeKHR::eFloat16)
    {
        std::vector<uint16_t> bufferData = GenerateBufferData<uint16_t>(
            [&](size_t index, std::initializer_list<size_t> coord) { return rad::fp16_ieee_from_fp32_value(dist(eng)); });
        m_buffer->Write(bufferData.data(), m_bufferOffset, m_bufferSize);
    }
    else if (m_dataType == vk::ComponentTypeKHR::eFloat32)
    {
        std::vector<float> bufferData = GenerateBufferData<float>(
            [&](size_t index, std::initializer_list<size_t> coord) { return dist(eng); });
        m_buffer->Write(bufferData.data(), m_bufferOffset, m_bufferSize);
    }
    else if (m_dataType == vk::ComponentTypeKHR::eFloat64)
    {
        std::uniform_real_distribution<double> dist64(minValue, maxValue);
        std::vector<double> bufferData = GenerateBufferData<double>(
            [&](size_t index, std::initializer_list<size_t> coord) { return dist64(eng); });
        m_buffer->Write(bufferData.data(), m_bufferOffset, m_bufferSize);
    }
    else if (m_dataType == vk::ComponentTypeKHR::eFloatE4M3NV)
    {
        std::vector<uint8_t> bufferData = GenerateBufferData<uint8_t>(
            [&](size_t index, std::initializer_list<size_t> coord) { return rad::fp8e4m3fn_from_fp32_value(dist(eng)); });
        m_buffer->Write(bufferData.data(), m_bufferOffset, m_bufferSize);
    }
    else if (m_dataType == vk::ComponentTypeKHR::eFloatE5M2NV)
    {
        std::vector<uint8_t> bufferData = GenerateBufferData<uint8_t>(
            [&](size_t index, std::initializer_list<size_t> coord) { return rad::fp8e5m2_from_fp32_value(dist(eng)); });
        m_buffer->Write(bufferData.data(), m_bufferOffset, m_bufferSize);
    }
}

void Tensor::FillRandom(int minValue, int maxValue)
{
    assert(IsIntegerType(m_dataType));
    assert(minValue < maxValue);
    std::random_device rd;
    std::default_random_engine eng(rd());
    std::uniform_int_distribution<int> dist(minValue, maxValue);
    if (m_dataType == vk::ComponentTypeKHR::eSint8)
    {
        std::vector<int8_t> bufferData = GenerateBufferData<int8_t>(
            [&](size_t index, std::initializer_list<size_t> coord) { return int8_t(dist(eng)); });
        m_buffer->Write(bufferData.data(), m_bufferOffset, m_bufferSize);
    }
    else if (m_dataType == vk::ComponentTypeKHR::eSint16)
    {
        std::vector<int16_t> bufferData = GenerateBufferData<int16_t>(
            [&](size_t index, std::initializer_list<size_t> coord) { return int16_t(dist(eng)); });
        m_buffer->Write(bufferData.data(), m_bufferOffset, m_bufferSize);
    }
    else if (m_dataType == vk::ComponentTypeKHR::eSint32)
    {
        std::vector<int32_t> bufferData = GenerateBufferData<int32_t>(
            [&](size_t index, std::initializer_list<size_t> coord) { return int32_t(dist(eng)); });
        m_buffer->Write(bufferData.data(), m_bufferOffset, m_bufferSize);
    }
    else if (m_dataType == vk::ComponentTypeKHR::eSint64)
    {
        std::vector<int64_t> bufferData = GenerateBufferData<int64_t>(
            [&](size_t index, std::initializer_list<size_t> coord) { return int64_t(dist(eng)); });
        m_buffer->Write(bufferData.data(), m_bufferOffset, m_bufferSize);
    }
    else if (m_dataType == vk::ComponentTypeKHR::eUint8)
    {
        assert(minValue >= 0);
        assert(maxValue > 0);
        std::vector<uint8_t> bufferData = GenerateBufferData<uint8_t>(
            [&](size_t index, std::initializer_list<size_t> coord) { return uint8_t(dist(eng)); });
        m_buffer->Write(bufferData.data(), m_bufferOffset, m_bufferSize);
    }
    else if (m_dataType == vk::ComponentTypeKHR::eUint16)
    {
        assert(minValue >= 0);
        assert(maxValue > 0);
        std::vector<uint16_t> bufferData = GenerateBufferData<uint16_t>(
            [&](size_t index, std::initializer_list<size_t> coord) { return uint16_t(dist(eng)); });
        m_buffer->Write(bufferData.data(), m_bufferOffset, m_bufferSize);
    }
    else if (m_dataType == vk::ComponentTypeKHR::eUint32)
    {
        assert(minValue >= 0);
        assert(maxValue > 0);
        std::vector<uint32_t> bufferData = GenerateBufferData<uint32_t>(
            [&](size_t index, std::initializer_list<size_t> coord) { return uint32_t(dist(eng)); });
        m_buffer->Write(bufferData.data(), m_bufferOffset, m_bufferSize);
    }
    else if (m_dataType == vk::ComponentTypeKHR::eUint64)
    {
        assert(minValue >= 0);
        assert(maxValue > 0);
        std::vector<uint64_t> bufferData = GenerateBufferData<uint64_t>(
            [&](size_t index, std::initializer_list<size_t> coord) { return uint64_t(dist(eng)); });
        m_buffer->Write(bufferData.data(), m_bufferOffset, m_bufferSize);
    }
}

} // namespace vkpp
