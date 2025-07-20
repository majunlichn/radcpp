#include <vkpp/Compute/Tensor.h>
#include <vkpp/Core/Command.h>
#include <rad/IO/Format.h>

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

    FillZeros();

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

std::vector<size_t> Tensor::ExpandSizeDimensions(rad::ArrayRef<size_t> sizes, size_t dimCount)
{
    if (dimCount > sizes.size())
    {
        std::vector<size_t> sizesExpanded(dimCount, 1);
        for (size_t i = 0; i < sizes.size(); ++i)
        {
            sizesExpanded[i + dimCount - sizes.size()] = sizes[i];
        }
        return sizesExpanded;
    }
    else
    {
        return sizes;
    }
}

std::vector<size_t> Tensor::ExpandStrideDimensions(rad::ArrayRef<size_t> strides, size_t dimCount)
{
    if (dimCount > strides.size())
    {
        size_t maxStride = *std::max_element(strides.begin(), strides.end());
        std::vector<size_t> stridesExpanded(dimCount, maxStride);
        for (size_t i = 0; i < strides.size(); ++i)
        {
            stridesExpanded[i + dimCount - strides.size()] = strides[i];
        }
        return stridesExpanded;
    }
    else
    {
        return strides;
    }
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

void Tensor::FillZeros()
{
    m_cmdStream = m_device->CreateCommandStream(QueueFamily::Universal);
    rad::Ref<CommandBuffer> cmdBuffer = m_cmdStream->m_cmdPoolTransientAlloc->AllocatePrimary();
    cmdBuffer->Begin();
    cmdBuffer->FillBuffer(m_buffer->m_handle, m_bufferOffset, m_buffer->GetSize(), 0);
    cmdBuffer->SetMemoryBarrier(
        vk::PipelineStageFlagBits2::eTransfer,
        vk::AccessFlagBits2::eTransferWrite,
        vk::PipelineStageFlagBits2::eAllCommands,
        vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eMemoryRead
    );
    cmdBuffer->End();
    m_cmdStream->SubmitAndWaitForCompletion(cmdBuffer->GetHandle(), {}, {});
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

void Tensor::DumpText(std::string_view fileName, TextFormat format)
{
    rad::File file;
    if (file.Open(fileName, "w"))
    {
        std::vector<uint8_t> bufferData(GetBufferSizeInBytes());
        Read(bufferData.data());
        std::string text;
        text.reserve(4 * 1024 * 1024); // reserve 4MB for formatted text.
        std::vector<size_t> coord(m_sizes.size(), 0);
        text += std::format("# Sizes = {}\n", rad::ToString(m_sizes));
        text += std::format("# Strides = {}\n", rad::ToString(m_strides));
        DumpText(bufferData, text, format, 0, coord);
        file.Write(text.data(), text.size());
        file.Close();
    }
}

static std::string DumpElementDec(vk::ComponentTypeKHR dataType, const void* data)
{
    if (dataType == vk::ComponentTypeKHR::eFloat16)
    {
        uint16_t value = *reinterpret_cast<const uint16_t*>(data);
        return std::format("{:11.4f}", rad::fp16_ieee_to_fp32_value(value));
    }
    else if (dataType == vk::ComponentTypeKHR::eFloat32)
    {
        float value = *reinterpret_cast<const float*>(data);
        if (value < 1000000)
        {
            return std::format("{:14.6f}", value);
        }
        else
        {
            return std::format("{:14.6e}", value);
        }
    }
    else if (dataType == vk::ComponentTypeKHR::eFloat64)
    {
        double value = *reinterpret_cast<const double*>(data);
        if (value < 1000000)
        {
            return std::format("{:14.6f}", value);
        }
        else
        {
            return std::format("{:14.6e}", value);
        }
    }
    else if (dataType == vk::ComponentTypeKHR::eSint8)
    {
        int8_t value = *reinterpret_cast<const int8_t*>(data);
        return std::format("{:4d}", value);
    }
    else if (dataType == vk::ComponentTypeKHR::eSint16)
    {
        int16_t value = *reinterpret_cast<const int16_t*>(data);
        return std::format("{:6d}", value);
    }
    else if (dataType == vk::ComponentTypeKHR::eSint32)
    {
        int32_t value = *reinterpret_cast<const int32_t*>(data);
        return std::format("{:11d}", value);
    }
    else if (dataType == vk::ComponentTypeKHR::eSint64)
    {
        int64_t value = *reinterpret_cast<const int64_t*>(data);
        return std::format("{:20d}", value);
    }
    else if (dataType == vk::ComponentTypeKHR::eUint8)
    {
        uint8_t value = *reinterpret_cast<const uint8_t*>(data);
        return std::format("{:4d}", value);
    }
    else if (dataType == vk::ComponentTypeKHR::eUint16)
    {
        uint16_t value = *reinterpret_cast<const uint16_t*>(data);
        return std::format("{:5d}", value);
    }
    else if (dataType == vk::ComponentTypeKHR::eUint32)
    {
        uint32_t value = *reinterpret_cast<const uint32_t*>(data);
        return std::format("{:10d}", value);
    }
    else if (dataType == vk::ComponentTypeKHR::eUint64)
    {
        uint64_t value = *reinterpret_cast<const uint64_t*>(data);
        return std::format("{:20d}", value);
    }
    else
    {
        RAD_UNREACHABLE();
        return {};
    }
}

static std::string DumpElementHex(vk::ComponentTypeKHR dataType, const void* data)
{
    size_t elementSize = GetComponentSizeInBytes(dataType);
    if (elementSize == 1)
    {
        uint8_t value = *reinterpret_cast<const uint8_t*>(data);
        return std::format("0x{:02X}", value);
    }
    else if (elementSize == 2)
    {
        uint16_t value = *reinterpret_cast<const uint16_t*>(data);
        return std::format("0x{:04X}", value);
    }
    else if (elementSize == 4)
    {
        uint32_t value = *reinterpret_cast<const uint32_t*>(data);
        return std::format("0x{:08X}", value);
    }
    else if (elementSize == 8)
    {
        uint64_t value = *reinterpret_cast<const uint64_t*>(data);
        return std::format("0x{:016X}", value);
    }
    else
    {
        RAD_UNREACHABLE();
        return {};
    }
}

void Tensor::DumpText(std::vector<uint8_t>& bufferData, std::string& text, TextFormat format, size_t dimIndex, std::vector<size_t>& coord)
{
    if (dimIndex == m_sizes.size() - 1)
    {
        // Iterate the last dimension:
        for (size_t i = 0; i < m_sizes[dimIndex]; ++i)
        {
            coord[dimIndex] = i;
            size_t index = std::inner_product(coord.begin(), coord.end(), m_strides.begin(), size_t(0));
            if (format == TextFormat::Dec)
            {
                text += DumpElementDec(m_dataType, &bufferData[index * GetElementSizeInBytes()]) + ", ";
            }
            else if (format == TextFormat::Hex)
            {
                text += DumpElementHex(m_dataType, &bufferData[index * GetElementSizeInBytes()]) + ", ";
            }
        }
        text.pop_back();
        text += "\n"; // New line after the last dimension
    }
    else
    {
        if (dimIndex == m_sizes.size() - 2)
        {
            for (size_t i = dimIndex; i < m_sizes.size(); ++i)
            {
                coord[i] = 0;
            }
            text += std::format("# Offset = {}\n", rad::ToString(coord));
        }
        // Iterate recursively:
        for (size_t i = 0; i < m_sizes[dimIndex]; ++i)
        {
            coord[dimIndex] = i;
            DumpText(bufferData, text, format, dimIndex + 1, coord);
        }
    }
}

} // namespace vkpp
