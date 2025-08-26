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

std::string Tensor::DumpText(TextFormat format)
{
    std::vector<uint8_t> bufferData(GetBufferSizeInBytes());
    Read(bufferData.data());
    std::string text;
    text.reserve(4 * 1024 * 1024); // Reserve 4MB for dump.
    text += std::format("# Sizes = {}\n", rad::ToString(m_sizes));
    text += std::format("# Strides = {}\n", rad::ToString(m_strides));
    TensorIterator iter(m_sizes);
    auto& indices = iter.m_indices;
    auto dump1D = [&]() {
        size_t dimCount = m_sizes.size();
        for (size_t i = 0; i < m_sizes[dimCount - 1]; ++i)
        {
            indices[dimCount - 1] = i;
            size_t index = std::inner_product(indices.begin(), indices.end(), m_strides.begin(), size_t(0));
            size_t offsetInBytes = index * GetElementSizeInBytes();
            if (offsetInBytes < bufferData.size())
            {
                if (format == TextFormat::Dec)
                {
                    text += FormatValueFixedWidthDec(m_dataType, &bufferData[offsetInBytes]) + ", ";
                }
                else if (format == TextFormat::Hex)
                {
                    text += FormatValueFixedWidthHex(m_dataType, &bufferData[offsetInBytes]) + ", ";
                }
            }
            else // Simulate robust buffer access:
            {
                uint64_t value = 0;
                if (format == TextFormat::Dec)
                {
                    text += FormatValueFixedWidthDec(m_dataType, &value) + ", ";
                }
                else if (format == TextFormat::Hex)
                {
                    text += FormatValueFixedWidthHex(m_dataType, &value) + ", ";
                }
            }
        }
        text.pop_back();
        text += "\n"; // New line after the last dimension
        };  // dump1D

    if (m_sizes.size() == 1)
    {
        dump1D();
    }
    else if (m_sizes.size() == 2)
    {
        do {
            iter.Reset1D();
            dump1D();
        } while (iter.Next1D());
    }
    else // >2D
    {
        do {
            iter.Reset2D();
            text += std::format("# Indices = {}\n", rad::ToString(indices));
            for (size_t row = 0; row < m_sizes[m_sizes.size() - 2]; ++row)
            {
                iter.m_indices[m_sizes.size() - 2] = row;
                dump1D();
            }
        } while (iter.Next2D());
    }
    return text;
}

bool Tensor::DumpTextToFile(std::string_view fileName, TextFormat format)
{
    rad::File file;
    if (file.Open(fileName, "w"))
    {
        std::string text = DumpText(format);
        if (text.size() > 0)
        {
            file.Write(text.data(), text.size());
            return true;
        }
    }
    return false;
}

} // namespace vkpp
