#pragma once

#include <vkpp/Core/Device.h>
#include <vkpp/Core/Buffer.h>
#include <functional>
#include <numeric>
#include <random>

#include <taskflow/taskflow.hpp>
#include <rad/System/Time.h>

namespace vkpp
{

class Tensor : public rad::RefCounted<Tensor>
{
public:
    Tensor(rad::Ref<Device> device);
    ~Tensor();

    bool Init(vk::ComponentTypeKHR dataType, rad::ArrayRef<size_t> sizes, rad::ArrayRef<size_t> strides = {},
        rad::Ref<Buffer> buffer = nullptr, VkDeviceSize bufferOffset = 0);

    vk::ComponentTypeKHR GetDataType() const { return m_dataType; }
    VkDeviceSize GetElementSizeInBytes() const { return GetComponentSizeInBytes(m_dataType); }
    bool IsFloatingPoint() const { return IsFloatingPointType(m_dataType); }
    bool IsSignedInteger() const { return IsSignedIntegerType(m_dataType); }
    bool IsUnsignedInteger() const { return IsUnsignedIntegerType(m_dataType); }
    bool IsInteger() const { return IsIntegerType(m_dataType); }

    static std::vector<size_t> MakeStrides(rad::ArrayRef<size_t> sizes);
    static std::vector<size_t> MakeStridesByMemoryOrder(rad::ArrayRef<size_t> sizes, rad::ArrayRef<size_t> memoryOrder);
    // Expand sizes to higher dimensions (same element count).
    static std::vector<size_t> ExpandSizeDimensions(rad::ArrayRef<size_t> sizes, size_t dimCount);
    // Expand strides to higher dimensions (same memory layout).
    static std::vector<size_t> ExpandStrideDimensions(rad::ArrayRef<size_t> strides, size_t dimCount);

    static VkDeviceSize GetBufferSizeInBytes(
        vk::ComponentTypeKHR dataType, rad::ArrayRef<size_t> sizes, rad::ArrayRef<size_t> strides = {});

    size_t GetDimensionCount() const { return m_sizes.size(); }
    size_t GetElementCount() const;

    VkDeviceSize GetBufferSizeInBytes() const { return m_bufferSize; }
    size_t GetBufferSizeInElements() const { return (m_bufferSize / GetElementSizeInBytes()); }
    bool IsContiguous() const { return m_isContiguous; }
    bool IsNCHW() const;
    bool IsNHWC() const;
    bool IsNCDHW() const;
    bool IsNDHWC() const;

    rad::Ref<Device> m_device;

    vk::ComponentTypeKHR m_dataType = vk::ComponentTypeKHR::eFloat16;
    // The total size of a tensor may not exceed (2^32 - 1) elements -
    // for example, 16GB for a Float32 tensor.
    std::vector<size_t> m_sizes;
    std::vector<size_t> m_strides;
    bool m_isContiguous = false;
    // The buffer allocated in device memory, can be shared by other tensors.
    rad::Ref<Buffer> m_buffer;
    // The base offset of the buffer range in bytes.
    // TODO: guarantee a minimum alignment in bytes for the base offset of the buffer range.
    VkDeviceSize m_bufferOffset = 0;
    // The buffer size required, must be rounded up to the nearest 4-byte boundary.
    VkDeviceSize m_bufferSize = 0;

    vk::PipelineStageFlags2 m_currentPipelineStage = vk::PipelineStageFlagBits2::eNone;
    vk::AccessFlags2 m_currentAccessFlags = vk::AccessFlagBits2::eNone;

    rad::Ref<CommandStream> m_cmdStream;

    void Read(void* data, vk::DeviceSize offset, vk::DeviceSize dataSize);
    void Read(void* data) { Read(data, 0, m_bufferSize); }
    void Write(const void* data, vk::DeviceSize offset, vk::DeviceSize dataSize);
    void Write(const void* data) { Write(data, 0, m_bufferSize); }

    template <rad::TriviallyCopyable T>
    std::vector<T> GenerateData(const std::function<T(rad::ArrayRef<size_t> indices)>& generator) const;
    template <rad::TriviallyCopyable T>
    void GenerateDataDimByDim(std::vector<T>& bufferData,
        const std::function<T(rad::ArrayRef<size_t> indices)>& generator,
        std::vector<size_t>& indices, size_t dimIndex) const;

    void FillZeros();

    template <rad::TriviallyCopyable T>
    void FillConstant(const T& value);

    template<typename Distribution>
    void FillRandomFloat(Distribution& dist);
    template<typename Distribution>
    void FillRandomInteger(Distribution& dist);

    void FillUniformDistribution(float minValue, float maxValue);
    void FillUniformDistribution(int minValue, int maxValue);
    void FillNormalDistribution(float mean = 0.0f, float stddev = 1.0f);

    enum class TextFormat
    {
        Dec,
        Hex,
    };

    std::string DumpText(TextFormat format = TextFormat::Dec);
    void DumpTextDimByDim(std::string& text, std::vector<uint8_t>& data, TextFormat format,
        std::vector<size_t>& indices, size_t dimIndex);

    bool DumpTextToFile(std::string_view fileName, TextFormat format = TextFormat::Dec);

}; // class Tensor

class TensorIterator
{
public:
    std::vector<size_t> m_sizes;

    TensorIterator(rad::ArrayRef<size_t> sizes) :
        m_sizes(sizes)
    {
    }

    ~TensorIterator() = default;

    void ForEach(const std::function<void(rad::ArrayRef<size_t> indices)>& operation)
    {
        size_t dimCount = m_sizes.size();
        std::vector<size_t> indices(m_sizes.size(), 0);
        while (true)
        {
            // Iterate the last dimension:
            for (size_t i = 0; i < m_sizes[dimCount - 1]; ++i)
            {
                indices[dimCount - 1] = i;
                operation(indices);
            }
            bool carryFlag = false;
            for (int dimIndex = int(dimCount - 2); dimIndex >= 0; --dimIndex)
            {
                if (indices[dimIndex] < m_sizes[dimIndex] - 1)
                {
                    ++indices[dimIndex];
                    carryFlag = true;
                    break;
                }
                else
                {
                    indices[dimIndex] = 0;
                }
            }
            if (!carryFlag)
            {
                break;
            }
        }
    }

}; // class TensorIterator

template<rad::TriviallyCopyable T>
inline std::vector<T> Tensor::GenerateData(const std::function<T(rad::ArrayRef<size_t> indices)>& generator) const
{
    assert(sizeof(T) == GetElementSizeInBytes());
    std::vector<T> bufferData(GetBufferSizeInElements(), T(0));
    std::vector<size_t> indices(m_sizes.size(), 0);
    GenerateDataDimByDim(bufferData, generator, indices, 0);
    return bufferData;
}

template<rad::TriviallyCopyable T>
inline void Tensor::GenerateDataDimByDim(std::vector<T>& bufferData,
    const std::function<T(rad::ArrayRef<size_t> indices)>& generator,
    std::vector<size_t>& indices, size_t dimIndex) const
{
    if (dimIndex == m_sizes.size() - 1)
    {
        // Iterate the last dimension:
        for (size_t i = 0; i < m_sizes[dimIndex]; ++i)
        {
            indices[dimIndex] = i;
            size_t bufferIndex = std::inner_product(indices.begin(), indices.end(), m_strides.begin(), size_t(0));
            bufferData[bufferIndex] = generator(indices);
        }
    }
    else
    {
        // Iterate recursively:
        for (size_t i = 0; i < m_sizes[dimIndex]; ++i)
        {
            indices[dimIndex] = i;
            GenerateDataDimByDim(bufferData, generator, indices, dimIndex + 1);
        }
    }
}

template<rad::TriviallyCopyable T>
inline void Tensor::FillConstant(const T& value)
{
    assert(sizeof(T) == GetElementSizeInBytes());
    if (IsContiguous())
    {
        std::vector<T> bufferData(GetBufferSizeInElements(), value);
        Write(bufferData.data());
    }
    else
    {
        std::vector<T> bufferData = GenerateData<T>(
            [&](rad::ArrayRef<size_t> indices) { return value; });
        Write(bufferData.data());
    }
}

template<typename Distribution>
inline void Tensor::FillRandomFloat(Distribution& dist)
{
    assert(IsFloatingPointType(m_dataType));

    std::random_device randomDevice;
    std::default_random_engine gen(randomDevice());

    if (m_dataType == vk::ComponentTypeKHR::eFloat16)
    {
        std::vector<uint16_t> bufferData = GenerateData<uint16_t>(
            [&](rad::ArrayRef<size_t> indices)
            { return rad::fp16_ieee_from_fp32_value(dist(gen)); }
        );
        Write(bufferData.data());
    }
    else if (m_dataType == vk::ComponentTypeKHR::eFloat32)
    {
        std::vector<float> bufferData = GenerateData<float>(
            [&](rad::ArrayRef<size_t> indices) { return dist(gen); });
        Write(bufferData.data());
    }
    else if (m_dataType == vk::ComponentTypeKHR::eFloat64)
    {
        std::vector<double> bufferData = GenerateData<double>(
            [&](rad::ArrayRef<size_t> indices) { return dist(gen); });
        Write(bufferData.data());
    }
    else if (m_dataType == vk::ComponentTypeKHR::eFloatE4M3NV)
    {
        std::vector<uint8_t> bufferData = GenerateData<uint8_t>(
            [&](rad::ArrayRef<size_t> indices)
            { return rad::fp8e4m3fn_from_fp32_value(dist(gen)); }
        );
        Write(bufferData.data());
    }
    else if (m_dataType == vk::ComponentTypeKHR::eFloatE5M2NV)
    {
        std::vector<uint8_t> bufferData = GenerateData<uint8_t>(
            [&](rad::ArrayRef<size_t> indices)
            { return rad::fp8e5m2_from_fp32_value(dist(gen)); }
        );
        Write(bufferData.data());
    }
}

template<typename Distribution>
inline void Tensor::FillRandomInteger(Distribution& dist)
{
    assert(IsIntegerType(m_dataType));

    std::random_device randomDevice;
    std::default_random_engine gen(randomDevice());

    if (m_dataType == vk::ComponentTypeKHR::eSint8)
    {
        std::vector<int8_t> bufferData = GenerateData<int8_t>(
            [&](rad::ArrayRef<size_t> indices) { return int8_t(dist(gen)); });
        Write(bufferData.data());
    }
    else if (m_dataType == vk::ComponentTypeKHR::eSint16)
    {
        std::vector<int16_t> bufferData = GenerateData<int16_t>(
            [&](rad::ArrayRef<size_t> indices) { return int16_t(dist(gen)); });
        Write(bufferData.data());
    }
    else if (m_dataType == vk::ComponentTypeKHR::eSint32)
    {
        std::vector<int32_t> bufferData = GenerateData<int32_t>(
            [&](rad::ArrayRef<size_t> indices) { return int32_t(dist(gen)); });
        Write(bufferData.data());
    }
    else if (m_dataType == vk::ComponentTypeKHR::eSint64)
    {
        std::vector<int64_t> bufferData = GenerateData<int64_t>(
            [&](rad::ArrayRef<size_t> indices) { return int64_t(dist(gen)); });
        Write(bufferData.data());
    }

    else if (m_dataType == vk::ComponentTypeKHR::eUint8)
    {
        std::vector<uint8_t> bufferData = GenerateData<uint8_t>(
            [&](rad::ArrayRef<size_t> indices) { return uint8_t(dist(gen)); });
        Write(bufferData.data());
    }
    else if (m_dataType == vk::ComponentTypeKHR::eUint16)
    {
        std::vector<uint16_t> bufferData = GenerateData<uint16_t>(
            [&](rad::ArrayRef<size_t> indices) { return uint16_t(dist(gen)); });
        Write(bufferData.data());
    }
    else if (m_dataType == vk::ComponentTypeKHR::eUint32)
    {
        std::vector<uint32_t> bufferData = GenerateData<uint32_t>(
            [&](rad::ArrayRef<size_t> indices) { return uint32_t(dist(gen)); });
        Write(bufferData.data());
    }
    else if (m_dataType == vk::ComponentTypeKHR::eUint64)
    {
        std::vector<uint64_t> bufferData = GenerateData<uint64_t>(
            [&](rad::ArrayRef<size_t> indices) { return uint64_t(dist(gen)); });
        Write(bufferData.data());
    }
}

} // namespace vkpp
