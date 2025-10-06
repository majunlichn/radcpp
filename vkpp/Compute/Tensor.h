#pragma once

#include <vkpp/Core/Device.h>
#include <vkpp/Core/Buffer.h>
#include <vkpp/Compute/TensorIterator.h>
#include <vkpp/Compute/HostTensor.h>

#include <taskflow/taskflow.hpp>

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

    static std::vector<size_t> MakeStrides(rad::ArrayRef<size_t> sizes,
        rad::ArrayRef<size_t> memoryOrder = {}, rad::ArrayRef<size_t> alignments = {});
    // Expand size dimensions (insert 1 at high dimensions, same element count).
    static std::vector<size_t> ExpandSizeND(rad::ArrayRef<size_t> sizes, size_t dimCount);
    // Expand stride dimensions (insert MaxStride at high dimensions, same memory layout).
    static std::vector<size_t> ExpandStrideND(rad::ArrayRef<size_t> strides, size_t dimCount);

    // Rounded up to the nearest 4-byte boundary (DWORD-aligned).
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

    static std::vector<size_t> GetMemoryOrder(rad::ArrayRef<size_t> strides);
    std::vector<size_t> GetMemoryOrder() const;

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
    std::vector<T> GenerateData(const std::function<T(rad::ArrayRef<size_t> coords)>& generator) const;

    template <rad::TriviallyCopyable T>
    void FillConstant(const T& value);

    void FillZeros();

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
    bool DumpTextToFile(std::string_view fileName, TextFormat format = TextFormat::Dec);

}; // class Tensor

template<rad::TriviallyCopyable T>
inline std::vector<T> Tensor::GenerateData(const std::function<T(rad::ArrayRef<size_t> coords)>& generator) const
{
    assert(sizeof(T) == GetElementSizeInBytes());
    std::vector<T> bufferData(GetBufferSizeInElements(), T(0));
    std::vector<size_t> coords(m_sizes.size(), 0);
    TensorIterator iter(m_sizes);
    iter.ForEachParallel([&](rad::ArrayRef<size_t> coords)
        {
            size_t bufferIndex = std::inner_product(coords.begin(), coords.end(), m_strides.begin(), size_t(0));
            bufferData[bufferIndex] = generator(coords);
        });
    return bufferData;
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
            [&](rad::ArrayRef<size_t> coords) { return value; });
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
            [&](rad::ArrayRef<size_t> coords)
            { return rad::fp16_ieee_from_fp32_value(dist(gen)); }
        );
        Write(bufferData.data());
    }
    else if (m_dataType == vk::ComponentTypeKHR::eFloat32)
    {
        std::vector<float> bufferData = GenerateData<float>(
            [&](rad::ArrayRef<size_t> coords) { return dist(gen); });
        Write(bufferData.data());
    }
    else if (m_dataType == vk::ComponentTypeKHR::eFloat64)
    {
        std::vector<double> bufferData = GenerateData<double>(
            [&](rad::ArrayRef<size_t> coords) { return dist(gen); });
        Write(bufferData.data());
    }
    else if (m_dataType == vk::ComponentTypeKHR::eFloatE4M3NV)
    {
        std::vector<uint8_t> bufferData = GenerateData<uint8_t>(
            [&](rad::ArrayRef<size_t> coords)
            { return rad::fp8e4m3fn_from_fp32_value(dist(gen)); }
        );
        Write(bufferData.data());
    }
    else if (m_dataType == vk::ComponentTypeKHR::eFloatE5M2NV)
    {
        std::vector<uint8_t> bufferData = GenerateData<uint8_t>(
            [&](rad::ArrayRef<size_t> coords)
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
            [&](rad::ArrayRef<size_t> coords) { return int8_t(dist(gen)); });
        Write(bufferData.data());
    }
    else if (m_dataType == vk::ComponentTypeKHR::eSint16)
    {
        std::vector<int16_t> bufferData = GenerateData<int16_t>(
            [&](rad::ArrayRef<size_t> coords) { return int16_t(dist(gen)); });
        Write(bufferData.data());
    }
    else if (m_dataType == vk::ComponentTypeKHR::eSint32)
    {
        std::vector<int32_t> bufferData = GenerateData<int32_t>(
            [&](rad::ArrayRef<size_t> coords) { return int32_t(dist(gen)); });
        Write(bufferData.data());
    }
    else if (m_dataType == vk::ComponentTypeKHR::eSint64)
    {
        std::vector<int64_t> bufferData = GenerateData<int64_t>(
            [&](rad::ArrayRef<size_t> coords) { return int64_t(dist(gen)); });
        Write(bufferData.data());
    }

    else if (m_dataType == vk::ComponentTypeKHR::eUint8)
    {
        std::vector<uint8_t> bufferData = GenerateData<uint8_t>(
            [&](rad::ArrayRef<size_t> coords) { return uint8_t(dist(gen)); });
        Write(bufferData.data());
    }
    else if (m_dataType == vk::ComponentTypeKHR::eUint16)
    {
        std::vector<uint16_t> bufferData = GenerateData<uint16_t>(
            [&](rad::ArrayRef<size_t> coords) { return uint16_t(dist(gen)); });
        Write(bufferData.data());
    }
    else if (m_dataType == vk::ComponentTypeKHR::eUint32)
    {
        std::vector<uint32_t> bufferData = GenerateData<uint32_t>(
            [&](rad::ArrayRef<size_t> coords) { return uint32_t(dist(gen)); });
        Write(bufferData.data());
    }
    else if (m_dataType == vk::ComponentTypeKHR::eUint64)
    {
        std::vector<uint64_t> bufferData = GenerateData<uint64_t>(
            [&](rad::ArrayRef<size_t> coords) { return uint64_t(dist(gen)); });
        Write(bufferData.data());
    }
}

} // namespace vkpp
