#pragma once

#include <vkpp/Core/Device.h>
#include <vkpp/Core/Buffer.h>
#include <functional>

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

    static constexpr size_t MaxDimensionCount = 8;
    static std::vector<size_t> MakeStrides(rad::ArrayRef<size_t> sizes);
    static std::vector<size_t> MakeStridesByMemoryOrder(rad::ArrayRef<size_t> sizes, rad::ArrayRef<size_t> memoryOrder);
    // Pad sizes to MaxDimensionCount.
    static std::vector<size_t> PadSizes(rad::ArrayRef<size_t> sizes);
    // Pad strides to MaxDimensionCount.
    static std::vector<size_t> PadStrides(rad::ArrayRef<size_t> sizes, rad::ArrayRef<size_t> strides);

    static VkDeviceSize GetBufferSizeInBytes(
        vk::ComponentTypeKHR dataType, rad::ArrayRef<size_t> sizes, rad::ArrayRef<size_t> strides = {});

    // Sizes padded to MaxDimensionCount.
    std::vector<size_t> GetPaddedSizes(rad::ArrayRef<size_t> sizes)
    {
        return PadSizes(sizes);
    }
    // Strides padded to MaxDimensionCount.
    std::vector<size_t> GetPaddedStrides(rad::ArrayRef<size_t> sizes, rad::ArrayRef<size_t> strides)
    {
        return PadStrides(sizes, strides);
    }

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
    std::vector<T> GenerateData(std::function<T(std::initializer_list<size_t> coord)> generator) const;

    template <rad::TriviallyCopyable T>
    void FillConstant(const T& value);

    void FillRandom(float minValue, float maxValue);
    void FillRandom(int minValue, int maxValue);

}; // class Tensor

template<rad::TriviallyCopyable T>
inline std::vector<T> Tensor::GenerateData(std::function<T(std::initializer_list<size_t> coord)> generator) const
{
    assert(sizeof(T) == GetElementSizeInBytes());
    std::vector<T> buffer(GetBufferSizeInElements(), T(0));
    std::vector<size_t> sizePadded = PadSizes(m_sizes);
    std::vector<size_t> stridePadded = PadStrides(m_sizes, m_strides);

    static_assert(MaxDimensionCount == 8);
    assert(sizePadded.size() == MaxDimensionCount);
    assert(stridePadded.size() == MaxDimensionCount);

    for (size_t c0 = 0; c0 < sizePadded[0]; ++c0)
    {
        for (size_t c1 = 0; c1 < sizePadded[1]; ++c1)
        {
            for (size_t c2 = 0; c2 < sizePadded[2]; ++c2)
            {
                for (size_t c3 = 0; c3 < sizePadded[3]; ++c3)
                {
                    for (size_t c4 = 0; c4 < sizePadded[4]; ++c4)
                    {
                        for (size_t c5 = 0; c5 < sizePadded[5]; ++c5)
                        {
                            for (size_t c6 = 0; c6 < sizePadded[6]; ++c6)
                            {
                                for (size_t c7 = 0; c7 < sizePadded[7]; ++c7)
                                {
                                    size_t index =
                                        c0 * stridePadded[0] + c1 * stridePadded[1] + c2 * stridePadded[2] + c3 * stridePadded[3] +
                                        c4 * stridePadded[4] + c5 * stridePadded[5] + c6 * stridePadded[6] + c7 * stridePadded[7];
                                    buffer[index] = generator({ c0, c1, c2, c3, c4, c5, c6, c7 });
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return buffer;
}

template<rad::TriviallyCopyable T>
inline void Tensor::FillConstant(const T& value)
{
    assert(sizeof(T) == GetElementSizeInBytes());
    std::vector<T> bufferData = GenerateData<T>(
        [&](std::initializer_list<size_t> coord) { return value; });
    Write(bufferData.data());
}

} // namespace vkpp
