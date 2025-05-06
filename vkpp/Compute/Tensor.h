#pragma once

#include <vkpp/Core/Device.h>
#include <vkpp/Core/Buffer.h>
#include <functional>

namespace vkpp
{

class Tensor : public rad::RefCounted<Tensor>
{
public:
    enum class MemoryLayout
    {
        Undefined,
        NCHW,
        NHWC,
        NCDHW,
        NDHWC,
    };

    Tensor(rad::Ref<Device> device);
    ~Tensor();

    VkDeviceSize GetElementSizeInBytes() const { return GetComponentSizeInBytes(m_dataType); }

    bool Init(vk::ComponentTypeKHR dataType, rad::ArrayRef<size_t> sizes, MemoryLayout layout, rad::ArrayRef<size_t> strides = {},
        rad::Ref<Buffer> buffer = nullptr, VkDeviceSize bufferOffset = 0);

    static std::vector<size_t> GetContiguousStrides(rad::ArrayRef<size_t> sizes, MemoryLayout layout);
    // The dimension order in memory.
    static std::vector<size_t> GetMemoryOrder(MemoryLayout layout);

    bool IsFloatingPoint() const { return IsFloatingPointType(m_dataType); }
    bool IsSignedInteger() const { return IsSignedIntegerType(m_dataType); }
    bool IsUnsignedInteger() const { return IsUnsignedIntegerType(m_dataType); }
    bool IsInteger() const { return IsIntegerType(m_dataType); }

    size_t GetDimensionCount() const { return m_sizes.size(); }
    size_t GetElementCount() const;
    // Element count that can fill the entire buffer,
    // which is "m_bufferSize / GetElementSizeInBytes()";
    size_t GetBufferElementCount() const;
    bool IsContiguous() const { return m_isContiguous; }

    rad::Ref<Device> m_device;

    vk::ComponentTypeKHR m_dataType = vk::ComponentTypeKHR::eFloat16;
    std::vector<size_t> m_sizes;
    std::vector<size_t> m_strides;
    MemoryLayout m_layout = MemoryLayout::Undefined;
    bool m_isContiguous = false;
    // The buffer allocated in device memory, can be shared by other tensors.
    rad::Ref<Buffer> m_buffer;
    VkDeviceSize m_bufferOffset = 0;
    // The buffer size must be multiple of 4.
    VkDeviceSize m_bufferSize = 0;

    template <rad::TriviallyCopyable T>
    std::vector<T> GenerateBufferData(std::function<T(size_t index, std::initializer_list<size_t> coord)> generator) const;
    template <rad::TriviallyCopyable T>
    std::vector<T> GenerateBufferData4D(std::function<T(size_t index, std::initializer_list<size_t> coord)> generator) const;
    template <rad::TriviallyCopyable T>
    std::vector<T> GenerateBufferData5D(std::function<T(size_t index, std::initializer_list<size_t> coord)> generator) const;

    void Read(void* data, vk::DeviceSize offset, vk::DeviceSize dataSize);
    void Read(void* data) { Read(data, 0, m_bufferSize); }
    void Write(const void* data, vk::DeviceSize offset, vk::DeviceSize dataSize);
    void Write(const void* data) { Write(data, 0, m_bufferSize); }

    template <rad::TriviallyCopyable T>
    void FillConstant(const T& value);

    void FillRandom(float minValue, float maxValue);
    void FillRandom(int minValue, int maxValue);

}; // class Tensor

template<rad::TriviallyCopyable T>
inline std::vector<T> Tensor::GenerateBufferData(std::function<T(size_t index, std::initializer_list<size_t> coord)> generator) const
{
    assert(sizeof(T) == GetElementSizeInBytes());
    if (m_sizes.size() == 4)
    {
        return GenerateBufferData4D(generator);
    }
    else
    {
        return GenerateBufferData5D(generator);
    }
    return {};
}

template<rad::TriviallyCopyable T>
inline std::vector<T> Tensor::GenerateBufferData4D(std::function<T(size_t index, std::initializer_list<size_t> coord)> generator) const
{
    assert(sizeof(T) == GetElementSizeInBytes());
    std::vector<T> buffer(m_bufferSize / GetElementSizeInBytes(), T(0));
    for (size_t n = 0; n < m_sizes[0]; ++n)
    {
        for (size_t c = 0; c < m_sizes[1]; ++c)
        {
            for (size_t h = 0; h < m_sizes[2]; ++h)
            {
                for (size_t w = 0; w < m_sizes[3]; ++w)
                {
                    size_t index = n * m_strides[0] + c * m_strides[1] + h * m_strides[2] + w * m_strides[3];
                    buffer[index] = generator(index, { n, c, h, w });
                }
            }
        }
    }
    return buffer;
}

template<rad::TriviallyCopyable T>
inline std::vector<T> Tensor::GenerateBufferData5D(std::function<T(size_t index, std::initializer_list<size_t> coord)> generator) const
{
    assert(sizeof(T) == GetElementSizeInBytes());
    std::vector<T> buffer(m_bufferSize / GetElementSizeInBytes(), T(0));
    for (size_t n = 0; n < m_sizes[0]; ++n)
    {
        for (size_t c = 0; c < m_sizes[1]; ++c)
        {
            for (size_t d = 0; d < m_sizes[2]; ++d)
            {
                for (size_t h = 0; h < m_sizes[3]; ++h)
                {
                    for (size_t w = 0; w < m_sizes[4]; ++w)
                    {
                        size_t index = n * m_strides[0] + c * m_strides[1] +
                            d * m_strides[2] + h * m_strides[3] + w * m_strides[4];
                        buffer[index] = generator(index, { n, c, d, h, w });
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
    std::vector<T> bufferData = GenerateBufferData<T>(
        [&](size_t index, std::initializer_list<size_t> coord) { return value; });
    m_buffer->Write(bufferData.data(), m_bufferOffset, m_bufferSize);
}

} // namespace vkpp
