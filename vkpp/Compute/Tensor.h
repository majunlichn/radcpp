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
    size_t GetNumberOfDimensions() const { return m_sizes.size(); }
    size_t GetElementCount() const;
    bool IsContiguous() const { return m_isContiguous; }

    rad::Ref<Device> m_device;

    vk::ComponentTypeKHR m_dataType = vk::ComponentTypeKHR::eFloat16;
    std::vector<size_t> m_sizes;
    std::vector<size_t> m_strides;
    MemoryLayout m_layout = MemoryLayout::Undefined;
    bool m_isContiguous = false;
    rad::Ref<Buffer> m_buffer;
    VkDeviceSize m_bufferOffset = 0;
    VkDeviceSize m_bufferSize = 0;

    template <rad::TriviallyCopyable T>
    std::vector<T> GenerateBufferData(std::function<T(size_t n, size_t c, size_t d, size_t h, size_t w)> generator) const
    {
        assert(sizeof(T) == GetElementSizeInBytes());
        std::vector<T> buffer(m_bufferSize / GetElementSizeInBytes(), T(0));
        if (m_sizes.size() == 4)
        {
            for (size_t n = 0; n < m_sizes[0]; ++n)
            {
                for (size_t c = 0; c < m_sizes[1]; ++c)
                {
                    for (size_t h = 0; h < m_sizes[2]; ++h)
                    {
                        for (size_t w = 0; w < m_sizes[3]; ++w)
                        {
                            size_t index = n * m_strides[0] + c * m_strides[1] + h * m_strides[2] + w * m_strides[3];
                            buffer[index] = generator(n, c, 0, h, w);
                        }
                    }
                }
            }
        }
        else
        {
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
                                buffer[index] = generator(n, c, d, h, w);
                            }
                        }
                    }
                }
            }
        }
        return buffer;
    }

    template <rad::TriviallyCopyable T>
    void FillConstant(const T& value)
    {
        assert(sizeof(T) == GetElementSizeInBytes());
        std::vector<T> bufferData = GenerateBufferData<T>(
            [&](size_t n, size_t c, size_t d, size_t h, size_t w) { return value; });
        m_buffer->Write(bufferData.data(), m_bufferOffset, m_bufferSize);
    }

    void FillRandom(float minValue, float maxValue);
    void FillRandom(int minValue, int maxValue);

}; // class Tensor

} // namespace vkpp
