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
    std::vector<T> GenerateData(const std::function<T(rad::ArrayRef<size_t> indices)>& generator) const;

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
    bool DumpTextToFile(std::string_view fileName, TextFormat format = TextFormat::Dec);

}; // class Tensor

class TensorIterator
{
public:
    std::vector<size_t> m_sizes;
    std::vector<size_t> m_indices;
    std::vector<size_t> m_permutation;

    TensorIterator(rad::ArrayRef<size_t> sizes) :
        m_sizes(sizes)
    {
        Reset();
    }

    ~TensorIterator() = default;

    void Reset()
    {
        m_indices.clear();
        m_indices.resize(m_sizes.size(), 0);
    }

    void ResetND(size_t n)
    {
        size_t dimCount = m_sizes.size();
        assert(dimCount >= n);
        if (m_permutation.empty())
        {
            std::fill_n(m_indices.end() - n, n, 0);
        }
        else
        {
            for (size_t i = 0; i < n; ++i)
            {
                size_t dimIndex = m_permutation[i];
                m_indices[dimIndex] = 0;
            }
        }
    }

    void Reset1D() { ResetND(1); }
    void Reset2D() { ResetND(2); }
    void Reset3D() { ResetND(3); }
    void Reset4D() { ResetND(4); }

    bool NextND(size_t n)
    {
        size_t dimCount = m_sizes.size();
        assert(dimCount > n);
        if (m_permutation.empty())
        {
            for (ptrdiff_t dimIndex = dimCount - ptrdiff_t(n) - 1; dimIndex >= 0; --dimIndex)
            {
                if (m_indices[dimIndex] < m_sizes[dimIndex] - 1)
                {
                    ++m_indices[dimIndex];
                    return true;
                }
                else
                {
                    m_indices[dimIndex] = 0;
                }
            }
            return false;
        }
        else
        {
            for (size_t i = n; i < dimCount; ++i)
            {
                size_t dimIndex = m_permutation[i];
                if (m_indices[dimIndex] < m_sizes[dimIndex] - 1)
                {
                    ++m_indices[dimIndex];
                    return true;
                }
                else
                {
                    m_indices[dimIndex] = 0;
                }
            }
            return false;
        }
    }

    bool Next1D() { return NextND(1); }
    bool Next2D() { return NextND(2); }
    bool Next3D() { return NextND(3); }
    bool Next4D() { return NextND(4); }

    using ElementWiseOp = std::function<void(rad::ArrayRef<size_t> indices)>;

    void ForEach(const ElementWiseOp& op)
    {
        Reset();
        if (m_permutation.empty())
        {
            // Iterate the last dimension:
            size_t dimCount = m_sizes.size();
            do
            {
                for (size_t i = 0; i < m_sizes[dimCount - 1]; ++i)
                {
                    m_indices[dimCount - 1] = i;
                    op(m_indices);
                }
            } while ((dimCount > 1) && Next1D());
        }
        else
        {
            assert(m_permutation.size() == m_sizes.size());
            size_t dimCount = m_sizes.size();
            do
            {
                size_t dimIndexPermuted = m_permutation[0];
                for (size_t i = 0; i < m_sizes[dimIndexPermuted]; ++i)
                {
                    m_indices[dimIndexPermuted] = i;
                    op(m_indices);
                }
            } while ((dimCount > 1) && Next1D());
        }
    }

    void ForEachRecursively(const ElementWiseOp& op, size_t dimIndex)
    {
        if (m_permutation.empty())
        {
            if (dimIndex == m_sizes.size() - 1)
            {
                // Iterate the last dimension:
                for (size_t i = 0; i < m_sizes[dimIndex]; ++i)
                {
                    m_indices[dimIndex] = i;
                    op(m_indices);
                }
            }
            else
            {
                for (size_t i = 0; i < m_sizes[dimIndex]; ++i)
                {
                    m_indices[dimIndex] = i;
                    ForEachRecursively(op, dimIndex + 1);
                }
            }
        }
        else
        {
            size_t dimCount = m_sizes.size();
            size_t dimIndexPermuted = m_permutation[dimCount - dimIndex - 1];
            if (dimIndex == m_sizes.size() - 1)
            {
                // Iterate the last dimension:
                for (size_t i = 0; i < m_sizes[dimIndexPermuted]; ++i)
                {
                    m_indices[dimIndexPermuted] = i;
                    op(m_indices);
                }
            }
            else
            {
                for (size_t i = 0; i < m_sizes[dimIndexPermuted]; ++i)
                {
                    m_indices[dimIndexPermuted] = i;
                    ForEachRecursively(op, dimIndex + 1);
                }
            }
        }
    }

    void ForEachRecursively(const ElementWiseOp& op)
    {
        Reset();
        ForEachRecursively(op, 0);
    }

    // @param dimGranularity: the number of dimensions to process in parallel.
    void ForEachParallelND(const ElementWiseOp& op, size_t dimGranularity)
    {
        if (dimGranularity >= m_sizes.size())
        {
            return ForEach(op);
        }
        Reset();
        tf::Executor executor;
        do {
            ResetND(dimGranularity);
            executor.silent_async([&, iter = *this]() mutable {
                iter.ForEachRecursively(op, m_sizes.size() - dimGranularity);
                });
        } while (NextND(dimGranularity));
        executor.wait_for_all();
    }

    void ForEachParallel(const ElementWiseOp& op)
    {
        size_t elementCount = 1;
        size_t dimGranularity = 0;
        while (dimGranularity < m_sizes.size())
        {
            elementCount *= m_sizes[m_sizes.size() - dimGranularity - 1];
            ++dimGranularity;
            if (elementCount >= 1000000)
            {
                break;
            }
        }
        ForEachParallelND(op, dimGranularity);
    }

}; // class TensorIterator

template<rad::TriviallyCopyable T>
inline std::vector<T> Tensor::GenerateData(const std::function<T(rad::ArrayRef<size_t> indices)>& generator) const
{
    assert(sizeof(T) == GetElementSizeInBytes());
    std::vector<T> bufferData(GetBufferSizeInElements(), T(0));
    std::vector<size_t> indices(m_sizes.size(), 0);
    TensorIterator iter(m_sizes);
    iter.ForEach([&](rad::ArrayRef<size_t> indices)
        {
            size_t bufferIndex = std::inner_product(indices.begin(), indices.end(), m_strides.begin(), size_t(0));
            bufferData[bufferIndex] = generator(indices);
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
