#pragma once

#include <rad/ML/MLDataType.h>
#include <rad/ML/MLTensorOptions.h>
#include <rad/Common/Algorithm.h>
#include <rad/Common/Memory.h>
#include <rad/Common/RefCounted.h>
#include <rad/Common/String.h>
#include <rad/Container/ArrayRef.h>
#include <rad/Container/SmallVector.h>
#include <rad/Container/Span.h>

namespace rad
{

class MLDevice;

class MLTensor : public RefCounted<MLTensor>
{
public:
    MLTensor(Ref<MLDevice> device) : m_device(std::move(device)) {}
    virtual ~MLTensor() = default;

    virtual MLDevice* GetDevice() = 0;

    static std::vector<size_t> MakeStrides(ArrayRef<size_t> sizes, ArrayRef<size_t> memoryOrder = {});

    size_t GetRank() const { return m_sizes.size(); }
    size_t GetElementCount() const;
    std::vector<size_t> GetMemoryOrder() const;

    virtual size_t GetDataSizeInElement() const;
    virtual size_t GetDataSize() const;

    size_t CoordToBufferIndex(ArrayRef<size_t> coord) const;
    size_t CoordToBufferOffset(ArrayRef<size_t> coord) const;

    bool IsNCHW() const;
    bool IsNHWC() const;
    bool IsNCDHW() const;
    bool IsNDHWC() const;

    // CPU backend only, nullptr if not available.
    virtual void* GetData() = 0;

    virtual void Read(void* data, size_t offset, size_t dataSize) = 0;
    virtual void Write(const void* data, size_t offset, size_t dataSize) = 0;

    enum class TextFormat
    {
        Dec,
        Hex,
    };
    std::string ToString(TextFormat format = TextFormat::Dec);

    MLTensor* FillConstant(float value);
    MLTensor* FillConstant(int value);

    Ref<MLTensor> Add(MLTensor* other);
    Ref<MLTensor> Add(MLTensor* other, float alpha);
    Ref<MLTensor> Add(MLTensor* other, int alpha);
    MLTensor* AddInPlace(MLTensor* other);
    MLTensor* AddInPlace(MLTensor* other, float alpha);
    MLTensor* AddInPlace(MLTensor* other, int alpha);

    Ref<MLDevice> m_device;
    MLDataType m_dataType = MLDataType::Unknown;
    std::vector<size_t> m_sizes;
    std::vector<size_t> m_strides;

}; // class MLTensor

} // namespace rad
