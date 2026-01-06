#pragma once

#include <MLCore/Common.h>

namespace ML
{

class Device;
class Context;

size_t GetTensorElementCount(rad::ArrayRef<size_t> sizes);
std::vector<size_t> MakeTensorStrides(rad::ArrayRef<size_t> sizes, rad::ArrayRef<size_t> memoryOrder = {});
size_t GetTensorDataSizeInElement(rad::ArrayRef<size_t> size, rad::ArrayRef<size_t> strides);

class TensorStorage : public rad::RefCounted<TensorStorage>
{
public:
    TensorStorage(rad::Ref<Device> device);
    virtual ~TensorStorage();

    virtual void Read(void* data, size_t offset, size_t dataSize) = 0;
    virtual void Write(const void* data, size_t offset, size_t dataSize) = 0;

    rad::Ref<Device> m_device;
    std::vector<size_t> m_sizes;
    std::vector<size_t> m_strides;
    DataType m_dataType = DataType::Unknown;

}; // class TensorStorage

class Tensor : public rad::RefCounted<Tensor>
{
public:
    rad::Ref<Device> m_device;
    rad::Ref<TensorStorage> m_storage;
    rad::Ref<Context> m_context;

    size_t m_bufferOffset = 0;
    std::vector<size_t> m_offsets;
    std::vector<size_t> m_sizes;
    std::vector<size_t> m_strides;
    DataType m_dataType = DataType::Unknown;

    Tensor(rad::Ref<TensorStorage> storage, rad::Ref<Context> context);
    virtual ~Tensor();

    void SetContext(rad::Ref<Context> context) { m_context = std::move(context); }

    size_t GetDimCount() const { return m_sizes.size(); }

    size_t GetElementCount() const;
    std::vector<size_t> GetMemoryOrder() const;

    virtual size_t GetDataSizeInElement() const;
    virtual size_t GetDataSize() const;

    bool IsContiguous() const;
    bool HasSameLayout(const Tensor* other) const;

    void Read(void* data, size_t offset, size_t dataSize);
    void Write(const void* data, size_t offset, size_t dataSize);

    size_t CoordToBufferIndex(rad::ArrayRef<size_t> coord) const;
    size_t CoordToBufferOffset(rad::ArrayRef<size_t> coord) const;

    bool IsNCHW() const;
    bool IsNHWC() const;
    bool IsNCDHW() const;
    bool IsNDHWC() const;

    enum class TextFormat
    {
        Dec,
        Hex,
    };
    std::string ToString(TextFormat format = TextFormat::Dec, rad::ArrayRef<size_t> offsets = {}, rad::ArrayRef<size_t> sizes = {});

    Tensor& FillConstant(float value);
    Tensor& FillConstant(int value);

    [[nodiscard]] Tensor AddScalar(float other);
    [[nodiscard]] Tensor AddScalar(int other);
    Tensor& AddScalarInPlace(float other);
    Tensor& AddScalarInPlace(int other);

    [[nodiscard]] Tensor Add(Tensor& other);
    [[nodiscard]] Tensor Add(Tensor& other, float alpha);
    [[nodiscard]] Tensor Add(Tensor& other, int alpha);
    Tensor& AddInPlace(Tensor& other);
    Tensor& AddInPlace(Tensor& other, float alpha);
    Tensor& AddInPlace(Tensor& other, int alpha);

    [[nodiscard]] Tensor SubtractScalar(float other);
    [[nodiscard]] Tensor SubtractScalar(int other);
    Tensor& SubtractScalarInPlace(float other);
    Tensor& SubtractScalarInPlace(int other);

    [[nodiscard]] Tensor Subtract(Tensor& other);
    [[nodiscard]] Tensor Subtract(Tensor& other, float alpha);
    [[nodiscard]] Tensor Subtract(Tensor& other, int alpha);
    Tensor& SubtractInPlace(Tensor& other);
    Tensor& SubtractInPlace(Tensor& other, float alpha);
    Tensor& SubtractInPlace(Tensor& other, int alpha);

    [[nodiscard]] Tensor MultiplyScalar(float other);
    [[nodiscard]] Tensor MultiplyScalar(int other);
    Tensor& MultiplyScalarInPlace(float other);
    Tensor& MultiplyScalarInPlace(int other);
    [[nodiscard]] Tensor Multiply(Tensor& other);
    Tensor& MultiplyInPlace(Tensor& other);

}; // class Tensor

Tensor MakeTensor(rad::ArrayRef<size_t> sizes, DataType dataType, const TensorOptions& options = {});
Tensor MakeTensorLike(Tensor& ref);
Tensor MakeTensorLike(Tensor* ref);

inline bool HaveSameLayout(const Tensor* a, const Tensor* b)
{
    return ((a->m_sizes == b->m_sizes) && (a->m_strides == b->m_strides));
}

} // namespace ML
