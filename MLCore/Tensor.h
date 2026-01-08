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

// Same as torch.Tensor, a multi-dimensional matrix containing elements of a single data type.
// Tensor is actually a view over a reference counted TensorStorage.
class Tensor : public rad::RefCounted<Tensor>
{
public:
    rad::Ref<TensorStorage> m_storage;
    rad::Ref<Context> m_context;

    std::vector<size_t> m_offsets;
    std::vector<size_t> m_sizes;
    std::vector<size_t> m_strides;
    DataType m_dataType = DataType::Unknown;

    size_t m_bufferOffset = 0;
    size_t m_bufferSize = 0;

    Tensor();
    Tensor(rad::Ref<TensorStorage> storage, rad::Ref<Context> context);
    Tensor(const Tensor& other) = default;
    Tensor(Tensor&& other) noexcept = default;
    Tensor& operator=(const Tensor& other) = default;
    Tensor& operator=(Tensor&& other) noexcept = default;
    virtual ~Tensor();

    operator bool() const { return m_storage != nullptr; }

    Device* GetDevice() const { return m_storage->m_device.get(); }
    void SetContext(rad::Ref<Context> context) { m_context = std::move(context); }

    bool IsFloatingPoint() const;
    bool IsInteger() const;
    bool IsSignedInteger() const;
    bool IsUnsignedInteger() const;

    size_t GetDimCount() const { return m_sizes.size(); }

    size_t GetElementCount() const;
    std::vector<size_t> GetMemoryOrder() const;

    virtual size_t GetDataSizeInElement() const;
    virtual size_t GetDataSize() const;

    bool IsContiguous() const;
    bool HasSameLayout(const Tensor& other) const;
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

    Tensor& FillConstant(Scalar value);

    [[nodiscard]] Tensor Add(Scalar other);
    Tensor& AddInPlace(Scalar other);
    [[nodiscard]] Tensor Add(const Tensor& other, Scalar alpha = 1);
    Tensor& AddInPlace(const Tensor& other, Scalar alpha = 1);

    [[nodiscard]] Tensor Subtract(Scalar other);
    Tensor& SubtractInPlace(Scalar other);
    [[nodiscard]] Tensor Subtract(const Tensor& other, Scalar alpha = 1);
    Tensor& SubtractInPlace(const Tensor& other, Scalar alpha = 1);

    [[nodiscard]] Tensor Multiply(Scalar other);
    Tensor& MultiplyInPlace(Scalar other);
    [[nodiscard]] Tensor Multiply(const Tensor& other);
    Tensor& MultiplyInPlace(const Tensor& other);

    [[nodiscard]] Tensor Divide(Scalar other);
    Tensor& DivideInPlace(Scalar other);
    [[nodiscard]] Tensor Divide(const Tensor& other);
    Tensor& DivideInPlace(const Tensor& other);

    Tensor& operator+=(Scalar other) { return AddInPlace(other); }
    Tensor& operator-=(Scalar other) { return SubtractInPlace(other); }
    Tensor& operator*=(Scalar other) { return MultiplyInPlace(other); }
    Tensor& operator/=(Scalar other) { return DivideInPlace(other); }

    Tensor& operator+=(const Tensor& other) { return AddInPlace(other); }
    Tensor& operator-=(const Tensor& other) { return SubtractInPlace(other); }
    Tensor& operator*=(const Tensor& other) { return MultiplyInPlace(other); }
    Tensor& operator/=(const Tensor& other) { return DivideInPlace(other); }

    friend Tensor operator+(Tensor lhs, float rhs)
    {
        return lhs += rhs;
    }
    friend Tensor operator+(Tensor lhs, int rhs)
    {
        return lhs += rhs;
    }

    friend Tensor operator-(Tensor lhs, float rhs)
    {
        return lhs -= rhs;
    }
    friend Tensor operator-(Tensor lhs, int rhs)
    {
        return lhs -= rhs;
    }

    friend Tensor operator*(Tensor lhs, float rhs)
    {
        return lhs *= rhs;
    }
    friend Tensor operator*(Tensor lhs, int rhs)
    {
        return lhs *= rhs;
    }

    friend Tensor operator/(Tensor lhs, float rhs)
    {
        return lhs /= rhs;
    }
    friend Tensor operator/(Tensor lhs, int rhs)
    {
        return lhs /= rhs;
    }

    friend Tensor operator+(Tensor lhs, Tensor& rhs)
    {
        return lhs += rhs;
    }

    friend Tensor operator-(Tensor lhs, Tensor& rhs)
    {
        return lhs -= rhs;
    }

    friend Tensor operator*(Tensor lhs, Tensor& rhs)
    {
        return lhs *= rhs;
    }

    friend Tensor operator/(Tensor lhs, Tensor& rhs)
    {
        return lhs /= rhs;
    }

}; // class Tensor

Tensor MakeTensor(rad::ArrayRef<size_t> sizes, DataType dataType, const TensorOptions& options = {});
Tensor MakeTensorLike(Tensor& ref);
Tensor MakeTensorLike(Tensor* ref);

inline bool HaveSameLayout(const Tensor& a, const Tensor& b)
{
    return ((a.m_sizes == b.m_sizes) && (a.m_strides == b.m_strides));
}

} // namespace ML
