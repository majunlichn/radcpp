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

enum class TextFormat
{
    Dec,
    Hex,
};

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
    bool IsBool() const;
    bool IsComplex() const;

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

    std::string ToString(rad::ArrayRef<size_t> offsets = {}, rad::ArrayRef<size_t> sizes = {}, TextFormat format = TextFormat::Dec);

    Tensor& Fill(const Scalar& value);

    [[nodiscard]] Tensor Add(const Scalar& other) const;
    Tensor& AddInPlace(const Scalar& other);
    [[nodiscard]] Tensor Add(const Tensor& other, const Scalar& alpha = 1) const;
    Tensor& AddInPlace(const Tensor& other, const Scalar& alpha = 1);

    Tensor& Add_(const Scalar& other) { return AddInPlace(other); }
    Tensor& Add_(const Tensor& other, const Scalar& alpha = 1) { return AddInPlace(other); }

    [[nodiscard]] Tensor Subtract(const Scalar& other) const;
    Tensor& SubtractInPlace(const Scalar& other);
    [[nodiscard]] Tensor Subtract(const Tensor& other, const Scalar& alpha = 1) const;
    Tensor& SubtractInPlace(const Tensor& other, const Scalar& alpha = 1);

    [[nodiscard]] Tensor Sub(const Scalar& other) const { return Subtract(other); }
    Tensor& Sub_(const Scalar& other) { return SubtractInPlace(other); }
    [[nodiscard]] Tensor Sub(const Tensor& other, const Scalar& alpha = 1) const { return Subtract(other); }
    Tensor& Sub_(const Tensor& other, const Scalar& alpha = 1) { return SubtractInPlace(other); }

    [[nodiscard]] Tensor Multiply(const Scalar& other) const;
    Tensor& MultiplyInPlace(const Scalar& other);
    [[nodiscard]] Tensor Multiply(const Tensor& other) const;
    Tensor& MultiplyInPlace(const Tensor& other);

    [[nodiscard]] Tensor Mul(const Scalar& other) const { return Multiply(other); }
    Tensor& Mul_(const Scalar& other) { return MultiplyInPlace(other); }
    [[nodiscard]] Tensor Mul(const Tensor& other) const { return Multiply(other); }
    Tensor& Mul_(const Tensor& other) { return MultiplyInPlace(other); }

    [[nodiscard]] Tensor Divide(const Scalar& other) const;
    Tensor& DivideInPlace(const Scalar& other);
    [[nodiscard]] Tensor Divide(const Tensor& other) const;
    Tensor& DivideInPlace(const Tensor& other);

    [[nodiscard]] Tensor Div(const Scalar& other) const { return Divide(other); }
    Tensor& Div_(const Scalar& other) { return DivideInPlace(other); }
    [[nodiscard]] Tensor Div(const Tensor& other) const { return Divide(other); }
    Tensor& Div_(const Tensor& other) { return DivideInPlace(other); }

    [[nodiscard]] Tensor Remainder(const Scalar& other) const;
    Tensor& Remainder_(const Scalar& other);
    [[nodiscard]] Tensor Remainder(const Tensor& other) const;
    Tensor& Remainder_(const Tensor& other);

    [[nodiscard]] Tensor BitwiseAnd(const Scalar& other) const;
    Tensor& BitwiseAnd_(const Scalar& other);
    [[nodiscard]] Tensor BitwiseAnd(const Tensor& other) const;
    Tensor& BitwiseAnd_(const Tensor& other);

    [[nodiscard]] Tensor BitwiseOr(const Scalar& other) const;
    Tensor& BitwiseOr_(const Scalar& other);
    [[nodiscard]] Tensor BitwiseOr(const Tensor& other) const;
    Tensor& BitwiseOr_(const Tensor& other);

    [[nodiscard]] Tensor BitwiseXor(const Scalar& other) const;
    Tensor& BitwiseXor_(const Scalar& other);
    [[nodiscard]] Tensor BitwiseXor(const Tensor& other) const;
    Tensor& BitwiseXor_(const Tensor& other);

    Tensor& operator+=(const Scalar& other) { return Add_(other); }
    Tensor& operator-=(const Scalar& other) { return Sub_(other); }
    Tensor& operator*=(const Scalar& other) { return Mul_(other); }
    Tensor& operator/=(const Scalar& other) { return Div_(other); }

    Tensor& operator+=(const Tensor& other) { return Add_(other); }
    Tensor& operator-=(const Tensor& other) { return Sub_(other); }
    Tensor& operator*=(const Tensor& other) { return Mul_(other); }
    Tensor& operator/=(const Tensor& other) { return Div_(other); }

}; // class Tensor

Tensor MakeTensor(rad::ArrayRef<size_t> sizes, DataType dataType, const TensorOptions& options = {});
Tensor MakeTensorLike(const Tensor& ref);
Tensor MakeTensorLike(const Tensor* ref);

inline bool HaveSameLayout(const Tensor& a, const Tensor& b)
{
    return ((a.m_sizes == b.m_sizes) && (a.m_strides == b.m_strides));
}

#define ML_DEFINE_TENSOR_BINARY_OPS(_)                              \
    _(+, x.Add(y), y.Add(x))                                        \
    _(-,                                                            \
      x.Sub(y),                                                     \
      MakeTensorLike(y).Fill(x).Sub_(y))                            \
    _(*, x.Mul(y), y.Mul(x))                                        \
    _(/,                                                            \
      x.Divide(y),                                                  \
      MakeTensorLike(y).Fill(x).Div_(y))                            \
    _(%,                                                            \
      x.Remainder(y),                                               \
      MakeTensorLike(y).Fill(x).Remainder_(y))                      \
    _(&, x.BitwiseAnd(y), y.BitwiseAnd(x))                          \
    _(|, x.BitwiseOr(y), y.BitwiseOr(x))                            \
    _(^, x.BitwiseXor(y), y.BitwiseXor(x))

#define ML_TENSOR_BINARY_OP(op, body, reverse_scalar_body)          \
    inline Tensor operator op(const Tensor& x, const Tensor& y) {   \
      return body;                                                  \
    }                                                               \
    inline Tensor operator op(const Tensor& x, const Scalar& y) {   \
      return body;                                                  \
    }                                                               \
    inline Tensor operator op(const Scalar& x, const Tensor& y) {   \
      return reverse_scalar_body;                                   \
    }

ML_DEFINE_TENSOR_BINARY_OPS(ML_TENSOR_BINARY_OP)

#undef ML_TENSOR_BINARY_OP
#undef ML_DEFINE_TENSOR_BINARY_OPS

} // namespace ML
