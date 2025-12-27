#pragma once

#include <MLCore/Common.h>

namespace ML
{

class Device;
class Context;

class Tensor : public rad::RefCounted<Tensor>
{
public:
    rad::Ref<Device> m_device;
    rad::Ref<Context> m_context;
    std::vector<size_t> m_sizes;
    std::vector<size_t> m_strides;
    DataType m_dataType = DataType::Unknown;

    Tensor();
    Tensor(rad::Ref<Device> device);

    virtual ~Tensor() = default;

    void SetContext(rad::Ref<Context> context) { m_context = std::move(context); }

    Device* GetDevice() { return m_device.get(); }
    Context* GetContext() { return m_context.get(); }

    static std::vector<size_t> MakeStrides(rad::ArrayRef<size_t> sizes, rad::ArrayRef<size_t> memoryOrder = {});

    size_t GetDimCount() const { return m_sizes.size(); }
    static size_t GetElementCount(rad::ArrayRef<size_t> sizes);
    static size_t GetElementCountND(rad::ArrayRef<size_t> sizes, size_t ndim);
    size_t GetElementCount() const;
    size_t GetElementCountND(size_t ndim) const;
    std::vector<size_t> GetMemoryOrder() const;

    static size_t GetDataSizeInElement(rad::ArrayRef<size_t> size, rad::ArrayRef<size_t> strides);
    virtual size_t GetDataSizeInElement() const;
    virtual size_t GetDataSize() const;

    bool IsContiguous() const;
    bool HasSameLayout(const Tensor* other) const;

    size_t CoordToBufferIndex(rad::ArrayRef<size_t> coord) const;
    size_t CoordToBufferOffset(rad::ArrayRef<size_t> coord) const;

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

    Tensor* FillConstant(float value);
    Tensor* FillConstant(int value);

    [[nodiscard]] rad::Ref<Tensor> AddScalar(float other);
    [[nodiscard]] rad::Ref<Tensor> AddScalar(int other);
    Tensor* AddScalarInPlace(float other);
    Tensor* AddScalarInPlace(int other);

    [[nodiscard]] rad::Ref<Tensor> Add(Tensor* other);
    [[nodiscard]] rad::Ref<Tensor> Add(Tensor* other, float alpha);
    [[nodiscard]] rad::Ref<Tensor> Add(Tensor* other, int alpha);
    Tensor* AddInPlace(Tensor* other);
    Tensor* AddInPlace(Tensor* other, float alpha);
    Tensor* AddInPlace(Tensor* other, int alpha);

    [[nodiscard]] rad::Ref<Tensor> SubtractScalar(float other);
    [[nodiscard]] rad::Ref<Tensor> SubtractScalar(int other);
    Tensor* SubtractScalarInPlace(float other);
    Tensor* SubtractScalarInPlace(int other);

    [[nodiscard]] rad::Ref<Tensor> Subtract(Tensor* other);
    [[nodiscard]] rad::Ref<Tensor> Subtract(Tensor* other, float alpha);
    [[nodiscard]] rad::Ref<Tensor> Subtract(Tensor* other, int alpha);
    Tensor* SubtractInPlace(Tensor* other);
    Tensor* SubtractInPlace(Tensor* other, float alpha);
    Tensor* SubtractInPlace(Tensor* other, int alpha);

}; // class Tensor

inline bool HaveSameLayout(const Tensor* a, const Tensor* b)
{
    return ((a->m_sizes == b->m_sizes) && (a->m_strides == b->m_strides));
}

} // namespace ML
