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

    size_t GetRank() const { return m_sizes.size(); }
    size_t GetElementCount() const;
    std::vector<size_t> GetMemoryOrder() const;

    virtual size_t GetDataSizeInElement() const;
    virtual size_t GetDataSize() const;

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

} // namespace ML
