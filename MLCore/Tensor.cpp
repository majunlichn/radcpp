#include <MLCore/Tensor.h>
#include <MLCore/Backend.h>
#include <MLCore/Device.h>
#include <MLCore/Context.h>
#include <MLCore/TensorIterator.h>
#include <MLCore/Global.h>

namespace ML
{

size_t GetTensorElementCount(rad::ArrayRef<size_t> sizes)
{
    if (sizes.empty())
    {
        return 0;
    }
    size_t count = sizes[0];
    for (size_t i = 1; i < sizes.size(); ++i)
    {
        count *= sizes[i];
    }
    return count;
}

std::vector<size_t> MakeTensorStrides(rad::ArrayRef<size_t> sizes, rad::ArrayRef<size_t> memoryOrder)
{
    assert(memoryOrder.empty() || (memoryOrder.size() == sizes.size()));

    std::vector<size_t> strides(sizes.size(), 0);
    if (memoryOrder.empty())
    {
        strides.back() = 1;
        std::partial_sum(
            sizes.rbegin(), sizes.rend() - 1, strides.rbegin() + 1, std::multiplies<size_t>());
    }
    else
    {
        size_t stride = 1;
        for (size_t i = 0; i < sizes.size(); ++i)
        {
            strides[memoryOrder[i]] = stride;
            stride *= sizes[memoryOrder[i]];
        }
    }
    return strides;
}

size_t GetTensorDataSizeInElement(rad::ArrayRef<size_t> size, rad::ArrayRef<size_t> strides)
{
    size_t indexOfTheLastElement = 0;
    for (size_t i = 0; i < size.size(); ++i)
    {
        indexOfTheLastElement += (size[i] - 1) * strides[i];
    }
    return (indexOfTheLastElement + 1);
}

TensorStorage::TensorStorage(rad::Ref<Device> device) :
    m_device(std::move(device))
{
}

TensorStorage::~TensorStorage()
{
}

Tensor::Tensor(rad::Ref<TensorStorage> storage, rad::Ref<Context> context) :
    m_device(storage->m_device),
    m_storage(std::move(storage)),
    m_context(std::move(context))
{
    assert(m_storage->m_device == m_context->m_device);
    m_sizes = m_storage->m_sizes;
    m_strides = m_storage->m_strides;
    m_dataType = m_storage->m_dataType;
}

Tensor::~Tensor()
{
}

Tensor MakeTensor(rad::ArrayRef<size_t> sizes, DataType dataType, const TensorOptions& options)
{
    rad::Ref<Device> device = GetCurrentDevice();
    rad::Ref<TensorStorage> storage = device->CreateTensorStorage(sizes, dataType, options);
    rad::Ref<Context> context = g_contextPool->GetContext(device.get());
    return Tensor(storage, context);
}

Tensor MakeTensorLike(Tensor& ref)
{
    TensorOptions options = {};
    options.m_strides = ref.m_strides;
    return Tensor(ref.m_device->CreateTensorStorage(ref.m_sizes, ref.m_dataType, options), ref.m_context);
}

Tensor MakeTensorLike(Tensor* ref)
{
    return MakeTensorLike(*ref);
}

size_t Tensor::GetElementCount() const
{
    return GetTensorElementCount(m_sizes);
}

std::vector<size_t> Tensor::GetMemoryOrder() const
{
    std::vector<size_t> order(m_strides.size());
    std::iota(order.begin(), order.end(), size_t(0));
    std::ranges::stable_sort(order,
        [&](size_t i, size_t j) {
            if (m_strides[i] < m_strides[j])
            {
                return true;
            }
            else if (m_strides[i] > m_strides[j])
            {
                return false;
            }
            else
            {
                return i > j;
            }
        }
    );
    return order;
}

size_t Tensor::GetDataSizeInElement() const
{
    return GetTensorDataSizeInElement(m_sizes, m_strides);
}

size_t Tensor::GetDataSize() const
{
    return GetDataSizeInElement() * GetElementSize(m_dataType);
}

bool Tensor::IsContiguous() const
{
    return (GetElementCount() == GetDataSizeInElement());
}

bool Tensor::HasSameLayout(const Tensor* other) const
{
    return HaveSameLayout(this, other);
}

void Tensor::Read(void* data, size_t offset, size_t dataSize)
{
    m_storage->Read(data, offset, dataSize);
}

void Tensor::Write(const void* data, size_t offset, size_t dataSize)
{
    m_storage->Write(data, offset, dataSize);
}

size_t Tensor::CoordToBufferIndex(rad::ArrayRef<size_t> coord) const
{
    assert(coord.size() == m_strides.size());
    return std::inner_product(coord.begin(), coord.end(), m_strides.begin(), size_t(0));
}

size_t Tensor::CoordToBufferOffset(rad::ArrayRef<size_t> coord) const
{
    return CoordToBufferIndex(coord) * GetElementSize(m_dataType);
}

bool Tensor::IsNCHW() const
{
    if ((m_sizes.size() == 4) &&
        (m_strides[0] > m_strides[1]) &&
        (m_strides[1] > m_strides[2]) &&
        (m_strides[2] > m_strides[3]) &&
        (m_strides[3] == 1))
    {
        return true;
    }
    return false;
}

bool Tensor::IsNHWC() const
{
    if ((m_sizes.size() == 4) &&
        (m_strides[0] > m_strides[2]) &&
        (m_strides[2] > m_strides[3]) &&
        (m_strides[3] > m_strides[1]) &&
        (m_strides[1] == 1))
    {
        return true;
    }
    return false;
}

bool Tensor::IsNCDHW() const
{
    if ((m_sizes.size() == 5) &&
        (m_strides[0] > m_strides[1]) &&
        (m_strides[1] > m_strides[2]) &&
        (m_strides[2] > m_strides[3]) &&
        (m_strides[3] > m_strides[4]) &&
        (m_strides[4] == 1))
    {
        return true;
    }
    return false;
}

bool Tensor::IsNDHWC() const
{
    if ((m_sizes.size() == 5) &&
        (m_strides[0] > m_strides[2]) &&
        (m_strides[2] > m_strides[3]) &&
        (m_strides[3] > m_strides[4]) &&
        (m_strides[4] > m_strides[1]) &&
        (m_strides[1] == 1))
    {
        return true;
    }
    return false;
}

std::string Tensor::ToString(TextFormat format, rad::ArrayRef<size_t> offsets, rad::ArrayRef<size_t> sizes)
{
    if (m_sizes.empty())
    {
        return {};
    }
    std::stringstream ss;
    std::vector<uint8_t> dataBuffer;
    dataBuffer.resize(GetDataSize());
    m_storage->Read(dataBuffer.data(), 0, dataBuffer.size());
    const uint8_t* data = dataBuffer.data();
    TensorIterator iter(this, offsets, sizes);
    size_t dimCount = m_sizes.size();
    if (dimCount == 1)
    {
        for (size_t w = 0; w < m_sizes[0]; ++w)
        {
            iter.m_coord[0] = w;
            ss << ToStringFixedWidthDec(data + w * GetElementSize(m_dataType), m_dataType);
        }
        ss << std::endl;
    }
    else
    {
        do {
            iter.Reset2D();
            size_t sizeH = m_sizes[dimCount - 2];
            size_t sizeW = m_sizes[dimCount - 1];
            ss << std::format("Indices = [{}]\n", rad::ToString(iter.m_coord, ", "));
            size_t bufferIndex = iter.m_bufferIndex;
            for (size_t h = 0; h < sizeH; ++h)
            {
                iter.m_coord[dimCount - 2] = h;
                bufferIndex += h * m_strides[dimCount - 2];
                for (size_t w = 0; w < sizeW; ++w)
                {
                    iter.m_coord[dimCount - 1] = w;
                    bufferIndex += w * m_strides[dimCount - 1];
                    if (format == TextFormat::Dec)
                    {
                        ss << ToStringFixedWidthDec(data + bufferIndex * GetElementSize(m_dataType), m_dataType) + ", ";
                    }
                    else // if (format == TextFormat::Hex)
                    {
                        ss << ToStringFixedWidthHex(data + bufferIndex * GetElementSize(m_dataType), m_dataType) + ", ";
                    }
                }
                ss << std::endl;
            }
        } while (iter.Next2D());
    }
    return ss.str();
}

Tensor& Tensor::FillConstant(float value)
{
    m_context->FillConstant(this, value);
    return *this;
}

Tensor& Tensor::FillConstant(int value)
{
    m_context->FillConstant(this, value);
    return *this;
}

Tensor Tensor::AddScalar(float other)
{
    Tensor output = MakeTensorLike(this);
    m_context->AddScalar(this, other, &output);
    return output;
}

Tensor Tensor::AddScalar(int other)
{
    Tensor output = MakeTensorLike(this);
    m_context->AddScalar(this, other, &output);
    return output;
}

Tensor& Tensor::AddScalarInPlace(float other)
{
    m_context->AddScalar(this, other);
    return *this;
}

Tensor& Tensor::AddScalarInPlace(int other)
{
    m_context->AddScalar(this, other);
    return *this;
}


Tensor Tensor::Add(Tensor& other)
{
    if (IsFloatingPointType(m_dataType))
    {
        return Add(other, 1.0f);
    }
    else
    {
        return Add(other, 1);
    }
}

Tensor Tensor::Add(Tensor& other, float alpha)
{
    Tensor output = MakeTensorLike(this);
    m_context->Add(this, &other, alpha, &output);
    return output;
}

Tensor Tensor::Add(Tensor& other, int alpha)
{
    Tensor output = MakeTensorLike(this);
    m_context->Add(this, &other, alpha, &output);
    return output;
}

Tensor& Tensor::AddInPlace(Tensor& other)
{
    if (IsFloatingPointType(m_dataType))
    {
        return AddInPlace(other, 1.0f);
    }
    else
    {
        return AddInPlace(other, 1);
    }
}

Tensor& Tensor::AddInPlace(Tensor& other, float alpha)
{
    m_context->Add(this, &other, alpha);
    return *this;
}

Tensor& Tensor::AddInPlace(Tensor& other, int alpha)
{
    m_context->Add(this, &other, alpha);
    return *this;
}

Tensor Tensor::SubtractScalar(float other)
{
    Tensor output = MakeTensorLike(this);
    m_context->SubtractScalar(this, other, &output);
    return output;
}

Tensor Tensor::SubtractScalar(int other)
{
    Tensor output = MakeTensorLike(this);
    m_context->SubtractScalar(this, other, &output);
    return output;
}

Tensor& Tensor::SubtractScalarInPlace(float other)
{
    m_context->SubtractScalar(this, other);
    return *this;
}

Tensor& Tensor::SubtractScalarInPlace(int other)
{
    m_context->SubtractScalar(this, other);
    return *this;
}

Tensor Tensor::Subtract(Tensor& other)
{
    if (IsFloatingPointType(m_dataType))
    {
        return Subtract(other, 1.0f);
    }
    else
    {
        return Subtract(other, 1);
    }
}

Tensor Tensor::Subtract(Tensor& other, float alpha)
{
    Tensor output = MakeTensorLike(this);
    m_context->Subtract(this, &other, alpha, &output);
    return output;
}

Tensor Tensor::Subtract(Tensor& other, int alpha)
{
    Tensor output = MakeTensorLike(this);
    m_context->Subtract(this, &other, alpha, &output);
    return output;
}

Tensor& Tensor::SubtractInPlace(Tensor& other)
{
    if (IsFloatingPointType(m_dataType))
    {
        return SubtractInPlace(other, 1.0f);
    }
    else
    {
        return SubtractInPlace(other, 1);
    }
}

Tensor& Tensor::SubtractInPlace(Tensor& other, float alpha)
{
    m_context->Subtract(this, &other, alpha);
    return *this;
}

Tensor& Tensor::SubtractInPlace(Tensor& other, int alpha)
{
    m_context->Subtract(this, &other, alpha);
    return *this;
}

Tensor Tensor::MultiplyScalar(float other)
{
    Tensor output = MakeTensorLike(this);
    m_context->MultiplyScalar(this, other, &output);
    return output;
}

Tensor Tensor::MultiplyScalar(int other)
{
    Tensor output = MakeTensorLike(this);
    m_context->MultiplyScalar(this, other, &output);
    return output;
}

Tensor& Tensor::MultiplyScalarInPlace(float other)
{
    m_context->MultiplyScalar(this, other, this);
    return *this;
}

Tensor& Tensor::MultiplyScalarInPlace(int other)
{
    m_context->MultiplyScalar(this, other, this);
    return *this;
}

Tensor Tensor::Multiply(Tensor& other)
{
    Tensor output = MakeTensorLike(this);
    m_context->Multiply(this, &other, &output);
    return output;
}

Tensor& Tensor::MultiplyInPlace(Tensor& other)
{
    m_context->Multiply(this, &other, this);
    return *this;
}

} // namespace ML
