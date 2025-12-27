#include <MLCore/Tensor.h>
#include <MLCore/Backend.h>
#include <MLCore/Device.h>
#include <MLCore/Context.h>
#include <MLCore/TensorIterator.h>
#include <MLCore/Global.h>

namespace ML
{

Tensor::Tensor()
{
    m_device = GetCurrentDevice();
    m_context = g_contextPool->GetContext(m_device.get());
}

Tensor::Tensor(rad::Ref<Device> device) :
    m_device(std::move(device))
{
    m_context = g_contextPool->GetContext(m_device.get());
}

std::vector<size_t> Tensor::MakeStrides(rad::ArrayRef<size_t> sizes, rad::ArrayRef<size_t> memoryOrder)
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

size_t Tensor::GetElementCount(rad::ArrayRef<size_t> sizes)
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

size_t Tensor::GetElementCountND(rad::ArrayRef<size_t> sizes, size_t ndim)
{
    if (sizes.empty())
    {
        return 0;
    }
    size_t dimIndex = (sizes.size() > ndim) ? (sizes.size() - ndim) : 0;
    size_t count = sizes[dimIndex];
    ++dimIndex;
    for (; dimIndex < sizes.size(); ++dimIndex)
    {
        count *= sizes[dimIndex];
    }
    return count;
}

size_t Tensor::GetElementCount() const
{
    return GetElementCount(m_sizes);
}

size_t Tensor::GetElementCountND(size_t ndim) const
{
    return GetElementCountND(m_sizes, ndim);
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

size_t Tensor::GetDataSizeInElement(rad::ArrayRef<size_t> size, rad::ArrayRef<size_t> strides)
{
    size_t indexOfTheLastElement = 0;
    for (size_t i = 0; i < size.size(); ++i)
    {
        indexOfTheLastElement += (size[i] - 1) * strides[i];
    }
    return (indexOfTheLastElement + 1);
}

size_t Tensor::GetDataSizeInElement() const
{
    return GetDataSizeInElement(m_sizes, m_strides);
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

std::string Tensor::ToString(TextFormat format)
{
    if (m_sizes.empty())
    {
        return {};
    }
    std::stringstream ss;
    const uint8_t* data = static_cast<const uint8_t*>(GetData());
    std::vector<uint8_t> dataBuffer;
    if (!data)
    {
        dataBuffer.resize(GetDataSize());
        Read(dataBuffer.data(), 0, dataBuffer.size());
        data = dataBuffer.data();
    }
    TensorIterator iter(this);
    size_t dimCount = m_sizes.size();
    if (dimCount == 1)
    {
        for (size_t w = 0; w < m_sizes[0]; ++w)
        {
            iter.m_coord[0] = w;
            ss << ToStringFixedWidthDec(data + iter.CoordToBufferOffset(), m_dataType);
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
            for (size_t h = 0; h < sizeH; ++h)
            {
                iter.m_coord[dimCount - 2] = h;
                for (size_t w = 0; w < sizeW; ++w)
                {
                    iter.m_coord[dimCount - 1] = w;
                    if (format == TextFormat::Dec)
                    {
                        ss << ToStringFixedWidthDec(data + iter.CoordToBufferOffset(), m_dataType) + ", ";
                    }
                    else // if (format == TextFormat::Hex)
                    {
                        ss << ToStringFixedWidthHex(data + iter.CoordToBufferOffset(), m_dataType) + ", ";
                    }
                }
                ss << std::endl;
            }
        } while (iter.Next2D());
    }
    return ss.str();
}

Tensor* Tensor::FillConstant(float value)
{
    m_context->FillConstant(this, value);
    return this;
}

Tensor* Tensor::FillConstant(int value)
{
    m_context->FillConstant(this, value);
    return this;
}

rad::Ref<Tensor> Tensor::AddScalar(float other)
{
    rad::Ref<Tensor> output = m_device->CreateTensorLike(this);
    m_context->AddScalar(this, other, output.get());
    return output;
}

rad::Ref<Tensor> Tensor::AddScalar(int other)
{
    rad::Ref<Tensor> output = m_device->CreateTensorLike(this);
    m_context->AddScalar(this, other, output.get());
    return output;
}

Tensor* Tensor::AddScalarInPlace(float other)
{
    m_context->AddScalar(this, other);
    return this;
}

Tensor* Tensor::AddScalarInPlace(int other)
{
    m_context->AddScalar(this, other);
    return this;
}


rad::Ref<Tensor> Tensor::Add(Tensor* other)
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

rad::Ref<Tensor> Tensor::Add(Tensor* other, float alpha)
{
    rad::Ref<Tensor> output = m_device->CreateTensorLike(this);
    m_context->Add(this, other, alpha, output.get());
    return output;
}

rad::Ref<Tensor> Tensor::Add(Tensor* other, int alpha)
{
    rad::Ref<Tensor> output = m_device->CreateTensorLike(this);
    m_context->Add(this, other, alpha, output.get());
    return output;
}

Tensor* Tensor::AddInPlace(Tensor* other)
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

Tensor* Tensor::AddInPlace(Tensor* other, float alpha)
{
    m_context->Add(this, other, alpha);
    return this;
}

Tensor* Tensor::AddInPlace(Tensor* other, int alpha)
{
    m_context->Add(this, other, alpha);
    return this;
}

rad::Ref<Tensor> Tensor::SubtractScalar(float other)
{
    rad::Ref<Tensor> output = m_device->CreateTensorLike(this);
    m_context->SubtractScalar(this, other, output.get());
    return output;
}

rad::Ref<Tensor> Tensor::SubtractScalar(int other)
{
    rad::Ref<Tensor> output = m_device->CreateTensorLike(this);
    m_context->SubtractScalar(this, other, output.get());
    return output;
}

Tensor* Tensor::SubtractScalarInPlace(float other)
{
    m_context->SubtractScalar(this, other);
    return this;
}

Tensor* Tensor::SubtractScalarInPlace(int other)
{
    m_context->SubtractScalar(this, other);
    return this;
}

rad::Ref<Tensor> Tensor::Subtract(Tensor* other)
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

rad::Ref<Tensor> Tensor::Subtract(Tensor* other, float alpha)
{
    rad::Ref<Tensor> output = m_device->CreateTensorLike(this);
    m_context->Subtract(this, other, alpha, output.get());
    return output;
}

rad::Ref<Tensor> Tensor::Subtract(Tensor* other, int alpha)
{
    rad::Ref<Tensor> output = m_device->CreateTensorLike(this);
    m_context->Subtract(this, other, alpha, output.get());
    return output;
}

Tensor* Tensor::SubtractInPlace(Tensor* other)
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

Tensor* Tensor::SubtractInPlace(Tensor* other, float alpha)
{
    m_context->Subtract(this, other, alpha);
    return this;
}

Tensor* Tensor::SubtractInPlace(Tensor* other, int alpha)
{
    m_context->Subtract(this, other, alpha);
    return this;
}

} // namespace ML
