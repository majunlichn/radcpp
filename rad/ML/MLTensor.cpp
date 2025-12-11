#include <rad/ML/MLTensor.h>
#include <rad/ML/MLDevice.h>
#include <rad/ML/MLContext.h>
#include <rad/ML/MLTensorIterator.h>
#include <rad/Common/Algorithm.h>

namespace rad
{

std::vector<size_t> MLTensor::MakeStrides(ArrayRef<size_t> sizes, ArrayRef<size_t> memoryOrder)
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

size_t MLTensor::GetElementCount() const
{
    if (m_sizes.empty())
    {
        return 0;
    }
    size_t count = m_sizes[0];
    for (size_t i = 1; i < m_sizes.size(); ++i)
    {
        count *= m_sizes[i];
    }
    return count;
}

std::vector<size_t> MLTensor::GetMemoryOrder() const
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

std::string MLTensor::ToString()
{
    if (m_sizes.empty())
    {
        return {};
    }
    std::stringstream ss;
    const uint8_t* data = static_cast<const uint8_t*>(GetData());
    MLTensorIterator iter(this);
    size_t dimCount = m_sizes.size();
    if (dimCount == 1)
    {
        for (size_t w = 0; w < m_sizes[0]; ++w)
        {
            iter.m_coord[0] = w;
            ss << FormatValueFixedWidthDec(data + iter.CoordToBufferOffset(), m_dataType);
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
                    ss << FormatValueFixedWidthDec(data + iter.CoordToBufferOffset(), m_dataType);
                }
                ss << std::endl;
            }
        } while (iter.Next2D());
    }
    return ss.str();
}

MLTensor* MLTensor::FillConstant(float value)
{
    MLContext* context = MLGetPerThreadContext(m_device->m_backend);
    context->FillConstant(this, value);
    return this;
}

MLTensor* MLTensor::FillConstant(int value)
{
    MLContext* context = MLGetPerThreadContext(m_device->m_backend);
    context->FillConstant(this, value);
    return this;
}

Ref<MLTensor> MLTensor::Add(MLTensor* other)
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

Ref<MLTensor> MLTensor::Add(MLTensor* other, float alpha)
{
    MLContext* context = MLGetPerThreadContext(m_device->m_backend);
    Ref<MLTensor> output = m_device->CreateTensorLike(this);
    context->Add(this, other, alpha, output.get());
    return output;
}

Ref<MLTensor> MLTensor::Add(MLTensor* other, int alpha)
{
    MLContext* context = MLGetPerThreadContext(m_device->m_backend);
    Ref<MLTensor> output = m_device->CreateTensorLike(this);
    context->Add(this, other, alpha, output.get());
    return output;
}

MLTensor* MLTensor::AddInPlace(MLTensor* other)
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

MLTensor* MLTensor::AddInPlace(MLTensor* other, float alpha)
{
    MLContext* context = MLGetPerThreadContext(m_device->m_backend);
    context->Add(this, other, alpha, nullptr);
    return this;
}

MLTensor* MLTensor::AddInPlace(MLTensor* other, int alpha)
{
    MLContext* context = MLGetPerThreadContext(m_device->m_backend);
    context->Add(this, other, alpha, nullptr);
    return this;
}

} // namespace rad
