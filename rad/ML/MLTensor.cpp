#include <rad/ML/MLTensor.h>
#include <rad/Common/Algorithm.h>

namespace rad
{

MLTensorCoord MLTensor::MakeStrides(const MLTensorCoord& sizes, ArrayRef<size_t> order)
{
    assert(order.empty() || (order.size() == sizes.size()));

    std::vector<size_t> strides(sizes.size(), 0);
    if (order.empty())
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
            strides[order[i]] = stride;
            stride *= sizes[order[i]];
        }
    }
    return strides;
}

size_t MLTensor::CalculateBufferSize()
{
    assert(m_sizes.size() > 0);
    assert(m_strides.empty() || (m_sizes.size() == m_strides.size()));

    size_t lastIndex = 0;
    for (size_t i = 0; i < m_sizes.size(); ++i)
    {
        lastIndex += (m_sizes[i] - 1) * m_strides[i];
    }

    return (lastIndex + 1) * GetElementSize(m_dataType);
}

} // namespace rad
