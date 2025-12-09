#include <rad/ML/MLTensor.h>
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

} // namespace rad
