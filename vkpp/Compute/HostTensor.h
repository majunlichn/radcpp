#pragma once

#include <vkpp/Compute/TensorIterator.h>

#include <rad/Common/Integer.h>
#include <rad/Common/Float.h>
#include <rad/Common/Algorithm.h>
#include <rad/Common/Numeric.h>
#include <rad/Common/Random.h>

#include <taskflow/taskflow.hpp>

namespace vkpp
{

std::vector<size_t> MakeTensorStrides(
    rad::ArrayRef<size_t> sizes, rad::ArrayRef<size_t> memoryOrder = {}, rad::ArrayRef<size_t> alignments = {});

template <typename T>
class HostTensor : public rad::RefCounted<HostTensor<T>>
{
public:
    std::vector<size_t> m_sizes;
    std::vector<size_t> m_strides;
    bool m_isContiguous = false;

    std::vector<T> m_data;

    HostTensor(rad::ArrayRef<size_t> sizes, rad::ArrayRef<size_t> strides = {}) :
        m_sizes(sizes),
        m_strides(strides)
    {
        assert(sizes.size() > 0);
        assert((sizes.size() == strides.size()) || strides.empty());

        m_sizes = sizes;
        m_strides = strides;
        if (m_strides.empty())
        {
            m_strides = MakeTensorStrides(m_sizes);
        }
        assert(m_sizes.size() == m_strides.size());

        size_t elementCount = GetElementCount();
        size_t lastIndex = 0;
        for (size_t i = 0; i < m_sizes.size(); ++i)
        {
            lastIndex += (m_sizes[i] - 1) * m_strides[i];
        }
        if (lastIndex + 1 == elementCount)
        {
            m_isContiguous = true;
        }

        m_data.resize(lastIndex + 1, T(0));
    }

    ~HostTensor()
    {
    }

    size_t GetDimensionCount() const { return m_sizes.size(); }
    size_t GetElementCount() const
    {
        size_t count = m_sizes[0];
        for (size_t i = 1; i < m_sizes.size(); ++i)
        {
            count *= m_sizes[i];
        }
        return count;
    }

    bool IsContiguous() const { return m_isContiguous; }
    bool IsNCHW() const
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

    bool IsNHWC() const
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

    bool IsNCDHW() const
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

    bool IsNDHWC() const
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

    std::vector<size_t> GetMemoryOrder() const
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

    size_t CoordToIndex(rad::ArrayRef<size_t> coords)
    {
        return std::inner_product(coords.begin(), coords.end(), m_strides.begin(), size_t(0));
    }

    void ForEach(const TensorIterator::ElementWiseOp& op)
    {
        TensorIterator iter(m_sizes);
        iter.ForEachParallel(op);
    }

    void FillConstant(T value)
    {
        if (IsContiguous())
        {
            std::fill_n(m_data.begin(), m_data.size(), value);
        }
        else
        {
            TensorIterator iter(m_sizes);
            iter.ForEachParallel([&](rad::ArrayRef<size_t> coords)
                {
                    m_data[CoordToIndex(coords)] = value;
                });
        }
    }

    void FillZeros()
    {
        FillConstant(T(0));
    }

}; // class HostTensor<T>

} // namespace vkpp
