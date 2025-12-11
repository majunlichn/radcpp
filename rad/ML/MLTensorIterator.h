#pragma once

#include <rad/ML/MLTensor.h>
#include <taskflow/taskflow.hpp>

namespace rad
{

// A helper class to calculate coordinates to iterate over tensor elements, support different iteration orders (permutations).
// For example, for a 4D tensor, order={ 1, 3, 2, 0 } means to iterate in the order of C, W, H, N.
class MLTensorIterator
{
public:
    MLTensor* m_tensor;
    std::vector<size_t> m_order;
    std::vector<size_t> m_coord;

    MLTensorIterator(MLTensor* tensor) :
        m_tensor(tensor)
    {
        Reset();
    }

    ~MLTensorIterator() = default;

    size_t CoordToBufferIndex(ArrayRef<size_t> coord)
    {
        return m_tensor->CoordToBufferIndex(coord);
    }

    size_t CoordToBufferIndex()
    {
        return m_tensor->CoordToBufferIndex(m_coord);
    }

    size_t CoordToBufferOffset(ArrayRef<size_t> coord)
    {
        return m_tensor->CoordToBufferOffset(coord);
    }

    size_t CoordToBufferOffset()
    {
        return m_tensor->CoordToBufferOffset(m_coord);
    }

    void Reset()
    {
        m_coord.clear();
        m_coord.resize(m_tensor->m_sizes.size(), 0);
    }

    void ResetND(size_t n)
    {
        size_t dimCount = m_tensor->m_sizes.size();
        assert(dimCount >= n);
        std::fill_n(m_coord.end() - n, n, 0);
    }

    void Reset1D() { ResetND(1); }
    void Reset2D() { ResetND(2); }
    void Reset3D() { ResetND(3); }
    void Reset4D() { ResetND(4); }

    void ResetNDPermuted(size_t n)
    {
        size_t dimCount = m_tensor->m_sizes.size();
        assert(dimCount >= n);
        for (size_t i = 0; i < n; ++i)
        {
            size_t dimIndex = m_order[i];
            m_coord[dimIndex] = 0;
        }
    }

    bool NextND(size_t n)
    {
        size_t dimCount = m_tensor->m_sizes.size();
        for (ptrdiff_t dimIndex = ptrdiff_t(dimCount - n - 1); dimIndex >= 0; --dimIndex)
        {
            if (m_coord[dimIndex] < m_tensor->m_sizes[dimIndex] - 1)
            {
                ++m_coord[dimIndex];
                return true;
            }
            else
            {
                m_coord[dimIndex] = 0;
            }
        }
        return false;
    }

    bool NextNDSubrangeND(size_t n, size_t subrangeND)
    {
        size_t dimCount = m_tensor->m_sizes.size();
        for (ptrdiff_t dimIndex = ptrdiff_t(dimCount - n - 1); dimIndex >= ptrdiff_t(dimCount - subrangeND); --dimIndex)
        {
            if (m_coord[dimIndex] < m_tensor->m_sizes[dimIndex] - 1)
            {
                ++m_coord[dimIndex];
                return true;
            }
            else
            {
                m_coord[dimIndex] = 0;
            }
        }
        return false;
    }

    bool Next1D() { return NextND(1); }
    bool Next2D() { return NextND(2); }
    bool Next3D() { return NextND(3); }
    bool Next4D() { return NextND(4); }

    bool NextNDPermuted(size_t n)
    {
        size_t dimCount = m_tensor->m_sizes.size();
        assert(dimCount > n);
        for (size_t i = n; i < dimCount; ++i)
        {
            size_t dimIndex = m_order[i];
            if (m_coord[dimIndex] < m_tensor->m_sizes[dimIndex] - 1)
            {
                ++m_coord[dimIndex];
                return true;
            }
            else
            {
                m_coord[dimIndex] = 0;
            }
        }
        return false;
    }

    bool Next1DPermuted() { return NextNDPermuted(1); }
    bool Next2DPermuted() { return NextNDPermuted(2); }
    bool Next3DPermuted() { return NextNDPermuted(3); }
    bool Next4DPermuted() { return NextNDPermuted(4); }

    bool NextNDPermutedSubrangeND(size_t n, size_t subrangeND)
    {
        for (size_t i = n; i < subrangeND; ++i)
        {
            size_t dimIndex = m_order[i];
            if (m_coord[dimIndex] < m_tensor->m_sizes[dimIndex] - 1)
            {
                ++m_coord[dimIndex];
                return true;
            }
            else
            {
                m_coord[dimIndex] = 0;
            }
        }
        return false;
    }

}; // class MLTensorIterator

} // namespace rad
