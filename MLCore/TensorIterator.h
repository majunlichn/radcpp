#pragma once

#include <MLCore/Tensor.h>

namespace ML
{

// A helper class to calculate coordinates to iterate over tensor elements, support different iteration orders (permutations).
// For example, for a 4D tensor, order={ 1, 3, 2, 0 } means to iterate in the order of C, W, H, N.
class TensorIterator
{
public:
    Tensor* m_tensor;
    std::vector<size_t> m_offsets;
    std::vector<size_t> m_sizes;
    std::vector<size_t> m_order;
    std::vector<size_t> m_coord;

    TensorIterator(Tensor* tensor, rad::ArrayRef<size_t> offsets = {}, rad::ArrayRef<size_t> sizes = {}) :
        m_tensor(tensor)
    {
        if (offsets.empty())
        {
            m_offsets.resize(m_tensor->GetDimCount(), 0);
        }
        else
        {
            assert(offsets.size() == m_tensor->GetDimCount());
            m_offsets = offsets;
        }

        if (sizes.empty())
        {
            m_sizes.resize(m_tensor->GetDimCount());
            for (size_t i = 0; i < m_tensor->GetDimCount(); ++i)
            {
                m_sizes[i] = m_tensor->m_sizes[i] - m_offsets[i];
            }
        }
        else
        {
            assert(sizes.size() == m_tensor->GetDimCount());
            m_sizes = sizes;
        }

        assert(IsValid());

        Reset();
    }

    ~TensorIterator() = default;

    bool IsValid() const
    {
        for (size_t i = 0; i < m_tensor->GetDimCount(); ++i)
        {
            if (m_offsets[i] + m_sizes[i] > m_sizes[i])
            {
                return false;
            }
        }
        return true;
    }

    size_t CoordToBufferIndex(rad::ArrayRef<size_t> coord)
    {
        return m_tensor->CoordToBufferIndex(coord);
    }

    size_t CoordToBufferIndex()
    {
        return m_tensor->CoordToBufferIndex(m_coord);
    }

    size_t CoordToBufferOffset(rad::ArrayRef<size_t> coord)
    {
        return m_tensor->CoordToBufferOffset(coord);
    }

    size_t CoordToBufferOffset()
    {
        return m_tensor->CoordToBufferOffset(m_coord);
    }

    void Reset()
    {
        m_coord = m_offsets;
    }

    void ResetND(size_t nd)
    {
        size_t dimCount = m_sizes.size();
        assert(nd <= dimCount);
        std::copy_n(m_offsets.end() - nd, nd, m_coord.end() - nd);
    }

    void Reset1D() { ResetND(1); }
    void Reset2D() { ResetND(2); }
    void Reset3D() { ResetND(3); }
    void Reset4D() { ResetND(4); }

    bool NextND(size_t nd)
    {
        size_t dimCount = m_tensor->GetDimCount();
        for (ptrdiff_t dimIndex = ptrdiff_t(dimCount - nd - 1); dimIndex >= 0; --dimIndex)
        {
            if (m_coord[dimIndex] < m_sizes[dimIndex] - 1)
            {
                ++m_coord[dimIndex];
                return true;
            }
            else
            {
                m_coord[dimIndex] = m_offsets[dimIndex];
            }
        }
        return false;
    }

    bool NextNDSubrangeND(size_t nd, size_t subrangeND)
    {
        size_t dimCount = m_sizes.size();
        for (ptrdiff_t dimIndex = ptrdiff_t(dimCount - nd - 1); dimIndex >= ptrdiff_t(dimCount - subrangeND); --dimIndex)
        {
            if (m_coord[dimIndex] < m_sizes[dimIndex] - 1)
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

    using ElementOp = std::function<void(rad::ArrayRef<size_t> coord)>;

    void ForEach(const ElementOp& op)
    {
        Reset();
        do
        {
            // Iterate the last dimension:
            size_t lastDimIndex = m_sizes.size() - 1;
            for (size_t i = 0; i < m_sizes[lastDimIndex]; ++i)
            {
                m_coord[lastDimIndex] = m_offsets[lastDimIndex] + i;
                op(m_coord);
            }
        } while (Next1D());
    }

    void ForEachSubrangeND(const ElementOp& op, size_t subrangeND)
    {
        ResetND(subrangeND);
        do
        {
            // Iterate the last dimension:
            size_t lastDimIndex = m_sizes.size() - 1;
            for (size_t i = 0; i < m_sizes[lastDimIndex]; ++i)
            {
                m_coord[lastDimIndex] = m_offsets[lastDimIndex] + i;
                op(m_coord);
            }
        } while (NextNDSubrangeND(1, subrangeND));
    }

    void ForEachRecursively(const ElementOp& op)
    {
        Reset();
        ForEachRecursively(op, 0);
    }

    void ForEachRecursively(const ElementOp& op, size_t dimIndex)
    {
        if (dimIndex == m_sizes.size() - 1)
        {
            // Iterate the last dimension:
            for (size_t i = 0; i < m_sizes[dimIndex]; ++i)
            {
                m_coord[dimIndex] = m_offsets[dimIndex] + i;
                op(m_coord);
            }
        }
        else
        {
            for (size_t i = 0; i < m_sizes[dimIndex]; ++i)
            {
                m_coord[dimIndex] = i;
                ForEachRecursively(op, dimIndex + 1);
            }
        }
    }

}; // class TensorIterator

} // namespace ML
