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
    std::vector<size_t> m_strides;
    std::vector<size_t> m_permutation;
    std::vector<size_t> m_coord;
    size_t m_bufferIndex = 0;

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

        m_strides = m_tensor->m_strides;

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

    void Reset()
    {
        m_coord = m_offsets;
        m_bufferIndex = 0;
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
                m_bufferIndex += m_strides[dimIndex];
                return true;
            }
            else
            {
                m_bufferIndex -= m_coord[dimIndex] * m_strides[dimIndex];
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
                m_bufferIndex += m_strides[dimIndex];
                return true;
            }
            else
            {
                m_coord[dimIndex] = m_offsets[dimIndex];
                m_bufferIndex = m_coord[dimIndex] * m_strides[dimIndex];
            }
        }
        return false;
    }

    bool Next() { return NextND(0); }
    bool Next1D() { return NextND(1); }
    bool Next2D() { return NextND(2); }
    bool Next3D() { return NextND(3); }
    bool Next4D() { return NextND(4); }

}; // class TensorIterator

void ForEach(TensorIterator& iter, const std::function<void(size_t bufferIndex)>& op);
void ForEachRecursively(TensorIterator& iter, const std::function<void(size_t bufferIndex)>& op);
void ForEachSubrangeND(TensorIterator& iter, const std::function<void(size_t bufferIndex)>& op, size_t subrangeND);
void ForEach(TensorIterator& input, TensorIterator& output,
    const std::function<void(size_t inputIndex, size_t outputIndex)>& op);
void ForEach(TensorIterator& input, TensorIterator& other, TensorIterator& output,
    const std::function<void(size_t inputIndex, size_t otherIndex, size_t outputIndex)>& op);

} // namespace ML
