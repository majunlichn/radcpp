#pragma once

#include <MLCore/Tensor.h>

namespace ML
{

// A helper class to calculate coordinates to iterate over tensor elements in different dimension orders (permutations).
class TensorIterator
{
public:
    std::vector<size_t> m_offsets;
    std::vector<size_t> m_sizes;
    std::vector<size_t> m_strides;
    std::vector<size_t> m_coord;
    size_t m_bufferIndex = 0;

    std::vector<size_t> m_permutation;
    std::vector<size_t> m_coordUnpermuted;

    TensorIterator(Tensor* tensor, rad::ArrayRef<size_t> offsets = {}, rad::ArrayRef<size_t> sizes = {})
    {
        if (offsets.empty())
        {
            m_offsets.resize(tensor->GetDimCount(), 0);
        }
        else
        {
            assert(offsets.size() == tensor->GetDimCount());
            m_offsets = offsets;
        }

        if (sizes.empty())
        {
            m_sizes.resize(tensor->GetDimCount());
            for (size_t i = 0; i < tensor->GetDimCount(); ++i)
            {
                m_sizes[i] = tensor->m_sizes[i] - m_offsets[i];
            }
        }
        else
        {
            assert(sizes.size() == tensor->GetDimCount());
            m_sizes = sizes;
        }

        m_strides = tensor->m_strides;

        for (size_t i = 0; i < m_sizes.size(); ++i)
        {
            assert(m_offsets[i] + m_sizes[i] <= tensor->m_sizes[i]);
        }

        Reset();
    }

    ~TensorIterator() = default;

    std::vector<size_t>& GetCoordUnpermuted()
    {
        if (m_permutation.empty())
        {
            return m_coord;
        }
        else
        {
            for (size_t i = 0; i < m_coord.size(); ++i)
            {
                m_coordUnpermuted[m_permutation[i]] = m_coord[i];
            }
            return m_coordUnpermuted;
        }
    }

    size_t CoordToBufferIndex() const
    {
        assert(m_sizes.size() == m_strides.size());
        return std::inner_product(m_coord.begin(), m_coord.end(), m_strides.begin(), size_t(0));
    }

    void Reset()
    {
        m_coord = m_offsets;
        m_bufferIndex = CoordToBufferIndex();
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

    // NCHW permutation: [0, 1, 2, 3]; NHWC permutation: [0, 2, 3, 1];
    void PermuteDims(rad::ArrayRef<size_t> permutation)
    {
        assert(permutation.size() == m_sizes.size());
        m_permutation = permutation;
        size_t dimCount = m_sizes.size();
        std::vector<size_t> offsets = m_offsets;
        std::vector<size_t> sizes = m_sizes;
        std::vector<size_t> strides = m_strides;
        std::vector<size_t> coord = m_coord;
        for (size_t i = 0; i < m_sizes.size(); ++i)
        {
            assert(permutation[i] < m_sizes.size());
            offsets[i] = m_offsets[m_permutation[i]];
            sizes[i] = m_sizes[m_permutation[i]];
            strides[i] = m_strides[m_permutation[i]];
            coord[i] = m_coord[m_permutation[i]];
        }
        m_offsets = std::move(offsets);
        m_sizes = std::move(sizes);
        m_strides = std::move(strides);
        m_coordUnpermuted = m_coord;
        m_coord = std::move(coord);
    }

    // NCHW order: [3, 2, 1, 0]; NHWC order: [1, 3, 2, 0];
    void SetDimOrder(rad::ArrayRef<size_t> order)
    {
        assert(order.size() == m_sizes.size());
        std::vector<size_t> permutation(m_sizes.size());
        size_t dimCount = m_sizes.size();
        for (size_t i = 0; i < m_sizes.size(); ++i)
        {
            permutation[i] = order[dimCount - i - 1];
        }
        PermuteDims(permutation);
    }

    bool NextND(size_t nd)
    {
        size_t dimCount = m_sizes.size();
        for (ptrdiff_t dimIndex = ptrdiff_t(dimCount - nd - 1); dimIndex >= 0; --dimIndex)
        {
            if (m_coord[dimIndex] < m_offsets[dimIndex] + m_sizes[dimIndex] - 1)
            {
                ++m_coord[dimIndex];
                m_bufferIndex += m_strides[dimIndex];
                return true;
            }
            else
            {
                assert(m_coord[dimIndex] >= m_offsets[dimIndex]);
                m_bufferIndex -= (m_coord[dimIndex] - m_offsets[dimIndex]) * m_strides[dimIndex];
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
            if (m_coord[dimIndex] < m_offsets[dimIndex] + m_sizes[dimIndex] - 1)
            {
                ++m_coord[dimIndex];
                m_bufferIndex += m_strides[dimIndex];
                return true;
            }
            else
            {
                assert(m_coord[dimIndex] >= m_offsets[dimIndex]);
                m_bufferIndex -= (m_coord[dimIndex] - m_offsets[dimIndex]) * m_strides[dimIndex];
                m_coord[dimIndex] = m_offsets[dimIndex];
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
