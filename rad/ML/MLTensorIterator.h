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

    size_t CoordToBufferOffset(ArrayRef<size_t> coord)
    {
        return m_tensor->CoordToBufferOffset(coord);
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

    using ElementOp = std::function<void(rad::ArrayRef<size_t> coord)>;

    void ForEach(const ElementOp& op)
    {
        Reset();
        size_t dimCount = m_tensor->m_sizes.size();
        do
        {
            // Iterate the last dimension:
            for (size_t i = 0; i < m_tensor->m_sizes[dimCount - 1]; ++i)
            {
                m_coord[dimCount - 1] = i;
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
            for (size_t i = 0; i < m_tensor->m_sizes[m_tensor->m_sizes.size() - 1]; ++i)
            {
                m_coord[m_tensor->m_sizes.size() - 1] = i;
                op(m_coord);
            }
        } while (NextNDSubrangeND(1, subrangeND));
    }

    void ForEachPermuted(const ElementOp& op)
    {
        Reset();
        assert(m_order.size() == m_tensor->m_sizes.size());
        size_t dimCount = m_tensor->m_sizes.size();
        do
        {
            size_t dimIndex = m_order[0];
            for (size_t i = 0; i < m_tensor->m_sizes[dimIndex]; ++i)
            {
                m_coord[dimIndex] = i;
                op(m_coord);
            }
        } while ((dimCount > 1) && NextNDPermuted(1));
    }

    void ForEachSubrangeNDPermuted(const ElementOp& op, size_t subrangeND)
    {
        ResetNDPermuted(subrangeND);
        assert(m_order.size() == m_tensor->m_sizes.size());
        size_t dimCount = m_tensor->m_sizes.size();
        do
        {
            size_t dimIndex = m_order[0];
            for (size_t i = 0; i < m_tensor->m_sizes[dimIndex]; ++i)
            {
                m_coord[dimIndex] = i;
                op(m_coord);
            }
        } while ((dimCount > 1) && NextNDPermutedSubrangeND(1, subrangeND));
    }

    void ForEachRecursively(const ElementOp& op, size_t dimIndex)
    {
        if (dimIndex == m_tensor->m_sizes.size() - 1)
        {
            // Iterate the last dimension:
            for (size_t i = 0; i < m_tensor->m_sizes[dimIndex]; ++i)
            {
                m_coord[dimIndex] = i;
                op(m_coord);
            }
        }
        else
        {
            for (size_t i = 0; i < m_tensor->m_sizes[dimIndex]; ++i)
            {
                m_coord[dimIndex] = i;
                ForEachRecursively(op, dimIndex + 1);
            }
        }
    }

    void ForEachRecursively(const ElementOp& op)
    {
        Reset();
        ForEachRecursively(op, 0);
    }

    void ForEachRecursivelyPermuted(const ElementOp& op, size_t dimIndex)
    {
        size_t dimCount = m_tensor->m_sizes.size();
        size_t dimIndexPermuted = m_order[dimCount - dimIndex - 1];
        if (dimIndex == m_tensor->m_sizes.size() - 1)
        {
            // Iterate the last dimension:
            for (size_t i = 0; i < m_tensor->m_sizes[dimIndexPermuted]; ++i)
            {
                m_coord[dimIndexPermuted] = i;
                op(m_coord);
            }
        }
        else
        {
            for (size_t i = 0; i < m_tensor->m_sizes[dimIndexPermuted]; ++i)
            {
                m_coord[dimIndexPermuted] = i;
                ForEachRecursivelyPermuted(op, dimIndex + 1);
            }
        }
    }

    void ForEachRecursivelyPermuted(const ElementOp& op)
    {
        assert(m_order.size() == m_tensor->m_sizes.size());
        Reset();
        ForEachRecursivelyPermuted(op, 0);
    }

    // @param granularityND: the number of dimensions processed by each thread (must <dimCount).
    void ForEachParallel(const ElementOp& op, size_t granularityND)
    {
        if (granularityND >= m_tensor->m_sizes.size())
        {
            return ForEach(op);
        }
        Reset();
        tf::Executor executor;
        do {
            executor.silent_async([&, iter = *this]() mutable {
                iter.ForEachSubrangeND(op, granularityND);
                });
        } while (NextND(granularityND));
        executor.wait_for_all();
    }

    void ForEachParallel(const ElementOp& op)
    {
        if (m_tensor->GetElementCount() < 1024)
        {
            ForEach(op);
            return;
        }
        const auto& sizes = m_tensor->m_sizes;
        size_t threadCount = 1;
        size_t granularityND = sizes.size();
        for (size_t i = 0; i < sizes.size() - 1; ++i)
        {
            threadCount *= sizes[i];
            if (threadCount >= std::thread::hardware_concurrency() / 2)
            {
                granularityND = sizes.size() - i - 1;
                break;
            }
        }
        ForEachParallel(op, granularityND);
    }

    // @param granularityND: the number of dimensions processed by each thread (must <dimCount).
    void ForEachParallelPermuted(const ElementOp& op, size_t granularityND)
    {
        size_t dimCount = m_tensor->m_sizes.size();
        assert(m_order.size() == dimCount);
        if (granularityND >= dimCount)
        {
            return ForEachPermuted(op);
        }
        Reset();
        tf::Executor executor;
        do {
            executor.silent_async([&, iter = *this]() mutable {
                iter.ForEachSubrangeNDPermuted(op, granularityND);
                });
        } while (NextNDPermuted(granularityND));
        executor.wait_for_all();
    }

    void ForEachParallelPermuted(const ElementOp& op)
    {
        if (m_tensor->GetElementCount() < 1024)
        {
            ForEachPermuted(op);
            return;
        }
        const auto& sizes = m_tensor->m_sizes;
        size_t threadCount = 1;
        size_t granularityND = sizes.size();
        for (size_t i = sizes.size() - 1; i > 0; --i)
        {
            threadCount *= sizes[m_order[i]];
            if (threadCount >= std::thread::hardware_concurrency() / 2)
            {
                granularityND = i;
                break;
            }
        }
        ForEachParallelPermuted(op, granularityND);
    }

}; // class MLTensorIterator

} // namespace rad
