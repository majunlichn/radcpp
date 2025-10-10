#pragma once

#include <vkpp/Core/Device.h>
#include <vkpp/Core/Buffer.h>

#include <functional>

#include <taskflow/taskflow.hpp>

namespace vkpp
{

class TensorIterator
{
public:
    std::vector<size_t> m_sizes;
    std::vector<size_t> m_coords;
    std::vector<size_t> m_permutation;

    TensorIterator(rad::ArrayRef<size_t> sizes) :
        m_sizes(sizes)
    {
        Reset();
    }

    ~TensorIterator() = default;

    void Reset()
    {
        m_coords.clear();
        m_coords.resize(m_sizes.size(), 0);
    }

    void ResetND(size_t n)
    {
        size_t dimCount = m_sizes.size();
        assert(dimCount >= n);
        std::fill_n(m_coords.end() - n, n, 0);
    }

    void Reset1D() { ResetND(1); }
    void Reset2D() { ResetND(2); }
    void Reset3D() { ResetND(3); }
    void Reset4D() { ResetND(4); }

    void ResetNDPermuted(size_t n)
    {
        size_t dimCount = m_sizes.size();
        assert(dimCount >= n);
        for (size_t i = 0; i < n; ++i)
        {
            size_t dimIndex = m_permutation[i];
            m_coords[dimIndex] = 0;
        }
    }

    bool NextND(size_t n)
    {
        size_t dimCount = m_sizes.size();
        assert(dimCount > n);
        for (ptrdiff_t dimIndex = ptrdiff_t(dimCount - n - 1); dimIndex >= ptrdiff_t(0); --dimIndex)
        {
            if (m_coords[dimIndex] < m_sizes[dimIndex] - 1)
            {
                ++m_coords[dimIndex];
                return true;
            }
            else
            {
                m_coords[dimIndex] = 0;
            }
        }
        return false;
    }

    bool IterateSubrangeND_NextND(size_t subrangeND, size_t n)
    {
        assert(n < subrangeND);
        for (ptrdiff_t dimIndex = ptrdiff_t(m_sizes.size() - n - 1); dimIndex >= ptrdiff_t(m_sizes.size() - subrangeND); --dimIndex)
        {
            if (m_coords[dimIndex] < m_sizes[dimIndex] - 1)
            {
                ++m_coords[dimIndex];
                return true;
            }
            else
            {
                m_coords[dimIndex] = 0;
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
        size_t dimCount = m_sizes.size();
        assert(dimCount > n);
        for (size_t i = n; i < dimCount; ++i)
        {
            size_t dimIndex = m_permutation[i];
            if (m_coords[dimIndex] < m_sizes[dimIndex] - 1)
            {
                ++m_coords[dimIndex];
                return true;
            }
            else
            {
                m_coords[dimIndex] = 0;
            }
        }
        return false;
    }

    bool Next1DPermuted() { return NextNDPermuted(1); }
    bool Next2DPermuted() { return NextNDPermuted(2); }
    bool Next3DPermuted() { return NextNDPermuted(3); }
    bool Next4DPermuted() { return NextNDPermuted(4); }

    bool IterateSubrangeND_NextNDPermuted(size_t subrangeND, size_t n)
    {
        assert(n < subrangeND);
        for (size_t i = n; i < subrangeND; ++i)
        {
            size_t dimIndex = m_permutation[i];
            if (m_coords[dimIndex] < m_sizes[dimIndex] - 1)
            {
                ++m_coords[dimIndex];
                return true;
            }
            else
            {
                m_coords[dimIndex] = 0;
            }
        }
        return false;
    }

    using ElementWiseOp = std::function<void(rad::ArrayRef<size_t> coords)>;

    void ForEach(const ElementWiseOp& op)
    {
        if (m_sizes.size() == 1)
        {
            for (size_t w = 0; w < m_sizes[0]; ++w)
            {
                m_coords[0] = w;
                op(m_coords);
            }
        }
        else if (m_sizes.size() == 2)
        {
            for (size_t h = 0; h < m_sizes[0]; ++h)
            {
                m_coords[0] = h;
                for (size_t w = 0; w < m_sizes[1]; ++w)
                {
                    m_coords[1] = w;
                    op(m_coords);
                }
            }
        }
        else if (m_sizes.size() == 3)
        {
            for (size_t c = 0; c < m_sizes[0]; ++c)
            {
                m_coords[0] = c;
                for (size_t h = 0; h < m_sizes[1]; ++h)
                {
                    m_coords[1] = h;
                    for (size_t w = 0; w < m_sizes[2]; ++w)
                    {
                        m_coords[2] = w;
                        op(m_coords);
                    }
                }
            }
        }
        else if (m_sizes.size() == 4)
        {
            for (size_t n = 0; n < m_sizes[0]; ++n)
            {
                m_coords[0] = n;
                for (size_t c = 0; c < m_sizes[1]; ++c)
                {
                    m_coords[1] = c;
                    for (size_t h = 0; h < m_sizes[2]; ++h)
                    {
                        m_coords[2] = h;
                        for (size_t w = 0; w < m_sizes[3]; ++w)
                        {
                            m_coords[3] = w;
                            op(m_coords);
                        }
                    }
                }
            }
        }
        else if (m_sizes.size() == 5)
        {
            for (size_t n = 0; n < m_sizes[0]; ++n)
            {
                m_coords[0] = n;
                for (size_t c = 0; c < m_sizes[1]; ++c)
                {
                    m_coords[1] = c;
                    for (size_t d = 0; d < m_sizes[2]; ++d)
                    {
                        m_coords[2] = d;
                        for (size_t h = 0; h < m_sizes[3]; ++h)
                        {
                            m_coords[3] = h;
                            for (size_t w = 0; w < m_sizes[4]; ++w)
                            {
                                m_coords[4] = w;
                                op(m_coords);
                            }
                        }
                    }
                }
            }
        }
        else
        {
            Reset();
            size_t dimCount = m_sizes.size();
            do
            {
                // Iterate the last dimension:
                for (size_t i = 0; i < m_sizes[dimCount - 1]; ++i)
                {
                    m_coords[dimCount - 1] = i;
                    op(m_coords);
                }
            } while ((dimCount > 1) && Next1D());
        }
    }

    void ForEachSubrangeND(size_t subrangeND, const ElementWiseOp& op)
    {
        std::span<size_t> subSizes = { m_sizes.end() - subrangeND, subrangeND };
        std::span<size_t> subCoords = { m_coords.end() - subrangeND, subrangeND };
        if (subrangeND == 1)
        {
            for (size_t w = 0; w < subSizes[0]; ++w)
            {
                subCoords[0] = w;
                op(m_coords);
            }
        }
        else if (subrangeND == 2)
        {
            for (size_t h = 0; h < subSizes[0]; ++h)
            {
                subCoords[0] = h;
                for (size_t w = 0; w < subSizes[1]; ++w)
                {
                    subCoords[1] = w;
                    op(m_coords);
                }
            }
        }
        else if (subrangeND == 3)
        {
            for (size_t c = 0; c < subSizes[0]; ++c)
            {
                subCoords[0] = c;
                for (size_t h = 0; h < subSizes[1]; ++h)
                {
                    subCoords[1] = h;
                    for (size_t w = 0; w < subSizes[2]; ++w)
                    {
                        subCoords[2] = w;
                        op(m_coords);
                    }
                }
            }
        }
        else if (subrangeND == 4)
        {
            for (size_t n = 0; n < subSizes[0]; ++n)
            {
                subCoords[0] = n;
                for (size_t c = 0; c < subSizes[1]; ++c)
                {
                    subCoords[1] = c;
                    for (size_t h = 0; h < subSizes[2]; ++h)
                    {
                        subCoords[2] = h;
                        for (size_t w = 0; w < subSizes[3]; ++w)
                        {
                            subCoords[3] = w;
                            op(m_coords);
                        }
                    }
                }
            }
        }
        else
        {
            Reset();
            do
            {
                // Iterate the last dimension:
                for (size_t i = 0; i < m_sizes[m_sizes.size() - 1]; ++i)
                {
                    m_coords[m_sizes.size() - 1] = i;
                    op(m_coords);
                }
            } while ((subrangeND > 1) && IterateSubrangeND_NextND(subrangeND, 1));
        }
    }

    void ForEachPermuted(const ElementWiseOp& op)
    {
        Reset();
        assert(m_permutation.size() == m_sizes.size());
        size_t dimCount = m_sizes.size();
        do
        {
            size_t dimIndexPermuted = m_permutation[0];
            for (size_t i = 0; i < m_sizes[dimIndexPermuted]; ++i)
            {
                m_coords[dimIndexPermuted] = i;
                op(m_coords);
            }
        } while ((dimCount > 1) && NextNDPermuted(1));
    }

    void ForEachSubrangeNDPermuted(size_t subrangeND, const ElementWiseOp& op)
    {
        Reset();
        assert(m_permutation.size() == m_sizes.size());
        size_t dimCount = m_sizes.size();
        do
        {
            size_t dimIndexPermuted = m_permutation[0];
            for (size_t i = 0; i < m_sizes[dimIndexPermuted]; ++i)
            {
                m_coords[dimIndexPermuted] = i;
                op(m_coords);
            }
        } while ((dimCount > 1) && IterateSubrangeND_NextNDPermuted(subrangeND, 1));
    }

    void ForEachRecursively(const ElementWiseOp& op, size_t dimIndex)
    {
        if (dimIndex == m_sizes.size() - 1)
        {
            // Iterate the last dimension:
            for (size_t i = 0; i < m_sizes[dimIndex]; ++i)
            {
                m_coords[dimIndex] = i;
                op(m_coords);
            }
        }
        else
        {
            for (size_t i = 0; i < m_sizes[dimIndex]; ++i)
            {
                m_coords[dimIndex] = i;
                ForEachRecursively(op, dimIndex + 1);
            }
        }
    }

    void ForEachRecursively(const ElementWiseOp& op)
    {
        Reset();
        ForEachRecursively(op, 0);
    }

    void ForEachRecursivelyPermuted(const ElementWiseOp& op, size_t dimIndex)
    {
        size_t dimCount = m_sizes.size();
        size_t dimIndexPermuted = m_permutation[dimCount - dimIndex - 1];
        if (dimIndex == m_sizes.size() - 1)
        {
            // Iterate the last dimension:
            for (size_t i = 0; i < m_sizes[dimIndexPermuted]; ++i)
            {
                m_coords[dimIndexPermuted] = i;
                op(m_coords);
            }
        }
        else
        {
            for (size_t i = 0; i < m_sizes[dimIndexPermuted]; ++i)
            {
                m_coords[dimIndexPermuted] = i;
                ForEachRecursivelyPermuted(op, dimIndex + 1);
            }
        }
    }

    void ForEachRecursivelyPermuted(const ElementWiseOp& op)
    {
        assert(m_permutation.size() == m_sizes.size());
        Reset();
        ForEachRecursivelyPermuted(op, 0);
    }

    // @param dimCountPerThread: the parallel granularity, which is the number of dimensions processed by each thread (must <dimCount).
    void ForEachParallel(const ElementWiseOp& op, size_t dimCountPerThread)
    {
        if (dimCountPerThread >= m_sizes.size())
        {
            return ForEach(op);
        }
        Reset();
        tf::Executor executor;
        do {
            ResetND(dimCountPerThread);
            executor.silent_async([&, iter = *this]() mutable {
                iter.ForEachSubrangeND(m_sizes.size() - dimCountPerThread, op);
                });
        } while (NextND(dimCountPerThread));
        executor.wait_for_all();
    }

    void ForEachParallel(const ElementWiseOp& op)
    {
        size_t elementCountPerThread = 1;
        size_t dimCountPerThread = 0;
        while (dimCountPerThread < m_sizes.size())
        {
            elementCountPerThread *= m_sizes[m_sizes.size() - dimCountPerThread - 1];
            ++dimCountPerThread;
            if (elementCountPerThread >= 64 * 64)
            {
                break;
            }
        }
        ForEachParallel(op, dimCountPerThread);
    }

    // @param dimCountPerThread: the parallel granularity, which is the number of dimensions processed by each thread (must <dimCount).
    void ForEachParallelPermuted(const ElementWiseOp& op, size_t dimCountPerThread)
    {
        assert(m_permutation.size() == m_sizes.size());
        if (dimCountPerThread >= m_sizes.size())
        {
            return ForEachPermuted(op);
        }
        Reset();
        tf::Executor executor;
        do {
            ResetNDPermuted(dimCountPerThread);
            executor.silent_async([&, iter = *this]() mutable {
                iter.ForEachSubrangeNDPermuted(m_sizes.size() - dimCountPerThread, op);
                });
        } while (NextNDPermuted(dimCountPerThread));
        executor.wait_for_all();
    }

    void ForEachParallelPermuted(const ElementWiseOp& op)
    {
        size_t elementCountPerThread = 1;
        size_t dimIndex = 0;
        for (; dimIndex < m_sizes.size(); ++dimIndex)
        {
            elementCountPerThread *= m_sizes[m_permutation[dimIndex]];
            if (elementCountPerThread >= 64 * 64)
            {
                break;
            }
        }
        ForEachParallelPermuted(op, dimIndex + 1);
    }

}; // class TensorIterator

} // namespace vkpp
