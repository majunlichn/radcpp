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
        if (m_permutation.empty())
        {
            std::fill_n(m_coords.end() - n, n, 0);
        }
        else
        {
            for (size_t i = 0; i < n; ++i)
            {
                size_t dimIndex = m_permutation[i];
                m_coords[dimIndex] = 0;
            }
        }
    }

    void Reset1D() { ResetND(1); }
    void Reset2D() { ResetND(2); }
    void Reset3D() { ResetND(3); }
    void Reset4D() { ResetND(4); }

    bool NextND(size_t n)
    {
        size_t dimCount = m_sizes.size();
        assert(dimCount > n);
        if (m_permutation.empty())
        {
            for (ptrdiff_t dimIndex = dimCount - ptrdiff_t(n) - 1; dimIndex >= 0; --dimIndex)
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
        else
        {
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
    }

    bool Next1D() { return NextND(1); }
    bool Next2D() { return NextND(2); }
    bool Next3D() { return NextND(3); }
    bool Next4D() { return NextND(4); }

    using ElementWiseOp = std::function<void(rad::ArrayRef<size_t> coords)>;

    void ForEach(const ElementWiseOp& op)
    {
        Reset();
        if (m_permutation.empty())
        {
            // Iterate the last dimension:
            size_t dimCount = m_sizes.size();
            do
            {
                for (size_t i = 0; i < m_sizes[dimCount - 1]; ++i)
                {
                    m_coords[dimCount - 1] = i;
                    op(m_coords);
                }
            } while ((dimCount > 1) && Next1D());
        }
        else
        {
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
            } while ((dimCount > 1) && Next1D());
        }
    }

    void ForEachRecursively(const ElementWiseOp& op, size_t dimIndex)
    {
        if (m_permutation.empty())
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
        else
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
                    ForEachRecursively(op, dimIndex + 1);
                }
            }
        }
    }

    void ForEachRecursively(const ElementWiseOp& op)
    {
        Reset();
        ForEachRecursively(op, 0);
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
                //iter.ForEachRecursively(op, m_sizes.size() - dimCountPerThread);
                iter.m_dimCount = dimCountPerThread;
                iter.ForEach(op);
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

}; // class TensorIterator

} // namespace vkpp
