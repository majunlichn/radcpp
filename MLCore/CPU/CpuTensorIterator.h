#pragma once

#include <MLCore/TensorIterator.h>

#include <rad/System/CpuInfo.h>
#include <taskflow/taskflow.hpp>


namespace ML
{

// A helper class to calculate coordinates to iterate over tensor elements, support different iteration orders (permutations).
// For example, for a 4D tensor, order={ 1, 3, 2, 0 } means to iterate in the order of C, W, H, N.
class CpuTensorIterator : public TensorIterator
{
public:
    CpuTensorIterator(Tensor* tensor) :
        TensorIterator(tensor)
    {
    }

    ~CpuTensorIterator() = default;

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
    void ForEachParallelND(const ElementOp& op, size_t granularityND)
    {
        if (granularityND >= m_tensor->m_sizes.size())
        {
            return ForEach(op);
        }
        Reset();
        tf::Executor executor;
        size_t threadCount = 0;
        size_t coreCount = rad::GetNumberOfPhysicalCores();
        do {
            executor.silent_async([&, iter = *this]() mutable {
                iter.ForEachSubrangeND(op, granularityND);
                });
            threadCount++;
            if (threadCount >= coreCount)
            {
                executor.wait_for_all();    // avoid thread competition
                threadCount = 0;
            }
        } while (NextND(granularityND));
        if (threadCount > 0)
        {
            executor.wait_for_all();
            threadCount = 0;
        }
    }

    void ForEachParallel(const ElementOp& op)
    {
        if (m_tensor->GetElementCount() < 1024)
        {
            ForEach(op);
            return;
        }
        // TODO: better parallel partitioning
        ForEachParallelND(op, 2);
    }

    // @param granularityND: the number of dimensions processed by each thread (must <dimCount).
    void ForEachParallelPermutedND(const ElementOp& op, size_t granularityND)
    {
        size_t dimCount = m_tensor->m_sizes.size();
        assert(m_order.size() == dimCount);
        if (granularityND >= dimCount)
        {
            return ForEachPermuted(op);
        }
        Reset();
        tf::Executor executor;
        size_t threadCount = 0;
        size_t coreCount = rad::GetNumberOfPhysicalCores();
        do {
            executor.silent_async([&, iter = *this]() mutable {
                iter.ForEachSubrangeNDPermuted(op, granularityND);
                });
            threadCount++;
            if (threadCount >= coreCount)
            {
                executor.wait_for_all();    // avoid thread competition
                threadCount = 0;
            }
        } while (NextNDPermuted(granularityND));
        if (threadCount > 0)
        {
            executor.wait_for_all();
            threadCount = 0;
        }
    }

    void ForEachParallelPermuted(const ElementOp& op)
    {
        if (m_tensor->GetElementCount() < 1024)
        {
            ForEachPermuted(op);
            return;
        }
        // TODO: better parallel partitioning
        ForEachParallelPermutedND(op, 2);
    }

}; // class CpuTensorIterator

} // namespace ML
