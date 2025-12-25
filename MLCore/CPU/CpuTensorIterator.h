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
        size_t dimCount = m_sizes.size();
        do
        {
            // Iterate the last dimension:
            for (size_t i = 0; i < m_sizes[dimCount - 1]; ++i)
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
            for (size_t i = 0; i < m_sizes[m_sizes.size() - 1]; ++i)
            {
                m_coord[m_sizes.size() - 1] = i;
                op(m_coord);
            }
        } while (NextNDSubrangeND(1, subrangeND));
    }

    void ForEachRecursively(const ElementOp& op, size_t dimIndex)
    {
        if (dimIndex == m_sizes.size() - 1)
        {
            // Iterate the last dimension:
            for (size_t i = 0; i < m_sizes[dimIndex]; ++i)
            {
                m_coord[dimIndex] = i;
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

    void ForEachRecursively(const ElementOp& op)
    {
        Reset();
        ForEachRecursively(op, 0);
    }

    // @param granularityND: the number of dimensions processed by each thread (must <dimCount).
    void ForEachParallelND(const ElementOp& op, size_t granularityND)
    {
        if (granularityND >= m_sizes.size())
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

}; // class CpuTensorIterator

} // namespace ML
