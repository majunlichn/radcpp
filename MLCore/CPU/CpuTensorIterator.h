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

}; // class CpuTensorIterator

} // namespace ML
