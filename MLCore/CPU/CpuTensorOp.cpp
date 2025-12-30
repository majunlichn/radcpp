#include <MLCore/CPU/CpuTensorOp.h>

#include <rad/System/CpuInfo.h>
#include <taskflow/taskflow.hpp>

namespace ML
{

void ForEachParallelND(TensorIterator& iter, const std::function<void(size_t bufferIndex)>& op, size_t granularityND)
{
    if (granularityND >= iter.m_sizes.size())
    {
        return ForEach(iter, op);
    }
    iter.Reset();
    tf::Executor executor;
    size_t threadCount = 0;
    size_t coreCount = rad::GetNumberOfPhysicalCores();
    do {
        executor.silent_async([&, subrangeIter = iter]() mutable {
            ForEachSubrangeND(subrangeIter, op, granularityND);
            });
        threadCount++;
        if (threadCount >= coreCount)
        {
            executor.wait_for_all();    // avoid thread competition
            threadCount = 0;
        }
    } while (iter.NextND(granularityND));
    if (threadCount > 0)
    {
        executor.wait_for_all();
        threadCount = 0;
    }
}

} // namespace ML
