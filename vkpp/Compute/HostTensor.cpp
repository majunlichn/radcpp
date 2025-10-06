#include <vkpp/Compute/HostTensor.h>

namespace vkpp
{

std::vector<size_t> MakeTensorStrides(rad::ArrayRef<size_t> sizes, rad::ArrayRef<size_t> memoryOrder, rad::ArrayRef<size_t> alignments)
{
    assert(memoryOrder.empty() || (memoryOrder.size() == sizes.size()));
    assert(alignments.empty() || (alignments.size() == sizes.size()));

    std::vector<size_t> strides(sizes.size(), 0);
    if (memoryOrder.empty())
    {
        strides.back() = 1;
        std::partial_sum(
            sizes.rbegin(), sizes.rend() - 1, strides.rbegin() + 1, std::multiplies<std::size_t>());
    }
    else
    {
        size_t stride = 1;
        for (size_t i = 0; i < sizes.size(); ++i)
        {
            strides[memoryOrder[i]] = stride;
            stride *= sizes[memoryOrder[i]];
        }
    }
    for (size_t i = 0; i < alignments.size(); ++i)
    {
        if (alignments[i] > 0)
        {
            strides[i] = rad::RoundUpToMultiple(strides[i], alignments[i]);
        }
    }
    return strides;
}

} // namespace vkpp
