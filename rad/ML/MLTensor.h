#pragma once

#include <rad/ML/MLDataType.h>
#include <rad/Common/Memory.h>
#include <rad/Common/RefCounted.h>
#include <rad/Container/Span.h>

namespace rad
{

using MLTensorCoord = std::vector<size_t>;

class MLTensor : public RefCounted<MLTensor>
{
public:
    MLTensor() = default;
    virtual ~MLTensor() = default;

    static MLTensorCoord MakeStrides(const MLTensorCoord& sizes, ArrayRef<size_t> memoryOrder = {});
    size_t CalculateBufferSize();

    MLTensorCoord m_sizes;
    MLTensorCoord m_strides;
    MLDataType m_dataType;

}; // class MLTensor

} // namespace rad
