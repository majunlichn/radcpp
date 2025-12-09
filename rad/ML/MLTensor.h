#pragma once

#include <rad/ML/MLDataType.h>
#include <rad/Common/Algorithm.h>
#include <rad/Common/Memory.h>
#include <rad/Common/RefCounted.h>
#include <rad/Container/Span.h>

namespace rad
{

class MLTensor : public RefCounted<MLTensor>
{
public:
    MLTensor() = default;
    virtual ~MLTensor() = default;

    static std::vector<size_t> MakeStrides(ArrayRef<size_t> sizes, ArrayRef<size_t> memoryOrder = {});

    size_t GetRank() const { return m_sizes.size(); }
    size_t GetElementCount() const;
    std::vector<size_t> GetMemoryOrder() const;

    size_t CoordToBufferIndex(rad::ArrayRef<size_t> coord)
    {
        assert(coord.size() == m_strides.size());
        return std::inner_product(coord.begin(), coord.end(), m_strides.begin(), size_t(0));
    }

    size_t CoordToBufferOffset(rad::ArrayRef<size_t> coord)
    {
        return CoordToBufferIndex(coord) * GetElementSize(m_dataType);
    }

    MLDataType m_dataType = MLDataType::Unknown;
    std::vector<size_t> m_sizes;
    std::vector<size_t> m_strides;

}; // class MLTensor

} // namespace rad
