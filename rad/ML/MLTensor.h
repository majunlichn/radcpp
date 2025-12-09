#pragma once

#include <rad/ML/MLDataType.h>
#include <rad/Common/Algorithm.h>
#include <rad/Common/Memory.h>
#include <rad/Common/RefCounted.h>
#include <rad/Container/ArrayRef.h>
#include <rad/Container/SmallVector.h>
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

    size_t CoordToBufferIndex(ArrayRef<size_t> coord)
    {
        assert(coord.size() == m_strides.size());
        return std::inner_product(coord.begin(), coord.end(), m_strides.begin(), size_t(0));
    }

    size_t CoordToBufferOffset(ArrayRef<size_t> coord)
    {
        return CoordToBufferIndex(coord) * GetElementSize(m_dataType);
    }

    bool IsNCHW() const
    {
        if ((m_sizes.size() == 4) &&
            (m_strides[0] > m_strides[1]) &&
            (m_strides[1] > m_strides[2]) &&
            (m_strides[2] > m_strides[3]) &&
            (m_strides[3] == 1))
        {
            return true;
        }
        return false;
    }

    bool IsNHWC() const
    {
        if ((m_sizes.size() == 4) &&
            (m_strides[0] > m_strides[2]) &&
            (m_strides[2] > m_strides[3]) &&
            (m_strides[3] > m_strides[1]) &&
            (m_strides[1] == 1))
        {
            return true;
        }
        return false;
    }

    bool IsNCDHW() const
    {
        if ((m_sizes.size() == 5) &&
            (m_strides[0] > m_strides[1]) &&
            (m_strides[1] > m_strides[2]) &&
            (m_strides[2] > m_strides[3]) &&
            (m_strides[3] > m_strides[4]) &&
            (m_strides[4] == 1))
        {
            return true;
        }
        return false;
    }

    bool IsNDHWC() const
    {
        if ((m_sizes.size() == 5) &&
            (m_strides[0] > m_strides[2]) &&
            (m_strides[2] > m_strides[3]) &&
            (m_strides[3] > m_strides[4]) &&
            (m_strides[4] > m_strides[1]) &&
            (m_strides[1] == 1))
        {
            return true;
        }
        return false;
    }

    MLDataType m_dataType = MLDataType::Unknown;
    std::vector<size_t> m_sizes;
    std::vector<size_t> m_strides;

}; // class MLTensor

} // namespace rad
