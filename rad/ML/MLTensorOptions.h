#pragma once

#include <rad/ML/MLDataType.h>
#include <rad/Common/String.h>
#include <rad/Container/ArrayRef.h>
#include <vector>

namespace rad
{

struct MLTensorOptions
{
    std::vector<size_t> m_strides;

    MLTensorOptions& SetStrides(ArrayRef<size_t> strides)
    {
        m_strides = strides;
        return *this;
    }
};

} // namespace rad
