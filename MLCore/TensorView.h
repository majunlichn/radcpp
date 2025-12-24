#pragma once

#include <MLCore/Tensor.h>

namespace ML
{

class TensorView
{
public:
    rad::Ref<Tensor> m_tensor;
    std::vector<size_t> m_offsets;
    std::vector<size_t> m_sizes;
    std::vector<size_t> m_strides;

    TensorView(rad::Ref<Tensor> tensor,
        rad::ArrayRef<size_t> offsets = {}, rad::ArrayRef<size_t> sizes = {}, rad::ArrayRef<size_t> strides = {});
    ~TensorView();

    void SetOffsets(rad::ArrayRef<size_t> offsets);
    void SetSizes(rad::ArrayRef<size_t> sizes);
    void SetStrides(rad::ArrayRef<size_t> strides);
    bool IsValid() const;

    size_t GetElementCount() const;
    size_t GetElementCountND(size_t ndim) const;
    size_t GetDataSizeInElement() const;

    bool IsContiguous() const;

}; // class TensorView

} // namespace ML
