#include <MLCore/TensorView.h>
#include <MLCore/Device.h>
#include <MLCore/Context.h>

namespace ML
{

TensorView::TensorView(rad::Ref<Tensor> tensor,
    rad::ArrayRef<size_t> offsets, rad::ArrayRef<size_t> sizes, rad::ArrayRef<size_t> strides) :
    m_tensor(std::move(tensor))
{
    if (offsets.empty())
    {
        m_offsets.resize(m_tensor->m_sizes.size(), 0);
    }
    else
    {
        assert(offsets.size() == m_tensor->GetDimCount());
        m_offsets = offsets;
    }

    if (sizes.empty())
    {
        m_sizes.resize(m_tensor->GetDimCount());
        for (size_t i = 0; i < m_tensor->GetDimCount(); ++i)
        {
            m_sizes[i] = m_tensor->m_sizes[i] - m_offsets[i];
        }
    }
    else
    {
        assert(sizes.size() == m_tensor->GetDimCount());
        m_sizes = sizes;
    }

    if (strides.empty())
    {
        m_strides = m_tensor->m_strides;
    }
    else
    {
        assert(strides.size() == m_tensor->m_sizes.size());
        m_strides = strides;
    }

    assert(IsValid());
}

TensorView::~TensorView()
{
}

void TensorView::SetOffsets(rad::ArrayRef<size_t> offsets)
{
    assert(offsets.size() == m_tensor->m_sizes.size());
    m_offsets = offsets;
}

void TensorView::SetSizes(rad::ArrayRef<size_t> sizes)
{
    assert(sizes.size() == m_tensor->m_sizes.size());
    m_sizes = sizes;
}

void TensorView::SetStrides(rad::ArrayRef<size_t> strides)
{
    assert(strides.size() == m_tensor->m_sizes.size());
    m_strides = strides;
}

bool TensorView::IsValid() const
{
    for (size_t i = 0; i < m_sizes.size(); ++i)
    {
        if ((m_offsets[i] + m_sizes[i]) > m_tensor->m_sizes[i])
        {
            return false;
        }
    }
    return true;
}

size_t TensorView::GetElementCount() const
{
    return Tensor::GetElementCount(m_sizes);
}

size_t TensorView::GetElementCountND(size_t ndim) const
{
    return Tensor::GetElementCountND(m_sizes, ndim);
}

size_t TensorView::GetDataSizeInElement() const
{
    return Tensor::GetDataSizeInElement(m_sizes, m_strides);
}

bool TensorView::IsContiguous() const
{
    return (GetElementCount() == GetDataSizeInElement());
}

} // namespace ML
