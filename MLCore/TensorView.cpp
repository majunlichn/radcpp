#include <MLCore/TensorView.h>
#include <MLCore/Device.h>
#include <MLCore/Context.h>

namespace ML
{

TensorView::TensorView(rad::Ref<Tensor> tensor,
    rad::ArrayRef<size_t> offsets, rad::ArrayRef<size_t> sizes, rad::ArrayRef<size_t> strides) :
    m_tensor(std::move(tensor))
{
    if (!offsets.empty())
    {
        assert(offsets.size() == m_tensor->m_sizes.size());
        m_offsets = offsets;
    }
    else
    {
        m_offsets.resize(m_tensor->m_sizes.size(), 0);
    }

    if (!sizes.empty())
    {
        assert(sizes.size() == m_tensor->m_sizes.size());
        m_sizes = sizes;
    }
    else
    {
        m_sizes = m_tensor->m_sizes;
    }

    if (!strides.empty())
    {
        assert(strides.size() == m_tensor->m_sizes.size());
        m_strides = strides;
    }
    else
    {
        m_strides = m_tensor->m_strides;
    }

    for (size_t i = 0; i < m_sizes.size(); ++i)
    {
        assert((m_offsets[i] + m_sizes[i]) <= m_tensor->m_sizes[i]);
    }
}

TensorView::~TensorView()
{
}

void TensorView::SetOffsets(rad::ArrayRef<size_t> offsets)
{
    if (!offsets.empty())
    {
        assert(offsets.size() == m_tensor->m_sizes.size());
        m_offsets = offsets;
    }
    else
    {
        m_offsets.resize(m_tensor->m_sizes.size(), 0);
    }
}

void TensorView::SetSizes(rad::ArrayRef<size_t> sizes)
{
    if (!sizes.empty())
    {
        assert(sizes.size() == m_tensor->m_sizes.size());
        m_sizes = sizes;
    }
    else
    {
        m_sizes = m_tensor->m_sizes;
    }
}

void TensorView::SetStrides(rad::ArrayRef<size_t> strides)
{
    if (!strides.empty())
    {
        assert(strides.size() == m_tensor->m_sizes.size());
        m_strides = strides;
    }
    else
    {
        m_strides = m_tensor->m_strides;
    }
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

size_t TensorView::GetDataSizeInElement() const
{
    return Tensor::GetDataSizeInElement(m_sizes, m_strides);
}

bool TensorView::IsContiguous() const
{
    return (GetElementCount() == GetDataSizeInElement());
}

} // namespace ML
