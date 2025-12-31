#include "TensorIterator.h"
#include <MLCore/Device.h>
#include <MLCore/Context.h>
#include <MLCore/Tensor.h>

namespace ML
{

void ForEach(TensorIterator& iter, const std::function<void(size_t bufferIndex)>& op)
{
    iter.Reset();
    do
    {
        // Iterate the last dimension:
        size_t lastDimIndex = iter.m_sizes.size() - 1;
        for (size_t i = 0; i < iter.m_sizes[lastDimIndex]; ++i)
        {
            iter.m_coord[lastDimIndex] = iter.m_offsets[lastDimIndex] + i;
            op(iter.m_bufferIndex + i * iter.m_strides[lastDimIndex]);
        }
    } while (iter.Next1D());
}

void ForEachRecursively(TensorIterator& iter, const std::function<void(size_t bufferIndex)>& op, size_t dimIndex, size_t bufferIndex)
{
    if (dimIndex == iter.m_sizes.size() - 1)
    {
        // Iterate the last dimension:
        for (size_t i = 0; i < iter.m_sizes[dimIndex]; ++i)
        {
            iter.m_coord[dimIndex] = iter.m_offsets[dimIndex] + i;
            op(bufferIndex + iter.m_coord[dimIndex] * iter.m_strides[dimIndex]);
        }
    }
    else
    {
        for (size_t i = 0; i < iter.m_sizes[dimIndex]; ++i)
        {
            iter.m_coord[dimIndex] = iter.m_offsets[dimIndex] + i;
            ForEachRecursively(iter, op, dimIndex + 1, bufferIndex + iter.m_coord[dimIndex] * iter.m_strides[dimIndex]);
        }
    }
}

void ForEachRecursively(TensorIterator& iter, const std::function<void(size_t bufferIndex)>& op)
{
    iter.Reset();
    ForEachRecursively(iter, op, 0, 0);
}

void ForEachSubrangeND(TensorIterator& iter, const std::function<void(size_t bufferIndex)>& op, size_t subrangeND)
{
    iter.ResetND(subrangeND);
    do
    {
        // Iterate the last dimension:
        size_t lastDimIndex = iter.m_sizes.size() - 1;
        for (size_t i = 0; i < iter.m_sizes[lastDimIndex]; ++i)
        {
            iter.m_coord[lastDimIndex] = iter.m_offsets[lastDimIndex] + i;
            op(iter.m_bufferIndex + i * iter.m_strides[lastDimIndex]);
        }
    } while (iter.NextNDSubrangeND(1, subrangeND));
}

void ForEach(TensorIterator& input, TensorIterator& output,
    const std::function<void(size_t inputIndex, size_t outputIndex)>& op)
{
    assert(input.m_sizes == output.m_sizes);
    input.Reset();
    output.Reset();
    do
    {
        // Iterate the last dimension:
        size_t lastDimIndex = input.m_sizes.size() - 1;
        for (size_t i = 0; i < input.m_sizes[lastDimIndex]; ++i)
        {
            input.m_coord[lastDimIndex] = input.m_offsets[lastDimIndex] + i;
            output.m_coord[lastDimIndex] = output.m_offsets[lastDimIndex] + i;
            size_t inputIndex = input.m_bufferIndex + input.m_coord[lastDimIndex] * input.m_strides[lastDimIndex];
            size_t outputIndex = output.m_bufferIndex + output.m_coord[lastDimIndex] * output.m_strides[lastDimIndex];
            op(inputIndex, outputIndex);
        }
    } while (input.Next1D() && output.Next1D());
}

void ForEach(TensorIterator& input, TensorIterator& other, TensorIterator& output,
    const std::function<void(size_t inputIndex, size_t otherIndex, size_t outputIndex)>& op)
{
    assert(input.m_sizes == other.m_sizes);
    assert(input.m_sizes == output.m_sizes);
    input.Reset();
    other.Reset();
    output.Reset();
    do
    {
        // Iterate the last dimension:
        size_t lastDimIndex = input.m_sizes.size() - 1;
        for (size_t i = 0; i < input.m_sizes[lastDimIndex]; ++i)
        {
            input.m_coord[lastDimIndex] = input.m_offsets[lastDimIndex] + i;
            other.m_coord[lastDimIndex] = other.m_offsets[lastDimIndex] + i;
            output.m_coord[lastDimIndex] = output.m_offsets[lastDimIndex] + i;
            size_t inputIndex = input.m_bufferIndex + input.m_coord[lastDimIndex] * input.m_strides[lastDimIndex];
            size_t otherIndex = other.m_bufferIndex + other.m_coord[lastDimIndex] * other.m_strides[lastDimIndex];
            size_t outputIndex = output.m_bufferIndex + output.m_coord[lastDimIndex] * output.m_strides[lastDimIndex];
            op(inputIndex, otherIndex, outputIndex);
        }
    } while (input.Next1D() && other.Next1D() && output.Next1D());
}

} // namespace ML
