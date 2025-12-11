#pragma once

#include <rad/ML/MLCpuDevice.h>
#include <rad/ML/MLCpuTensor.h>
#include <rad/ML/MLCpuContext.h>
#include <rad/ML/MLCpuTensorIterator.h>

namespace rad
{

template <typename T>
struct MLCpuTensorOpForEach
{
    void operator()(MLTensor* input, const std::function<T()>& op)
    {
        MLCpuTensor* cpuInput = static_cast<MLCpuTensor*>(input);

        MLCpuTensorIterator inputIter(input);
        T* inputData = (T*)cpuInput->m_buffer.data();
        inputIter.ForEachParallel([&](ArrayRef<size_t> coord) {
            inputData[inputIter.CoordToBufferIndex(coord)] = op();
            });
    }
}; // struct MLCpuTensorOpForEach

template <typename T, typename ComputeType = T>
struct MLCpuTensorOpElementWiseUnary
{
    void operator()(MLTensor* input, MLTensor* output, const std::function<ComputeType(ComputeType x)>& op)
    {
        if (output == nullptr)
        {
            output = input;
        }
        MLCpuTensor* cpuInput = static_cast<MLCpuTensor*>(input);
        MLCpuTensor* cpuOutput = static_cast<MLCpuTensor*>(input);
        assert(cpuInput->m_dataType == cpuOutput->m_dataType);
        assert(cpuInput->m_sizes == cpuOutput->m_sizes);

        MLCpuTensorIterator inputIter(input);
        MLCpuTensorIterator outputIter(output);

        const T* inputData = (const T*)cpuInput->m_buffer.data();
        T* outputData = (T*)cpuOutput->m_buffer.data();

        inputIter.ForEachParallel([&](ArrayRef<size_t> coord) {
            ComputeType x = inputData[inputIter.CoordToBufferIndex(coord)];
            outputData[outputIter.CoordToBufferIndex(coord)] = op(x);
            });
    }
}; // struct MLCpuTensorOpElementWiseUnary

template <typename T, typename ComputeType = T>
struct MLCpuTensorOpElementWiseBinary
{
    void operator()(MLTensor* input, MLTensor* other, MLTensor* output,
        const std::function<ComputeType(ComputeType a, ComputeType b)>& op)
    {
        if (output == nullptr)
        {
            output = input;
        }
        MLCpuTensor* cpuInput = static_cast<MLCpuTensor*>(input);
        MLCpuTensor* cpuOther = static_cast<MLCpuTensor*>(other);
        MLCpuTensor* cpuOutput = static_cast<MLCpuTensor*>(output);
        assert(cpuInput->m_dataType == cpuOther->m_dataType);
        assert(cpuInput->m_dataType == cpuOutput->m_dataType);
        assert(cpuInput->m_sizes == cpuOther->m_sizes);
        assert(cpuInput->m_sizes == cpuOutput->m_sizes);

        MLCpuTensorIterator inputIter(input);
        MLCpuTensorIterator otherIter(other);
        MLCpuTensorIterator outputIter(output);

        const T* inputData = (const T*)cpuInput->m_buffer.data();
        const T* otherData = (const T*)cpuOther->m_buffer.data();
        T* outputData = (T*)cpuOutput->m_buffer.data();

        inputIter.ForEachParallel([&](ArrayRef<size_t> coord) {
            ComputeType a = inputData[inputIter.CoordToBufferIndex(coord)];
            ComputeType b = otherData[otherIter.CoordToBufferIndex(coord)];
            outputData[outputIter.CoordToBufferIndex(coord)] = op(a, b);
            });
    }
}; // struct MLCpuTensorOpElementWiseBinary

} // namespace rad
