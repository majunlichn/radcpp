#pragma once

#include <MLCore/CPU/CpuDevice.h>
#include <MLCore/CPU/CpuTensor.h>
#include <MLCore/CPU/CpuContext.h>
#include <MLCore/CPU/CpuTensorIterator.h>

namespace ML
{

template <typename T>
struct CpuTensorOpForEach
{
    void operator()(Tensor* input, const std::function<T()>& op)
    {
        CpuTensor* cpuInput = static_cast<CpuTensor*>(input);

        CpuTensorIterator inputIter(input);
        T* inputData = (T*)cpuInput->m_buffer.data();
        inputIter.ForEachParallel([&](rad::ArrayRef<size_t> coord) {
            inputData[inputIter.CoordToBufferIndex(coord)] = op();
            });
    }

}; // struct CpuTensorOpForEach

template <typename T, typename ComputeType = T>
struct CpuTensorOpElementWiseUnary
{
    void operator()(Tensor* input, Tensor* output, const std::function<ComputeType(ComputeType x)>& op)
    {
        if (output == nullptr)
        {
            output = input;
        }
        CpuTensor* cpuInput = static_cast<CpuTensor*>(input);
        CpuTensor* cpuOutput = static_cast<CpuTensor*>(input);
        assert(cpuInput->m_sizes == cpuOutput->m_sizes);
        assert(cpuInput->m_dataType == cpuOutput->m_dataType);

        CpuTensorIterator inputIter(input);
        CpuTensorIterator outputIter(output);

        const T* inputData = (const T*)cpuInput->m_buffer.data();
        T* outputData = (T*)cpuOutput->m_buffer.data();

        inputIter.ForEachParallel([&](rad::ArrayRef<size_t> coord) {
            ComputeType x = inputData[inputIter.CoordToBufferIndex(coord)];
            outputData[outputIter.CoordToBufferIndex(coord)] = op(x);
            });
    }

}; // struct CpuTensorOpElementWiseUnary

template <typename T, typename ComputeType = T>
struct CpuTensorOpElementWiseBinary
{
    void operator()(Tensor* input, Tensor* other, Tensor* output,
        const std::function<ComputeType(ComputeType a, ComputeType b)>& op)
    {
        if (output == nullptr)
        {
            output = input;
        }
        CpuTensor* cpuInput = static_cast<CpuTensor*>(input);
        CpuTensor* cpuOther = static_cast<CpuTensor*>(other);
        CpuTensor* cpuOutput = static_cast<CpuTensor*>(output);
        assert(cpuInput->m_sizes == cpuOther->m_sizes);
        assert(cpuInput->m_sizes == cpuOutput->m_sizes);
        assert(cpuInput->m_dataType == cpuOther->m_dataType);
        assert(cpuInput->m_dataType == cpuOutput->m_dataType);

        CpuTensorIterator inputIter(input);
        CpuTensorIterator otherIter(other);
        CpuTensorIterator outputIter(output);

        const T* inputData = (const T*)cpuInput->m_buffer.data();
        const T* otherData = (const T*)cpuOther->m_buffer.data();
        T* outputData = (T*)cpuOutput->m_buffer.data();

        inputIter.ForEachParallel([&](rad::ArrayRef<size_t> coord) {
            ComputeType a = inputData[inputIter.CoordToBufferIndex(coord)];
            ComputeType b = otherData[otherIter.CoordToBufferIndex(coord)];
            outputData[outputIter.CoordToBufferIndex(coord)] = op(a, b);
            });
    }

}; // struct CpuTensorOpElementWiseBinary

} // namespace ML
