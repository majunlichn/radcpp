#pragma once

#include <MLCore/CPU/CpuDevice.h>
#include <MLCore/CPU/CpuTensor.h>
#include <MLCore/CPU/CpuContext.h>
#include <MLCore/CPU/CpuTensorIterator.h>

#include <execution>

namespace ML
{

template <typename T>
struct CpuTensorOpForEach
{
    void operator()(Tensor* input, const std::function<T()>& op)
    {
        CpuTensor* cpuInput = static_cast<CpuTensor*>(input);
        T* inputData = (T*)cpuInput->m_buffer.data();

        if (cpuInput->IsContiguous())
        {
            auto& buffer = cpuInput->m_buffer;
            std::generate_n(std::execution::par_unseq, inputData, input->GetElementCount(), [&]() { return op(); });
        }
        else
        {
            CpuTensorIterator inputIter(input);
            inputIter.ForEachParallelND([&](rad::ArrayRef<size_t> coord) {
                inputData[inputIter.CoordToBufferIndex(coord)] = op();
                }, 2);
        }
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

        const T* inputData = (const T*)cpuInput->m_buffer.data();
        T* outputData = (T*)cpuOutput->m_buffer.data();

        if (input->IsContiguous() && HaveSameLayout(input, output))
        {
            std::transform(std::execution::par_unseq,
                inputData, inputData + input->GetElementCount(), outputData,
                [&](T x) {
                    return static_cast<T>(op(static_cast<ComputeType>(x)));
                });
        }
        else
        {
            CpuTensorIterator inputIter(input);
            CpuTensorIterator outputIter(output);
            inputIter.ForEachParallelND([&](rad::ArrayRef<size_t> coord) {
                ComputeType x = static_cast<ComputeType>(inputData[inputIter.CoordToBufferIndex(coord)]);
                outputData[outputIter.CoordToBufferIndex(coord)] = static_cast<T>(op(x));
                }, 2);
        }
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

        const T* inputData = (const T*)cpuInput->m_buffer.data();
        const T* otherData = (const T*)cpuOther->m_buffer.data();
        T* outputData = (T*)cpuOutput->m_buffer.data();

        if (input->IsContiguous() && HaveSameLayout(input, other) && HaveSameLayout(input, output))
        {
            std::transform(std::execution::par_unseq,
                inputData, inputData + input->GetElementCount(), otherData, outputData,
                [&](T a, T b) {
                    return static_cast<T>(op(static_cast<ComputeType>(a), static_cast<ComputeType>(b)));
                });
        }
        else
        {
            CpuTensorIterator inputIter(input);
            CpuTensorIterator otherIter(other);
            CpuTensorIterator outputIter(output);
            inputIter.ForEachParallelND([&](rad::ArrayRef<size_t> coord) {
                ComputeType a = inputData[inputIter.CoordToBufferIndex(coord)];
                ComputeType b = otherData[otherIter.CoordToBufferIndex(coord)];
                outputData[outputIter.CoordToBufferIndex(coord)] = op(a, b);
                }, 2);
        }
    }

}; // struct CpuTensorOpElementWiseBinary

} // namespace ML
