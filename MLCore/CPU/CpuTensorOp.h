#pragma once

#include <MLCore/CPU/CpuDevice.h>
#include <MLCore/CPU/CpuTensor.h>
#include <MLCore/CPU/CpuContext.h>
#include <MLCore/TensorIterator.h>

#include <execution>

namespace ML
{

void ForEachParallelND(TensorIterator& iter, const std::function<void(size_t bufferIndex)>& op, size_t granularityND);

template <typename T>
struct CpuTensorOpForEach
{
    void operator()(const Tensor& input, const std::function<T()>& op)
    {
        CpuTensorStorage* inputStorage = static_cast<CpuTensorStorage*>(input.m_storage.get());
        T* inputData = (T*)inputStorage->m_buffer.data();

        if (input.IsContiguous())
        {
            auto& buffer = inputStorage->m_buffer;
            std::generate_n(std::execution::par_unseq, inputData, input.GetElementCount(), [&]() { return op(); });
        }
        else
        {
            TensorIterator inputIter(input);
            ForEachParallelND(inputIter, [&](size_t bufferIndex) {
                inputData[bufferIndex] = op();
                }, 2);
        }
    }

}; // struct CpuTensorOpForEach

template <typename T, typename ComputeType = T>
struct CpuTensorOpElementWiseUnary
{
    void operator()(const Tensor& input, Tensor& output, const std::function<ComputeType(ComputeType x)>& op)
    {
        assert(input.m_sizes == output.m_sizes);
        assert(input.m_dataType == output.m_dataType);

        CpuTensorStorage* inputStorage = static_cast<CpuTensorStorage*>(input.m_storage.get());
        CpuTensorStorage* outputStorage = static_cast<CpuTensorStorage*>(output.m_storage.get());
        const T* inputData = (const T*)inputStorage->m_buffer.data();
        T* outputData = (T*)outputStorage->m_buffer.data();

        if (input.IsContiguous() && HaveSameLayout(input, output))
        {
            std::transform(std::execution::par_unseq,
                inputData, inputData + input.GetElementCount(), outputData,
                [&](T x) {
                    return static_cast<T>(op(static_cast<ComputeType>(x)));
                });
        }
        else
        {
            TensorIterator inputIter(input);
            TensorIterator outputIter(output);
            ForEach(inputIter, outputIter, [&](size_t inputIndex, size_t outputIndex) {
                ComputeType x = static_cast<ComputeType>(inputData[inputIndex]);
                outputData[outputIndex] = static_cast<T>(op(x));
                });
        }
    }

}; // struct CpuTensorOpElementWiseUnary

template <typename T, typename ComputeType = T>
struct CpuTensorOpElementWiseBinary
{
    void operator()(const Tensor& input, const Tensor& other, Tensor& output,
        const std::function<ComputeType(ComputeType a, ComputeType b)>& op)
    {
        assert(input.m_sizes == other.m_sizes);
        assert(input.m_sizes == output.m_sizes);
        assert(input.m_dataType == other.m_dataType);
        assert(input.m_dataType == output.m_dataType);

        CpuTensorStorage* inputStorage = static_cast<CpuTensorStorage*>(input.m_storage.get());
        CpuTensorStorage* otherStorage = static_cast<CpuTensorStorage*>(other.m_storage.get());
        CpuTensorStorage* outputStorage = static_cast<CpuTensorStorage*>(output.m_storage.get());
        const T* inputData = (const T*)inputStorage->m_buffer.data();
        const T* otherData = (const T*)otherStorage->m_buffer.data();
        T* outputData = (T*)outputStorage->m_buffer.data();

        if (input.IsContiguous() && HaveSameLayout(input, other) && HaveSameLayout(input, output))
        {
            std::transform(std::execution::par_unseq,
                inputData, inputData + input.GetElementCount(), otherData, outputData,
                [&](T a, T b) {
                    return static_cast<T>(op(static_cast<ComputeType>(a), static_cast<ComputeType>(b)));
                });
        }
        else
        {
            TensorIterator inputIter(input);
            TensorIterator otherIter(other);
            TensorIterator outputIter(output);
            ForEach(inputIter, otherIter, outputIter, [&](size_t inputIndex, size_t otherIndex, size_t outputIndex) {
                ComputeType a = static_cast<ComputeType>(inputData[inputIndex]);
                ComputeType b = static_cast<ComputeType>(otherData[otherIndex]);
                outputData[outputIndex] = static_cast<T>(op(a, b));
                });
        }
    }

}; // struct CpuTensorOpElementWiseBinary

} // namespace ML
