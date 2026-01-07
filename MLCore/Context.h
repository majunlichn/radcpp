#pragma once

#include <MLCore/Common.h>

namespace ML
{

class Device;
class Tensor;

class Context : public rad::RefCounted<Context>
{
public:
    rad::Ref<Device> m_device;
    Context(rad::Ref<Device> device) : m_device(std::move(device)) {}
    virtual ~Context() = default;

    virtual void FillConstant(Tensor* input, float value) = 0;
    virtual void FillConstant(Tensor* input, int value) = 0;

    // Add input tensor by other scalar.
    // @param output If nullptr, results are written back to the input.
    virtual void AddScalar(Tensor* input, float other, Tensor* output = nullptr) = 0;
    virtual void AddScalar(Tensor* input, int other, Tensor* output = nullptr) = 0;

    // Add input tensor by other tensor.
    // @param output If nullptr, results are written back to the input.
    // @param alpha  The multiplier for other.
    virtual void Add(Tensor* input, Tensor* other, float alpha = 1.0f, Tensor* output = nullptr) = 0;
    virtual void Add(Tensor* input, Tensor* other, int alpha = 1, Tensor* output = nullptr) = 0;

    // Subtract a scalar from input.
    // @param output If nullptr, results are written back to the input.
    virtual void SubtractScalar(Tensor* input, float other, Tensor* output = nullptr) = 0;
    virtual void SubtractScalar(Tensor* input, int other, Tensor* output = nullptr) = 0;

    // Subtract a tensor from input.
    // @param output If nullptr, results are written back to the input.
    // @param alpha  The multiplier for other.
    virtual void Subtract(Tensor* input, Tensor* other, float alpha = 1.0f, Tensor* output = nullptr) = 0;
    virtual void Subtract(Tensor* input, Tensor* other, int alpha = 1, Tensor* output = nullptr) = 0;

    // Multiply input tensor by other scalar.
    // @param output If nullptr, results are written back to the input.
    virtual void MultiplyScalar(Tensor* input, float other, Tensor* output = nullptr) = 0;
    virtual void MultiplyScalar(Tensor* input, int other, Tensor* output = nullptr) = 0;

    // Multiply input tensor by other tensor.
    // @param output If nullptr, results are written back to the input.
    virtual void Multiply(Tensor* input, Tensor* other, Tensor* output = nullptr) = 0;

    // Divide input tensor by other scalar. For integer divisions, the result is truncated.
    // @param output If nullptr, results are written back to the input.
    virtual void DivideScalar(Tensor* input, float other, Tensor* output = nullptr) = 0;
    virtual void DivideScalar(Tensor* input, int other, Tensor* output = nullptr) = 0;

    // Divide input tensor by other tensor. For integer divisions, the result is truncated.
    // @param output If nullptr, results are written back to the input.
    virtual void Divide(Tensor* input, Tensor* other, Tensor* output = nullptr) = 0;

}; // class Context

} // namespace ML
