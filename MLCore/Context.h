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

    virtual void FillConstant(const Tensor& input, Scalar value) = 0;

    // output = input + alpha * other;
    // @param output If nullptr, results are written back to the input.
    virtual void Add(const Tensor& input, const Scalar other, const Tensor& output) = 0;
    virtual void Add(const Tensor& input, const Tensor& other, const Scalar alpha, const Tensor& output) = 0;

    // output = input - alpha * other;
    // @param output If nullptr, results are written back to the input.
    virtual void Subtract(const Tensor& input, const Scalar other, const Tensor& output) = 0;
    virtual void Subtract(const Tensor& input, const Tensor& other, const Scalar alpha, const Tensor& output) = 0;

    // Multiply input tensor by other element-wise.
    // @param output If nullptr, results are written back to the input.
    virtual void Multiply(const Tensor& input, const Scalar other, const Tensor& output) = 0;
    virtual void Multiply(const Tensor& input, const Tensor& other, const Tensor& output) = 0;

    // Divide input tensor by other element-wise. For integer divisions, the result is truncated.
    // @param output If nullptr, results are written back to the input.
    virtual void Divide(const Tensor& input, const Scalar other, const Tensor& output) = 0;
    virtual void Divide(const Tensor& input, const Tensor& other, const Tensor& output) = 0;

}; // class Context

} // namespace ML
