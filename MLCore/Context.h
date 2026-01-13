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

    virtual void Fill(const Tensor& input, const Scalar& value) = 0;

    // output = input + alpha * other;
    virtual void Add(const Tensor& input, const Scalar& other, Tensor& output) = 0;
    virtual void Add(const Tensor& input, const Tensor& other, const Scalar& alpha, Tensor& output) = 0;

    // output = input - alpha * other;
    virtual void Subtract(const Tensor& input, const Scalar& other, Tensor& output) = 0;
    virtual void Subtract(const Tensor& input, const Tensor& other, const Scalar& alpha, Tensor& output) = 0;

    // Multiply input tensor by other element-wise.
    virtual void Multiply(const Tensor& input, const Scalar& other, Tensor& output) = 0;
    virtual void Multiply(const Tensor& input, const Tensor& other, Tensor& output) = 0;

    // Divide input tensor by other element-wise. For integer divisions, the result is truncated.
    virtual void Divide(const Tensor& input, const Scalar& other, Tensor& output) = 0;
    virtual void Divide(const Tensor& input, const Tensor& other, Tensor& output) = 0;

    // Computes Python's modulus operation entrywise.
    // The result has the same sign as the divisor other and its absolute value is less than that of other.
    virtual void Remainder(const Tensor& input, const Scalar& other, Tensor& output) = 0;
    virtual void Remainder(const Tensor& input, const Tensor& other, Tensor& output) = 0;

    virtual void BitwiseAnd(const Tensor& input, const Scalar& other, Tensor& output) = 0;
    virtual void BitwiseAnd(const Tensor& input, const Tensor& other, Tensor& output) = 0;

    virtual void BitwiseOr(const Tensor& input, const Scalar& other, Tensor& output) = 0;
    virtual void BitwiseOr(const Tensor& input, const Tensor& other, Tensor& output) = 0;

    virtual void BitwiseXor(const Tensor& input, const Scalar& other, Tensor& output) = 0;
    virtual void BitwiseXor(const Tensor& input, const Tensor& other, Tensor& output) = 0;

}; // class Context

} // namespace ML
