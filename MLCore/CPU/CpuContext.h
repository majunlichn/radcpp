#pragma once

#include <MLCore/Context.h>

namespace ML
{

class CpuDevice;

class CpuContext : public Context
{
public:
    CpuContext(rad::Ref<CpuDevice> device);
    ~CpuContext() override;

    virtual void FillConstant(const Tensor& input, Scalar value) override;

    // output = input + alpha * other;
    // @param output If nullptr, results are written back to the input.
    virtual void Add(const Tensor& input, const Scalar other, const Tensor& output) override;
    virtual void Add(const Tensor& input, const Tensor& other, const Scalar alpha, const Tensor& output) override;

    // output = input - alpha * other;
    // @param output If nullptr, results are written back to the input.
    virtual void Subtract(const Tensor& input, const Scalar other, const Tensor& output) override;
    virtual void Subtract(const Tensor& input, const Tensor& other, const Scalar alpha, const Tensor& output) override;

    // Multiply input tensor by other element-wise.
    // @param output If nullptr, results are written back to the input.
    virtual void Multiply(const Tensor& input, const Scalar other, const Tensor& output) override;
    virtual void Multiply(const Tensor& input, const Tensor& other, const Tensor& output) override;

    // Divide input tensor by other element-wise. For integer divisions, the result is truncated.
    // @param output If nullptr, results are written back to the input.
    virtual void Divide(const Tensor& input, const Scalar other, const Tensor& output) override;
    virtual void Divide(const Tensor& input, const Tensor& other, const Tensor& output) override;

}; // class CpuContext

} // namespace ML
