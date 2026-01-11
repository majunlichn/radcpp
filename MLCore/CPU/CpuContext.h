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

    virtual void Fill(const Tensor& input, Scalar value) override;

    virtual void Add(const Tensor& input, const Scalar other, Tensor& output) override;
    virtual void Add(const Tensor& input, const Tensor& other, const Scalar alpha, Tensor& output) override;

    virtual void Subtract(const Tensor& input, const Scalar other, Tensor& output) override;
    virtual void Subtract(const Tensor& input, const Tensor& other, const Scalar alpha, Tensor& output) override;

    virtual void Multiply(const Tensor& input, const Scalar other, Tensor& output) override;
    virtual void Multiply(const Tensor& input, const Tensor& other, Tensor& output) override;

    virtual void Divide(const Tensor& input, const Scalar other, Tensor& output) override;
    virtual void Divide(const Tensor& input, const Tensor& other, Tensor& output) override;

    virtual void Remainder(const Tensor& input, const Scalar other, Tensor& output) override;
    virtual void Remainder(const Tensor& input, const Tensor& other, Tensor& output) override;

}; // class CpuContext

} // namespace ML
