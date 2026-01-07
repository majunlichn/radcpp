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

    virtual void FillConstant(Tensor* input, float value) override;
    virtual void FillConstant(Tensor* input, int value) override;

    virtual void AddScalar(Tensor* input, float other, Tensor* output = nullptr) override;
    virtual void AddScalar(Tensor* input, int other, Tensor* output = nullptr) override;

    virtual void Add(Tensor* input, Tensor* other, float alpha = 1.0f, Tensor* output = nullptr) override;
    virtual void Add(Tensor* input, Tensor* other, int alpha = 1, Tensor* output = nullptr) override;

    virtual void SubtractScalar(Tensor* input, float other, Tensor* output = nullptr) override;
    virtual void SubtractScalar(Tensor* input, int other, Tensor* output = nullptr) override;

    virtual void Subtract(Tensor* input, Tensor* other, float alpha = 1.0f, Tensor* output = nullptr) override;
    virtual void Subtract(Tensor* input, Tensor* other, int alpha = 1, Tensor* output = nullptr) override;

    virtual void MultiplyScalar(Tensor* input, float other, Tensor* output = nullptr) override;
    virtual void MultiplyScalar(Tensor* input, int other, Tensor* output = nullptr) override;

    virtual void Multiply(Tensor* input, Tensor* other, Tensor* output = nullptr) override;

    virtual void DivideScalar(Tensor* input, float other, Tensor* output = nullptr) override;
    virtual void DivideScalar(Tensor* input, int other, Tensor* output = nullptr) override;

    virtual void Divide(Tensor* input, Tensor* other, Tensor* output = nullptr) override;

}; // class CpuContext

} // namespace ML
