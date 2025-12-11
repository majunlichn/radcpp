#pragma once

#include <rad/ML/MLContext.h>

namespace rad
{

class MLCpuDevice;

class MLCpuContext : public MLContext
{
public:
    MLCpuContext(Ref<MLCpuDevice> device);
    ~MLCpuContext() override;

    virtual void FillConstant(MLTensor* input, float value) override;
    virtual void FillConstant(MLTensor* input, int value) override;

    virtual void Add(MLTensor* input, MLTensor* other, float alpha = 1.0f, MLTensor* output = nullptr) override;
    virtual void Add(MLTensor* input, MLTensor* other, int alpha = 1, MLTensor* output = nullptr) override;

    Ref<MLCpuDevice> m_device;

}; // class MLCpuContext

} // namespace rad
