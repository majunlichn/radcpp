#pragma once

#include <rad/ML/MLDevice.h>
#include <rad/ML/MLTensor.h>

namespace rad
{

class MLContext : public RefCounted<MLContext>
{
public:
    MLContext() = default;
    virtual ~MLContext() = default;

    virtual void FillConstant(MLTensor* input, float value) = 0;
    virtual void FillConstant(MLTensor* input, int value) = 0;

    // @param output If nullptr, results are written back to the input (in-place).
    // @param alpha  The multiplier for other.
    virtual void Add(MLTensor* input, MLTensor* other, float alpha = 1.0f, MLTensor* output = nullptr) = 0;
    virtual void Add(MLTensor* input, MLTensor* other, int alpha = 1, MLTensor* output = nullptr) = 0;

}; // class MLContext

} // namespace rad
