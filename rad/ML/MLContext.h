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

    virtual void Add(MLTensor* input, MLTensor* other, float alpha = 1.0f, MLTensor* output = nullptr) = 0;

}; // class MLContext

} // namespace rad
