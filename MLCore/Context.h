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

    // @param output If nullptr, results are written back to the input (in-place).
    // @param alpha  The multiplier for other.
    virtual void Add(Tensor* input, Tensor* other, float alpha = 1.0f, Tensor* output = nullptr) = 0;
    virtual void Add(Tensor* input, Tensor* other, int alpha = 1, Tensor* output = nullptr) = 0;

}; // class Context

} // namespace ML
