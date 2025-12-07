#pragma once

#include <rad/Common/Memory.h>
#include <rad/Common/RefCounted.h>

namespace rad
{

class HostTensor;

class MLTensor : public RefCounted<MLTensor>
{
public:
    MLTensor() = default;
    virtual ~MLTensor() = default;

}; // class MLTensor

} // namespace rad
