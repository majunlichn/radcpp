#pragma once

#include <rad/Common/Memory.h>
#include <rad/Common/RefCounted.h>

namespace rad
{

class MLContext : public RefCounted<MLContext>
{
public:
    MLContext() = default;
    virtual ~MLContext() = default;

}; // class MLContext

} // namespace rad
