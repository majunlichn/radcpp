#pragma once

#include <rad/Common/Memory.h>
#include <rad/Common/RefCounted.h>

namespace rad
{

class MLDevice : public RefCounted<MLDevice>
{
public:
    MLDevice() = default;
    virtual ~MLDevice() = default;

}; // class MLDevice

} // namespace rad
