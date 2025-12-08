#pragma once

#include <rad/Common/Memory.h>
#include <rad/Common/RefCounted.h>

namespace rad
{

enum class MLDeviceType
{
    CPU,
    GPU,
    NPU,
};

class MLDevice : public RefCounted<MLDevice>
{
public:
    MLDevice() = default;
    virtual ~MLDevice() = default;

    MLDeviceType GetType() const { return m_type; }

    MLDeviceType m_type;

}; // class MLDevice

} // namespace rad
