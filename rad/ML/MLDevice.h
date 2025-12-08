#pragma once

#include <rad/Common/Memory.h>
#include <rad/Common/RefCounted.h>
#include <rad/Common/String.h>

namespace rad
{

enum class MLDeviceType
{
    Unknown,
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

    MLDeviceType m_type = MLDeviceType::Unknown;
    std::string m_name;
    std::string m_driverVersion;

}; // class MLDevice

} // namespace rad
