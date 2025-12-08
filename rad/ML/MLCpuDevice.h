#pragma once

#include <rad/ML/MLDevice.h>
#include <rad/System/CpuInfo.h>

namespace rad
{

class MLCpuDevice : public MLDevice
{
public:
    MLCpuDevice();
    ~MLCpuDevice();

    uint32_t GetPhysicalCoreCount() const;
    uint32_t GetLogicalCoreCount() const;

}; // class MLCpuDevice

} // namespace rad
