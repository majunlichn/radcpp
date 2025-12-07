#pragma once

#include <rad/ML/MLDevice.h>

namespace rad
{

class MLCpuDevice : public MLDevice
{
public:
    MLCpuDevice();
    ~MLCpuDevice();

}; // class MLCpuDevice

} // namespace rad
