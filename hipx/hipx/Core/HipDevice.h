#pragma once

#include <hipx/Core/HipError.h>
#include <vector>

class HipDevice
{
public:
    HipDevice(int index);
    ~HipDevice();

    const char* GetName() const noexcept { return m_prop.name; }
    void SetForCurrentThread();

    int m_index = 0;
    hipDevice_t m_handle = {};
    hipDeviceProp_t m_prop = {};

}; // class HipDevice
