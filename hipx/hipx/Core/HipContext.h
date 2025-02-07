#pragma once

#include <hipx/Core/HipError.h>
#include <vector>

class HipContext
{
public:
    HipContext();
    ~HipContext();

    bool Init();
    int GetRuntimeVersion() const noexcept { return m_runtimeVersion; }

    int m_runtimeVersion = 0;
    int m_deviceCount = 0;
    std::vector<hipDevice_t> m_devices;
    std::vector<hipDeviceProp_t> m_deviceProps;

}; // class HipContext
