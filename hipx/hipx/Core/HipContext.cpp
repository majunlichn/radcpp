#include "HipContext.h"

HipContext::HipContext()
{
}

HipContext::~HipContext()
{

}

bool HipContext::Init()
{
    HIP_CHECK(hipInit(0));
    HIP_CHECK(hipRuntimeGetVersion(&m_runtimeVersion));
    printf("HIP Runtime: %d\n", m_runtimeVersion);
    HIP_CHECK(hipGetDeviceCount(&m_deviceCount));
    m_devices.resize(m_deviceCount);
    m_deviceProps.reserve(m_deviceCount);
    for (int i = 0; i < m_deviceCount; ++i)
    {
        HIP_CHECK(hipDeviceGet(&m_devices[i], i));
        hipDeviceProp_t prop = {};
        HIP_CHECK(hipGetDeviceProperties(&prop, i));
        printf("HIP Device#%d: %s\n", i, prop.name);
        m_deviceProps.push_back(prop);
    }
    return true;
}
