#include "HipDevice.h"

HipDevice::HipDevice(int index) :
    m_index(index)
{
    HIP_CHECK(hipDeviceGet(&m_handle, index));
    HIP_CHECK(hipGetDeviceProperties(&m_prop, index));
}

HipDevice::~HipDevice()
{
}

void HipDevice::SetForCurrentThread()
{
    HIP_CHECK(hipSetDevice(m_index));
}
