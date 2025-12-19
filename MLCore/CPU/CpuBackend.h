#pragma once

#include <MLCore/Backend.h>

namespace ML
{

class CpuDevice;

class CpuBackend : public Backend
{
public:
    rad::Ref<CpuDevice> m_device;

    CpuBackend();
    virtual ~CpuBackend();

    bool Init();

    virtual size_t GetDeviceCount() const override;
    virtual Device* GetDevice(size_t index) override;

}; // class CpuBackend

Backend* InitCpuBackend(std::string_view name = "CPU");

} // namespace ML
