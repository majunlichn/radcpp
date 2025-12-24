#pragma once

#include <MLCore/Backend.h>
#include <vkpp/Core/Instance.h>
#include <vkpp/Core/Device.h>

namespace ML
{

class VulkanDevice;

class VulkanBackend : public Backend
{
public:
    rad::Ref<vkpp::Instance> m_instance;
    std::vector<rad::Ref<VulkanDevice>> m_devices;

    VulkanBackend();
    virtual ~VulkanBackend();

    bool Init();

    virtual size_t GetDeviceCount() const override;
    virtual Device* GetDevice(size_t index) override;

}; // class VulkanBackend

Backend* InitVulkanBackend(std::string_view name = "Vulkan");

} // namespace ML
