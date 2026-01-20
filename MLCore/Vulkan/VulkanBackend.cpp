#include <MLCore/Vulkan/VulkanBackend.h>
#include <MLCore/Vulkan/VulkanDevice.h>
#include <MLCore/Global.h>
#include <MLCore/Logging.h>

namespace ML
{

VulkanBackend::VulkanBackend()
{
}

VulkanBackend::~VulkanBackend()
{
}

bool VulkanBackend::Init()
{
    m_instance = RAD_NEW vkpp::Instance();
    if (!m_instance->Init(
        "MLCore", VK_MAKE_VERSION(0, 0, 0),
        "MLCore", VK_MAKE_VERSION(0, 0, 0)))
    {
        VKPP_LOG(err, "Failed to init the Vulkan instance!");
        return false;
    }
    if (m_instance->m_physicalDevices.empty())
    {
        VKPP_LOG(err, "No Vulkan device available!");
        return false;
    }
    m_devices.clear();
    for (auto& physicalDevice : m_instance->m_physicalDevices)
    {
        vkpp::DeviceConfig deviceConfig = {};
        deviceConfig.enableVulkan11Features = true;
        deviceConfig.enableVulkan12Features = true;
        deviceConfig.enableVulkan13Features = true;
        deviceConfig.enableVulkan14Features = true;
        deviceConfig.enableBFloat16 = true;
        deviceConfig.enableFloat8 = true;
        m_devices.push_back(RAD_NEW VulkanDevice(m_instance->CreateDevice(physicalDevice, deviceConfig)));
    }

    for (size_t i = 0; i < GetDeviceCount(); ++i)
    {
        Device* device = GetDevice(i);
        if (device != nullptr)
        {
            if (auto context = device->CreateContext())
            {
                SetDefaultContext(device, context);
            }
            else
            {
                ML_LOG(err, "Failed to create context for device#{}: {}", i, device->GetName());
                return false;
            }
        }
    }
    return true;
}

size_t VulkanBackend::GetDeviceCount() const
{
    return m_devices.size();
}

Device* VulkanBackend::GetDevice(size_t index)
{
    if (index < m_devices.size())
    {
        return m_devices[index].get();
    }
    else
    {
        return nullptr;
    }
}

rad::Ref<VulkanBackend> CreateVulkanBackend()
{
    rad::Ref<VulkanBackend> backend = RAD_NEW VulkanBackend();
    if (backend->Init())
    {
        ML_LOG(info, "Vulkan backend created.");
        for (size_t i = 0; i < backend->GetDeviceCount(); ++i)
        {
            Device* device = backend->GetDevice(i);
            ML_LOG(info, "Device#{}: {} (Driver {})", i, device->m_name, device->m_driverVersion);
        }
        return backend.get();
    }
    return nullptr;
}

} // namespace ML
