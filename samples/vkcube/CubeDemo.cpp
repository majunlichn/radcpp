#include "CubeDemo.h"
#include <glm/ext.hpp>

#include "lunarg.ppm.h"

CubeDemo::CubeDemo()
{
}

CubeDemo::~CubeDemo()
{
}

void CubeDemo::ParseCommandLine(int argc, char* argv[])
{
    for (int i = 1; i < argc; i++)
    {
        if (rad::StrCaseEqual(argv[i], "--gpu-index") && (i < argc - 1))
        {
            m_gpuIndex = std::stoi(argv[i + 1]);
            i++;
            continue;
        }

        if (rad::StrCaseEqual(argv[i], "--present-mode") && (i < argc - 1))
        {
            m_presentMode = static_cast<vk::PresentModeKHR>(std::stoi(argv[i + 1]));
            i++;
            continue;
        }

        if (rad::StrCaseEqual(argv[i], "--width"))
        {
            if (i < argc - 1)
            {
                int32_t widthInput = std::stoi(argv[i + 1]);
                if (widthInput > 0)
                {
                    m_width = static_cast<uint32_t>(widthInput);
                    i++;
                    continue;
                }
            }
        }
        if (rad::StrCaseEqual(argv[i], "--height"))
        {
            if (i < argc - 1)
            {
                int32_t heightInput = std::stoi(argv[i + 1]);
                if (heightInput > 0)
                {
                    m_height = static_cast<uint32_t>(heightInput);
                    i++;
                    continue;
                }
            }
        }
    }
}

bool CubeDemo::Init(int argc, char* argv[])
{
    glm::vec3 eye = { 0.0f, 3.0f, 5.0f };
    glm::vec3 origin = { 0, 0, 0 };
    glm::vec3 up = { 0.0f, 1.0f, 0.0f };

    m_presentMode = vk::PresentModeKHR::eFifo;
    m_frameCount = UINT32_MAX;
    m_width = 500;
    m_height = 500;
    m_gpuIndex = -1;

    ParseCommandLine(argc, argv);

    m_initialized = false;

    m_instance = RAD_NEW vkpp::Instance();
    if (!m_instance->Init(m_name, m_instance->GetApiVersion()))
    {
        return false;
    }

    m_spinAngle = 4.0f;
    m_spinIncrement = 0.2f;
    m_spinPause = false;

    m_projectionMatrix = glm::perspective(glm::radians(45.0f), 1.0f, 0.1f, 100.0f);
    m_projectionMatrix[1][1] *= -1; // Flip projection matrix from GL to Vulkan orientation.
    m_viewMatrix = glm::lookAt(eye, origin, up);
    m_modelMatrix = glm::identity<glm::mat4>();

    if (!VulkanWindow::Create(m_name, m_width, m_height, SDL_WINDOW_VULKAN))
    {
        return false;
    }

    const auto& physicalDevices = m_instance->m_physicalDevices;
    if (m_gpuIndex == -1)
    {
        int priorityPrev = 0;
        for (uint32_t i = 0; i < physicalDevices.size(); i++)
        {
            const auto physicalDeviceProperties = physicalDevices[i].getProperties();
            assert(physicalDeviceProperties.deviceType <= vk::PhysicalDeviceType::eCpu);

            auto surfaceSupport = physicalDevices[i].getSurfaceSupportKHR(0, m_surface->GetHandle());
            if (surfaceSupport != vk::True)
            {
                continue;
            }

            std::map<vk::PhysicalDeviceType, int> deviceTypePriorities =
            {
                { vk::PhysicalDeviceType::eDiscreteGpu,     5 },
                { vk::PhysicalDeviceType::eIntegratedGpu,   4 },
                { vk::PhysicalDeviceType::eVirtualGpu,      3 },
                { vk::PhysicalDeviceType::eCpu,             2 },
                { vk::PhysicalDeviceType::eOther,           1 },
            };

            int priority = -1;
            if (deviceTypePriorities.find(physicalDeviceProperties.deviceType) != deviceTypePriorities.end())
            {
                priority = deviceTypePriorities[physicalDeviceProperties.deviceType];
            }

            if (priority > priorityPrev)
            {
                m_gpuIndex = i;
                priorityPrev = priority;
            }
        }
    }

    vk::raii::PhysicalDevice physicalDevice = physicalDevices[m_gpuIndex];
    m_device = m_instance->CreateDevice(physicalDevice);

    m_swapchain = CreateSwapchain(m_width, m_height);

    m_isFirstSwapchainFrame = true;
    m_frameIndex = 0;

    m_textures.resize(1);
    m_textures[0] = vkpp::CreateTextureFromMemory_R8G8B8A8_SRGB(m_device, lunarg_ppm, lunarg_ppm_len);

    return true;
}
