#include <SDFramework/Gui/VulkanGuiContext.h>
#include <SDFramework/Gui/VulkanWindow.h>

// Implementation references:
// https://github.com/KhronosGroup/Vulkan-Tools/blob/main/cube/cube.c
// https://github.com/ocornut/imgui/blob/master/examples/example_sdl3_vulkan/main.cpp

namespace sdf
{

VulkanGuiContext::VulkanGuiContext(VulkanWindow* window, rad::Ref<vkpp::Device> device) :
    m_window(window),
    m_device(std::move(device))
{
}

VulkanGuiContext::~VulkanGuiContext()
{
}

static void CheckVulkanResult(VkResult result)
{
    if (result < 0)
    {
        VKPP_LOG(err, "VulkanGuiRenderer: {}", string_VkResult(result));
    }
}

bool VulkanGuiContext::Init()
{
    int width = 0;
    int height = 0;
    m_window->GetSizeInPixels(&width, &height);
    vk::Format renderTargetFormat = vk::Format::eR8G8B8A8Unorm;
    m_renderTarget = m_device->CreateImage2D_ColorAttachment(
        renderTargetFormat, width, height, vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eColorAttachment);
    m_renderTargetView = m_renderTarget->CreateView2D();

    std::vector<vk::DescriptorPoolSize> descPoolSizes =
    {
        { vk::DescriptorType::eCombinedImageSampler, 4096 },
    };
    m_descPool = m_device->CreateDescriptorPool(1024, descPoolSizes);

    IMGUI_CHECKVERSION();
    m_context = ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;   // Enable Keyboard Controls
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;    // Enable Gamepad Controls

    ImGui::StyleColorsDark();

    ImGui_ImplSDL3_InitForVulkan(m_window->GetHandle());
    ImGui_ImplVulkan_InitInfo initInfo = {};
    initInfo.ApiVersion = m_device->m_properties.apiVersion;
    initInfo.Instance = m_device->m_instance->GetHandle();
    initInfo.PhysicalDevice = static_cast<vk::PhysicalDevice>(m_device->m_physicalDevice);
    initInfo.Device = m_device->GetHandle();
    initInfo.QueueFamily = m_device->GetQueueFamilyIndex(vkpp::QueueFamily::Graphics);
    initInfo.Queue = m_device->GetQueue(vkpp::QueueFamily::Graphics);
    initInfo.DescriptorPool = m_descPool->GetHandle();
    initInfo.RenderPass = VK_NULL_HANDLE;
    initInfo.MinImageCount = m_window->GetSwapchain()->GetImageCount();
    initInfo.ImageCount = m_window->GetSwapchain()->GetImageCount();
    initInfo.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    // (Optional)
    initInfo.PipelineCache;
    initInfo.Subpass;
    // (Optional) Set to create internal descriptor pool instead of using DescriptorPool
    initInfo.DescriptorPoolSize;
    // (Optional) Dynamic Rendering
    // Need to explicitly enable VK_KHR_dynamic_rendering extension to use this, even for Vulkan 1.3.
    initInfo.UseDynamicRendering = true;
    initInfo.PipelineRenderingCreateInfo;
    // (Optional) Allocation, Debugging
    initInfo.Allocator = nullptr;
    initInfo.CheckVkResultFn = CheckVulkanResult;
    initInfo.MinAllocationSize = 1024 * 1024;
    ImGui_ImplVulkan_Init(&initInfo);

    SDL_DisplayID displayID = SDL_GetDisplayForWindow(m_window->GetHandle());
    SDL_Rect rect = {};
    float fontSize = 24.0f;
#if defined(_WIN32)
    auto fonts = ImGui::GetIO().Fonts;
    fonts->AddFontFromFileTTF("C:\\Windows\\Fonts\\consola.ttf", fontSize);
#endif
    return true;
}

void VulkanGuiContext::Destroy()
{
    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplSDL3_Shutdown();
    ImGui::DestroyContext();
}

bool VulkanGuiContext::ProcessEvent(const SDL_Event& event)
{
    return ImGui_ImplSDL3_ProcessEvent(&event);
}

void VulkanGuiContext::NewFrame()
{
    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplSDL3_NewFrame();
    ImGui::NewFrame();
}

void VulkanGuiContext::Render()
{
    ImGui::Render();
    ImDrawData* drawData = ImGui::GetDrawData();
    const bool isMinimized = ((drawData->DisplaySize.x <= 0.0f) || (drawData->DisplaySize.y <= 0.0f));
    if (!isMinimized)
    {
        vkpp::CommandBuffer* cmdBuffer = m_cmdBuffers[m_cmdBufferIndex].get();
        ImGui_ImplVulkan_RenderDrawData(drawData, cmdBuffer->GetHandle());
    }
}

} // namespace sdf
