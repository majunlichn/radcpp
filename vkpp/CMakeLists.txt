set(VKPP_ROOT ${CMAKE_CURRENT_SOURCE_DIR})

set(VKPP_HEADERS
    Core/Common.h
    Core/Instance.h
    Core/Device.h
    Core/Command.h
    Core/Fence.h
    Core/Semaphore.h
    Core/Event.h
    Core/Buffer.h
    Core/Image.h
    Core/RenderPass.h
    Core/Framebuffer.h
    Core/Descriptor.h
    Core/ShaderMacro.h
    Core/ShaderIncluder.h
    Core/ShaderCompiler.h
    Core/Pipeline.h
    Core/Swapchain.h
    Compute/Tensor.h
    Compute/TensorOp.h
)

set(VKPP_SOURCE_FILES
    ${VKPP_HEADERS}
    Core/Common.cpp
    Core/Instance.cpp
    Core/Device.cpp
    Core/Command.cpp
    Core/Fence.cpp
    Core/Semaphore.cpp
    Core/Event.cpp
    Core/Buffer.cpp
    Core/Image.cpp
    Core/RenderPass.cpp
    Core/Framebuffer.cpp
    Core/Descriptor.cpp
    Core/ShaderCompiler.cpp
    Core/Pipeline.cpp
    Core/Swapchain.cpp
    Compute/Tensor.cpp
    Compute/TensorOp.cpp
)

source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${VKPP_SOURCE_FILES})

add_library(vkpp STATIC ${VKPP_SOURCE_FILES})

target_include_directories(vkpp
    PUBLIC "$ENV{VULKAN_SDK}/Include"
)

target_link_libraries(vkpp
    PUBLIC rad
)

if(WIN32)
    target_link_libraries(vkpp
        PUBLIC "$ENV{VULKAN_SDK}/Lib/shaderc_shared.lib"
        PUBLIC "$ENV{VULKAN_SDK}/Lib/SPIRV-Tools-shared.lib"
    )
endif()
