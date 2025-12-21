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
    Core/Sampler.h
    Core/RenderPass.h
    Core/Framebuffer.h
    Core/Descriptor.h
    Core/ShaderMacro.h
    Core/ShaderIncluder.h
    Core/ShaderCompiler.h
    Core/SlangSession.h
    Core/Pipeline.h
    Core/Surface.h
    Core/Swapchain.h
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
    Core/Sampler.cpp
    Core/RenderPass.cpp
    Core/Framebuffer.cpp
    Core/Descriptor.cpp
    Core/ShaderCompiler.cpp
    Core/SlangSession.cpp
    Core/Pipeline.cpp
    Core/Surface.cpp
    Core/Swapchain.cpp
)

source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${VKPP_SOURCE_FILES})

add_library(vkpp STATIC ${VKPP_SOURCE_FILES})

target_link_libraries(vkpp
    PUBLIC rad
)

set(VULKAN_SDK_INCLUDE_DIR "$ENV{VULKAN_SDK}/include")
set(VULKAN_SDK_LIB_DIR "$ENV{VULKAN_SDK}/lib")

if (WIN32)
    set(VULKAN_SDK_INCLUDE_DIR "$ENV{VULKAN_SDK}/Include")
    set(VULKAN_SDK_LIB_DIR "$ENV{VULKAN_SDK}/Lib")
endif()

target_include_directories(vkpp
    PUBLIC ${VULKAN_SDK_INCLUDE_DIR}
)

if(WIN32)
    target_link_libraries(vkpp
        PUBLIC "${VULKAN_SDK_LIB_DIR}/shaderc_shared.lib"
        PUBLIC "${VULKAN_SDK_LIB_DIR}/SPIRV-Tools-shared.lib"
    )
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_libraries(vkpp
        PUBLIC "${VULKAN_SDK_LIB_DIR}/libshaderc_shared.so"
        PUBLIC "${VULKAN_SDK_LIB_DIR}/libSPIRV-Tools-shared.so"
    )
endif()

find_package(slang CONFIG REQUIRED
    PATHS
        ${RADCPP_ROOT}/imported/slang/cmake
        ${RADCPP_ROOT}/imported/slang/lib/cmake/slang
)
target_include_directories(vkpp
    PUBLIC "${RADCPP_ROOT}/imported/slang/include"
)
target_link_libraries(vkpp
    PUBLIC slang::slang
)
