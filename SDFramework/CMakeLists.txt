set(SDF_SOURCE_FILES
    Core/Common.h
    Core/Common.cpp
    Core/Application.h
    Core/Application.cpp
    Core/EventHandler.h
    Core/EventHandler.cpp
    Core/Timer.h
    Core/Timer.cpp
    Gui/Frame.cpp
    Gui/Frame.h
    Gui/Renderer.cpp
    Gui/Renderer.h
    Gui/Surface.cpp
    Gui/Surface.h
    Gui/Texture.cpp
    Gui/Texture.h
    Gui/Window.cpp
    Gui/Window.h
    Multimedia/Audio.cpp
    Multimedia/Audio.h
    Multimedia/AudioMixer.cpp
    Multimedia/AudioMixer.h
    Multimedia/Camera.cpp
    Multimedia/Camera.h
    RHI/GpuDevice.h
    RHI/GpuDevice.cpp
    RHI/GpuBuffer.h
    RHI/GpuBuffer.cpp
    RHI/GpuTexture.h
    RHI/GpuTexture.cpp
    RHI/GpuShader.h
    RHI/GpuShader.cpp
    RHI/GpuCommandBuffer.h
    RHI/GpuCommandBuffer.cpp
)

if (TARGET vkpp)
    list(APPEND SDF_SOURCE_FILES
        Gui/VulkanWindow.h
        Gui/VulkanWindow.cpp
        Gui/VulkanFrame.h
        Gui/VulkanFrame.cpp
    )
endif()

source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SDF_SOURCE_FILES})

add_library(SDFramework STATIC ${SDF_SOURCE_FILES})

find_package(SDL3 CONFIG REQUIRED PATHS ${RADCPP_ROOT}/imported/SDL/build/installed)
find_package(SDL3_mixer CONFIG REQUIRED PATHS ${RADCPP_ROOT}/imported/SDL_mixer/build/installed)

target_link_libraries(SDFramework
    PUBLIC rad
    PUBLIC SDL3::SDL3
    PUBLIC SDL3_mixer::SDL3_mixer
    PUBLIC imgui
)

if (TARGET vkpp)
    target_link_libraries(SDFramework
        PUBLIC vkpp
    )
endif()

find_package(ktx CONFIG REQUIRED PATHS ${RADCPP_ROOT}/imported/KTX-Software/build/installed/lib/cmake/ktx)

target_link_libraries(SDFramework
    PUBLIC KTX::ktx
)
