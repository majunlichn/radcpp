set(SDF_SOURCE_FILES
    Core/Common.h
    Core/Common.cpp
    Core/Application.h
    Core/Application.cpp
    Core/EventHandler.h
    Core/EventHandler.cpp
    Core/Timer.h
    Core/Timer.cpp
    Gui/GuiContext.cpp
    Gui/GuiContext.h
    Gui/Renderer.cpp
    Gui/Renderer.h
    Gui/Surface.cpp
    Gui/Surface.h
    Gui/Texture.cpp
    Gui/Texture.h
    Gui/Window.cpp
    Gui/Window.h
    Media/Audio.cpp
    Media/Audio.h
    Media/AudioMixer.cpp
    Media/AudioMixer.h
    Media/Camera.cpp
    Media/Camera.h
)

if (TARGET vkpp)
    list(APPEND SDF_SOURCE_FILES
        Gui/VulkanWindow.h
        Gui/VulkanWindow.cpp
        Gui/VulkanContext.h
        Gui/VulkanContext.cpp
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
