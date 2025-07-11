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
    Rendering/VulkanTexture.h
    Rendering/VulkanTexture.cpp
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

# https://github.com/GPUOpen-Tools/Compressonator/releases/tag/V4.5.52
set(Compressonator_INSTALL_DIR "C:/Compressonator_4.5.52")

add_library(CMP_Core SHARED IMPORTED GLOBAL)
if (WIN32)
    set_target_properties(CMP_Core PROPERTIES
        IMPORTED_LOCATION ${Compressonator_INSTALL_DIR}/lib/bin/x64/CMP_Core_MT_DLL.dll
        IMPORTED_IMPLIB ${Compressonator_INSTALL_DIR}/lib/bin/x64/CMP_Core_MT_DLL.lib
    )
endif()
add_library(CMP_Framework SHARED IMPORTED GLOBAL)
if (WIN32)
    set_target_properties(CMP_Framework PROPERTIES
        IMPORTED_LOCATION ${Compressonator_INSTALL_DIR}/lib/bin/x64/CMP_Framework_MT_DLL.dll
        IMPORTED_IMPLIB ${Compressonator_INSTALL_DIR}/lib/bin/x64/CMP_Framework_MT_DLL.lib
    )
endif()
add_library(Compressonator SHARED IMPORTED GLOBAL)
if (WIN32)
    set_target_properties(Compressonator PROPERTIES
        IMPORTED_LOCATION ${Compressonator_INSTALL_DIR}/lib/bin/x64/Compressonator_MT_DLL.dll
        IMPORTED_IMPLIB ${Compressonator_INSTALL_DIR}/lib/bin/x64/Compressonator_MT_DLL.lib
    )
endif()

target_include_directories(vkpp
    PUBLIC ${Compressonator_INSTALL_DIR}/include
)

target_link_libraries(SDFramework
    PUBLIC CMP_Core CMP_Framework Compressonator
)
