set(IMGUI_SOURCE_FILES
    imconfig.h
    imgui_demo.cpp
    imgui_draw.cpp
    imgui_internal.h
    imgui_tables.cpp
    imgui_widgets.cpp
    imgui.cpp
    imgui.h
    imstb_rectpack.h
    imstb_textedit.h
    imstb_truetype.h
    backends/imgui_impl_sdl3.cpp
    backends/imgui_impl_sdl3.h
    backends/imgui_impl_sdlrenderer3.cpp
    backends/imgui_impl_sdlrenderer3.h
    implot/implot_demo.cpp
    implot/implot_internal.h
    implot/implot_items.cpp
    implot/implot.cpp
    implot/implot.h
)

if(DEFINED ENV{VULKAN_SDK})
    list(APPEND IMGUI_SOURCE_FILES
        backends/imgui_impl_vulkan.cpp
        backends/imgui_impl_vulkan.h
    )
endif()

source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${IMGUI_SOURCE_FILES})

add_library(imgui STATIC ${IMGUI_SOURCE_FILES})

target_include_directories(imgui
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)
if(DEFINED ENV{VULKAN_SDK})
    target_include_directories(imgui
        PUBLIC $ENV{VULKAN_SDK}/Include
    )
endif()

target_link_libraries(imgui
    PUBLIC SDL3::SDL3
)
