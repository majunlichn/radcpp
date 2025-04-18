set(VKPP_ROOT ${CMAKE_CURRENT_SOURCE_DIR})

set(VKPP_HEADERS
    Core/Common.h
    Core/Instance.h
    Core/Device.h
    Core/Command.h
    Core/Buffer.h
    Core/Image.h
    Core/Descriptor.h
    Core/ShaderMacro.h
    Core/ShaderIncluder.h
    Core/ShaderCompiler.h
)

set(VKPP_SOURCE_FILES
    ${VKPP_HEADERS}
    Core/Common.cpp
    Core/Instance.cpp
    Core/Device.cpp
    Core/Command.cpp
    Core/Buffer.cpp
    Core/Image.cpp
    Core/Descriptor.cpp
    Core/ShaderCompiler.cpp
)

source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${VKPP_SOURCE_FILES})

add_library(vkpp STATIC ${VKPP_SOURCE_FILES})

target_include_directories(vkpp
    PUBLIC "$ENV{VULKAN_SDK}/Include"
)

target_link_libraries(vkpp
    PUBLIC rad
)
