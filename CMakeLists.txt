cmake_minimum_required(VERSION 3.21)

if(POLICY CMP0091)
    cmake_policy(SET CMP0091 NEW)
endif()

if((NOT DEFINED CMAKE_TOOLCHAIN_FILE) AND (DEFINED ENV{VCPKG_ROOT}))
    list(APPEND CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake")
endif()
cmake_path(CONVERT ${CMAKE_TOOLCHAIN_FILE} TO_CMAKE_PATH_LIST ${CMAKE_TOOLCHAIN_FILE} NORMALIZE)
list(REMOVE_DUPLICATES ${CMAKE_TOOLCHAIN_FILE})
message(STATUS "CMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}")

project(
    radcpp
    VERSION 0.0.0
    DESCRIPTION "Great collections of C++ programming language"
    LANGUAGES CXX
)

include(CMakeDependentOption)

if(DEFINED ENV{VULKAN_SDK})
    set(HAS_VULKAN_SDK True)
endif()

option(BUILD_TESTS "Build tests" ${PROJECT_IS_TOP_LEVEL})
option(BUILD_SAMPLES "Build samples" ${PROJECT_IS_TOP_LEVEL})
option(BUILD_VKPP "Build vkpp" ${HAS_VULKAN_SDK})
option(BUILD_GUI "Build GUI libraries" True)
option(BUILD_SDFramework "Build SDFramework" True)

set(RADCPP_ROOT ${CMAKE_CURRENT_SOURCE_DIR})

set(CMAKE_BUILD_TYPE
    "Release"
    CACHE STRING "Default build type for single-config generators"
)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    set(CMAKE_C_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
    set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
endif()

include(cmake/radCompileOptions.cmake)
include(cmake/radSanitizers.cmake)

# Print build infos
message(STATUS "============================================================")
message(STATUS "Build Informations")
message(STATUS "CMake version: ${CMAKE_VERSION}")
get_property(
    is_generator_multi_config GLOBAL
    PROPERTY GENERATOR_IS_MULTI_CONFIG
)
message(STATUS "CMake Generator: ${CMAKE_GENERATOR}")
if (NOT is_generator_multi_config)
    message(STATUS "CMake Build Type: ${CMAKE_BUILD_TYPE}")
endif()
message(STATUS "CMake Install Prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "============================================================")

add_subdirectory(rad)

if(BUILD_VKPP)
    add_subdirectory(vkpp)
endif()

if(BUILD_SDFramework)
    add_subdirectory(SDFramework)
endif()

if(BUILD_GUI)
    find_package(SDL3 CONFIG REQUIRED PATHS ${RADCPP_ROOT}/imported/SDL/build/installed)
    add_subdirectory(imported/imgui)
    set_target_properties(imgui PROPERTIES FOLDER "imported")
endif()

if(BUILD_TESTS)
    include(CTest)
    if(TARGET rad)
        add_subdirectory(tests/rad)
        set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT rad_tests)
    endif()
    if(TARGET vkpp)
        add_subdirectory(tests/vkpp)
    endif()
endif()

if(BUILD_SAMPLES)
    if(TARGET SDFramework)
        add_subdirectory(samples/Painter)
        if(TARGET vkpp)
            add_subdirectory(samples/vkcube)
        endif()
    endif()
endif()

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
