cmake_minimum_required(VERSION 3.21)

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

option(RADCPP_BUILD_TESTS "Whether build tests" ${PROJECT_IS_TOP_LEVEL})

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

if(RADCPP_BUILD_TESTS)
    add_subdirectory(tests)
    set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT HelloWorld)
endif()

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
