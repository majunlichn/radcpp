cmake_minimum_required(VERSION 3.21)

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
message(STATUS "CMake version: ${CMAKE_VERSION}")
get_property(
    is_generator_multi_config GLOBAL
    PROPERTY GENERATOR_IS_MULTI_CONFIG
)
message(STATUS "radcpp: Generator: ${CMAKE_GENERATOR}")
if (NOT is_generator_multi_config)
    message(STATUS "radcpp: Build Type: ${CMAKE_BUILD_TYPE}")
endif()

add_subdirectory(rad)

if(RADCPP_BUILD_TESTS)
    add_subdirectory(tests)
endif()
