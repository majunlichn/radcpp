cmake_minimum_required(VERSION 3.10)

message(STATUS "CMake version: ${CMAKE_VERSION}")

if(POLICY CMP0167)
    # find_package(Boost): without the CONFIG or NO_MODULE options, find the upstream BoostConfig.cmake directly.
    # https://cmake.org/cmake/help/latest/policy/CMP0167.html
    cmake_policy(SET CMP0167 NEW)
endif()

if((NOT DEFINED CMAKE_TOOLCHAIN_FILE) AND (DEFINED ENV{VCPKG_ROOT}))
    list(APPEND CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake")
endif()
message("CMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}")

project(radcpp)

if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
    set(RADCPP_SUBPROJECT OFF)
else()
    set(RADCPP_SUBPROJECT ON)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include(cmake/CompilerOptions.cmake)
include(cmake/Sanitizers.cmake)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(RADCPP_ROOT ${CMAKE_CURRENT_SOURCE_DIR})

include_directories(
    ${RADCPP_ROOT}
)

add_subdirectory(rad)

if(NOT ${RADCPP_SUBPROJECT})
    add_subdirectory(tests)
endif()
