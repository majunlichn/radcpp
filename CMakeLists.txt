cmake_minimum_required(VERSION 3.10)

if (DEFINED ENV{VCPKG_ROOT})
    list(APPEND CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake")
endif()
message("CMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}")

project(radcpp)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(RADCPP_ROOT ${CMAKE_CURRENT_SOURCE_DIR})

include_directories(
    ${RADCPP_ROOT}
)

add_subdirectory(rad)
add_subdirectory(tests)
