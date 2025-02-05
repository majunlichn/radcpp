cmake_minimum_required(VERSION 3.23)

message("Configuring radcpp ...")
message("CMake Version: ${CMAKE_VERSION}")
message("CMake Generator: ${CMAKE_GENERATOR} (${CMAKE_GENERATOR_PLATFORM})")

if(POLICY CMP0167)
    # find_package(Boost): without the CONFIG or NO_MODULE options, find the upstream BoostConfig.cmake directly.
    # https://cmake.org/cmake/help/latest/policy/CMP0167.html
    cmake_policy(SET CMP0167 NEW)
endif()

if((NOT DEFINED CMAKE_TOOLCHAIN_FILE) AND (DEFINED ENV{VCPKG_ROOT}))
    list(APPEND CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake")
endif()
cmake_path(CONVERT ${CMAKE_TOOLCHAIN_FILE} TO_CMAKE_PATH_LIST ${CMAKE_TOOLCHAIN_FILE} NORMALIZE)
list(REMOVE_DUPLICATES ${CMAKE_TOOLCHAIN_FILE})
message("CMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}")

set(RADCPP_VERSION_MAJOR 0)
set(RADCPP_VERSION_MINOR 0)
set(RADCPP_VERSION_PATCH 0)
set(RADCPP_VERSION "${RADCPP_VERSION_MAJOR}.${RADCPP_VERSION_MINOR}.${RADCPP_VERSION_PATCH}")

project(radcpp
    VERSION ${RADCPP_VERSION}
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

option(RADCPP_BUILD_TESTS "Enable tests." ON)
option(RADCPP_INSTALL "Enable install." ON)

if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
    set(RADCPP_IS_SUBPROJECT False)
else()
    set(RADCPP_IS_SUBPROJECT True)
endif()

include(cmake/CompilerOptions.cmake)
include(cmake/Sanitizers.cmake)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(RADCPP_ROOT ${CMAKE_CURRENT_SOURCE_DIR})

include_directories(
    ${RADCPP_ROOT}
)

add_subdirectory(imported/reflect-cpp)
set(STDEXEC_BUILD_TESTS OFF)
set(STDEXEC_BUILD_EXAMPLES OFF)
add_subdirectory(imported/stdexec)
set_target_properties(stdexec PROPERTIES
    FOLDER "imported"
)
set_target_properties(system_context PROPERTIES
    FOLDER "imported"
)
add_subdirectory(rad)
