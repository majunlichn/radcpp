cmake_minimum_required(VERSION 3.23)

message(STATUS "Configuring radcpp ...")
message(STATUS "CMake Version: ${CMAKE_VERSION}")
message(STATUS "CMake Generator: ${CMAKE_GENERATOR}")

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

set(RADCPP_ROOT ${CMAKE_CURRENT_SOURCE_DIR})

option(RADCPP_ENABLE_VULKAN "Enable radcpp Vulkan support." ON)
if (RADCPP_ENABLE_VULKAN AND (NOT DEFINED ENV{VULKAN_SDK}))
    message("ENV{VULKAN_SDK} is not defined, build with no Vulkan support.")
    set(RADCPP_ENABLE_VULKAN OFF)
endif()
option(RADCPP_BUILD_TESTS "Build radcpp tests." ${PROJECT_IS_TOP_LEVEL})
option(RADCPP_INSTALL "Enable radcpp intall." ON)

# PROJECT_IS_TOP_LEVEL
if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
    set(RADCPP_IS_TOP_LEVEL False)
else()
    set(RADCPP_IS_TOP_LEVEL True)
endif()

include(cmake/CompilerOptions.cmake)
include(cmake/Sanitizers.cmake)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

add_subdirectory(imported/reflect-cpp)

set(STDEXEC_BUILD_TESTS OFF)
set(STDEXEC_BUILD_DOCS OFF)
set(STDEXEC_BUILD_EXAMPLES OFF)
add_subdirectory(imported/stdexec)
set_target_properties(stdexec PROPERTIES
    FOLDER "imported"
)
set_target_properties(system_context PROPERTIES
    FOLDER "imported"
)

list(APPEND RADCPP_HEADERS
    ${RADCPP_ROOT}/radcpp/Algorithm/Sort.h
    ${RADCPP_ROOT}/radcpp/Container/SmallVector.h
    ${RADCPP_ROOT}/radcpp/Container/Span.h
    ${RADCPP_ROOT}/radcpp/Container/Vector.h
    ${RADCPP_ROOT}/radcpp/Core/BFloat16.h
    ${RADCPP_ROOT}/radcpp/Core/Flags.h
    ${RADCPP_ROOT}/radcpp/Core/Float.h
    ${RADCPP_ROOT}/radcpp/Core/Float8.h
    ${RADCPP_ROOT}/radcpp/Core/Float16.h
    ${RADCPP_ROOT}/radcpp/Core/Format.h
    ${RADCPP_ROOT}/radcpp/Core/Integer.h
    ${RADCPP_ROOT}/radcpp/Core/Literals.h
    ${RADCPP_ROOT}/radcpp/Core/Math.h
    ${RADCPP_ROOT}/radcpp/Core/Memory.h
    ${RADCPP_ROOT}/radcpp/Core/Numeric.h
    ${RADCPP_ROOT}/radcpp/Core/Platform.h
    ${RADCPP_ROOT}/radcpp/Core/pystring.h
    ${RADCPP_ROOT}/radcpp/Core/Random.h
    ${RADCPP_ROOT}/radcpp/Core/RefCounted.h
    ${RADCPP_ROOT}/radcpp/Core/String.h
    ${RADCPP_ROOT}/radcpp/Core/TypeTraits.h
    ${RADCPP_ROOT}/radcpp/IO/Compression.h
    ${RADCPP_ROOT}/radcpp/IO/File.h
    ${RADCPP_ROOT}/radcpp/IO/FileSystem.h
    ${RADCPP_ROOT}/radcpp/IO/Image.h
    ${RADCPP_ROOT}/radcpp/IO/Json.h
    ${RADCPP_ROOT}/radcpp/IO/Logging.h
    ${RADCPP_ROOT}/radcpp/System/Application.h
    ${RADCPP_ROOT}/radcpp/System/CpuInfo.h
    ${RADCPP_ROOT}/radcpp/System/OS.h
    ${RADCPP_ROOT}/radcpp/System/Process.h
    ${RADCPP_ROOT}/radcpp/System/Thread.h
    ${RADCPP_ROOT}/radcpp/System/Time.h
)

list(APPEND RADCPP_SOURCES
    ${RADCPP_ROOT}/radcpp/Core/BFloat16.cpp
    ${RADCPP_ROOT}/radcpp/Core/Float.cpp
    ${RADCPP_ROOT}/radcpp/Core/Float8.cpp
    ${RADCPP_ROOT}/radcpp/Core/Float16.cpp
    ${RADCPP_ROOT}/radcpp/Core/Format.cpp
    ${RADCPP_ROOT}/radcpp/Core/Integer.cpp
    ${RADCPP_ROOT}/radcpp/Core/Math.cpp
    ${RADCPP_ROOT}/radcpp/Core/Memory.cpp
    ${RADCPP_ROOT}/radcpp/Core/Numeric.cpp
    ${RADCPP_ROOT}/radcpp/Core/pystring.cpp
    ${RADCPP_ROOT}/radcpp/Core/Random.cpp
    ${RADCPP_ROOT}/radcpp/Core/String.cpp
    ${RADCPP_ROOT}/radcpp/IO/Compression.cpp
    ${RADCPP_ROOT}/radcpp/IO/File.cpp
    ${RADCPP_ROOT}/radcpp/IO/FileSystem.cpp
    ${RADCPP_ROOT}/radcpp/IO/Image.cpp
    ${RADCPP_ROOT}/radcpp/IO/Json.cpp
    ${RADCPP_ROOT}/radcpp/IO/Logging.cpp
    ${RADCPP_ROOT}/radcpp/System/Application.cpp
    ${RADCPP_ROOT}/radcpp/System/CpuInfo.cpp
    ${RADCPP_ROOT}/radcpp/System/OS.cpp
    ${RADCPP_ROOT}/radcpp/System/Process.cpp
    ${RADCPP_ROOT}/radcpp/System/Thread.cpp
    ${RADCPP_ROOT}/radcpp/System/Time.cpp
)

message("RADCPP_ENABLE_VULKAN=${RADCPP_ENABLE_VULKAN}")
if (RADCPP_ENABLE_VULKAN)
    list(APPEND RADCPP_HEADERS
        ${RADCPP_ROOT}/radcpp/GPU/VulkanCommon.h
        ${RADCPP_ROOT}/radcpp/GPU/VulkanInstance.h
        ${RADCPP_ROOT}/radcpp/GPU/VulkanDevice.h
    )
    list(APPEND RADCPP_SOURCES
        ${RADCPP_ROOT}/radcpp/GPU/VulkanCommon.cpp
        ${RADCPP_ROOT}/radcpp/GPU/VulkanInstance.cpp
        ${RADCPP_ROOT}/radcpp/GPU/VulkanDevice.cpp
    )
endif()

set(RADCPP_SOURCES ${RADCPP_HEADERS} ${RADCPP_SOURCES})

source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR}/radcpp FILES ${RADCPP_SOURCES})

add_library(radcpp STATIC ${RADCPP_SOURCES})

target_sources(radcpp PUBLIC
    FILE_SET HEADERS FILES ${RADCPP_HEADERS}
)

find_package(Boost REQUIRED COMPONENTS json)
find_package(Eigen3 CONFIG REQUIRED)
find_package(fmt CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)
find_package(Backward CONFIG REQUIRED)
find_package(CpuFeatures CONFIG REQUIRED)
# vcpkg install minizip-ng[zlib,bzip2,lzma,zstd]:x64-windows
find_package(minizip-ng CONFIG REQUIRED)

find_package(Python3 COMPONENTS Development)

target_include_directories(radcpp
    PUBLIC $<BUILD_INTERFACE:${RADCPP_ROOT}>
    PUBLIC $<BUILD_INTERFACE:${RADCPP_ROOT}/imported/utfcpp/source>
    PUBLIC $<INSTALL_INTERFACE:include>
    PUBLIC $<INSTALL_INTERFACE:include/utfcpp/source>
    PUBLIC ${Boost_INCLUDE_DIRS}
)

if (RADCPP_ENABLE_VULKAN)
    target_include_directories(radcpp
        PUBLIC $ENV{VULKAN_SDK}/Include
    )
    target_compile_definitions(radcpp
        PUBLIC VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1
    )
endif()

target_link_libraries(radcpp
    PUBLIC ${Boost_LIBRARIES}
    PUBLIC Eigen3::Eigen
    PUBLIC fmt::fmt
    PUBLIC spdlog::spdlog
    PUBLIC Backward::Backward
    PUBLIC CpuFeatures::cpu_features
    PUBLIC MINIZIP::minizip-ng
    PUBLIC Python3::Python
    PUBLIC STDEXEC::stdexec
)

if (WIN32)
    target_compile_definitions(radcpp
        PRIVATE NOMINMAX=1
        PRIVATE WIN32_LEAN_AND_MEAN=1
    )
    target_compile_options(radcpp
        PRIVATE "$<$<CXX_COMPILER_ID:MSVC>:/utf-8>"
    )
endif()

if (RADCPP_INSTALL)
    set(RADCPP_INSTALL_CMAKEDIR "share/radcpp")
    install(DIRECTORY "${RADCPP_ROOT}/imported/utfcpp" DESTINATION include)
    install(TARGETS radcpp EXPORT radcppTargets
        FILE_SET HEADERS DESTINATION include
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
    )
    install(EXPORT radcppTargets
        FILE radcppTargets.cmake
        DESTINATION ${RADCPP_INSTALL_CMAKEDIR}
    )
    include(CMakePackageConfigHelpers)
    configure_package_config_file(${RADCPP_ROOT}/cmake/radcppConfig.cmake.in
        "${CMAKE_CURRENT_BINARY_DIR}/radcppConfig.cmake"
        INSTALL_DESTINATION ${RADCPP_INSTALL_CMAKEDIR}
        NO_SET_AND_CHECK_MACRO
        NO_CHECK_REQUIRED_COMPONENTS_MACRO
    )
    write_basic_package_version_file(
        "${CMAKE_CURRENT_BINARY_DIR}/radcppConfigVersion.cmake"
        VERSION "${RADCPP_VERSION_MAJOR}.${RADCPP_VERSION_MINOR}.${RADCPP_VERSION_PATCH}"
        COMPATIBILITY AnyNewerVersion
    )
    install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/radcppConfig.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/radcppConfigVersion.cmake
        DESTINATION ${RADCPP_INSTALL_CMAKEDIR}
    )
    export(EXPORT radcppTargets
        FILE "${CMAKE_CURRENT_BINARY_DIR}/radcppTargets.cmake"
    )
endif()

if (RADCPP_BUILD_TESTS)
    add_subdirectory(tests)
endif()
