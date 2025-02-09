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

set(RADCPP_ROOT ${CMAKE_CURRENT_SOURCE_DIR})

option(RADCPP_BUILD_TESTS "Build radcpp tests." ON)
option(RADCPP_INSTALL "Enable radcpp intall." ON)

if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
    set(RADCPP_IS_SUBPROJECT False)
else()
    set(RADCPP_IS_SUBPROJECT True)
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
    rad/Algorithm/Sort.h
    rad/Container/SmallVector.h
    rad/Container/Span.h
    rad/Container/Vector.h
    rad/Core/BFloat16.h
    rad/Core/Flags.h
    rad/Core/Float.h
    rad/Core/Float16.h
    rad/Core/Format.h
    rad/Core/Integer.h
    rad/Core/Literals.h
    rad/Core/Math.h
    rad/Core/Memory.h
    rad/Core/Numeric.h
    rad/Core/Platform.h
    rad/Core/pystring.h
    rad/Core/Random.h
    rad/Core/RefCounted.h
    rad/Core/String.h
    rad/Core/TypeTraits.h
    rad/IO/Compression.h
    rad/IO/File.h
    rad/IO/FileSystem.h
    rad/IO/Image.h
    rad/IO/Json.h
    rad/IO/Logging.h
    rad/System/Application.h
    rad/System/CpuInfo.h
    rad/System/OS.h
    rad/System/Process.h
    rad/System/Time.h
)

list(APPEND RADCPP_SOURCES
    ${RADCPP_HEADERS}
    rad/Core/BFloat16.cpp
    rad/Core/Float.cpp
    rad/Core/Float16.cpp
    rad/Core/Format.cpp
    rad/Core/Integer.cpp
    rad/Core/Math.cpp
    rad/Core/Memory.cpp
    rad/Core/Numeric.cpp
    rad/Core/pystring.cpp
    rad/Core/Random.cpp
    rad/Core/String.cpp
    rad/IO/Compression.cpp
    rad/IO/File.cpp
    rad/IO/FileSystem.cpp
    rad/IO/Image.cpp
    rad/IO/Json.cpp
    rad/IO/Logging.cpp
    rad/System/Application.cpp
    rad/System/CpuInfo.cpp
    rad/System/OS.cpp
    rad/System/Process.cpp
    rad/System/Time.cpp
)

source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR}/rad FILES ${RADCPP_SOURCES})

add_library(radcpp STATIC ${RADCPP_SOURCES})

target_sources(radcpp PUBLIC
    FILE_SET HEADERS FILES ${RADCPP_HEADERS}
)

find_package(Boost REQUIRED json)
find_package(Eigen3 CONFIG REQUIRED)
find_package(fmt CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)
find_package(Backward CONFIG REQUIRED)
find_package(CpuFeatures CONFIG REQUIRED)
# vcpkg install minizip-ng[zlib,bzip2,lzma,zstd]:x64-windows
find_package(minizip-ng CONFIG REQUIRED)

find_package(Python3 COMPONENTS Development)

target_include_directories(radcpp
    PUBLIC ${RADCPP_ROOT}
    PUBLIC ${Boost_INCLUDE_DIRS}
    PUBLIC ${RADCPP_ROOT}/imported/utfcpp/source
)

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
    install(TARGETS radcpp
        FILE_SET HEADERS DESTINATION include
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
    )
endif()

if (RADCPP_BUILD_TESTS)
    add_subdirectory(tests)
endif()
