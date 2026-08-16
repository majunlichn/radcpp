cmake_minimum_required(VERSION 3.25)

# Setup vcpkg toolchain file automatically.
if(NOT DEFINED CMAKE_TOOLCHAIN_FILE AND DEFINED ENV{VCPKG_ROOT})
    set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake"
        CACHE FILEPATH "vcpkg toolchain file" FORCE)
endif()
if(DEFINED CMAKE_TOOLCHAIN_FILE)
    cmake_path(NORMAL_PATH CMAKE_TOOLCHAIN_FILE)
endif()
message(STATUS "CMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}")

project(radcpp VERSION 0.0.0 LANGUAGES CXX)

set(RAD_PROJECT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Build options
option(RAD_BUILD_TESTS "Build tests" ON)
option(RAD_ENABLE_INSTALL "Enable installation rules" ON)

include(GNUInstallDirs)
include(cmake/radCompileOptions.cmake)
include(cmake/radSanitizers.cmake)

get_property(IS_MULTICONFIG_GENERATOR GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)

# Set output directories
if(IS_MULTICONFIG_GENERATOR)
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib/$<CONFIG>")
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib/$<CONFIG>")
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/$<CONFIG>")
else()
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
endif()

# Set default install prefix (if not specified by user)
if(WIN32 AND CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set_property(CACHE CMAKE_INSTALL_PREFIX PROPERTY VALUE "${CMAKE_BINARY_DIR}/install")
    message(STATUS "Setting default install prefix: ${CMAKE_INSTALL_PREFIX}")
endif()

if(RAD_BUILD_TESTS)
    enable_testing()
    find_package(GTest CONFIG REQUIRED)
    include(GoogleTest)
endif()

set(RAD_SOURCES
    src/rad/Container/SmallVector.h
    src/rad/Core/BFloat16.h
    src/rad/Core/BFloat16.cpp
    src/rad/Core/Float.h
    src/rad/Core/Float.cpp
    src/rad/Core/Float16.h
    src/rad/Core/Float16.cpp
    src/rad/Core/Float8.h
    src/rad/Core/Float8.cpp
    src/rad/Core/Flags.h
    src/rad/Core/Hash.h
    src/rad/Core/Hash.cpp
    src/rad/Core/Integer.h
    src/rad/Core/IntegerFallback.h
    src/rad/Core/Integer.cpp
    src/rad/Core/Memory.h
    src/rad/Core/Memory.cpp
    src/rad/Core/Platform.h
    src/rad/Core/Platform.cpp
    src/rad/Core/Random.h
    src/rad/Core/Random.cpp
    src/rad/Core/Range.h
    src/rad/Core/RefCounted.h
    src/rad/Core/Result.h
    src/rad/Core/Span.h
    src/rad/Core/Sort.h
    src/rad/Core/String.h
    src/rad/Core/String.cpp
    src/rad/Core/TypeTraits.h
    src/rad/Core/Unicode.h
    src/rad/Core/Unicode.cpp
    src/rad/Diagnostics/Exception.h
    src/rad/Diagnostics/Exception.cpp
    src/rad/Diagnostics/StackTrace.h
    src/rad/Diagnostics/StackTrace.cpp
    src/rad/IO/Image.h
    src/rad/IO/Image.cpp
    src/rad/IO/Logging.h
    src/rad/IO/Logging.cpp
    src/rad/System/Application.h
    src/rad/System/Application.cpp
    src/rad/System/CpuInfo.h
    src/rad/System/CpuInfo.cpp
    src/rad/System/OS.h
    src/rad/System/OS.cpp
    src/rad/System/Process.h
    src/rad/System/Process.cpp
    src/rad/System/Thread.h
    src/rad/System/Thread.cpp
    src/rad/System/Time.h
    src/rad/System/Time.cpp
)

set(RAD_TEST_SOURCES
    src/rad/TestMain.cpp
    src/rad/Core/BFloat16.test.cpp
    src/rad/Core/Float.test.cpp
    src/rad/Core/Float8.test.cpp
    src/rad/Core/Flags.test.cpp
    src/rad/Core/Integer.test.cpp
    src/rad/Core/Memory.test.cpp
    src/rad/Core/Platform.test.cpp
    src/rad/Core/Span.test.cpp
    src/rad/Core/Sort.test.cpp
    src/rad/Core/String.test.cpp
    src/rad/Core/Unicode.test.cpp
    src/rad/Diagnostics/Exception.test.cpp
    src/rad/Diagnostics/StackTrace.test.cpp
    src/rad/IO/Image.test.cpp
    src/rad/IO/Logging.test.cpp
    src/rad/System/Application.test.cpp
    src/rad/System/CpuInfo.test.cpp
    src/rad/System/OS.test.cpp
    src/rad/System/Process.test.cpp
    src/rad/System/Thread.test.cpp
    src/rad/System/Time.test.cpp
)

add_library(pcg_cpp INTERFACE)
target_include_directories(pcg_cpp
    INTERFACE
        "${RAD_PROJECT_ROOT}/thirdparty/pcg-cpp/include"
)

add_library(rad STATIC)
add_library(rad::rad ALIAS rad)

target_sources(rad PRIVATE ${RAD_SOURCES})
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR}/src/rad FILES ${RAD_SOURCES})

find_package(Imath CONFIG REQUIRED)
find_package(Backward CONFIG REQUIRED)
find_package(boost_algorithm CONFIG REQUIRED)
find_package(boost_container CONFIG REQUIRED)
find_package(boost_exception CONFIG REQUIRED)
find_package(boost_filesystem CONFIG REQUIRED)
find_package(boost_locale CONFIG REQUIRED)
find_package(boost_nowide CONFIG REQUIRED)
find_package(boost_outcome CONFIG REQUIRED)
find_package(boost_process CONFIG REQUIRED)
find_package(boost_smart_ptr CONFIG REQUIRED)
find_package(CpuFeatures CONFIG REQUIRED)
if(WIN32)
    find_package(boost_stacktrace_windbg CONFIG REQUIRED)
    set(RAD_STACKTRACE_BACKEND Boost::stacktrace_windbg)
else()
    find_package(boost_stacktrace_basic CONFIG REQUIRED)
    set(RAD_STACKTRACE_BACKEND Boost::stacktrace_basic)
endif()
find_package(pystring CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)
find_package(Stb REQUIRED)
find_package(Threads REQUIRED)
find_package(xxHash CONFIG REQUIRED)

target_include_directories(rad
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
        "${Stb_INCLUDE_DIR}"
)

target_link_libraries(rad
    PRIVATE Boost::algorithm
    PUBLIC Boost::container
    PUBLIC Boost::exception
    PRIVATE Boost::filesystem
    PRIVATE Boost::locale
    PRIVATE Boost::nowide
    PUBLIC Boost::outcome
    PRIVATE Boost::process
    PRIVATE Backward::Backward
    PUBLIC CpuFeatures::cpu_features
    PUBLIC ${RAD_STACKTRACE_BACKEND}
    PRIVATE "$<BUILD_INTERFACE:pcg_cpp>"
    PUBLIC Boost::smart_ptr
    PRIVATE pystring::pystring
    PUBLIC spdlog::spdlog
    PRIVATE Threads::Threads
    PUBLIC Imath::Imath
    PUBLIC xxHash::xxhash
)

if(DEFINED RAD_ENABLE_MEMORY_TRACKING)
    if(RAD_ENABLE_MEMORY_TRACKING)
        target_compile_definitions(rad PUBLIC RAD_ENABLE_MEMORY_TRACKING=1)
    else()
        target_compile_definitions(rad PUBLIC RAD_ENABLE_MEMORY_TRACKING=0)
    endif()
endif()

if(WIN32)
    target_link_libraries(rad PRIVATE bcrypt advapi32)
    target_compile_definitions(rad
        PUBLIC NOMINMAX=1
        PUBLIC WIN32_LEAN_AND_MEAN=1
        PUBLIC _WIN32_WINNT=0x0A00
    )
    target_compile_options(rad
        PUBLIC "$<$<CXX_COMPILER_ID:MSVC>:/utf-8>"
        PUBLIC "$<$<CXX_COMPILER_ID:MSVC>:/Zc:preprocessor>"
    )
endif()

if(RAD_BUILD_TESTS)
    add_executable(rad_tests)
    target_sources(rad_tests PRIVATE ${RAD_TEST_SOURCES})
    source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR}/src/rad FILES ${RAD_TEST_SOURCES})
    target_link_libraries(rad_tests
        PRIVATE rad
        PRIVATE GTest::gtest
    )
    set_target_properties(rad_tests PROPERTIES
        VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/bin/$<CONFIG>"
    )
    gtest_discover_tests(rad_tests)
endif()

if(RAD_ENABLE_INSTALL)
    include(CMakePackageConfigHelpers)

    install(
        TARGETS rad
        EXPORT radTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )

    install(
        DIRECTORY "${RAD_PROJECT_ROOT}/src/rad/"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/rad"
        FILES_MATCHING
            PATTERN "*.h"
    )

    set(RAD_INSTALL_CMAKEDIR "${CMAKE_INSTALL_LIBDIR}/cmake/rad")

    install(
        EXPORT radTargets
        NAMESPACE rad::
        DESTINATION "${RAD_INSTALL_CMAKEDIR}"
    )

    configure_package_config_file(
        cmake/radConfig.cmake.in
        "${CMAKE_CURRENT_BINARY_DIR}/radConfig.cmake"
        INSTALL_DESTINATION "${RAD_INSTALL_CMAKEDIR}"
    )

    write_basic_package_version_file(
        "${CMAKE_CURRENT_BINARY_DIR}/radConfigVersion.cmake"
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY SameMajorVersion
    )

    install(
        FILES
            "${CMAKE_CURRENT_BINARY_DIR}/radConfig.cmake"
            "${CMAKE_CURRENT_BINARY_DIR}/radConfigVersion.cmake"
        DESTINATION "${RAD_INSTALL_CMAKEDIR}"
    )
endif()

# https://cmake.org/cmake/help/latest/prop_gbl/USE_FOLDERS.html
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
