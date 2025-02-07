cmake_minimum_required(VERSION 3.23)

project(hipx LANGUAGES CXX)

set(HIPX_ROOT ${CMAKE_CURRENT_SOURCE_DIR})

if(WIN32)
    set(ROCM_ROOT
        "$ENV{HIP_PATH}"
        CACHE PATH
        "Root directory of the ROCm installation"
    )
else()
    set(ROCM_ROOT
        "/opt/rocm"
        CACHE PATH
        "Root directory of the ROCm installation"
    )
endif()
list(APPEND CMAKE_PREFIX_PATH "${ROCM_ROOT}")

if (WIN32)
    set(CMAKE_C_COMPILER ${ROCM_ROOT}/bin/hipcc.bin.exe)
    set(CMAKE_CXX_COMPILER ${ROCM_ROOT}/bin/hipcc.bin.exe)
endif()

set(GPU_RUNTIME "HIP" CACHE STRING "Switches between HIP and CUDA")
enable_language(${GPU_RUNTIME})

set(CMAKE_${GPU_RUNTIME}_STANDARD 20)
set(CMAKE_${GPU_RUNTIME}_STANDARD_REQUIRED ON)
set(CMAKE_${GPU_RUNTIME}_EXTENSIONS OFF)

find_package(hip REQUIRED PATHS ${ROCM_ROOT})

list(APPEND HIPX_SOURCES
    hipx/Core/HipError.h
    hipx/Core/HipContext.h
    hipx/Core/HipContext.cpp
    hipx/Core/HipDevice.h
    hipx/Core/HipDevice.cpp
)

set_source_files_properties(${HIPX_SOURCES} PROPERTIES LANGUAGE ${GPU_RUNTIME})

list(APPEND TEST_SOURCES
    ${HIPX_SOURCES}
    hipx/test.cpp
)

set_source_files_properties(${TEST_SOURCES} PROPERTIES LANGUAGE ${GPU_RUNTIME})

add_executable(hipx-test
    ${TEST_SOURCES}
)

target_include_directories(hipx-test
    PUBLIC ${HIPX_ROOT}
)
