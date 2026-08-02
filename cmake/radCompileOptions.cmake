include_guard()

# C++ standard configuration
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
    # https://learn.microsoft.com/en-us/cpp/build/reference/permissive-standards-conformance
    add_compile_options(/permissive-)
endif()

cmake_host_system_information(RESULT CPU_CORE_COUNT QUERY NUMBER_OF_PHYSICAL_CORES)

if(CPU_CORE_COUNT LESS_EQUAL 0)
    include(ProcessorCount)
    ProcessorCount(CPU_CORE_COUNT)
endif()
if(CPU_CORE_COUNT LESS_EQUAL 0)
    set(CPU_CORE_COUNT 1)
endif()
message(STATUS "Detected CPU cores: ${CPU_CORE_COUNT}")

if(MSVC)
    add_compile_options(/MP)
endif()
