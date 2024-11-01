set(RAD_SOURCE_FILES
    Core/Platform.h
    Core/Integer.h
    Core/Integer.cpp
    Core/Float.h
    Core/Float.cpp
    Core/Float16.h
    Core/Float16.cpp
    Core/BFloat16.h
    Core/BFloat16.cpp
    Core/Numeric.h
    Core/Numeric.cpp
    Core/TypeTraits.h
    Core/Memory.h
    Core/Memory.cpp
    Core/RefCounted.h
    Core/String.h
    Core/String.cpp
    Core/pystring.h
    Core/pystring.cpp
    Core/Format.h
    Core/Format.cpp
    Core/Flags.h
    Core/Sort.h
    Core/Math.h
    Core/Math.cpp
    Container/Span.h
    Container/SmallVector.h
    IO/File.h
    IO/File.cpp
    IO/FileSystem.h
    IO/FileSystem.cpp
    IO/Json.h
    IO/Json.cpp
    IO/Logging.h
    IO/Logging.cpp
    IO/Compression.h
    IO/Compression.cpp
    System/Application.h
    System/Application.cpp
    System/Process.h
    System/Process.cpp
    System/Time.h
    System/Time.cpp
    System/CpuInfo.h
    System/CpuInfo.cpp
    System/OS.h
    System/OS.cpp
)

source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${RAD_SOURCE_FILES})

add_library(rad STATIC ${RAD_SOURCE_FILES})

find_package(Boost REQUIRED json)
find_package(fmt CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)
find_package(Backward CONFIG REQUIRED)
find_package(CpuFeatures CONFIG REQUIRED)
# vcpkg install minizip-ng[zlib,bzip2,lzma,zstd]:x64-windows
find_package(minizip-ng CONFIG REQUIRED)

find_package (Python3 COMPONENTS Development)

target_include_directories(rad
    PUBLIC ${RADCPP_ROOT}
    PUBLIC ${Boost_INCLUDE_DIRS}
)

target_link_libraries(rad
    PUBLIC ${Boost_LIBRARIES}
    PUBLIC fmt::fmt
    PUBLIC spdlog::spdlog
    PUBLIC Backward::Backward
    PUBLIC CpuFeatures::cpu_features
    PUBLIC MINIZIP::minizip-ng
    PUBLIC Python3::Python
)

if (WIN32)
target_compile_definitions(rad
    PRIVATE NOMINMAX=1
    PRIVATE WIN32_LEAN_AND_MEAN=1
)
target_compile_options(rad
    PRIVATE "$<$<CXX_COMPILER_ID:MSVC>:/utf-8>"
)
endif()
