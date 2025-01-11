set(RAD_HEADERS
    Algorithm/Sort.h
    Container/SmallVector.h
    Container/Span.h
    Container/Vector.h
    Core/BFloat16.h
    Core/Flags.h
    Core/Float.h
    Core/Float16.h
    Core/Format.h
    Core/Integer.h
    Core/Literals.h
    Core/Math.h
    Core/Memory.h
    Core/Numeric.h
    Core/Platform.h
    Core/pystring.h
    Core/Random.h
    Core/RefCounted.h
    Core/String.h
    Core/TypeTraits.h
    IO/Compression.h
    IO/File.h
    IO/FileSystem.h
    IO/Image.h
    IO/Json.h
    IO/Logging.h
    System/Application.h
    System/CpuInfo.h
    System/OS.h
    System/Process.h
    System/Time.h
)

set(RAD_SOURCES
    ${RAD_HEADERS}
    Core/BFloat16.cpp
    Core/Float.cpp
    Core/Float16.cpp
    Core/Format.cpp
    Core/Integer.cpp
    Core/Math.cpp
    Core/Memory.cpp
    Core/Numeric.cpp
    Core/pystring.cpp
    Core/Random.cpp
    Core/String.cpp
    IO/Compression.cpp
    IO/File.cpp
    IO/FileSystem.cpp
    IO/Image.cpp
    IO/Json.cpp
    IO/Logging.cpp
    System/Application.cpp
    System/CpuInfo.cpp
    System/OS.cpp
    System/Process.cpp
    System/Time.cpp
)

source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${RAD_SOURCES})

add_library(rad STATIC ${RAD_SOURCES})

target_sources(rad PUBLIC
    FILE_SET HEADERS
    FILES ${RAD_HEADERS}
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

target_include_directories(rad
    PUBLIC ${RADCPP_ROOT}
    PUBLIC ${Boost_INCLUDE_DIRS}
    PUBLIC ${RADCPP_ROOT}/imported/utfcpp/source
)

target_link_libraries(rad
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
target_compile_definitions(rad
    PRIVATE NOMINMAX=1
    PRIVATE WIN32_LEAN_AND_MEAN=1
)
target_compile_options(rad
    PRIVATE "$<$<CXX_COMPILER_ID:MSVC>:/utf-8>"
)
endif()

install(TARGETS rad
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin
        FILE_SET HEADERS DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/rad
)
