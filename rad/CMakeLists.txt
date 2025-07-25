add_library(rad STATIC)

configure_file(Config.h.in rad/Config.h)

target_include_directories(rad
    PUBLIC ${RADCPP_ROOT}
    PUBLIC ${CMAKE_CURRENT_BINARY_DIR}
)

list(APPEND RAD_HEADERS
    Container/SmallVector.h
    Container/Span.h
    Core/Algorithm.h
    Core/Flags.h
    Core/Float.h
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
    Core/ThreadPool.h
    IO/File.h
    IO/Format.h
    IO/Image.h
    IO/Json.h
    IO/Logging.h
    System/Application.h
    System/CpuInfo.h
    System/OS.h
    System/Thread.h
    System/Time.h
)
list(APPEND RAD_PRIVATE_SOURCES
    Core/BFloat16.cpp
    Core/Float.cpp
    Core/Float16.cpp
    Core/Float8.cpp
    Core/Integer.cpp
    Core/Math.cpp
    Core/Memory.cpp
    Core/Numeric.cpp
    Core/Platform.cpp
    Core/Random.cpp
    Core/String.cpp
    Core/pystring.cpp
    Core/ThreadPool.cpp
    IO/File.cpp
    IO/Format.cpp
    IO/Image.cpp
    IO/Json.cpp
    IO/Logging.cpp
    System/Application.cpp
    System/CpuInfo.cpp
    System/OS.cpp
    System/Thread.cpp
    System/Time.cpp
)

source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${RAD_HEADERS} ${RAD_PRIVATE_SOURCES})

target_sources(rad
    PUBLIC FILE_SET HEADERS BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR} FILES ${RAD_HEADERS}
    PRIVATE ${RAD_PRIVATE_SOURCES}
)
target_sources(rad PUBLIC
    FILE_SET HEADERS FILES ${RADCPP_HEADERS}
)

target_include_directories(rad
    PUBLIC ${RADCPP_ROOT}/imported/taskflow
)

find_package(Backward CONFIG REQUIRED)
find_package(boost_json REQUIRED CONFIG)
find_package(boost_locale REQUIRED CONFIG)
find_package(boost_math REQUIRED CONFIG)
find_package(boost_nowide REQUIRED CONFIG)
find_package(boost_process REQUIRED CONFIG)
find_package(CpuFeatures CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)
find_package(Threads REQUIRED)
find_package(libdeflate CONFIG REQUIRED)

target_link_libraries(rad
    PUBLIC Backward::Backward
    PUBLIC Boost::json
    PUBLIC Boost::locale
    PUBLIC Boost::math
    PUBLIC Boost::nowide
    PUBLIC Boost::process
    PUBLIC CpuFeatures::cpu_features
    PUBLIC glm::glm
    PUBLIC spdlog::spdlog
    PUBLIC Threads::Threads
    PUBLIC STDEXEC::stdexec
    PUBLIC $<IF:$<TARGET_EXISTS:libdeflate::libdeflate_shared>,libdeflate::libdeflate_shared,libdeflate::libdeflate_static>
)

target_compile_features(rad PUBLIC cxx_std_20)

if (WIN32)
    target_compile_definitions(rad
        PRIVATE NOMINMAX=1
        PRIVATE WIN32_LEAN_AND_MEAN=1
    )
    target_compile_options(rad
        PRIVATE "$<$<CXX_COMPILER_ID:MSVC>:/utf-8>"
    )
    target_link_libraries(rad
        PUBLIC ${CMAKE_BINARY_DIR}/vcpkg_installed/x64-windows/lib/hwloc.lib
    )
endif()

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/rad/Config.h DESTINATION include/rad)

install(
    TARGETS rad
    COMPONENT radcpp.rad
    FILE_SET HEADERS DESTINATION include/rad
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
