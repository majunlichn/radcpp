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
    Core/Memory.h
    Core/Platform.h
    Core/pystring.h
    Core/RefCounted.h
    Core/String.h
    Core/TypeTraits.h
    IO/File.h
    IO/Json.h
    IO/Logging.h
)
list(APPEND RAD_PRIVATE_SOURCES
    Core/BFloat16.cpp
    Core/Float.cpp
    Core/Float16.cpp
    Core/Float8.cpp
    Core/Integer.cpp
    Core/Memory.cpp
    Core/Platform.cpp
    Core/String.cpp
    Core/pystring.cpp
    IO/File.cpp
    IO/Json.cpp
    IO/Logging.cpp
)

source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${RAD_HEADERS} ${RAD_PRIVATE_SOURCES})

target_sources(rad
    PUBLIC FILE_SET HEADERS BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR} FILES ${RAD_HEADERS}
    PRIVATE ${RAD_PRIVATE_SOURCES}
)
target_sources(rad PUBLIC
    FILE_SET HEADERS FILES ${RADCPP_HEADERS}
)

find_package(boost_json REQUIRED CONFIG)
find_package(boost_locale REQUIRED CONFIG)
find_package(boost_math REQUIRED CONFIG)
find_package(spdlog CONFIG REQUIRED)
find_package(Threads REQUIRED)
target_link_libraries(rad
    PUBLIC Boost::json
    PUBLIC Boost::locale
    PUBLIC Boost::math
    PUBLIC spdlog::spdlog
    PUBLIC Threads::Threads
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
