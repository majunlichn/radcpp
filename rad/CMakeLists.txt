add_library(rad STATIC)

configure_file(Config.h.in rad/Config.h)

target_include_directories(rad
    PUBLIC ${RADCPP_ROOT}
    PUBLIC ${CMAKE_CURRENT_BINARY_DIR}
)

list(APPEND RAD_HEADERS
    Integer.h
    Platform.h
)
list(APPEND RAD_PRIVATE_SOURCES
    Integer.cpp
    Platform.cpp
)

source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${RAD_HEADERS} ${RAD_PRIVATE_SOURCES})

target_sources(rad
    PUBLIC FILE_SET HEADERS BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR} FILES ${RAD_HEADERS}
    PRIVATE ${RAD_PRIVATE_SOURCES}
)
target_sources(rad PUBLIC
    FILE_SET HEADERS FILES ${RADCPP_HEADERS}
)

find_package(Threads REQUIRED)
target_link_libraries(rad
    Threads::Threads
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

install(
    TARGETS rad
    COMPONENT radcpp.rad
    FILE_SET HEADERS DESTINATION include/rad
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
