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
target_sources(rad
    PUBLIC ${RAD_HEADERS}
    PRIVATE ${RAD_PRIVATE_SOURCES}
)
find_package(Threads REQUIRED)
target_link_libraries(rad
    Threads::Threads
)
target_compile_features(rad PUBLIC cxx_std_20)

install(
    TARGETS rad
    COMPONENT radcpp.rad
)
