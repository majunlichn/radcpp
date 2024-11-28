set(REFLECT_CPP_SOURCES
    src/reflectcpp.cpp
    src/reflectcpp_json.cpp
    src/yyjson.c
)

source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${REFLECT_CPP_SOURCES})

add_library(reflect-cpp STATIC
    ${REFLECT_CPP_SOURCES}
)

target_include_directories(reflect-cpp
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
)

set_target_properties(reflect-cpp PROPERTIES
    FOLDER "imported"
)
