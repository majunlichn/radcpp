set(REFLECT_CPP_SOURCES
    src/reflectcpp.cpp
    src/reflectcpp_json.cpp
    src/yyjson.c
)

set_source_files_properties(src/yyjson.c PROPERTIES LANGUAGE CXX)

source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${REFLECT_CPP_SOURCES})

add_library(reflect-cpp STATIC
    ${REFLECT_CPP_SOURCES}
)

target_include_directories(reflect-cpp
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/rfl/thirdparty
)

set_target_properties(reflect-cpp PROPERTIES
    FOLDER "imported"
)
