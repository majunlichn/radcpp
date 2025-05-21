set(VKCUBE_SOURCE_FILES
    main.cpp
    CubeDemo.h
    CubeDemo.cpp
)

source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${VKCUBE_SOURCE_FILES})

add_executable(vkcube ${VKCUBE_SOURCE_FILES})

target_include_directories(vkcube
    PUBLIC ${RADCPP_ROOT}
)

target_link_libraries(vkcube
    PRIVATE SDFramework
)

set_target_properties(vkcube PROPERTIES
    FOLDER "samples"
)

add_custom_command(TARGET vkcube POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_RUNTIME_DLLS:vkcube> $<TARGET_FILE_DIR:vkcube>
    COMMAND_EXPAND_LISTS
)
