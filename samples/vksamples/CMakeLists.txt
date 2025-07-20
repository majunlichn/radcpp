set(VKSAMPLES_SOURCE_FILES
    main.cpp
    Common/Sample.h
    Common/Sample.cpp
    Cube/CubeDemo.h
    Cube/CubeDemo.cpp
)

source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${VKSAMPLES_SOURCE_FILES})

add_executable(vksamples ${VKSAMPLES_SOURCE_FILES})

target_include_directories(vksamples
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE ${RADCPP_ROOT}
)

target_link_libraries(vksamples
    PRIVATE SDFramework
)

set_target_properties(vksamples PROPERTIES
    FOLDER "samples"
)

add_custom_command(TARGET vksamples POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_RUNTIME_DLLS:vksamples> $<TARGET_FILE_DIR:vksamples>
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/Cube/cube.vert ${CMAKE_CURRENT_BINARY_DIR}/Cube/cube.vert
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/Cube/cube.frag ${CMAKE_CURRENT_BINARY_DIR}/Cube/cube.frag
    COMMAND_EXPAND_LISTS
)

set_property(TARGET vksamples PROPERTY VS_DPI_AWARE "PerMonitor")
