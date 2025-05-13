set(PAINTER_SOURCE_FILES
    main.cpp
    PaintManager.h
    PaintManager.cpp
    MainWindow.h
    MainWindow.cpp
    Widget.h
    Widget.cpp
    MainMenu.h
    MainMenu.cpp
)

source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${PAINTER_SOURCE_FILES})

add_executable(Painter ${PAINTER_SOURCE_FILES})

target_include_directories(Painter
    PUBLIC ${RADCPP_ROOT}
)

target_link_libraries(Painter
    PRIVATE SDFramework
)

set_target_properties(Painter PROPERTIES
    FOLDER "samples"
)

add_custom_command(TARGET Painter POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_RUNTIME_DLLS:Painter> $<TARGET_FILE_DIR:Painter>
    COMMAND_EXPAND_LISTS
)
