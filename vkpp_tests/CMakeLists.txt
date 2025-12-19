add_executable(vkpp_tests)

list(APPEND vkpp_tests_SOURCES
    main.cpp
    SlangTest.cpp
)
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${vkpp_tests_SOURCES})
target_sources(vkpp_tests PRIVATE ${vkpp_tests_SOURCES})

find_package(GTest CONFIG REQUIRED)

target_include_directories(vkpp_tests
    PUBLIC ${RADCPP_ROOT}
)

target_link_libraries(vkpp_tests
    PRIVATE vkpp
    PRIVATE GTest::gtest GTest::gtest_main GTest::gmock GTest::gmock_main
)

add_custom_command(TARGET vkpp_tests POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_RUNTIME_DLLS:vkpp_tests> $<TARGET_FILE_DIR:vkpp_tests>
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${RADCPP_ROOT}/vkpp/Shaders ${CMAKE_CURRENT_BINARY_DIR}/Shaders
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/hello-world.slang ${CMAKE_CURRENT_BINARY_DIR}/hello-world.slang
    COMMAND_EXPAND_LISTS
)

if(WIN32)
    add_custom_command(TARGET vkpp_tests POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
            ${RADCPP_ROOT}/imported/slang/bin/slang-glslang.dll
            ${RADCPP_ROOT}/imported/slang/bin/slang-glsl-module.dll
            $<TARGET_FILE_DIR:vkpp_tests>
        COMMAND_EXPAND_LISTS
    )
endif()

install(
    TARGETS vkpp_tests
    COMPONENT radcpp.tests
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
