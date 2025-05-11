add_executable(vkpp_tests)

target_include_directories(vkpp_tests
    PUBLIC ${RADCPP_ROOT}
)
list(APPEND vkpp_tests_SOURCES
    main.cpp
    TensorTest.cpp
)

source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${vkpp_tests_SOURCES})
target_sources(vkpp_tests PRIVATE ${vkpp_tests_SOURCES})
find_package(GTest CONFIG REQUIRED)
target_link_libraries(vkpp_tests
    PRIVATE vkpp
    PRIVATE GTest::gtest GTest::gtest_main GTest::gmock GTest::gmock_main
)

install(
    TARGETS vkpp_tests
    COMPONENT radcpp.tests
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
