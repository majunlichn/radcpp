set(TESTS_SOURCE_FILES
    Core/Integer.cpp
    main.cpp
)

source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${TESTS_SOURCE_FILES})

add_executable(tests ${TESTS_SOURCE_FILES})

find_package(GTest CONFIG REQUIRED)
target_link_libraries(tests
    PRIVATE rad
    PRIVATE GTest::gtest GTest::gtest_main GTest::gmock GTest::gmock_main
)
