set(RAD_TESTS_SOURCE_FILES
    Core/Integer.cpp
    main.cpp
)

source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${RAD_TESTS_SOURCE_FILES})

add_executable(rad-tests ${RAD_TESTS_SOURCE_FILES})

find_package(GTest CONFIG REQUIRED)
target_link_libraries(rad-tests
    PRIVATE rad
    PRIVATE GTest::gtest GTest::gtest_main GTest::gmock GTest::gmock_main
)
