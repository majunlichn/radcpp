set(TEST_SOURCES
    ExecutionTest.cpp
    IntegerTest.cpp
    StringTest.cpp
    TimeTest.cpp
    main.cpp
)

if (RADCPP_ENABLE_VULKAN)
    list(APPEND TEST_SOURCES
        VulkanTest.cpp
    )
endif()

source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${TEST_SOURCES})

add_executable(radcpp-tests ${TEST_SOURCES})

find_package(GTest CONFIG REQUIRED)

target_link_libraries(radcpp-tests
    PRIVATE radcpp
    PRIVATE GTest::gtest GTest::gtest_main GTest::gmock GTest::gmock_main
    PRIVATE reflect-cpp
)
