add_executable(rad_tests)

list(APPEND rad_tests_SOURCES
    ContainerTest.cpp
    FloatTest.cpp
    IntegerTest.cpp
    MemoryTest.cpp
    StackTraceTest.cpp
    TaskflowTest.cpp
    ExecutionTest.cpp
    GeometryTest.cpp
    TableTest.cpp
    main.cpp
)
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${rad_tests_SOURCES})
target_sources(rad_tests PRIVATE ${rad_tests_SOURCES})

find_package(GTest CONFIG REQUIRED)

target_include_directories(rad_tests
    PUBLIC ${RADCPP_ROOT}
)

target_link_libraries(rad_tests
    PRIVATE rad
    PRIVATE GTest::gtest GTest::gtest_main GTest::gmock GTest::gmock_main
)

install(
    TARGETS rad_tests
    COMPONENT radcpp.tests
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
