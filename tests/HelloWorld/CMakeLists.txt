add_executable(HelloWorld)
target_include_directories(HelloWorld
    PUBLIC ${RADCPP_ROOT}
)
list(APPEND HelloWorld_SOURCES
    ContainerTest.cpp
    FloatTest.cpp
    IntegerTest.cpp
    MemoryTest.cpp
    main.cpp
)
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${HelloWorld_SOURCES})
target_sources(HelloWorld PRIVATE ${HelloWorld_SOURCES})
find_package(GTest CONFIG REQUIRED)
target_link_libraries(HelloWorld
    PRIVATE rad
    PRIVATE GTest::gtest GTest::gtest_main GTest::gmock GTest::gmock_main
)
target_compile_features(HelloWorld PUBLIC cxx_std_20)

set_target_properties(HelloWorld PROPERTIES
    FOLDER "tests"
)

install(
    TARGETS HelloWorld
    COMPONENT radcpp.tests
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
