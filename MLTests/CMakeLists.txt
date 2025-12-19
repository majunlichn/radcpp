add_executable(MLTests)

target_include_directories(MLTests
    PUBLIC ${RADCPP_ROOT}
)

list(APPEND MLTests_SOURCES
    TensorOpElementWiseBinary.cpp
    main.cpp
)
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${MLTests_SOURCES})
target_sources(MLTests PRIVATE ${MLTests_SOURCES})

find_package(GTest CONFIG REQUIRED)
target_link_libraries(MLTests
    PRIVATE MLCore
    PRIVATE GTest::gtest GTest::gtest_main GTest::gmock GTest::gmock_main
)

install(
    TARGETS MLTests
    COMPONENT radcpp.MLTests
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
