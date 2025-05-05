add_executable(HelloVulkan)

target_include_directories(HelloVulkan
    PUBLIC ${RADCPP_ROOT}
)
list(APPEND HelloVulkan_SOURCES
    main.cpp
    TestTensor.cpp
)

source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${HelloVulkan_SOURCES})
target_sources(HelloVulkan PRIVATE ${HelloVulkan_SOURCES})
find_package(GTest CONFIG REQUIRED)
target_link_libraries(HelloVulkan
    PRIVATE vkpp
    PRIVATE GTest::gtest GTest::gtest_main GTest::gmock GTest::gmock_main
)
target_compile_features(HelloVulkan PUBLIC cxx_std_20)

set_target_properties(HelloVulkan PROPERTIES
    FOLDER "tests"
)

install(
    TARGETS HelloVulkan
    COMPONENT radcpp.tests
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
