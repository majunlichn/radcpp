add_executable(HelloWorld)
target_include_directories(HelloWorld
    PUBLIC ${RADCPP_ROOT}
)
target_sources(HelloWorld PRIVATE
    main.cpp
)
find_package(Threads REQUIRED)
target_link_libraries(HelloWorld
    rad
)
target_compile_features(HelloWorld PUBLIC cxx_std_20)
