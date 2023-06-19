
add_executable(example example.cpp)
target_include_directories(example PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(example PUBLIC mgbrain)