
add_executable(example example.cpp)
target_include_directories(example PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(example PUBLIC mgbrain)

add_executable(train_mnist train_mnist.cpp)
target_include_directories(train_mnist PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(train_mnist PUBLIC mgbrain)