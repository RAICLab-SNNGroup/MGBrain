find_package(Catch2 REQUIRED)

include(CTest)
include(Catch)

#添加测试文件
add_executable(tmodel test_model.cpp)
target_include_directories(tmodel PRIVATE ${CMAKE_SOURCE_DIR}/model)
target_link_libraries(tmodel PRIVATE Catch2::Catch2WithMain model)
