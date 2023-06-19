

# 查找当前目录下的所有源文件
# 并将名称保存到 DIR_SRCS 变量
aux_source_directory(. DIR_SRCS)

# 添加sub1子目录
add_subdirectory(util)
add_subdirectory(model)
add_subdirectory(partition)
add_subdirectory(simulator)

add_library(mgbrain macro.h macro.cpp mgbrain.h)
target_link_libraries(mgbrain PUBLIC model part simulator util)

# 指定生成目标
add_executable(main main.cpp)

# 添加链接库
target_link_libraries(main mgbrain)

