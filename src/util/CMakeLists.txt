# 查找当前目录下的所有源文件
# 并将名称保存到 LIB_MODEL 变量
# aux_source_directory(. LIB_UTIL)
FILE(GLOB SRC_UTIL "*.cpp" "*.h" )
# 生成链接库
add_library (util ${SRC_UTIL})