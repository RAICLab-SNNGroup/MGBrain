# 查找当前目录下的所有源文件
# 并将名称保存到 LIB_MODEL 变量
aux_source_directory(. LIB_MODEL)
# 生成链接库
add_library (model 
    ${LIB_MODEL}
)
target_compile_options(model PUBLIC ${OpenMP_CXX_FLAGS})
target_link_libraries(model PUBLIC gomp)
