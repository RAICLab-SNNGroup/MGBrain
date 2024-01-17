# 查找当前目录下的所有源文件
# 并将名称保存到 LIB_MODEL 变量

FILE(GLOB SRC_MODEL "./*.cpp" "./*.h")
FILE(GLOB SRC_MODEL_CU "./*.cu" "./*.cuh")
# 生成链接库
set(CUDA_FAST_FLAG "-std=c++11;-O3;-arch=sm_75;-lcufft")
if(${CMAKE_VERSION} VERSION_GREATER "3.16.0")
    # 生成链接库
    add_library(modelcu
        ${SRC_MODEL_CU} 
    )
    target_include_directories(modelcu PUBLIC /usr/local/cuda-11.4/include)
    add_library(model
        ${SRC_MODEL}
    )
    target_link_libraries(model PUBLIC nvidia-ml ${OpenMP_CXX_LIBRARIES} modelcu rt)
    target_compile_options(model PRIVATE ${OpenMP_CXX_FLAGS} )
    target_include_directories(model PUBLIC /usr/local/cuda-11.4/include)
else()
    # 生成链接库
    cuda_add_library(modelcu
        ${SRC_MODEL_CU}
    )
    add_library(
        ${SRC_MODEL}
    )
    target_link_libraries(model PUBLIC nvidia-ml modelcu gomp rt)
    target_compile_options(model PUBLIC ${OpenMP_CXX_FLAGS})
endif()
