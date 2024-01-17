# 查找当前目录下的所有源文件
# 并将名称保存到 LIB_MODEL 变量

## 单CPU源文件
FILE(GLOB SRC_SCSIM "scsim/*.cpp" "scsim/*.h")

## 多GPU源文件
FILE(GLOB SRC_MCSIM "mcsim/*.cpp" "mcsim/*.h")

## 单GPU源文件
# FILE(GLOB SRC_SGSIM "sgsim/*.cpp" "sgsim/*.h")
# FILE(GLOB SRC_SGSIM_CU "sgsim/sgnet/*.cu" "sgsim/sgnet/*.cuh")

## 多GPU源文件
FILE(GLOB SRC_MGSIM "mgsim/adj/*.cpp" "mgsim/adj/*.h")
FILE(GLOB SRC_MGSIM_CU "mgsim/adj/*.cu" "mgsim/adj/*.cuh")

set(CUDA_FAST_FLAG "-std=c++11;-O3;-arch=sm_75;-lcufft")
if(${CMAKE_VERSION} VERSION_GREATER "3.16.0")
    # 生成链接库
    add_library(mgcuda
        cudamem.cuh
        ${SRC_MGSIM_CU} 
        # ${SRC_SGSIM_CU}
    )
    # target_compile_options(mgcuda ${CUDA_FAST_FLAG})
    target_include_directories(mgcuda PUBLIC /usr/local/cuda-11.4/include)
    add_library(simulator
        sim.h
        struct.h
        ustruct.h
        # ${SRC_SCSIM}
        # ${SRC_MCSIM}
        # ${SRC_SGSIM}
        ${SRC_MGSIM}
    )
    target_link_libraries(simulator PUBLIC nvidia-ml ${OpenMP_CXX_LIBRARIES} mgcuda rt)
    target_compile_options(simulator PRIVATE ${OpenMP_CXX_FLAGS} )
    target_include_directories(simulator PUBLIC /usr/local/cuda-11.4/include)
else()
    # 生成链接库
    cuda_add_library(mgcuda
        cudamem.cuh
        ${SRC_MGSIM_CU} 
        # ${SRC_SGSIM_CU}
    )
    add_library(simulator
        sim.h
        structure.h
        structure.cpp
        # ${SRC_SCSIM}
        # ${SRC_MCSIM}
        # ${SRC_SGSIM}
        ${SRC_MGSIM}
    )
    target_link_libraries(simulator PUBLIC nvidia-ml mgcuda gomp rt)
    target_compile_options(simulator PUBLIC ${OpenMP_CXX_FLAGS})
    # if (OpenMP_CXX_FOUND)
    #     # target_link_libraries(simulator PUBLIC OpenMP::OpenMP_CXX)
    #     target_link_libraries(simulator PUBLIC nvidia-ml rt mgcuda gomp)
    #     target_compile_options(simulator PUBLIC ${OpenMP_CXX_FLAGS})
    # endif ()

endif()



