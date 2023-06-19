# CMake 最低版本号要求
cmake_minimum_required (VERSION 3.10)
message("CMAKE_VERSION:${CMAKE_VERSION}")

# 项目名
project (mgbrain)

find_package(OpenMP REQUIRED)

if ( OpenMP_FOUND )

  message ( STATUS "OpenMP_CXX_FOUND  = ${OpenMP_CXX_FOUND}" )
  message ( STATUS "OpenMP_CXX_FLAGS  = ${OpenMP_CXX_FLAGS}" )
  message ( STATUS "OpenMP_CXX_LIBRARIES  = ${OpenMP_CXX_LIBRARIES}" )
  message ( STATUS "OpenMP_CXX_INTERFACE_LINK_OPTIONS  = ${OpenMP_CXX_INTERFACE_LINK_OPTIONS}" )

endif ()

# CMake 指定计算能力
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 61)
endif()

#设置METIS自定义安装位置
set(CUSTOM_METIS_INSTALL "YES")
if(${CMAKE_VERSION} VERSION_GREATER "3.10.2")
set(METIS_INSTALL_LOCATION ~/envs/metis)
set(GKLIB_INSTALL_LOCATION ~/envs/gklib)
else()
set(METIS_INSTALL_LOCATION ~/zfz/env/metis)
set(GKLIB_INSTALL_LOCATION ~/zfz/env/gklib)
endif()



if(${CMAKE_VERSION} VERSION_GREATER "3.16.0")

    enable_language(CUDA)
    message("CUDA:${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
    message("NVCC:${CMAKE_CUDA_COMPILER}")
    message("CUDART:${CUDA_LIBRARIES}")

elseif(${CMAKE_VERSION} VERSION_GREATER "3.13.0")
    find_package(CUDA REQUIRED)
    message("CUDA:${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
    message("NVCC:${CMAKE_CUDA_COMPILER}")
    message("CUDART:${CUDA_LIBRARIES}")
else()
    find_package(CUDA REQUIRED)
    message("CUDA:${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
    message("NVCC:${CMAKE_CUDA_COMPILER}")
    message("CUDART:${CUDA_LIBRARIES}")
    include_directories(${CUDA_INCLUDE_DIRS})
    link_directories(${GKLIB_INSTALL_LOCATION}/lib)
    link_directories(${METIS_INSTALL_LOCATION}/lib)
endif()

add_subdirectory(src)
add_subdirectory(bench)
add_subdirectory(examples)