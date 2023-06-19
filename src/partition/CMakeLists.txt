
aux_source_directory(. LIB_PART)
# 生成链接库
add_library (part 
    ${LIB_PART} 
)
target_link_libraries(part PUBLIC metis GKlib)

if(CUSTOM_METIS_INSTALL)
    #设置METIS自定义安装位置
    if(${CMAKE_VERSION} VERSION_GREATER "3.13.0")
        target_link_directories(part PUBLIC ${GKLIB_INSTALL_LOCATION}/lib)
        target_link_directories(part PUBLIC ${METIS_INSTALL_LOCATION}/lib)
    endif()
    target_include_directories(part PUBLIC ${GKLIB_INSTALL_LOCATION}/include)
    target_include_directories(part PUBLIC ${METIS_INSTALL_LOCATION}/include)
endif()
