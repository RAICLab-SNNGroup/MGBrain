#pragma once
#include "../mgstruct.h"
namespace MGBrain
{

    /// @brief 多区域子网络
    struct GSubNetUVA
    {
        /// @brief 子网络号
        int id;
        /// @brief 神经元数量
        int neus_size;
        /// @brief 神经元数据块
        NEUBlock neus;
        /// @brief 内部突触数量
        int syns_size;
        /// @brief 内部突触数据块
        SYNBlock syns;
        /// @brief 外部突触数量
        int outs_size;
        /// @brief 外部突触数据块
        OUTSYNBlock *outs;
        /// @brief 其他设备外部突触连接的数量
        int ins_size;
    };
    struct GSubNetAddrsUVA{
        // // 用于异步传输的流
        // cudaStream_t* outs_streams;
        // 网络的脉冲接受地址
        real *tar_exc_buffer;
        // 网络的脉冲接受地址
        real *tar_inh_buffer;
        // 网络的神经元状态地址
        bool *tar_fired_buffer;
        // 网络的神经元状态地址
        int *tar_last_fired_buffer;
    };
    struct MultiNetUVA{
        //GPU内存中子网络
        std::vector<GSubNetUVA*> gnets;
        //主机内存中子网络
        std::vector<GSubNetUVA*> cnets;
        //子网络的数据地址
        std::vector<GSubNetAddrsUVA*> addrs;
        //GPU内存中的子网络数据地址
        std::vector<std::vector<GSubNetAddrsUVA*>> gaddrs;
        std::vector<std::vector<cudaStream_t>> gstreams;
        int max_delay;
        int blocksize;
    };
    /// @brief 初始化子网络的神经元空间（主机内存）
    void initGSubNetUVANeus(GSubNetUVA *cnet,int max_delay);
    /// @brief 初始化子网络的内部突触空间（主机内存）
    void initGSubNetUVASyns(GSubNetUVA *cnet);
    /// @brief 初始化子网络的外部突触空间（主机内存）
    void initGSubNetUVAOutSyns(SYNBlock *block, int num);
    /// @brief 释放子网络空间（主机内存）
    void freeGSubNetUVA(GSubNetUVA *cnet);
    /// @brief 释放子网络空间（GPU内存）
    void gpuFreeGSubNetUVA(GSubNetUVA *gnet);


    /// @brief 拷贝到GPU内存
    GSubNetUVA *copy_subnetuva_gpu(GSubNetUVA *cnet,GSubNetAddrsUVA* addr,int max_delay);
    /// @brief 拷贝常量到GPU常量区
    void copy_constsuva_gpu(int max_delay,LIFConsts* lifconst=nullptr,STDPConsts* stdpconst=nullptr);
    
    
    /// @brief CUDA核函数:仿真神经元
    /// @param net 子网络
    /// @param step 当前时间片
    __global__ void mgsimNeusUVA(GSubNetUVA *net, int step);
    /// @brief CUDA核函数:仿真子网络内部突触
    /// @param net 子网络
    /// @param step 当前时间片
    __global__ void mgsimSynsUVA(GSubNetUVA *net, int step);
    /// @brief CUDA核函数:仿真外部突触，也即脉冲同步操作
    /// @param net 子网络
    /// @param index 邻接设备索引
    /// @param step 当前时间片
    __global__ void mgsimOutsUVA(GSubNetUVA *net, int index, int step,GSubNetAddrsUVA* tar_addrs);
    /// @brief 仿真单个时间片
    void mgsimStepUVA(GSubNetUVA *gnet, GSubNetUVA *cnet, int step, int blocksize,std::vector<GSubNetAddrsUVA*>addrs,std::vector<cudaStream_t> streams);
};